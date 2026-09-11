"""Lazy, save-scoped SQLite conversation memory.

One database file is created per world UUID.  The database still stores the
full world/player/NPC key on every row so accidental cross-scope reads are
guarded by SQL predicates as well as by the file boundary.  No per-NPC tables
are created.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbrainz.retrieval_dictionary import DEFAULT_TOKEN_EXPANSIONS

from .policy import is_context_eligible
from .types import (
    ConversationSession,
    ConversationTurn,
    DaySynopsis,
    MemoryEpisode,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    MemoryVisibility,
    RetrievalMatch,
    StructuredFact,
    TurnWriteResult,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{2,64}")
_MAX_TEXT = 12000
_MAX_UI_MEMORY_PAGE = 200
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _object(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _safe_text(value: Any, limit: int = _MAX_TEXT) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _compact_strings(values: Any, limit: int, item_limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set)):
        return ()
    output: list[str] = []
    for value in values:
        text = _safe_text(value, item_limit)
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return tuple(output)


def _visibility(
    value: Any,
    default: MemoryVisibility = MemoryVisibility.PRIVATE,
) -> MemoryVisibility:
    try:
        return value if isinstance(value, MemoryVisibility) else MemoryVisibility(str(value))
    except ValueError:
        return default


def _visibility_value(value: Any) -> str:
    return _visibility(value).value


def _json_list(value: Any, limit: int = 32, item_limit: int = 256) -> list[str]:
    return list(_compact_strings(value, limit, item_limit))


def memory_root_for_settings(settings: Any) -> Path:
    """Resolve the save-memory directory using the same rules as the service."""

    configured_root = getattr(settings, "memory_root", None)
    if configured_root:
        return Path(configured_root).expanduser()
    database_path = getattr(settings, "database_path", None)
    if database_path:
        return Path(database_path).expanduser().parent / "memory"
    return Path.home() / ".config" / "PBrainZ" / "memory"


class SQLiteMemoryStore:
    """MemoryStore/MemoryRetriever implementation backed by one world DB."""

    def __init__(self, root: str | Path, world_uuid: str) -> None:
        self.root = Path(root).expanduser()
        self.world_uuid = _safe_text(world_uuid, 256)
        if not self.world_uuid:
            raise ValueError("world_uuid is required")
        digest = hashlib.sha256(self.world_uuid.encode("utf-8")).hexdigest()[:24]
        self.path = self.root / f"world-{digest}.db"
        self._initialized = False
        self.fts_enabled = False

    @classmethod
    def list_worlds(cls, root: str | Path, limit: int = 64) -> list[dict[str, Any]]:
        """List existing world databases without creating or migrating files."""

        memory_root = Path(root).expanduser()
        if not memory_root.is_dir():
            return []
        bounded = max(1, min(int(limit), 128))
        try:
            paths = sorted(
                memory_root.glob("world-*.db"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:bounded]
        except OSError:
            return []
        worlds: list[dict[str, Any]] = []
        for path in paths:
            try:
                with sqlite3.connect(path, timeout=1) as connection:
                    connection.row_factory = sqlite3.Row
                    metadata = connection.execute(
                        "SELECT value FROM world_metadata WHERE key = 'world_uuid'"
                    ).fetchone()
                    if metadata is None or not str(metadata["value"]).strip():
                        continue
                    world_uuid = str(metadata["value"])
                    counts = {
                        name: cls._table_count(connection, name)
                        for name in (
                            "memories",
                            "conversation_sessions",
                            "conversation_turns",
                            "conversation_episodes",
                            "structured_facts",
                        )
                    }
                worlds.append(
                    {
                        "world_uuid": world_uuid,
                        "path": str(path),
                        "memory_count": counts["memories"],
                        "session_count": counts["conversation_sessions"],
                        "turn_count": counts["conversation_turns"],
                        "episode_count": counts["conversation_episodes"],
                        "fact_count": counts["structured_facts"],
                    }
                )
            except (OSError, sqlite3.Error):
                continue
        return worlds

    @staticmethod
    def _table_count(connection: sqlite3.Connection, table: str) -> int:
        try:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.Error:
            return 0

    def initialize(self) -> None:
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;
                CREATE TABLE IF NOT EXISTS world_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_entities (
                    world_uuid TEXT NOT NULL,
                    entity_uuid TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'memory',
                    PRIMARY KEY(world_uuid, entity_uuid, entity_kind)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_entities_lookup
                    ON memory_entities(world_uuid, entity_uuid, entity_kind);
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_activity_at TEXT NOT NULL,
                    ended_at TEXT,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    summary TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_scope
                    ON conversation_sessions(world_uuid, player_uuid, npc_uuid);
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    message_id TEXT,
                    game_day INTEGER,
                    world_age_hours REAL,
                    speaker_uuid TEXT,
                    speaker_name TEXT,
                    speaker_kind TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON conversation_turns(session_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_turns_scope
                    ON conversation_turns(world_uuid, player_uuid, npc_uuid, id DESC);
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    session_id TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    state TEXT NOT NULL DEFAULT 'active',
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_recalled_at TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    visibility TEXT NOT NULL DEFAULT 'PRIVATE',
                    game_day INTEGER,
                    participants_json TEXT NOT NULL DEFAULT '[]',
                    topic_tags_json TEXT NOT NULL DEFAULT '[]',
                    entity_refs_json TEXT NOT NULL DEFAULT '[]',
                    transcript_ref TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                    ON memories(world_uuid, player_uuid, npc_uuid, active);
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                    ON memories(world_uuid, player_uuid, npc_uuid, importance DESC);
                CREATE TABLE IF NOT EXISTS commitments (
                    commitment_id TEXT PRIMARY KEY,
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    importance REAL NOT NULL DEFAULT 0.7,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_commitments_scope
                    ON commitments(world_uuid, player_uuid, npc_uuid, status);
                CREATE TABLE IF NOT EXISTS conversation_episodes (
                    episode_id TEXT PRIMARY KEY,
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    game_day INTEGER,
                    participants_json TEXT NOT NULL DEFAULT '[]',
                    witnesses_json TEXT NOT NULL DEFAULT '[]',
                    topic_tags_json TEXT NOT NULL DEFAULT '[]',
                    entity_refs_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL,
                    key_facts_json TEXT NOT NULL DEFAULT '[]',
                    emotional_tone TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    visibility TEXT NOT NULL DEFAULT 'PUBLIC',
                    transcript_ref TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_scope
                    ON conversation_episodes(world_uuid, player_uuid, npc_uuid, active);
                CREATE INDEX IF NOT EXISTS idx_episodes_visibility
                    ON conversation_episodes(world_uuid, player_uuid, visibility, active);
                CREATE TABLE IF NOT EXISTS day_synopses (
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    game_day INTEGER NOT NULL,
                    synopsis TEXT NOT NULL DEFAULT '',
                    commitments_json TEXT NOT NULL DEFAULT '[]',
                    plans_json TEXT NOT NULL DEFAULT '[]',
                    claims_json TEXT NOT NULL DEFAULT '[]',
                    unresolved_topics_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(world_uuid, player_uuid, npc_uuid, game_day)
                );
                CREATE TABLE IF NOT EXISTS structured_facts (
                    fact_id TEXT PRIMARY KEY,
                    world_uuid TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    npc_uuid TEXT NOT NULL,
                    conversation_id TEXT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    subject_uuid TEXT,
                    source_uuid TEXT,
                    truth_status TEXT NOT NULL DEFAULT 'unverified',
                    visibility TEXT NOT NULL DEFAULT 'PUBLIC',
                    game_day INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    importance REAL NOT NULL DEFAULT 0.5,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_facts_scope
                    ON structured_facts(world_uuid, player_uuid, npc_uuid, active);
                """,
            )
            self._migrate_turn_columns(connection)
            self._migrate_memory_columns(connection)
            existing = connection.execute(
                "SELECT value FROM world_metadata WHERE key = 'world_uuid'"
            ).fetchone()
            if existing is not None and existing["value"] != self.world_uuid:
                raise ValueError("memory database belongs to a different world")
            connection.execute(
                "INSERT INTO world_metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO NOTHING",
                ("world_uuid", self.world_uuid),
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        world_uuid UNINDEXED,
                        player_uuid UNINDEXED,
                        npc_uuid UNINDEXED,
                        content,
                        tags
                    )
                    """
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False
            connection.commit()
        self._initialized = True
        if self.fts_enabled:
            self._rebuild_fts()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _migrate_turn_columns(connection: sqlite3.Connection) -> None:
        """Add canonical-message fields without rewriting existing save data."""
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversation_turns)")
        }
        additions = {
            "message_id": "TEXT",
            "game_day": "INTEGER",
            "world_age_hours": "REAL",
            "speaker_uuid": "TEXT",
            "speaker_name": "TEXT",
            "speaker_kind": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE conversation_turns ADD COLUMN {name} {definition}"
                )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_message_id "
            "ON conversation_turns(world_uuid, message_id) "
            "WHERE message_id IS NOT NULL AND message_id != ''"
        )

    @staticmethod
    def _migrate_memory_columns(connection: sqlite3.Connection) -> None:
        """Add visibility and retrieval metadata to pre-layer databases."""
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(memories)")
        }
        additions = {
            "visibility": "TEXT NOT NULL DEFAULT 'PRIVATE'",
            "game_day": "INTEGER",
            "participants_json": "TEXT NOT NULL DEFAULT '[]'",
            "topic_tags_json": "TEXT NOT NULL DEFAULT '[]'",
            "entity_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            "transcript_ref": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE memories ADD COLUMN {name} {definition}"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_visibility "
            "ON memories(world_uuid, player_uuid, visibility, active)"
        )

    def ensure_session(
        self,
        session_id: str,
        scope: MemoryScope,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession:
        self.initialize()
        session_id = _safe_text(session_id, 256) or uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT world_uuid, player_uuid, npc_uuid "
                "FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None and (
                existing["world_uuid"],
                existing["player_uuid"],
                existing["npc_uuid"],
            ) != (scope.world_uuid, scope.player_uuid, scope.npc_uuid):
                raise ValueError("conversation session belongs to a different memory scope")
            connection.execute(
                """
                INSERT INTO conversation_sessions(
                    session_id, world_uuid, player_uuid, npc_uuid,
                    started_at, last_activity_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_activity_at = excluded.last_activity_at,
                    ended_at = NULL,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                    timestamp,
                    timestamp,
                    _json(metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            connection.commit()
        return self._session(row)

    def upsert_entity(
        self,
        entity_uuid: str,
        display_name: str,
        *,
        entity_kind: str = "npc",
        source: str = "memory",
    ) -> None:
        """Remember a user-facing name inside this world database only."""

        entity_uuid = _safe_text(entity_uuid, 256)
        display_name = _safe_text(display_name, 256)
        entity_kind = _safe_text(entity_kind, 32) or "npc"
        source = _safe_text(source, 64) or "memory"
        if not entity_uuid or not display_name:
            return
        self.initialize()
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._upsert_entity_connection(
                connection,
                entity_uuid,
                display_name,
                entity_kind,
                source,
                timestamp,
            )
            connection.commit()

    def _upsert_entity_connection(
        self,
        connection: sqlite3.Connection,
        entity_uuid: str,
        display_name: str,
        entity_kind: str,
        source: str,
        timestamp: str,
    ) -> None:
        entity_uuid = _safe_text(entity_uuid, 256)
        display_name = _safe_text(display_name, 256)
        if not entity_uuid or not display_name:
            return
        connection.execute(
            """
            INSERT INTO memory_entities(
                world_uuid, entity_uuid, entity_kind, display_name,
                last_seen_at, source
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(world_uuid, entity_uuid, entity_kind) DO UPDATE SET
                display_name = excluded.display_name,
                last_seen_at = excluded.last_seen_at,
                source = excluded.source
            """,
            (
                self.world_uuid,
                entity_uuid,
                _safe_text(entity_kind, 32) or "npc",
                display_name,
                timestamp,
                _safe_text(source, 64) or "memory",
            ),
        )

    def add_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        return self._write_turn(
            session_id,
            scope,
            role,
            content,
            message_id=None,
            metadata=metadata,
        ).turn

    def record_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        *,
        message_id: str,
        metadata: dict[str, Any] | None = None,
        game_day: int | None = None,
        world_age_hours: float | None = None,
        speaker_uuid: str | None = None,
        speaker_name: str | None = None,
        speaker_kind: str | None = None,
    ) -> TurnWriteResult:
        message_id = _safe_text(message_id, 256)
        if not message_id:
            raise ValueError("conversation message_id is required")
        return self._write_turn(
            session_id,
            scope,
            role,
            content,
            message_id=message_id,
            metadata=metadata,
            game_day=game_day,
            world_age_hours=world_age_hours,
            speaker_uuid=speaker_uuid,
            speaker_name=speaker_name,
            speaker_kind=speaker_kind,
        )

    def _write_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        *,
        message_id: str | None,
        metadata: dict[str, Any] | None = None,
        game_day: int | None = None,
        world_age_hours: float | None = None,
        speaker_uuid: str | None = None,
        speaker_name: str | None = None,
        speaker_kind: str | None = None,
    ) -> TurnWriteResult:
        self.initialize()
        content = _safe_text(content)
        if not content:
            raise ValueError("conversation turn content is required")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if message_id:
                existing = connection.execute(
                    "SELECT * FROM conversation_turns "
                    "WHERE world_uuid = ? AND message_id = ?",
                    (scope.world_uuid, message_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["player_uuid"],
                        existing["npc_uuid"],
                    ) != (scope.player_uuid, scope.npc_uuid):
                        raise ValueError(
                            "conversation message ID belongs to a different memory scope"
                        )
                    return TurnWriteResult(self._turn(existing), duplicate=True)
            row = connection.execute(
                "SELECT turn_count FROM conversation_sessions WHERE session_id = ? "
                "AND world_uuid = ? AND player_uuid = ? AND npc_uuid = ?",
                (session_id, scope.world_uuid, scope.player_uuid, scope.npc_uuid),
            ).fetchone()
            if row is None:
                raise ValueError("conversation session does not match memory scope")
            turn_index = int(row["turn_count"]) + 1
            default_kind = "player" if str(role).casefold() == "user" else "npc"
            self._upsert_entity_connection(
                connection,
                speaker_uuid
                or (scope.player_uuid if default_kind == "player" else scope.npc_uuid),
                speaker_name or "",
                _safe_text(speaker_kind, 32) or default_kind,
                "conversation",
                timestamp,
            )
            for participant in (metadata or {}).get("participants", ()):
                if not isinstance(participant, dict):
                    continue
                self._upsert_entity_connection(
                    connection,
                    participant.get("id") or participant.get("speakerID") or "",
                    participant.get("name")
                    or participant.get("speakerName")
                    or "",
                    participant.get("kind")
                    or participant.get("speakerKind")
                    or "npc",
                    "conversation",
                    timestamp,
                )
            connection.execute(
                """
                INSERT INTO conversation_turns(
                    session_id, world_uuid, player_uuid, npc_uuid,
                    turn_index, role, content, created_at, metadata_json,
                    message_id, game_day, world_age_hours,
                    speaker_uuid, speaker_name, speaker_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                    turn_index,
                    _safe_text(role, 32),
                    content,
                    timestamp,
                    _json(metadata or {}),
                    message_id,
                    int(game_day) if game_day is not None else None,
                    float(world_age_hours) if world_age_hours is not None else None,
                    _safe_text(speaker_uuid, 256) or None,
                    _safe_text(speaker_name, 256) or None,
                    _safe_text(speaker_kind, 32) or None,
                ),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET turn_count = ?, last_activity_at = ?, ended_at = NULL
                WHERE session_id = ?
                """,
                (turn_index, timestamp, session_id),
            )
            connection.commit()
        return TurnWriteResult(
            ConversationTurn(
                role=role,
                content=content,
                created_at=timestamp,
                turn_index=turn_index,
                metadata=metadata or {},
                message_id=message_id,
                game_day=int(game_day) if game_day is not None else None,
                world_age_hours=(
                    float(world_age_hours) if world_age_hours is not None else None
                ),
                speaker_uuid=_safe_text(speaker_uuid, 256) or None,
                speaker_name=_safe_text(speaker_name, 256) or None,
                speaker_kind=_safe_text(speaker_kind, 32) or None,
            )
        )

    def recent_turns(
        self,
        session_id: str,
        scope: MemoryScope,
        limit: int = 8,
    ) -> list[ConversationTurn]:
        self.initialize()
        bounded = max(1, min(int(limit), 64))
        fetch_limit = min(256, max(bounded, bounded * 4))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at, turn_index, metadata_json,
                       message_id, game_day, world_age_hours,
                       speaker_uuid, speaker_name, speaker_kind
                FROM conversation_turns
                WHERE session_id = ? AND world_uuid = ?
                  AND player_uuid = ? AND npc_uuid = ?
                ORDER BY id DESC LIMIT ?
                """,
                (
                    session_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                    fetch_limit,
                ),
            ).fetchall()
        turns = [self._turn(row) for row in reversed(rows)]
        eligible = [
            turn
            for turn in turns
            if is_context_eligible(
                turn.content,
                role=turn.role,
                metadata=turn.metadata,
            )
        ]
        return eligible[-bounded:]

    def search_turns(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 4,
    ) -> list[ConversationTurn]:
        """Find a small, scope-isolated recall window across past sessions."""
        self.initialize()
        bounded = max(1, min(int(limit), 8))
        fetch_limit = min(32, max(bounded, bounded * 4))
        tokens = [
            token.casefold()
            for token in _TOKEN_RE.findall(_safe_text(query, 2000))
            if len(token) >= 3
        ][:12]
        if not tokens:
            return []
        clauses = " OR ".join("LOWER(content) LIKE ?" for _ in tokens)
        like_args = tuple(f"%{token}%" for token in tokens)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT role, content, created_at, turn_index, metadata_json,
                       message_id, game_day, world_age_hours,
                       speaker_uuid, speaker_name, speaker_kind
                FROM conversation_turns
                WHERE world_uuid = ? AND player_uuid = ? AND npc_uuid = ?
                  AND ({clauses})
                ORDER BY id DESC LIMIT ?
                """,
                (
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                    *like_args,
                    fetch_limit,
                ),
            ).fetchall()
        turns = [self._turn(row) for row in reversed(rows)]
        eligible = [
            turn
            for turn in turns
            if is_context_eligible(
                turn.content,
                role=turn.role,
                metadata=turn.metadata,
            )
        ]
        return eligible[-bounded:]

    def remember(self, memory: MemoryRecord) -> MemoryRecord:
        self.initialize()
        timestamp = _now()
        content = _safe_text(memory.content)
        if not content:
            raise ValueError("memory content is required")
        tags = tuple(
            dict.fromkeys(
                _safe_text(tag, 64) for tag in memory.tags if _safe_text(tag, 64)
            )
        )
        importance = max(0.0, min(1.0, float(memory.importance)))
        memory_id = memory.memory_id or uuid.uuid4().hex
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT world_uuid, player_uuid, npc_uuid FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if existing is not None and (
                existing["world_uuid"],
                existing["player_uuid"],
                existing["npc_uuid"],
            ) != (
                memory.scope.world_uuid,
                memory.scope.player_uuid,
                memory.scope.npc_uuid,
            ):
                raise ValueError("memory ID belongs to a different memory scope")
            connection.execute(
                """
                INSERT INTO memories(
                    memory_id, world_uuid, player_uuid, npc_uuid, session_id,
                    memory_type, content, tags_json, importance, state,
                    provenance_json, created_at, updated_at, last_recalled_at, active,
                    visibility, game_day, participants_json, topic_tags_json,
                    entity_refs_json, transcript_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    content = excluded.content,
                    tags_json = excluded.tags_json,
                    importance = excluded.importance,
                    state = excluded.state,
                    provenance_json = excluded.provenance_json,
                    updated_at = excluded.updated_at,
                    last_recalled_at = excluded.last_recalled_at,
                    active = excluded.active,
                    visibility = excluded.visibility,
                    game_day = excluded.game_day,
                    participants_json = excluded.participants_json,
                    topic_tags_json = excluded.topic_tags_json,
                    entity_refs_json = excluded.entity_refs_json,
                    transcript_ref = excluded.transcript_ref
                """,
                (
                    memory_id,
                    memory.scope.world_uuid,
                    memory.scope.player_uuid,
                    memory.scope.npc_uuid,
                    memory.session_id,
                    memory.memory_type.value,
                    content,
                    _json(tags),
                    importance,
                    _safe_text(memory.state, 32) or "active",
                    _json(memory.provenance),
                    memory.created_at or timestamp,
                    timestamp,
                    memory.last_recalled_at,
                    int(memory.active),
                    _visibility_value(memory.visibility),
                    int(memory.game_day) if memory.game_day is not None else None,
                    _json(_compact_strings(memory.participants, 32, 32)),
                    _json(_compact_strings(memory.topic_tags, 32, 64)),
                    _json(_compact_strings(memory.entity_refs, 32, 128)),
                    _safe_text(memory.transcript_ref, 256) or None,
                ),
            )
            if self.fts_enabled:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
                connection.execute(
                    "INSERT INTO memory_fts("
                    "memory_id, world_uuid, player_uuid, npc_uuid, content, tags"
                    ") "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        memory.scope.world_uuid,
                        memory.scope.player_uuid,
                        memory.scope.npc_uuid,
                        content,
                        " ".join(tags),
                    ),
                )
            connection.commit()
        return MemoryRecord(
            memory_id=memory_id,
            scope=memory.scope,
            memory_type=memory.memory_type,
            content=content,
            tags=tags,
            importance=importance,
            state=memory.state,
            provenance=memory.provenance,
            session_id=memory.session_id,
            created_at=memory.created_at or timestamp,
            updated_at=timestamp,
            last_recalled_at=memory.last_recalled_at,
            active=memory.active,
            visibility=_visibility(memory.visibility),
            game_day=memory.game_day,
            participants=tuple(memory.participants),
            topic_tags=tuple(memory.topic_tags),
            entity_refs=tuple(memory.entity_refs),
            transcript_ref=memory.transcript_ref,
        )

    def add_commitment(
        self,
        scope: MemoryScope,
        content: str,
        *,
        session_id: str | None = None,
        importance: float = 0.8,
        provenance: dict[str, Any] | None = None,
        commitment_id: str | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            memory_id=commitment_id or uuid.uuid4().hex,
            scope=scope,
            memory_type=MemoryType.COMMITMENT,
            content=content,
            tags=("commitment",),
            importance=importance,
            provenance=provenance or {},
            session_id=session_id,
        )
        saved = self.remember(record)
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO commitments(
                    commitment_id, world_uuid, player_uuid, npc_uuid, session_id,
                    content, status, importance, provenance_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(commitment_id) DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    provenance_json = excluded.provenance_json,
                    updated_at = excluded.updated_at
                """,
                (
                    saved.memory_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                    session_id,
                    saved.content,
                    saved.importance,
                    _json(provenance or {}),
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return saved

    def remember_hearsay(
        self,
        scope: MemoryScope,
        content: str,
        *,
        source_npc_uuid: str,
        subject_npc_uuid: str | None = None,
        session_id: str | None = None,
        importance: float = 0.45,
    ) -> MemoryRecord:
        """Store a claim as hearsay, never as authoritative world fact."""

        return self.remember(
            MemoryRecord(
                memory_id=uuid.uuid4().hex,
                scope=scope,
                memory_type=MemoryType.HEARSAY,
                content=content,
                tags=("hearsay", "claim"),
                importance=importance,
                provenance={
                    "source": "npc_claim",
                    "source_npc_uuid": str(source_npc_uuid),
                    "subject_npc_uuid": str(subject_npc_uuid or ""),
                    "session_id": session_id,
                },
                session_id=session_id,
            )
        )

    def remember_episode(self, episode: MemoryEpisode) -> MemoryEpisode:
        """Upsert one bounded, searchable conversation/event episode."""
        self.initialize()
        summary = _safe_text(episode.summary, 4000)
        if not summary:
            raise ValueError("episode summary is required")
        timestamp = _now()
        participants = _compact_strings(episode.participants, 32, 256)
        witnesses = _compact_strings(episode.witnesses, 32, 256)
        topics = _compact_strings(episode.topic_tags, 32, 64)
        entities = _compact_strings(episode.entity_refs, 32, 128)
        facts = _compact_strings(episode.key_facts, 16, 600)
        importance = max(0.0, min(1.0, float(episode.importance)))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_episodes(
                    episode_id, world_uuid, player_uuid, npc_uuid, conversation_id,
                    game_day, participants_json, witnesses_json, topic_tags_json,
                    entity_refs_json, summary, key_facts_json, emotional_tone,
                    importance, visibility, transcript_ref, created_at, updated_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(episode_id) DO UPDATE SET
                    game_day = excluded.game_day,
                    participants_json = excluded.participants_json,
                    witnesses_json = excluded.witnesses_json,
                    topic_tags_json = excluded.topic_tags_json,
                    entity_refs_json = excluded.entity_refs_json,
                    summary = excluded.summary,
                    key_facts_json = excluded.key_facts_json,
                    emotional_tone = excluded.emotional_tone,
                    importance = excluded.importance,
                    visibility = excluded.visibility,
                    transcript_ref = excluded.transcript_ref,
                    updated_at = excluded.updated_at,
                    active = 1
                """,
                (
                    _safe_text(episode.episode_id, 256),
                    episode.scope.world_uuid,
                    episode.scope.player_uuid,
                    episode.scope.npc_uuid,
                    _safe_text(episode.conversation_id, 256),
                    episode.game_day,
                    _json(participants),
                    _json(witnesses),
                    _json(topics),
                    _json(entities),
                    summary,
                    _json(facts),
                    _safe_text(episode.emotional_tone, 128) or None,
                    importance,
                    _visibility_value(episode.visibility),
                    _safe_text(episode.transcript_ref, 256) or None,
                    episode.created_at or timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return MemoryEpisode(
            episode_id=_safe_text(episode.episode_id, 256),
            scope=episode.scope,
            conversation_id=_safe_text(episode.conversation_id, 256),
            game_day=episode.game_day,
            participants=participants,
            witnesses=witnesses,
            topic_tags=topics,
            entity_refs=entities,
            summary=summary,
            key_facts=facts,
            emotional_tone=_safe_text(episode.emotional_tone, 128) or None,
            importance=importance,
            visibility=_visibility(episode.visibility, MemoryVisibility.PUBLIC),
            transcript_ref=_safe_text(episode.transcript_ref, 256) or None,
            created_at=episode.created_at or timestamp,
            updated_at=timestamp,
        )

    def get_day_synopsis(self, scope: MemoryScope, game_day: int) -> DaySynopsis | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM day_synopses
                WHERE world_uuid = ? AND player_uuid = ? AND npc_uuid = ? AND game_day = ?
                """,
                (scope.world_uuid, scope.player_uuid, scope.npc_uuid, int(game_day)),
            ).fetchone()
        return self._day_synopsis(row, scope) if row else None

    def save_day_synopsis(self, synopsis: DaySynopsis) -> DaySynopsis:
        """Persist a compact current-day synopsis; raw turns remain separate."""
        self.initialize()
        timestamp = _now()
        synopsis_text = _safe_text(synopsis.synopsis, 2400)
        values = (
            synopsis.scope.world_uuid,
            synopsis.scope.player_uuid,
            synopsis.scope.npc_uuid,
            int(synopsis.game_day),
            synopsis_text,
            _json(_json_list(synopsis.commitments, 16, 600)),
            _json(_json_list(synopsis.plans, 16, 600)),
            _json(_json_list(synopsis.claims, 16, 600)),
            _json(_json_list(synopsis.unresolved_topics, 16, 300)),
            timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO day_synopses(
                    world_uuid, player_uuid, npc_uuid, game_day, synopsis,
                    commitments_json, plans_json, claims_json, unresolved_topics_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(world_uuid, player_uuid, npc_uuid, game_day) DO UPDATE SET
                    synopsis = excluded.synopsis,
                    commitments_json = excluded.commitments_json,
                    plans_json = excluded.plans_json,
                    claims_json = excluded.claims_json,
                    unresolved_topics_json = excluded.unresolved_topics_json,
                    updated_at = excluded.updated_at
                """,
                values,
            )
            connection.commit()
        return DaySynopsis(
            scope=synopsis.scope,
            game_day=int(synopsis.game_day),
            synopsis=synopsis_text,
            commitments=tuple(_json_list(synopsis.commitments, 16, 600)),
            plans=tuple(_json_list(synopsis.plans, 16, 600)),
            claims=tuple(_json_list(synopsis.claims, 16, 600)),
            unresolved_topics=tuple(_json_list(synopsis.unresolved_topics, 16, 300)),
            updated_at=timestamp,
        )

    def save_structured_fact(self, fact: StructuredFact) -> StructuredFact:
        self.initialize()
        content = _safe_text(fact.content, 1200)
        if not content:
            raise ValueError("structured fact content is required")
        timestamp = _now()
        importance = max(0.0, min(1.0, float(fact.importance)))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO structured_facts(
                    fact_id, world_uuid, player_uuid, npc_uuid, conversation_id,
                    kind, content, subject_uuid, source_uuid, truth_status, visibility,
                    game_day, status, importance, provenance_json, created_at,
                    updated_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    content = excluded.content,
                    subject_uuid = excluded.subject_uuid,
                    source_uuid = excluded.source_uuid,
                    truth_status = excluded.truth_status,
                    visibility = excluded.visibility,
                    game_day = excluded.game_day,
                    status = excluded.status,
                    importance = excluded.importance,
                    provenance_json = excluded.provenance_json,
                    updated_at = excluded.updated_at,
                    active = excluded.active
                """,
                (
                    _safe_text(fact.fact_id, 256),
                    fact.scope.world_uuid,
                    fact.scope.player_uuid,
                    fact.scope.npc_uuid,
                    _safe_text(fact.conversation_id, 256) or None,
                    _safe_text(fact.kind, 64),
                    content,
                    _safe_text(fact.subject_uuid, 256) or None,
                    _safe_text(fact.source_uuid, 256) or None,
                    _safe_text(fact.truth_status, 32) or "unverified",
                    _visibility_value(fact.visibility),
                    fact.game_day,
                    _safe_text(fact.status, 32) or "active",
                    importance,
                    _json(fact.provenance),
                    fact.created_at or timestamp,
                    timestamp,
                    int(fact.active),
                ),
            )
            connection.commit()
        return StructuredFact(
            fact_id=_safe_text(fact.fact_id, 256),
            scope=fact.scope,
            kind=_safe_text(fact.kind, 64),
            content=content,
            subject_uuid=_safe_text(fact.subject_uuid, 256) or None,
            source_uuid=_safe_text(fact.source_uuid, 256) or None,
            truth_status=_safe_text(fact.truth_status, 32) or "unverified",
            visibility=_visibility(fact.visibility, MemoryVisibility.PUBLIC),
            game_day=fact.game_day,
            status=_safe_text(fact.status, 32) or "active",
            importance=importance,
            provenance=fact.provenance,
            conversation_id=_safe_text(fact.conversation_id, 256) or None,
            created_at=fact.created_at or timestamp,
            updated_at=timestamp,
            active=fact.active,
        )

    def list_structured_facts(
        self, query: MemoryQuery, limit: int = 12
    ) -> list[StructuredFact]:
        self.initialize()
        bounded = max(1, min(int(limit), 32))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM structured_facts
                WHERE world_uuid = ? AND player_uuid = ? AND active = 1
                  AND status = 'active'
                ORDER BY importance DESC, updated_at DESC LIMIT 256
                """,
                (query.scope.world_uuid, query.scope.player_uuid),
            ).fetchall()
        facts = [
            self._fact(row, query.scope)
            for row in rows
            if self._row_visible(row, query.actor_id, fact=True)
        ]
        if query.current_day is not None:
            facts.sort(
                key=lambda fact: (
                    fact.game_day == query.current_day,
                    fact.importance,
                    fact.updated_at,
                ),
                reverse=True,
            )
        return facts[:bounded]

    def retrieve(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 6,
    ) -> list[RetrievalMatch]:
        return self.retrieve_query(
            MemoryQuery(
                scope=scope,
                actor_id=scope.npc_uuid,
                current_message=query,
                max_results=limit,
            )
        )

    def retrieve_query(self, query: MemoryQuery) -> list[RetrievalMatch]:
        """Filter by actor visibility before bounded hybrid relevance ranking.

        FTS5 supplies an expanded lexical candidate set when available, while
        the bounded scope query remains the correctness fallback.  Primitive
        tags such as ``first_meeting`` narrow the SQL candidate set before any
        ranking, which keeps exact facts cheap and avoids unrelated episodes.
        """
        self.initialize()
        bounded = max(1, min(int(query.max_results), 32))
        tokens = self._query_tokens(query)
        with self._connect() as connection:
            memory_where = [
                "world_uuid = ?",
                "player_uuid = ?",
                "active = 1",
            ]
            memory_args: list[object] = [
                query.scope.world_uuid,
                query.scope.player_uuid,
            ]
            if query.requested_kinds:
                placeholders = ",".join("?" for _ in query.requested_kinds)
                memory_where.append(f"memory_type IN ({placeholders})")
                memory_args.extend(kind.value for kind in query.requested_kinds)
            if query.requested_tags:
                tag_clauses = []
                for tag in query.requested_tags[:8]:
                    tag_clauses.append("tags_json LIKE ?")
                    memory_args.append(f'%"{_safe_text(tag, 64)}"%')
                if tag_clauses:
                    memory_where.append("(" + " OR ".join(tag_clauses) + ")")
            memory_rows = connection.execute(
                "SELECT * FROM memories WHERE "
                + " AND ".join(memory_where)
                + " ORDER BY importance DESC, updated_at DESC LIMIT 512",
                memory_args,
            ).fetchall()
            fts_rows = self._fts_rows(connection, query.scope, tokens)
            episode_rows = []
            if not query.requested_tags:
                episode_rows = connection.execute(
                    """
                    SELECT * FROM conversation_episodes
                    WHERE world_uuid = ? AND player_uuid = ? AND active = 1
                    ORDER BY importance DESC, updated_at DESC LIMIT 256
                    """,
                    (query.scope.world_uuid, query.scope.player_uuid),
                ).fetchall()

        # FTS can find lower-importance records outside the bounded fallback
        # window.  De-duplicate by ID before applying actor visibility.
        all_memory_rows = {row["memory_id"]: row for row in memory_rows}
        all_memory_rows.update({row["memory_id"]: row for row in fts_rows})
        records: list[MemoryRecord] = []
        for row in all_memory_rows.values():
            if not self._row_visible(row, query.actor_id):
                continue
            memory = self._memory(row, query.scope)
            if not is_context_eligible(
                memory.content,
                role="assistant",
                metadata=memory.provenance,
            ):
                continue
            if query.requested_kinds and memory.memory_type not in query.requested_kinds:
                continue
            if not self._matches_participants(
                memory.participants, query.participants, query.actor_id
            ):
                continue
            records.append(memory)
        for row in episode_rows:
            if not self._row_visible(row, query.actor_id, episode=True):
                continue
            episode = self._episode(row, query.scope)
            episode_memory = self._episode_memory(episode, query.scope)
            if not is_context_eligible(
                episode_memory.content,
                role="assistant",
                metadata=episode_memory.provenance,
            ):
                continue
            if query.requested_kinds and MemoryType.EPISODE not in query.requested_kinds:
                continue
            if not self._matches_participants(
                episode.participants, query.participants, query.actor_id
            ):
                continue
            records.append(episode_memory)

        matches = [self._rank_record(record, query, tokens) for record in records]
        matches.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        selected: list[RetrievalMatch] = []
        budget_chars = max(400, int(query.token_budget) * 4)
        used_chars = 0
        for match in matches:
            content_size = len(match.memory.content)
            if selected and used_chars + content_size > budget_chars:
                continue
            selected.append(match)
            used_chars += content_size
            if len(selected) >= bounded:
                break
        memory_ids = [
            item.memory.memory_id
            for item in selected
            if not item.memory.memory_id.startswith("episode:")
        ]
        if memory_ids:
            self._mark_recalled(memory_ids)
        return selected

    def end_session(self, session_id: str, scope: MemoryScope, summary: str | None = None) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversation_sessions
                SET ended_at = ?, last_activity_at = ?, summary = COALESCE(?, summary)
                WHERE session_id = ? AND world_uuid = ? AND player_uuid = ? AND npc_uuid = ?
                """,
                (
                    _now(),
                    _now(),
                    summary,
                    session_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                ),
            )
            connection.commit()

    def set_summary(self, session_id: str, scope: MemoryScope, summary: str) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversation_sessions SET summary = ?, last_activity_at = ?
                WHERE session_id = ? AND world_uuid = ? AND player_uuid = ? AND npc_uuid = ?
                """,
                (
                    _safe_text(summary, 4000),
                    _now(),
                    session_id,
                    scope.world_uuid,
                    scope.player_uuid,
                    scope.npc_uuid,
                ),
            )
            connection.commit()

    def list_saved_memories(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str = "",
        record_kind: str = "all",
    ) -> dict[str, Any]:
        """Return a bounded browser view of durable memory-layer records.

        Raw conversation turns are intentionally excluded. They remain the
        append-only transcript used for consolidation; this view contains the
        records that the NPC retrieval pipeline can actually reuse.
        """

        self.initialize()
        bounded_limit = max(1, min(int(limit), _MAX_UI_MEMORY_PAGE))
        bounded_offset = max(0, min(int(offset), 100_000))
        kind = _safe_text(record_kind, 32).casefold() or "all"
        if kind not in {"all", "memory", "episode", "fact", "day_synopsis"}:
            raise ValueError("record_kind must be all, memory, episode, fact, or day_synopsis")
        needle = _safe_text(search, 128).casefold()
        union = """
            SELECT 'memory' AS record_kind, memory_id AS record_id,
                   world_uuid, player_uuid, npc_uuid, game_day,
                   memory_type AS record_type, content, visibility, importance,
                   state, created_at, updated_at, tags_json AS metadata_json,
                   provenance_json
            FROM memories
            WHERE active = 1
            UNION ALL
            SELECT 'episode' AS record_kind, episode_id AS record_id,
                   world_uuid, player_uuid, npc_uuid, game_day,
                   'EPISODE' AS record_type,
                   summary || CASE WHEN key_facts_json = '[]' THEN ''
                       ELSE char(10) || 'Facts: ' || key_facts_json END AS content,
                   visibility, importance, CASE WHEN active = 1 THEN 'active' ELSE 'deleted' END,
                   created_at, updated_at, key_facts_json AS metadata_json,
                   '{}' AS provenance_json
            FROM conversation_episodes
            WHERE active = 1
            UNION ALL
            SELECT 'fact' AS record_kind, fact_id AS record_id,
                   world_uuid, player_uuid, npc_uuid, game_day,
                   'FACT/' || kind AS record_type, content, visibility, importance,
                   status, created_at, updated_at, provenance_json AS metadata_json,
                   provenance_json
            FROM structured_facts
            WHERE active = 1 AND status = 'active'
            UNION ALL
            SELECT 'day_synopsis' AS record_kind,
                   player_uuid || ':' || npc_uuid || ':' || game_day AS record_id,
                   world_uuid, player_uuid, npc_uuid, game_day,
                   'DAY_SYNOPSIS' AS record_type, synopsis, 'SYSTEM' AS visibility,
                   0.7 AS importance, 'active' AS state, updated_at, updated_at,
                   commitments_json AS metadata_json, '{}' AS provenance_json
            FROM day_synopses
            WHERE synopsis != '' OR commitments_json != '[]' OR plans_json != '[]'
                OR claims_json != '[]' OR unresolved_topics_json != '[]'
        """
        predicates = ["world_uuid = ?", "(? = 'all' OR record_kind = ?)"]
        query_args: list[Any] = [self.world_uuid, kind, kind]
        if needle:
            predicates.append(
                "(LOWER(content) LIKE ? OR LOWER(record_type) LIKE ? "
                "OR LOWER(player_uuid) LIKE ? OR LOWER(npc_uuid) LIKE ? "
                "OR EXISTS (SELECT 1 FROM memory_entities AS entity "
                "WHERE entity.world_uuid = saved.world_uuid "
                "AND (entity.entity_uuid = saved.player_uuid "
                "OR entity.entity_uuid = saved.npc_uuid) "
                "AND LOWER(entity.display_name) LIKE ?))"
            )
            pattern = f"%{needle}%"
            query_args.extend([pattern] * 5)
        where = " AND ".join(predicates)
        names: dict[tuple[str, str], str] = {}
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM ({union}) AS saved WHERE {where}", query_args
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM ({union}) AS saved
                WHERE {where}
                ORDER BY updated_at DESC, record_id DESC
                LIMIT ? OFFSET ?
                """,
                [*query_args, bounded_limit, bounded_offset],
            ).fetchall()
            entity_ids = {
                (str(row["npc_uuid"]), "npc")
                for row in rows
                if row["npc_uuid"]
            } | {
                (str(row["player_uuid"]), "player")
                for row in rows
                if row["player_uuid"]
            }
            if entity_ids:
                entity_rows = connection.execute(
                    "SELECT entity_uuid, entity_kind, display_name "
                    "FROM memory_entities WHERE world_uuid = ?",
                    (self.world_uuid,),
                ).fetchall()
                names = {
                    (str(row["entity_uuid"]), str(row["entity_kind"])): str(
                        row["display_name"]
                    )
                    for row in entity_rows
                    if (str(row["entity_uuid"]), str(row["entity_kind"])) in entity_ids
                }
        return {
            "items": [self._saved_memory_row(row, names) for row in rows],
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "world_uuid": self.world_uuid,
        }

    @staticmethod
    def _saved_memory_row(
        row: sqlite3.Row,
        names: dict[tuple[str, str], str] | None = None,
    ) -> dict[str, Any]:
        content = _safe_text(row["content"], 4000)
        names = names or {}
        provenance = _object(row["provenance_json"], {})
        event_time = provenance.get("event_time")
        if not isinstance(event_time, dict):
            event_time = None
        return {
            "record_kind": row["record_kind"],
            "record_id": row["record_id"],
            "world_uuid": row["world_uuid"],
            "player_uuid": row["player_uuid"],
            "npc_uuid": row["npc_uuid"],
            "player_name": names.get((str(row["player_uuid"]), "player")),
            "npc_name": names.get((str(row["npc_uuid"]), "npc")),
            "game_day": row["game_day"],
            "record_type": row["record_type"],
            "content": content,
            "preview": content[:280] + ("…" if len(content) > 280 else ""),
            "visibility": row["visibility"],
            "importance": float(row["importance"]),
            "state": row["state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "primitive_type": provenance.get("primitive_type"),
            "event_time": event_time,
        }

    def delete_saved_memory(
        self,
        record_kind: str,
        record_id: str,
        *,
        player_uuid: str | None = None,
        npc_uuid: str | None = None,
        game_day: int | None = None,
    ) -> bool:
        """Permanently remove one selected memory-layer record in this world."""

        self.initialize()
        kind = _safe_text(record_kind, 32).casefold()
        record_id = _safe_text(record_id, 256)
        if kind not in {"memory", "episode", "fact", "day_synopsis"} or not record_id:
            raise ValueError("record_kind and record_id identify a deletable memory record")
        if kind == "day_synopsis":
            if not player_uuid or not npc_uuid or game_day is None:
                raise ValueError(
                    "day_synopsis deletion requires player_uuid, npc_uuid, and game_day"
                )
            table = "day_synopses"
            where = "world_uuid = ? AND player_uuid = ? AND npc_uuid = ? AND game_day = ?"
            args: tuple[Any, ...] = (
                self.world_uuid,
                _safe_text(player_uuid, 256),
                _safe_text(npc_uuid, 256),
                int(game_day),
            )
        else:
            table, column = {
                "memory": ("memories", "memory_id"),
                "episode": ("conversation_episodes", "episode_id"),
                "fact": ("structured_facts", "fact_id"),
            }[kind]
            where = f"world_uuid = ? AND {column} = ?"
            args = (self.world_uuid, record_id)
            if player_uuid:
                where += " AND player_uuid = ?"
                args += (_safe_text(player_uuid, 256),)
            if npc_uuid:
                where += " AND npc_uuid = ?"
                args += (_safe_text(npc_uuid, 256),)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(f"SELECT 1 FROM {table} WHERE {where}", args).fetchone() is None:
                connection.rollback()
                return False
            if kind == "memory":
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (record_id,))
                connection.execute(
                    "DELETE FROM commitments WHERE world_uuid = ? AND commitment_id = ?",
                    (self.world_uuid, record_id),
                )
            connection.execute(f"DELETE FROM {table} WHERE {where}", args)
            connection.commit()
        return True

    def stats(self, scope: MemoryScope | None = None) -> dict[str, Any]:
        self.initialize()
        where = "WHERE world_uuid = ?" if scope else ""
        args = (scope.world_uuid,) if scope else ()
        with self._connect() as connection:
            memory_count = connection.execute(
                f"SELECT COUNT(*) FROM memories {where}", args
            ).fetchone()[0]
            session_count = connection.execute(
                f"SELECT COUNT(*) FROM conversation_sessions {where}", args
            ).fetchone()[0]
            turn_count = connection.execute(
                f"SELECT COUNT(*) FROM conversation_turns {where}", args
            ).fetchone()[0]
            episode_count = connection.execute(
                f"SELECT COUNT(*) FROM conversation_episodes {where}", args
            ).fetchone()[0]
            fact_count = connection.execute(
                f"SELECT COUNT(*) FROM structured_facts {where}", args
            ).fetchone()[0]
        return {
            "path": str(self.path),
            "world_uuid": self.world_uuid,
            "memory_count": memory_count,
            "session_count": session_count,
            "turn_count": turn_count,
            "episode_count": episode_count,
            "fact_count": fact_count,
            "fts_enabled": self.fts_enabled,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _query_tokens(query: MemoryQuery) -> list[str]:
        source = " ".join(
            [
                query.current_message,
                query.current_topic or "",
                *query.mentioned_entities,
            ]
        )
        expansions = (
            dict(query.token_expansions)
            if query.token_expansions is not None
            else DEFAULT_TOKEN_EXPANSIONS
        )
        tokens: list[str] = []
        for token in _TOKEN_RE.findall(_safe_text(source, 4000)):
            normalized = token.casefold()
            if len(normalized) < 3:
                continue
            for candidate in (normalized, *expansions.get(normalized, ())):
                if candidate not in tokens:
                    tokens.append(candidate)
                if len(tokens) >= 24:
                    return tokens
        return tokens

    @staticmethod
    def _row_visible(
        row: sqlite3.Row,
        actor_id: str,
        *,
        episode: bool = False,
        fact: bool = False,
    ) -> bool:
        owner = str(row["npc_uuid"] or "")
        if owner == str(actor_id):
            return True
        visibility = _visibility(row["visibility"])
        if visibility in {
            MemoryVisibility.PRIVATE,
            MemoryVisibility.TOLD,
            MemoryVisibility.SYSTEM,
        }:
            return False
        if fact:
            provenance = _object(row["provenance_json"], {})
            values = provenance.get("participants", []) if isinstance(provenance, dict) else []
            if str(actor_id) in {str(value) for value in values}:
                return True
        else:
            fields = ["participants_json"]
            if episode:
                fields.append("witnesses_json")
            for field_name in fields:
                values = _object(row[field_name], [])
                if str(actor_id) in {str(value) for value in values}:
                    return True
        # A public fact/memory with no audience list is deliberately not
        # broadcast. Producers must name the actors who heard it.
        return False

    @staticmethod
    def _matches_participants(
        record_participants: tuple[str, ...],
        requested_participants: tuple[str, ...],
        actor_id: str,
    ) -> bool:
        if not requested_participants or not record_participants:
            return True
        requested = {str(value) for value in requested_participants}
        requested.add(str(actor_id))
        return bool(requested.intersection(str(value) for value in record_participants))

    @staticmethod
    def _rank_record(
        memory: MemoryRecord,
        query: MemoryQuery,
        tokens: list[str],
    ) -> RetrievalMatch:
        haystack = " ".join(
            [
                memory.content,
                *memory.tags,
                *memory.topic_tags,
                *memory.entity_refs,
            ]
        ).casefold()
        reasons: list[str] = ["visibility_allowed"]
        exact_tokens = [token for token in tokens if token in haystack]
        if exact_tokens:
            reasons.append("lexical_match")
        if query.current_topic and query.current_topic.casefold() in haystack:
            reasons.append("topic_match")
        if query.mentioned_entities and any(
            entity.casefold() in haystack for entity in query.mentioned_entities
        ):
            reasons.append("entity_match")
        age_days = 0.0
        try:
            created = datetime.fromisoformat(memory.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400)
        except (TypeError, ValueError):
            pass
        recency = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
        score = memory.importance * 0.55 + recency * 0.2
        if exact_tokens:
            score += min(0.4, 0.12 * len(exact_tokens))
        if "topic_match" in reasons:
            score += 0.18
        if "entity_match" in reasons:
            score += 0.18
        if memory.memory_type is MemoryType.COMMITMENT and memory.state == "active":
            reasons.append("active_commitment")
            score += 0.35
        if query.current_day is not None and memory.game_day == query.current_day:
            reasons.append("current_day")
            score += 0.08
        if not exact_tokens and not query.current_topic and not query.mentioned_entities:
            reasons.append("importance")
        reasons.append("recency")
        return RetrievalMatch(memory, score, tuple(dict.fromkeys(reasons)))

    @staticmethod
    def _turn(row: sqlite3.Row) -> ConversationTurn:
        return ConversationTurn(
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            turn_index=row["turn_index"],
            metadata=_object(row["metadata_json"], {}),
            message_id=row["message_id"],
            game_day=row["game_day"],
            world_age_hours=row["world_age_hours"],
            speaker_uuid=row["speaker_uuid"],
            speaker_name=row["speaker_name"],
            speaker_kind=row["speaker_kind"],
        )

    @staticmethod
    def _episode(row: sqlite3.Row, scope: MemoryScope) -> MemoryEpisode:
        return MemoryEpisode(
            episode_id=row["episode_id"],
            scope=scope,
            conversation_id=row["conversation_id"],
            game_day=row["game_day"],
            participants=tuple(str(value) for value in _object(row["participants_json"], [])),
            witnesses=tuple(str(value) for value in _object(row["witnesses_json"], [])),
            topic_tags=tuple(str(value) for value in _object(row["topic_tags_json"], [])),
            entity_refs=tuple(str(value) for value in _object(row["entity_refs_json"], [])),
            summary=row["summary"],
            key_facts=tuple(str(value) for value in _object(row["key_facts_json"], [])),
            emotional_tone=row["emotional_tone"],
            importance=float(row["importance"]),
            visibility=_visibility(row["visibility"], MemoryVisibility.PUBLIC),
            transcript_ref=row["transcript_ref"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _episode_memory(episode: MemoryEpisode, scope: MemoryScope) -> MemoryRecord:
        facts = " ".join(f"Fact: {fact}" for fact in episode.key_facts)
        content = f"{episode.summary} {facts}".strip()
        return MemoryRecord(
            memory_id=f"episode:{episode.episode_id}",
            scope=scope,
            memory_type=MemoryType.EPISODE,
            content=content,
            tags=episode.topic_tags,
            importance=episode.importance,
            provenance={
                "visibility": episode.visibility.value,
                "participants": list(episode.participants),
                "witnesses": list(episode.witnesses),
                "transcript_ref": episode.transcript_ref,
            },
            session_id=episode.conversation_id,
            created_at=episode.created_at,
            updated_at=episode.updated_at,
            visibility=episode.visibility,
            game_day=episode.game_day,
            participants=episode.participants,
            topic_tags=episode.topic_tags,
            entity_refs=episode.entity_refs,
            transcript_ref=episode.transcript_ref,
        )

    @staticmethod
    def _day_synopsis(row: sqlite3.Row, scope: MemoryScope) -> DaySynopsis:
        return DaySynopsis(
            scope=scope,
            game_day=int(row["game_day"]),
            synopsis=row["synopsis"],
            commitments=tuple(str(value) for value in _object(row["commitments_json"], [])),
            plans=tuple(str(value) for value in _object(row["plans_json"], [])),
            claims=tuple(str(value) for value in _object(row["claims_json"], [])),
            unresolved_topics=tuple(
                str(value) for value in _object(row["unresolved_topics_json"], [])
            ),
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _fact(row: sqlite3.Row, scope: MemoryScope) -> StructuredFact:
        return StructuredFact(
            fact_id=row["fact_id"],
            scope=scope,
            kind=row["kind"],
            content=row["content"],
            subject_uuid=row["subject_uuid"],
            source_uuid=row["source_uuid"],
            truth_status=row["truth_status"],
            visibility=_visibility(row["visibility"], MemoryVisibility.PUBLIC),
            game_day=row["game_day"],
            status=row["status"],
            importance=float(row["importance"]),
            provenance=_object(row["provenance_json"], {}),
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            active=bool(row["active"]),
        )

    def _fts_rows(
        self,
        connection: sqlite3.Connection,
        scope: MemoryScope,
        tokens: list[str],
    ) -> list[sqlite3.Row]:
        if not self.fts_enabled or not tokens:
            return []
        match = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:12])
        try:
            return connection.execute(
                """
                SELECT m.* FROM memory_fts f
                JOIN memories m ON m.memory_id = f.memory_id
                WHERE f.memory_fts MATCH ? AND m.world_uuid = ?
                  AND m.player_uuid = ? AND m.active = 1
                LIMIT 128
                """,
                (match, scope.world_uuid, scope.player_uuid),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    def _rebuild_fts(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_fts")
            connection.execute(
                """
                INSERT INTO memory_fts(
                    memory_id, world_uuid, player_uuid, npc_uuid, content, tags
                )
                SELECT memory_id, world_uuid, player_uuid, npc_uuid, content, tags_json
                FROM memories
                WHERE active = 1
                """
            )
            connection.commit()

    def _mark_recalled(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE memories SET last_recalled_at = ? WHERE memory_id IN ({placeholders})",
                (_now(), *memory_ids),
            )
            connection.commit()

    @staticmethod
    def _session(row: sqlite3.Row) -> ConversationSession:
        return ConversationSession(
            session_id=row["session_id"],
            scope=MemoryScope(row["world_uuid"], row["player_uuid"], row["npc_uuid"]),
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            ended_at=row["ended_at"],
            turn_count=row["turn_count"],
            summary=row["summary"],
            metadata=_object(row["metadata_json"], {}),
        )

    @staticmethod
    def _memory(row: sqlite3.Row, scope: MemoryScope) -> MemoryRecord:
        try:
            memory_type = MemoryType(row["memory_type"])
        except ValueError:
            memory_type = MemoryType.FACT
        tags_value = _object(row["tags_json"], [])
        return MemoryRecord(
            memory_id=row["memory_id"],
            scope=scope,
            memory_type=memory_type,
            content=row["content"],
            tags=tuple(str(tag) for tag in tags_value if str(tag)),
            importance=float(row["importance"]),
            state=row["state"],
            provenance=_object(row["provenance_json"], {}),
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_recalled_at=row["last_recalled_at"],
            active=bool(row["active"]),
            visibility=_visibility(row["visibility"]),
            game_day=row["game_day"],
            participants=tuple(
                str(value) for value in _object(row["participants_json"], [])
            ),
            topic_tags=tuple(
                str(value) for value in _object(row["topic_tags_json"], [])
            ),
            entity_refs=tuple(
                str(value) for value in _object(row["entity_refs_json"], [])
            ),
            transcript_ref=row["transcript_ref"],
        )
