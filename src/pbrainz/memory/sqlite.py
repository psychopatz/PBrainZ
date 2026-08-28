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

from .types import (
    ConversationSession,
    ConversationTurn,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievalMatch,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{2,64}")
_MAX_TEXT = 12000


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
                    metadata_json TEXT NOT NULL DEFAULT '{}'
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
                    active INTEGER NOT NULL DEFAULT 1
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
                """,
            )
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

    def add_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        self.initialize()
        content = _safe_text(content)
        if not content:
            raise ValueError("conversation turn content is required")
        timestamp = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT turn_count FROM conversation_sessions WHERE session_id = ? "
                "AND world_uuid = ? AND player_uuid = ? AND npc_uuid = ?",
                (session_id, scope.world_uuid, scope.player_uuid, scope.npc_uuid),
            ).fetchone()
            if row is None:
                raise ValueError("conversation session does not match memory scope")
            turn_index = int(row["turn_count"]) + 1
            connection.execute(
                """
                INSERT INTO conversation_turns(
                    session_id, world_uuid, player_uuid, npc_uuid,
                    turn_index, role, content, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        return ConversationTurn(
            role=role,
            content=content,
            created_at=timestamp,
            turn_index=turn_index,
            metadata=metadata or {},
        )

    def recent_turns(
        self,
        session_id: str,
        scope: MemoryScope,
        limit: int = 8,
    ) -> list[ConversationTurn]:
        self.initialize()
        bounded = max(1, min(int(limit), 64))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at, turn_index, metadata_json
                FROM conversation_turns
                WHERE session_id = ? AND world_uuid = ?
                  AND player_uuid = ? AND npc_uuid = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, scope.world_uuid, scope.player_uuid, scope.npc_uuid, bounded),
            ).fetchall()
        return [
            ConversationTurn(
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                turn_index=row["turn_index"],
                metadata=_object(row["metadata_json"], {}),
            )
            for row in reversed(rows)
        ]

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
                    provenance_json, created_at, updated_at, last_recalled_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    content = excluded.content,
                    tags_json = excluded.tags_json,
                    importance = excluded.importance,
                    state = excluded.state,
                    provenance_json = excluded.provenance_json,
                    updated_at = excluded.updated_at,
                    last_recalled_at = excluded.last_recalled_at,
                    active = excluded.active
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

    def retrieve(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 6,
    ) -> list[RetrievalMatch]:
        self.initialize()
        bounded = max(1, min(int(limit), 32))
        tokens = [token.casefold() for token in _TOKEN_RE.findall(_safe_text(query, 2000))]
        with self._connect() as connection:
            rows = self._fts_rows(connection, scope, tokens)
            if not rows:
                if tokens:
                    clauses = " OR ".join(
                        "(LOWER(content) LIKE ? OR LOWER(tags_json) LIKE ?)"
                        for _ in tokens[:12]
                    )
                    like_args = tuple(
                        value
                        for token in tokens[:12]
                        for value in (f"%{token}%", f"%{token}%")
                    )
                    rows = connection.execute(
                        f"SELECT * FROM memories WHERE world_uuid = ? "
                        f"AND player_uuid = ? AND npc_uuid = ? AND active = 1 "
                        f"AND ({clauses})",
                        (scope.world_uuid, scope.player_uuid, scope.npc_uuid, *like_args),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT * FROM memories
                        WHERE world_uuid = ? AND player_uuid = ? AND npc_uuid = ? AND active = 1
                        """,
                        (scope.world_uuid, scope.player_uuid, scope.npc_uuid),
                    ).fetchall()
            commitment_rows = connection.execute(
                """
                SELECT m.* FROM memories m
                JOIN commitments c ON c.commitment_id = m.memory_id
                WHERE c.world_uuid = ? AND c.player_uuid = ? AND c.npc_uuid = ?
                  AND c.status = 'active' AND m.active = 1
                """,
                (scope.world_uuid, scope.player_uuid, scope.npc_uuid),
            ).fetchall()
        by_id = {row["memory_id"]: row for row in rows}
        for row in commitment_rows:
            by_id[row["memory_id"]] = row
        matches: list[RetrievalMatch] = []
        for row in by_id.values():
            memory = self._memory(row, scope)
            haystack = f"{memory.content} {' '.join(memory.tags)}".casefold()
            reasons: list[str] = []
            if tokens and any(token in haystack for token in tokens):
                reasons.append("fts_text")
            if memory.memory_type is MemoryType.COMMITMENT and memory.state == "active":
                reasons.append("active_commitment")
            if memory.tags and tokens and any(token in memory.tags for token in tokens):
                reasons.append("tag_match")
            age_days = 0.0
            try:
                created = datetime.fromisoformat(memory.created_at).replace(tzinfo=UTC)
                age_days = max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400)
            except (TypeError, ValueError):
                pass
            recency = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
            score = memory.importance * 0.55 + recency * 0.25
            if "fts_text" in reasons:
                score += 0.35
            if "tag_match" in reasons:
                score += 0.2
            if "active_commitment" in reasons:
                score += 0.45
            if not tokens:
                reasons.append("importance")
            reasons.append("recency")
            matches.append(RetrievalMatch(memory, score, tuple(dict.fromkeys(reasons))))
        matches.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        selected = matches[:bounded]
        if selected:
            self._mark_recalled([item.memory.memory_id for item in selected])
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
        return {
            "path": str(self.path),
            "world_uuid": self.world_uuid,
            "memory_count": memory_count,
            "session_count": session_count,
            "turn_count": turn_count,
            "fts_enabled": self.fts_enabled,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _fts_rows(
        self,
        connection: sqlite3.Connection,
        scope: MemoryScope,
        tokens: list[str],
    ) -> list[sqlite3.Row]:
        if not self.fts_enabled or not tokens:
            return []
        match = " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:12])
        try:
            return connection.execute(
                """
                SELECT m.* FROM memory_fts f
                JOIN memories m ON m.memory_id = f.memory_id
                WHERE f.memory_fts MATCH ? AND m.world_uuid = ?
                  AND m.player_uuid = ? AND m.npc_uuid = ? AND m.active = 1
                """,
                (match, scope.world_uuid, scope.player_uuid, scope.npc_uuid),
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
        )
