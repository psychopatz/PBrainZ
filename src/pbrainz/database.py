"""SQLite persistence for PBrainZ settings, catalogs, and activity logs."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbrainz.branding import DATABASE_ENV, DATABASE_NAME, PORTABLE_ROOT_ENV

LOGGER = logging.getLogger(__name__)
DEFAULT_DATABASE_NAME = DATABASE_NAME
DEFAULT_ACTIVITY_LIMIT = 50
MAX_ACTIVITY_LIMIT = 500
DEFAULT_TRACE_LIMIT = 100
MAX_TRACE_LIMIT = 300
MAX_TRACE_PAYLOAD_CHARS = 50000
PERSISTED_SETTINGS = (
    "app_name",
    "host",
    "port",
    "log_level",
    "default_provider",
    "default_model",
    "enabled_providers",
    "request_timeout",
    "max_retries",
    "bridge_required",
    "bridge_root",
    "zomboid_path",
    "bridge_poll_interval",
    "open_gui",
    "openai_api_key",
    "openai_base_url",
    "openai_models",
    "ollama_api_key",
    "ollama_base_url",
    "ollama_models",
    "lmstudio_api_key",
    "lmstudio_base_url",
    "lmstudio_models",
    "custom_api_key",
    "custom_base_url",
    "custom_models",
    "horde_api_key",
    "horde_base_url",
    "horde_models",
    "gemini_api_key",
    "gemini_models",
    "auto_refresh_models",
    "model_refresh_interval",
    "ui_theme",
    "memory_root",
    "context_max_chars",
    "memory_recent_turns",
    "memory_retrieval_limit",
    "memory_consolidation_turns",
    "template_profiles_json",
    "active_template_profile_id",
    "llm_diagnostics",
    "llm_trace_capture",
    "tts_enabled",
    "tts_piper_executable",
    "tts_model_root",
    "tts_metadata_path",
    "tts_voice_catalog_url",
    "tts_voice_catalog_ttl_seconds",
    "tts_voice_catalog_language",
    "tts_output_device",
    "tts_master_volume",
    "tts_synthesis_workers",
    "tts_model_cache_size",
    "tts_max_simultaneous_playback",
    "tts_max_generated_ahead",
    "tts_max_tts_ready_ahead",
    "tts_natural_gap_ms",
    "tts_synthesis_timeout",
    "tts_audio_buffer_ms",
    "tts_voice_presets_json",
)


class SettingsDatabase:
    """Small SQLite store with one connection per operation for thread safety."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or os.getenv(DATABASE_ENV)
        self.path = (
            Path(configured_path).expanduser()
            if configured_path
            else _default_database_path()
        )

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_catalog (
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (provider, model_id)
                );
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_activity_log_created
                    ON activity_log(created_at DESC);
                CREATE TABLE IF NOT EXISTS llm_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    npc_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_trace_created
                    ON llm_trace(created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_llm_trace_request
                    ON llm_trace(request_id);
                """
            )
            connection.commit()
        finally:
            connection.close()
        if os.name != "nt":
            try:
                self.path.chmod(0o600)
            except OSError as error:
                LOGGER.warning(
                    "Could not restrict database permissions for %s: %s", self.path, error
                )

    def load_settings(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        finally:
            connection.close()
        values: dict[str, Any] = {}
        for row in rows:
            try:
                values[row["key"]] = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                values[row["key"]] = row["value"]
        return values

    def save_settings(self, values: Mapping[str, object]) -> None:
        timestamp = _now()
        connection = self._connect()
        try:
            connection.executemany(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                [
                    (key, json.dumps(value), timestamp)
                    for key, value in values.items()
                    if key in PERSISTED_SETTINGS
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def replace_model_catalog(
        self,
        provider: str,
        model_ids: Iterable[str],
        *,
        source: str = "provider_api",
    ) -> list[str]:
        normalized = list(
            dict.fromkeys(model_id.strip() for model_id in model_ids if model_id.strip())
        )
        timestamp = _now()
        connection = self._connect()
        try:
            connection.execute("DELETE FROM model_catalog WHERE provider = ?", (provider,))
            connection.executemany(
                """
                INSERT INTO model_catalog(provider, model_id, display_name, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(provider, model_id, model_id, source, timestamp) for model_id in normalized],
            )
            connection.commit()
        finally:
            connection.close()
        return normalized

    def model_catalog(self, provider: str) -> list[dict[str, str]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT model_id, display_name, source, updated_at
                FROM model_catalog
                WHERE provider = ?
                ORDER BY model_id
                """,
                (provider,),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]

    def add_log(self, level: str, source: str, message: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO activity_log(created_at, level, source, message) VALUES (?, ?, ?, ?)",
                (_now(), level, source, message[:2000]),
            )
            connection.execute(
                """
                DELETE FROM activity_log
                WHERE id <= COALESCE((SELECT MAX(id) FROM activity_log), 0) - 2000
                """
            )
            connection.commit()
        finally:
            connection.close()

    def recent_logs(self, limit: int = DEFAULT_ACTIVITY_LIMIT) -> list[dict[str, str]]:
        bounded_limit = max(1, min(limit, MAX_ACTIVITY_LIMIT))
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT created_at, level, source, message
                FROM activity_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in reversed(rows)]

    def add_llm_trace(
        self,
        *,
        source: str,
        phase: str,
        request_id: str = "",
        npc_id: str = "",
        session_id: str = "",
        payload: object = None,
    ) -> None:
        """Store one bounded opt-in LLM diagnostic event.

        This is intentionally separate from activity_log and save-scoped memory:
        traces are local troubleshooting data, never NPC memory. Callers should
        check their capture setting before constructing large payloads.
        """
        serialized = json.dumps(
            _json_safe(payload), ensure_ascii=False, separators=(",", ":")
        )
        if len(serialized) > MAX_TRACE_PAYLOAD_CHARS:
            serialized = json.dumps(
                {
                    "truncated": True,
                    "original_chars": len(serialized),
                    "preview": serialized[: MAX_TRACE_PAYLOAD_CHARS - 160],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO llm_trace(
                    created_at, source, phase, request_id, npc_id, session_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    str(source)[:160],
                    str(phase)[:160],
                    str(request_id)[:256],
                    str(npc_id)[:256],
                    str(session_id)[:256],
                    serialized,
                ),
            )
            connection.execute(
                """
                DELETE FROM llm_trace
                WHERE id <= COALESCE((SELECT MAX(id) FROM llm_trace), 0) - ?
                """,
                (MAX_TRACE_LIMIT,),
            )
            connection.commit()
        finally:
            connection.close()

    def recent_llm_traces(
        self,
        limit: int = DEFAULT_TRACE_LIMIT,
        *,
        request_id: str = "",
        search: str = "",
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), MAX_TRACE_LIMIT))
        filters: list[str] = []
        parameters: list[object] = []
        if request_id.strip():
            filters.append("request_id = ?")
            parameters.append(request_id.strip()[:256])
        if search.strip():
            search_value = f"%{search.strip()[:160]}%"
            filters.append(
                "(request_id LIKE ? OR npc_id LIKE ? OR session_id LIKE ? "
                "OR source LIKE ? OR phase LIKE ? OR payload_json LIKE ?)"
            )
            parameters.extend([search_value] * 6)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT id, created_at, source, phase, request_id, npc_id, session_id,
                       payload_json
                FROM llm_trace
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                (*parameters, bounded_limit),
            ).fetchall()
        finally:
            connection.close()
        result: list[dict[str, Any]] = []
        for row in reversed(rows):
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except (TypeError, json.JSONDecodeError):
                item["payload"] = {"unavailable": True}
                item.pop("payload_json", None)
            result.append(item)
        return result

    def clear_llm_traces(self) -> int:
        connection = self._connect()
        try:
            cursor = connection.execute("DELETE FROM llm_trace")
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_safe(value: object, depth: int = 0) -> object:
    """Make arbitrary provider/game diagnostics bounded and JSON-compatible."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return value[:12000]
        return value
    if depth >= 6:
        return "[depth-limit]"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= 96:
                result["[truncated]"] = "96+ entries"
                break
            result[str(key)[:256]] = _json_safe(child, depth + 1)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child, depth + 1) for child in list(value)[:96]]
    return str(value)[:12000]


def _default_database_path() -> Path:
    """Choose the portable data directory beside the running application."""
    return application_root() / "data" / DEFAULT_DATABASE_NAME


def application_root() -> Path:
    """Return the directory that should contain portable application data.

    The launchers set ``PBRAINZ_PORTABLE_ROOT`` explicitly for source runs and
    AppImage launches. The frozen executable and AppImage environment fallbacks
    keep direct launches working when a launcher is bypassed.
    """
    configured_root = os.getenv(PORTABLE_ROOT_ENV)
    if configured_root and configured_root.strip():
        return Path(configured_root).expanduser().resolve()
    appimage_path = os.getenv("APPIMAGE")
    if appimage_path and appimage_path.strip():
        return Path(appimage_path).expanduser().resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()
