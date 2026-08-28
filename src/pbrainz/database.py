"""SQLite persistence for P BrainZ settings, catalogs, and activity logs."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbrainz.branding import DATABASE_ENV, DATABASE_NAME, PORTABLE_ROOT_ENV

DEFAULT_DATABASE_NAME = DATABASE_NAME
DEFAULT_ACTIVITY_LIMIT = 50
MAX_ACTIVITY_LIMIT = 500
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
    "llm_diagnostics",
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
                """
            )
            connection.commit()
        finally:
            connection.close()
        if os.name != "nt":
            try:
                self.path.chmod(0o600)
            except OSError:
                pass

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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_database_path() -> Path:
    """Choose the portable data directory beside the running application."""
    return application_root() / "data" / DEFAULT_DATABASE_NAME


def application_root() -> Path:
    """Return the directory that should contain portable application data."""
    configured_root = os.getenv(PORTABLE_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    appimage_path = os.getenv("APPIMAGE")
    if appimage_path:
        return Path(appimage_path).expanduser().resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()

