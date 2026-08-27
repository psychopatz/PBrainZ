"""SQLite persistence for HoomansLLM settings, model catalogs, and activity logs."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATABASE_NAME = "hoomansllm.db"
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
)


class SettingsDatabase:
    """Small SQLite store with one connection per operation for thread safety."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or os.getenv("HOOMANSLLM_DB")
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

    def recent_logs(self, limit: int = 100) -> list[dict[str, str]]:
        bounded_limit = max(1, min(limit, 500))
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

    def import_legacy_if_unconfigured(self, source_path: str | Path) -> bool:
        """Import a legacy database once when this store is still untouched.

        Linux source launches historically stored the database in the working
        directory, while frozen AppImages use the XDG config directory. Keep
        existing installs seamless by importing useful settings and catalogs
        only when the new store has no user configuration yet.
        """
        source = Path(source_path).expanduser()
        try:
            if not source.is_file() or source.resolve() == self.path.resolve():
                return False
        except OSError:
            return False

        target_settings = self.load_settings()
        if self._has_user_configuration(target_settings) or self._catalog_count():
            return False

        try:
            with sqlite3.connect(source) as connection:
                connection.row_factory = sqlite3.Row
                source_settings = {
                    row["key"]: _decode_value(row["value"])
                    for row in connection.execute("SELECT key, value FROM settings")
                }
                catalog_rows = connection.execute(
                    """
                    SELECT provider, model_id
                    FROM model_catalog
                    ORDER BY provider, model_id
                    """
                ).fetchall()
        except (OSError, sqlite3.Error, KeyError, TypeError, json.JSONDecodeError):
            return False

        if not self._has_user_configuration(source_settings) and not catalog_rows:
            return False

        self.save_settings(source_settings)
        models_by_provider: dict[str, list[str]] = {}
        for row in catalog_rows:
            models_by_provider.setdefault(row["provider"], []).append(row["model_id"])
        for provider, model_ids in models_by_provider.items():
            self.replace_model_catalog(provider, model_ids, source="legacy_database")
        self.add_log("INFO", "hoomans_llm.database", f"Imported legacy database from {source}")
        return True

    def _catalog_count(self) -> int:
        try:
            connection = self._connect()
            try:
                return int(connection.execute("SELECT COUNT(*) FROM model_catalog").fetchone()[0])
            finally:
                connection.close()
        except sqlite3.Error:
            return 0

    @staticmethod
    def _has_user_configuration(values: Mapping[str, Any]) -> bool:
        credential_keys = (
            "openai_api_key",
            "ollama_api_key",
            "lmstudio_api_key",
            "custom_api_key",
            "gemini_api_key",
        )
        if any(str(values.get(key) or "").strip() for key in credential_keys):
            return True
        if str(values.get("custom_base_url") or "").strip():
            return True
        if str(values.get("default_provider") or "openai").strip().lower() != "openai":
            return True
        return values.get("default_model") not in (None, "", "null")

    def read_legacy_dotenv(self) -> dict[str, str]:
        """Read old dotenv files only for the one-time migration into SQLite."""
        values: dict[str, str] = {}
        for path in (Path(".env"), Path(".env.local")):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[7:].lstrip()
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip().lower()
                if key not in PERSISTED_SETTINGS:
                    continue
                values[key] = value.strip().strip("'\"")
        return values

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _decode_value(value: object) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _default_database_path() -> Path:
    """Choose a writable persistent location for source and frozen builds."""
    if sys.platform.startswith("linux"):
        config_root = os.getenv("XDG_CONFIG_HOME")
        base = Path(config_root).expanduser() if config_root else Path.home() / ".config"
        return base / "HoomansLLM" / DEFAULT_DATABASE_NAME
    if not getattr(sys, "frozen", False):
        return Path.cwd() / DEFAULT_DATABASE_NAME
    if os.name == "nt":
        return Path(sys.executable).resolve().parent / DEFAULT_DATABASE_NAME
    return Path.cwd() / DEFAULT_DATABASE_NAME


def legacy_database_candidates() -> tuple[Path, ...]:
    """Return safe sidecar locations that may contain a pre-XDG database."""
    candidates = [Path.cwd() / DEFAULT_DATABASE_NAME]
    if getattr(sys, "frozen", False):
        appimage_path = os.getenv("APPIMAGE")
        if appimage_path:
            appimage_directory = Path(appimage_path).expanduser().resolve().parent
            candidates.extend(
                directory / DEFAULT_DATABASE_NAME
                for directory in (appimage_directory, *appimage_directory.parents[:3])
            )
        candidates.append(Path(sys.executable).resolve().parent / DEFAULT_DATABASE_NAME)
    return tuple(dict.fromkeys(candidates))
