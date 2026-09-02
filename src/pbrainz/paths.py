"""Cross-platform Project Zomboid user-data and bridge paths."""

from __future__ import annotations

import os
from pathlib import Path

ZOMBOID_PATH_ENV = "ZOMBOID_PATH"
ZOMBOID_ROOT_ENV = "ZOMBOID_ROOT"
BRIDGE_ROOT_ENV = "ZOMBOID_BRIDGE_ROOT"
BRIDGE_CONFIG_ENV = "ZOMBOID_BRIDGE_CONFIG"


def normalize_zomboid_path(value: str | Path | None) -> Path:
    """Return a user-supplied Zomboid directory without requiring it to exist."""

    if value is None:
        return default_zomboid_path()
    text = os.fspath(value).strip()
    return Path(text).expanduser() if text else default_zomboid_path()


def default_zomboid_path() -> Path:
    """Find the conventional Project Zomboid data directory for this OS.

    ``Path.home()`` follows the active operating system and user account. The
    first candidate preserves the existing PBrainZ default; the additional
    candidates cover common Linux, macOS, and redirected Documents layouts.
    An explicit environment value always wins and is useful for portable or
    custom installations.
    """

    configured = os.getenv(ZOMBOID_PATH_ENV) or os.getenv(ZOMBOID_ROOT_ENV)
    if configured and configured.strip():
        return Path(configured).expanduser()

    home = Path.home()
    candidates = (
        home / "Zomboid",
        home / "Documents" / "Zomboid",
        home / ".local" / "share" / "Zomboid",
        home / "Library" / "Application Support" / "Zomboid",
    )
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return candidates[0]


def bridge_root_for(
    zomboid_path: str | Path | None = None,
    explicit_bridge_root: str | Path | None = None,
) -> Path:
    """Return the bridge directory, honoring an explicit bridge override."""

    configured = explicit_bridge_root or os.getenv(BRIDGE_ROOT_ENV)
    if configured and os.fspath(configured).strip():
        return Path(configured).expanduser()
    return normalize_zomboid_path(zomboid_path) / "Lua" / "PsychopatzBridge"


def bridge_config_path_for(
    zomboid_path: str | Path | None = None,
    explicit_config_path: str | Path | None = None,
) -> Path:
    """Return the Core bridge toggle file path for a Zomboid data directory."""

    configured = explicit_config_path or os.getenv(BRIDGE_CONFIG_ENV)
    if configured and os.fspath(configured).strip():
        return Path(configured).expanduser()
    return normalize_zomboid_path(zomboid_path) / "Lua" / "PsychopatzCore_Bridge.txt"
