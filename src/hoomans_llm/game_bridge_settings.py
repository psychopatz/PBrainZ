"""Read and update the Project Hoomans/PsychopatzCore bridge setting."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

CONFIG_FILENAME = "PsychopatzCore_Bridge.txt"


def default_config_path() -> Path:
    """Return the cross-platform Project Zomboid bridge configuration path."""

    configured = os.getenv("ZOMBOID_BRIDGE_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Zomboid" / "Lua" / CONFIG_FILENAME


@dataclass(frozen=True, slots=True)
class GameBridgeConfig:
    """The small INI-like configuration consumed by PsychopatzCore."""

    enabled: bool = False
    transport: str = "file"
    poll_interval_ms: int = 250
    version: int = 1

    def serialize(self) -> str:
        return (
            f"config_version={self.version}\n"
            f"bridge_enabled={str(self.enabled).lower()}\n"
            f"bridge_transport={self.transport}\n"
            f"bridge_poll_interval_ms={self.poll_interval_ms}\n"
        )


def parse_config(text: str) -> GameBridgeConfig:
    """Parse the same bounded setting format as the game-side bootstrap."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        content = line.split("#", 1)[0].split(";", 1)[0].strip()
        key, separator, value = content.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip()
    enabled = values.get("bridge_enabled", "false").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    transport = values.get("bridge_transport", "file").casefold()
    if transport != "file":
        transport = "file"
    try:
        interval = int(values.get("bridge_poll_interval_ms", "250"))
    except ValueError:
        interval = 250
    return GameBridgeConfig(
        enabled=enabled,
        transport=transport,
        poll_interval_ms=max(100, min(5000, interval)),
    )


class GameBridgeSettings:
    """Persist the game bridge flag atomically without owning game runtime state."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path else default_config_path()

    def read(self) -> GameBridgeConfig:
        try:
            return parse_config(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, OSError, UnicodeError):
            return GameBridgeConfig()

    def set_enabled(self, enabled: bool) -> GameBridgeConfig:
        config = replace(self.read(), enabled=enabled is True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(config.serialize(), encoding="utf-8")
        temporary.replace(self.path)
        return config
