"""Read-only access to the PsychopatzCore bridge runtime state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pbrainz.paths import bridge_root_for

from .protocol import PROTOCOL_VERSION

MAX_RUNTIME_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class BridgeState:
    """A bounded, validated view of the current bridge runtime."""

    available: bool
    enabled: bool = False
    ready: bool = False
    runtime_id: str | None = None
    protocol_version: int | None = None
    lifecycle: str | None = None
    authority: str | None = None
    transport: str | None = None
    tool_catalog_id: str | None = None
    tool_catalog_version: int | None = None
    packet_channels: tuple[dict[str, Any], ...] = ()
    message: str = "bridge runtime unavailable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "ready": self.ready,
            "runtime_id": self.runtime_id,
            "protocol_version": self.protocol_version,
            "lifecycle": self.lifecycle,
            "authority": self.authority,
            "transport": self.transport,
            "tool_catalog_id": self.tool_catalog_id,
            "tool_catalog_version": self.tool_catalog_version,
            "packet_channels": list(self.packet_channels),
            "message": self.message,
        }


class BridgeRuntimeMonitor:
    """Read the same cross-platform state directory used by PsychopatzCore."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        zomboid_path: str | Path | None = None,
    ) -> None:
        self.set_root(root, zomboid_path=zomboid_path)

    def set_root(
        self,
        root: str | Path | None = None,
        *,
        zomboid_path: str | Path | None = None,
    ) -> None:
        """Point monitoring at a new bridge root without touching the filesystem."""

        self.root = bridge_root_for(zomboid_path, root)
        self.state_dir = self.root / "state"

    def read(self) -> BridgeState:
        marker_path = self.state_dir / "runtime.ready.txt"
        runtime_path = self.state_dir / "runtime.json"
        try:
            marker = marker_path.read_text(encoding="utf-8").strip()
            if runtime_path.stat().st_size > MAX_RUNTIME_BYTES:
                return BridgeState(False, message="bridge runtime exceeds size limit")
            value = json.loads(runtime_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return BridgeState(False)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return BridgeState(False, message="bridge runtime is unreadable or incomplete")

        if not isinstance(value, dict):
            return BridgeState(False, message="bridge runtime is not an object")
        runtime_id = value.get("runtime_id")
        valid = (
            value.get("protocol_version") == PROTOCOL_VERSION
            and isinstance(runtime_id, str)
            and bool(runtime_id)
            and marker == runtime_id
        )
        if not valid:
            return BridgeState(False, message="bridge runtime marker does not match runtime state")

        enabled = value.get("enabled") is True
        lifecycle = str(value.get("lifecycle") or "")
        ready = enabled and lifecycle == "READY"
        tool_catalog_id = value.get("tool_catalog_id")
        packet_channels = value.get("packet_channels")
        normalized_channels = tuple(
            dict(row) for row in packet_channels[:32]
            if isinstance(row, dict)
        ) if isinstance(packet_channels, list) else ()
        return BridgeState(
            available=True,
            enabled=enabled,
            ready=ready,
            runtime_id=runtime_id,
            protocol_version=PROTOCOL_VERSION,
            lifecycle=lifecycle or None,
            authority=str(value.get("authority") or "") or None,
            transport=str(value.get("transport") or "") or None,
            tool_catalog_id=(
                tool_catalog_id if isinstance(tool_catalog_id, str) and tool_catalog_id else None
            ),
            tool_catalog_version=(
                value["tool_catalog_version"]
                if isinstance(value.get("tool_catalog_version"), int)
                and not isinstance(value.get("tool_catalog_version"), bool)
                else None
            ),
            packet_channels=normalized_channels,
            message="bridge ready" if ready else f"bridge lifecycle is {lifecycle or 'unknown'}",
        )
