"""Read-only access to the PsychopatzCore bridge runtime state.

The game-side bridge remains the authority. This module only validates the
runtime marker and JSON state; it never writes requests or dispatches commands.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
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
            "message": self.message,
        }


class BridgeRuntimeMonitor:
    """Read the same cross-platform state directory used by PsychopatzCore."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root or os.getenv("ZOMBOID_BRIDGE_ROOT")
        self.root = (
            Path(configured_root)
            if configured_root
            else Path.home() / "Zomboid" / "Lua" / "PsychopatzBridge"
        )
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
        return BridgeState(
            available=True,
            enabled=enabled,
            ready=ready,
            runtime_id=runtime_id,
            protocol_version=PROTOCOL_VERSION,
            lifecycle=lifecycle or None,
            authority=str(value.get("authority") or "") or None,
            transport=str(value.get("transport") or "") or None,
            message="bridge ready" if ready else f"bridge lifecycle is {lifecycle or 'unknown'}",
        )
