"""Atomic heartbeat published by the optional Project Hoomans worker."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class ProviderPresenceWriter:
    """Publish a small, runtime-bound worker/provider presence document."""

    STATE_FILENAME = "provider_state.json"
    READY_FILENAME = "provider_state.ready.txt"
    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_dir = self.root / "state"

    def write(
        self,
        *,
        runtime_id: str,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        heartbeat_ms: int | None = None,
    ) -> None:
        runtime = str(runtime_id or "").strip()
        if not runtime:
            raise ValueError("runtime_id is required")
        normalized_status = str(status or "unavailable").strip() or "unavailable"
        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "runtime_id": runtime,
            "status": normalized_status,
            "ready": normalized_status == "ready",
            "heartbeat_ms": int(
                heartbeat_ms if heartbeat_ms is not None else time.time() * 1000
            ),
        }
        if provider:
            payload["provider"] = str(provider)
        if model:
            payload["model"] = str(model)
        if reason:
            payload["reason"] = str(reason)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.state_dir / self.STATE_FILENAME
        marker_path = self.state_dir / self.READY_FILENAME
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._atomic_write(state_path, encoded)
        self._atomic_write(marker_path, runtime)

    def _atomic_write(self, path: Path, value: str) -> None:
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            temporary.write_text(value, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
