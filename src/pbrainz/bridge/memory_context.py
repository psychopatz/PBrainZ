"""Short-lived active-world context reported by the game bridge."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pbrainz.memory import MemoryIdentity, MemoryLocationError


@dataclass(frozen=True, slots=True)
class ActiveMemoryContext:
    """A validated game identity with a monotonic freshness timestamp."""

    identity: MemoryIdentity
    runtime_id: str
    player_uuid: str = ""
    player_name: str = ""
    observed_at: float = 0.0


class ActiveMemoryContextCache:
    """Keep the latest game identity without persisting session presence."""

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._value: ActiveMemoryContext | None = None

    def clear(self) -> None:
        self._value = None

    def update(
        self,
        context: Mapping[str, Any] | object,
        runtime_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Validate and store one game-provided memory context."""

        if not isinstance(context, Mapping):
            return False
        try:
            identity = MemoryIdentity.from_context(context)
        except (MemoryLocationError, TypeError, ValueError):
            return False
        current = time.monotonic() if now is None else float(now)
        self._value = ActiveMemoryContext(
            identity=identity,
            runtime_id=str(runtime_id or ""),
            player_uuid=str(
                context.get("player_uuid") or context.get("playerUUID") or ""
            ).strip(),
            player_name=str(
                context.get("player_name") or context.get("playerName") or ""
            ).strip(),
            observed_at=current,
        )
        return True

    def as_dict(self, *, now: float | None = None) -> dict[str, Any]:
        value = self._value
        if value is None:
            return {"status": "unavailable", "reason": "no_game_context"}
        current = time.monotonic() if now is None else float(now)
        age = max(0.0, current - value.observed_at)
        if age > self.ttl_seconds:
            return {
                "status": "unavailable",
                "reason": "game_context_stale",
                "runtime_id": value.runtime_id or None,
                "age_seconds": round(age, 3),
            }
        identity = value.identity
        return {
            "status": "active",
            "world_uuid": identity.world_uuid,
            "world_mode": identity.storage_mode,
            "save_relative_path": identity.save_relative_path or None,
            "server_instance_id": identity.server_instance_id or None,
            "server_world_generation": identity.server_world_generation or None,
            "player_uuid": value.player_uuid or None,
            "player_name": value.player_name or None,
            "runtime_id": value.runtime_id or None,
            "age_seconds": round(age, 3),
        }
