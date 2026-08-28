"""Generic client for Core snapshot/event channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import BridgeClient
from .protocol import CORE_NAMESPACE, POLL_PACKETS_COMMAND, BridgeClientError
from .state import BridgeState

MAX_SUBSCRIPTIONS = 32


@dataclass(slots=True)
class PacketStreamClient:
    """Track cursors for arbitrary Core channels without knowing their payloads."""

    cursors: dict[str, int] = field(default_factory=dict)

    async def poll(
        self,
        client: BridgeClient,
        state: BridgeState,
        subscriptions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not state.runtime_id:
            raise BridgeClientError("bridge runtime is unavailable")
        bounded = []
        for subscription in subscriptions[:MAX_SUBSCRIPTIONS]:
            if not isinstance(subscription, dict):
                raise BridgeClientError("packet subscription is invalid")
            namespace = subscription.get("namespace")
            channel = subscription.get("channel")
            if not isinstance(namespace, str) or not isinstance(channel, str):
                raise BridgeClientError("packet subscription is invalid")
            stream_id = f"{namespace}:{channel}"
            bounded.append({
                "namespace": namespace,
                "channel": channel,
                "after": self.cursors.get(stream_id, 0),
                "limit": subscription.get("limit", 32),
                "include_snapshot": subscription.get("include_snapshot", False),
            })

        result = await client.call(
            CORE_NAMESPACE,
            POLL_PACKETS_COMMAND,
            {"subscriptions": bounded},
            state.runtime_id,
        )
        if not isinstance(result, dict) or not isinstance(result.get("streams"), list):
            raise BridgeClientError("packet stream response is invalid")
        for stream in result["streams"]:
            self._advance(stream)
        return result

    def reset(self) -> None:
        self.cursors.clear()

    def _advance(self, stream: object) -> None:
        if not isinstance(stream, dict):
            raise BridgeClientError("packet stream contains an invalid row")
        namespace = stream.get("namespace")
        channel = stream.get("channel")
        if not isinstance(namespace, str) or not isinstance(channel, str):
            raise BridgeClientError("packet stream row has no channel")
        if stream.get("error"):
            return
        stream_id = f"{namespace}:{channel}"
        events = stream.get("events")
        if not isinstance(events, list):
            raise BridgeClientError("packet stream row has invalid events")
        gap = stream.get("gap") is True
        if gap and stream.get("snapshot") is None:
            return
        if gap:
            sequence = stream.get("sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                self.cursors[stream_id] = sequence
            return
        for event in events:
            if not isinstance(event, dict):
                raise BridgeClientError("packet stream contains an invalid event")
            sequence = event.get("sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise BridgeClientError("packet stream event has invalid sequence")
            self.cursors[stream_id] = max(self.cursors.get(stream_id, 0), sequence)
