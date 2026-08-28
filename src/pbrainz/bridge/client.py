"""Async request client for one bounded bridge command at a time."""

from __future__ import annotations

import asyncio
import time

from .protocol import (
    BridgeClientError,
    BridgeCommandError,
    BridgeRequest,
    BridgeTimeoutError,
)
from .transport import FileBridgeTransport


class BridgeClient:
    """Synchronous-slot, async-waiting client for one bridge command at a time."""

    def __init__(
        self,
        transport: FileBridgeTransport,
        *,
        timeout: float = 8.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.transport = transport
        self.timeout = timeout
        self.poll_interval = poll_interval

    async def call(
        self,
        namespace: str,
        command: str,
        arguments: dict[str, object],
        target_runtime_id: str,
    ) -> dict[str, object]:
        request = BridgeRequest.create(namespace, command, arguments, target_runtime_id)
        slot = self.transport.write_request(request)
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                response = self.transport.read_response(slot, request.request_id)
                if response is not None:
                    if response.runtime_id != target_runtime_id:
                        raise BridgeClientError("bridge response came from a different runtime")
                    if response.status == "error":
                        error = response.error or {}
                        raise BridgeCommandError(
                            str(error.get("code") or "INTERNAL_ERROR"),
                            str(error.get("message") or "bridge command failed"),
                        )
                    return response.result or {}
                await asyncio.sleep(self.poll_interval)
            raise BridgeTimeoutError(f"bridge command '{command}' timed out")
        finally:
            self.transport.release(slot)

