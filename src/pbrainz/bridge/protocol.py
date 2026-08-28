"""Shared protocol contracts for the PsychopatzCore file bridge."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
SLOT_COUNT = 16
MAX_REQUEST_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_STRING = 4096
NAMESPACE = "projecthoomans.llm"
CORE_NAMESPACE = "psychopatzcore.bridge"
TOOL_CATALOG_COMMAND = "toolCatalog"
POLL_PACKETS_COMMAND = "pollPackets"
MAX_DELIVERY_TEXT = MAX_STRING - 128


class BridgeClientError(RuntimeError):
    """Base class for local bridge transport and protocol failures."""


class BridgeTimeoutError(BridgeClientError):
    """Raised when the game does not answer a bridge request in time."""


class BridgeCommandError(BridgeClientError):
    """Raised when the game returns a protocol-level command error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class BridgeRequest:
    """A validated request compatible with PsychopatzCore protocol v1."""

    request_id: str
    namespace: str
    command: str
    arguments: dict[str, Any]
    target_runtime_id: str

    @classmethod
    def create(
        cls,
        namespace: str,
        command: str,
        arguments: dict[str, Any],
        target_runtime_id: str,
    ) -> BridgeRequest:
        return cls(
            request_id=uuid.uuid4().hex,
            namespace=namespace,
            command=command,
            arguments=arguments,
            target_runtime_id=target_runtime_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_type": "request",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "namespace": self.namespace,
            "command": self.command,
            "arguments": self.arguments,
            "created_at": int(time.time() * 1000),
            "target_runtime_id": self.target_runtime_id,
        }


@dataclass(frozen=True, slots=True)
class BridgeResponse:
    """A validated response read after the response-ready marker appears."""

    request_id: str
    runtime_id: str
    status: str
    result: dict[str, Any] | None
    error: dict[str, Any] | None

    @classmethod
    def from_dict(cls, value: object, request_id: str) -> BridgeResponse:
        if not isinstance(value, dict):
            raise BridgeClientError("bridge response is not an object")
        if value.get("message_type") != "response":
            raise BridgeClientError("bridge response has an invalid message type")
        if value.get("protocol_version") != PROTOCOL_VERSION:
            raise BridgeClientError("bridge response uses an unsupported protocol")
        if value.get("request_id") != request_id:
            raise BridgeClientError("bridge response ID does not match the request")
        runtime_id = value.get("runtime_id")
        status = value.get("status")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise BridgeClientError("bridge response has no runtime ID")
        if status not in {"ok", "error"}:
            raise BridgeClientError("bridge response has an invalid status")
        result = value.get("result")
        error = value.get("error")
        if result is not None and not isinstance(result, dict):
            raise BridgeClientError("bridge response result is not an object")
        if error is not None and not isinstance(error, dict):
            raise BridgeClientError("bridge response error is not an object")
        return cls(request_id, runtime_id, status, result, error)
