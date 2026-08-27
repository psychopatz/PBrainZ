"""Client for the bounded PsychopatzCore file bridge.

Project Hoomans owns the conversation and exposes two narrow bridge commands:
pollChat and deliverChat. This module is intentionally a small protocol client
rather than a general-purpose command router.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.bridge import BridgeRuntimeMonitor, BridgeState
from hoomans_llm.config import Settings
from hoomans_llm.exceptions import ProviderError
from hoomans_llm.providers.registry import ProviderRegistry

PROTOCOL_VERSION = 1
SLOT_COUNT = 16
MAX_REQUEST_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_STRING = 4096
NAMESPACE = "projecthoomans.llm"
MAX_DELIVERY_TEXT = MAX_STRING - 128
LOGGER = logging.getLogger(__name__)


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


class FileBridgeTransport:
    """Bounded fixed-slot transport matching PsychopatzBridgeFileTransport."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root or os.getenv("ZOMBOID_BRIDGE_ROOT")
        self.root = (
            Path(configured_root)
            if configured_root
            else Path.home() / "Zomboid" / "Lua" / "PsychopatzBridge"
        )
        self.requests = self.root / "requests"
        self.responses = self.root / "responses"
        self.state = self.root / "state"

    def ensure_directories(self) -> None:
        for directory in (self.requests, self.responses, self.state):
            directory.mkdir(parents=True, exist_ok=True)

    def write_request(self, request: BridgeRequest) -> int:
        self.ensure_directories()
        for slot in range(SLOT_COUNT):
            lock_path = self._path(self.requests, slot, ".lock")
            try:
                lock_path.open("x", encoding="ascii").close()
            except FileExistsError:
                continue
            if self._occupied(slot):
                lock_path.unlink(missing_ok=True)
                continue
            request_path = self._path(self.requests, slot, ".json")
            temporary_path = self.requests / f"{self._name(slot)}.{request.request_id}.tmp"
            encoded = json.dumps(
                request.as_dict(), ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
            if len(encoded) > MAX_REQUEST_BYTES:
                lock_path.unlink(missing_ok=True)
                raise BridgeClientError("bridge request exceeds the protocol size limit")
            try:
                temporary_path.write_bytes(encoded)
                os.replace(temporary_path, request_path)
            except OSError as error:
                temporary_path.unlink(missing_ok=True)
                lock_path.unlink(missing_ok=True)
                raise BridgeClientError(f"could not publish bridge request: {error}") from error
            return slot
        raise BridgeClientError("all PsychopatzCore bridge request slots are busy")

    def read_response(self, slot: int, request_id: str) -> BridgeResponse | None:
        marker_path = self._path(self.responses, slot, ".ready.txt")
        try:
            marker = marker_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as error:
            raise BridgeClientError(f"could not read bridge response marker: {error}") from error
        if marker != request_id:
            return None
        response_path = self._path(self.responses, slot, ".json")
        try:
            if response_path.stat().st_size > MAX_RESPONSE_BYTES:
                raise BridgeClientError("bridge response exceeds the protocol size limit")
            value = json.loads(response_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BridgeClientError(f"could not decode bridge response: {error}") from error
        return BridgeResponse.from_dict(value, request_id)

    def release(self, slot: int) -> None:
        for path in (
            self._path(self.requests, slot, ".json"),
            self._path(self.responses, slot, ".json"),
            self._path(self.responses, slot, ".ready.txt"),
            self._path(self.requests, slot, ".lock"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _occupied(self, slot: int) -> bool:
        return any(
            path.exists()
            for path in (
                self._path(self.requests, slot, ".json"),
                self._path(self.responses, slot, ".json"),
                self._path(self.responses, slot, ".ready.txt"),
            )
        )

    @staticmethod
    def _name(slot: int) -> str:
        return f"slot-{slot:02d}"

    @classmethod
    def _path(cls, directory: Path, slot: int, suffix: str) -> Path:
        return directory / f"{cls._name(slot)}{suffix}"


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
        arguments: dict[str, Any],
        target_runtime_id: str,
    ) -> dict[str, Any]:
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


async def run_bridge_pump(
    settings: Settings,
    providers: ProviderRegistry,
    monitor: BridgeRuntimeMonitor,
) -> None:
    """Poll Project Hoomans chat requests and deliver provider responses."""
    transport = FileBridgeTransport(settings.bridge_root)
    client = BridgeClient(
        transport,
        timeout=max(2.0, min(settings.request_timeout, 30.0)),
    )
    while True:
        state = monitor.read()
        if not state.ready or not state.runtime_id:
            await asyncio.sleep(settings.bridge_poll_interval)
            continue
        try:
            request = await client.call(NAMESPACE, "pollChat", {}, state.runtime_id)
            if request.get("status") != "pending":
                await asyncio.sleep(settings.bridge_poll_interval)
                continue
            await _complete_and_deliver(providers, client, state, request)
        except asyncio.CancelledError:
            raise
        except (BridgeClientError, ProviderError, ValueError, TypeError) as error:
            LOGGER.warning("Project Hoomans bridge cycle failed: %s", error)
            await asyncio.sleep(settings.bridge_poll_interval)


async def _complete_and_deliver(
    providers: ProviderRegistry,
    client: BridgeClient,
    state: BridgeState,
    request: dict[str, Any],
) -> None:
    request_id = str(request.get("request_id") or "")
    npc_id = str(request.get("npc_id") or "")
    if not request_id or not npc_id:
        raise ValueError("Project Hoomans returned an incomplete LLM request")
    LOGGER.info("NPC chat request received npc=%s", npc_id)
    provider_name = "unknown"
    model_name = "unknown"
    failure_reason: str | None = None
    try:
        body = ChatCompletionRequest.model_validate(
            {
                "model": request.get("model") or "default",
                "provider": request.get("provider"),
                "messages": request.get("messages"),
                "temperature": request.get("temperature"),
                "max_tokens": request.get("max_tokens"),
                "metadata": {
                    **(request.get("metadata") or {}),
                    "bridge_runtime_id": state.runtime_id,
                    "source": "project-hoomans",
                    "npc_id": npc_id,
                },
            }
        )
        provider_name, model_name = providers.resolve(body.provider, body.model)
        result = await providers.complete(
            provider_name,
            body.model_copy(update={"model": model_name}),
        )
        arguments = {
            "request_id": request_id,
            "npc_id": npc_id,
            "response_text": result.text[:MAX_DELIVERY_TEXT],
        }
    except ProviderError as error:
        failure_reason = error.code
        arguments = {
            "request_id": request_id,
            "npc_id": npc_id,
            "error": error.message[:1024],
        }
    except Exception as error:
        failure_reason = type(error).__name__
        arguments = {
            "request_id": request_id,
            "npc_id": npc_id,
            "error": f"LLM completion failed: {error}"[:1024],
        }
    await client.call(NAMESPACE, "deliverChat", arguments, state.runtime_id)
    if failure_reason:
        LOGGER.warning(
            "NPC chat failed npc=%s provider=%s model=%s reason=%s",
            npc_id,
            provider_name,
            model_name,
            failure_reason,
        )
    else:
        LOGGER.info(
            "NPC chat delivered npc=%s provider=%s model=%s",
            npc_id,
            provider_name,
            model_name,
        )
