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
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.bridge import BridgeRuntimeMonitor, BridgeState
from hoomans_llm.config import Settings
from hoomans_llm.conversation_service import ConversationRequest, ConversationService
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

    def recover_stale_slots(self, runtime_id: str) -> int:
        """Release slot files that target a previous game runtime.

        A server or game restart can leave request/response files behind. The
        game intentionally rejects those requests as stale, and the leftover
        slots can delay a fresh request. Only files whose validated runtime
        identity disagrees with the current runtime are removed.
        """
        self.ensure_directories()
        recovered = 0
        for slot in range(SLOT_COUNT):
            request = self._read_object(self._path(self.requests, slot, ".json"))
            response = self._read_object(self._path(self.responses, slot, ".json"))
            request_runtime = request.get("target_runtime_id") if request else None
            response_runtime = response.get("runtime_id") if response else None
            stale_request = isinstance(request_runtime, str) and bool(
                request_runtime
            ) and request_runtime != runtime_id
            stale_response = isinstance(response_runtime, str) and bool(
                response_runtime
            ) and response_runtime != runtime_id
            if stale_request or stale_response:
                self.release(slot)
                recovered += 1
        return recovered

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
    def _read_object(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

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
    conversation_service = ConversationService(settings, providers)
    observed_runtime_id: str | None = None
    observed_state: tuple[bool, bool, str | None, str | None] | None = None
    while True:
        state = monitor.read()
        state_signature = (state.available, state.ready, state.runtime_id, state.lifecycle)
        if state_signature != observed_state:
            LOGGER.info(
                "Project Hoomans bridge state available=%s ready=%s "
                "lifecycle=%s runtime=%s message=%s",
                state.available,
                state.ready,
                state.lifecycle or "unknown",
                state.runtime_id or "unknown",
                state.message,
            )
            observed_state = state_signature
        if state.ready and state.runtime_id and state.runtime_id != observed_runtime_id:
            recovered = transport.recover_stale_slots(state.runtime_id)
            LOGGER.info(
                "Project Hoomans bridge runtime selected runtime=%s "
                "stale_slots_recovered=%s root=%s",
                state.runtime_id,
                recovered,
                transport.root,
            )
            observed_runtime_id = state.runtime_id
        if not state.ready or not state.runtime_id:
            await asyncio.sleep(settings.bridge_poll_interval)
            continue
        try:
            request = await client.call(NAMESPACE, "pollChat", {}, state.runtime_id)
            if request.get("status") != "pending":
                await asyncio.sleep(settings.bridge_poll_interval)
                continue
            LOGGER.info(
                "NPC task received from Project Hoomans npc=%s request=%s message=%s",
                str(request.get("npc_id") or "unknown"),
                str(request.get("request_id") or "unknown"),
                _preview(_request_message(request)),
            )
            await _complete_and_deliver(
                providers,
                client,
                state,
                request,
                conversation_service=conversation_service,
            )
        except asyncio.CancelledError:
            raise
        except (BridgeClientError, ProviderError, ValueError, TypeError) as error:
            LOGGER.warning(
                "Project Hoomans bridge cycle failed root=%s runtime=%s: %s",
                transport.root,
                state.runtime_id or "unknown",
                error,
            )
            await asyncio.sleep(settings.bridge_poll_interval)


async def _complete_and_deliver(
    providers: ProviderRegistry,
    client: BridgeClient,
    state: BridgeState,
    request: dict[str, Any],
    *,
    conversation_service: ConversationService | None = None,
) -> None:
    request_id = str(request.get("request_id") or "")
    npc_id = str(request.get("npc_id") or "")
    if not request_id or not npc_id:
        raise ValueError("Project Hoomans returned an incomplete LLM request")
    LOGGER.info(
        "NPC provider task started npc=%s request=%s message=%s",
        npc_id,
        request_id,
        _preview(_request_message(request)),
    )
    provider_name = "unknown"
    model_name = "unknown"
    failure_reason: str | None = None
    try:
        structured = bool(
            request.get("conversation_context")
            or request.get("world_uuid")
            or request.get("worldUUID")
        )
        if structured:
            if conversation_service is None:
                raise ValueError("conversation service is unavailable")
            conversation_request = ConversationRequest.from_mapping(request)
            conversation_result = await conversation_service.complete(conversation_request)
            result = conversation_result.completion
            provider_name = str(conversation_result.diagnostics.get("provider", "unknown"))
            model_name = str(conversation_result.diagnostics.get("model", result.model))
            if conversation_service.settings.llm_diagnostics:
                LOGGER.info(
                    "NPC context built npc=%s session=%s recent=%s retrieved=%s chars=%s",
                    npc_id,
                    conversation_result.session_id,
                    conversation_result.diagnostics.get("context", {}).get("recent_turns", 0),
                    len(conversation_result.retrieved_memories),
                    conversation_result.diagnostics.get("context", {}).get("context_chars", 0),
                )
        else:
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
        if structured and conversation_service and conversation_service.settings.llm_diagnostics:
            arguments["diagnostics"] = conversation_result.diagnostics
        if structured:
            semantic_tool_calls = _semantic_tool_calls(
                result.tool_calls,
                request,
            )
            if semantic_tool_calls:
                arguments["semantic_tool_calls"] = semantic_tool_calls
        LOGGER.info(
            "NPC response received from provider npc=%s request=%s provider=%s model=%s text=%s",
            npc_id,
            request_id,
            provider_name,
            model_name,
            _preview(result.text),
        )
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
    LOGGER.info(
        "NPC task sent to Project Hoomans npc=%s request=%s response=%s error=%s",
        npc_id,
        request_id,
        _preview(arguments.get("response_text")),
        _preview(arguments.get("error")),
    )
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
            "NPC chat delivered npc=%s request=%s provider=%s model=%s",
            npc_id,
            request_id,
            provider_name,
            model_name,
        )


def _request_message(request: dict[str, Any]) -> str:
    context = request.get("conversation_context") or request.get("context")
    if isinstance(context, dict):
        for key in ("message", "current_player_message", "currentPlayerMessage"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            value = message.get("content")
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _preview(value: object, limit: int = 1200) -> str:
    rendered = " ".join(str(value or "").split())
    if not rendered:
        return "<empty>"
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _semantic_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Forward only tools Project Hoomans explicitly exposed for this turn.

    The returned values are still untrusted semantic intents. Lua validates the
    command ID through the existing client/authority command registry before
    any gameplay action can be sent.
    """
    if not tool_calls:
        return []
    context = request.get("conversation_context") or request.get("context") or request
    if not isinstance(context, dict):
        return []
    exposed_tools = context.get("available_tools", context.get("availableTools", []))
    if not isinstance(exposed_tools, list):
        return []
    exposed: set[str] = set()
    for tool in exposed_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if name:
            exposed.add(str(name))
    normalized: list[dict[str, Any]] = []
    for call in tool_calls[:8]:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or "").strip()
        if not name or name not in exposed:
            continue
        raw_arguments = function.get("arguments") or {}
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except JSONDecodeError:
                raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        normalized.append(
            {
                "id": str(call.get("id") or ""),
                "name": name,
                "arguments": {str(key): value for key, value in list(raw_arguments.items())[:16]},
            }
        )
    return normalized
