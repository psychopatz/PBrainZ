"""Complete one game request and deliver its response and presentation state."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from typing import Any

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.conversation_runtime import Utterance
from hoomans_llm.conversation_service import ConversationRequest, ConversationService
from hoomans_llm.exceptions import ProviderError
from hoomans_llm.providers.registry import ProviderRegistry
from hoomans_llm.tts import TTSService

from .client import BridgeClient
from .delivery import (
    _build_tts_utterance,
    _publish_speech_fallback,
    _speech_failed_callback,
    _speech_finished_callback,
    _speech_started_callback,
)
from .protocol import MAX_DELIVERY_TEXT, NAMESPACE
from .state import BridgeState

LOGGER = logging.getLogger(__name__)


async def complete_and_deliver(
    providers: ProviderRegistry,
    client: BridgeClient,
    state: BridgeState,
    request: dict[str, Any],
    *,
    conversation_service: ConversationService | None = None,
    tts_service: TTSService | None = None,
) -> None:
    request_id = str(request.get("request_id") or "")
    npc_id = str(request.get("npc_id") or "")
    if not request_id or not npc_id:
        raise ValueError("Project Hoomans returned an incomplete LLM request")
    tts_utterance: Utterance | None = None
    LOGGER.info(
        "NPC provider task started npc=%s request=%s message=%s",
        npc_id,
        request_id,
        preview(request_message(request)),
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
        arguments: dict[str, Any] = {
            "request_id": request_id,
            "npc_id": npc_id,
            "response_text": result.text[:MAX_DELIVERY_TEXT],
        }
        if structured and conversation_service and conversation_service.settings.llm_diagnostics:
            arguments["diagnostics"] = conversation_result.diagnostics
        if structured:
            semantic_tool_calls = semantic_tool_calls_for(result.tool_calls, request)
            if semantic_tool_calls:
                arguments["semantic_tool_calls"] = semantic_tool_calls
        if not failure_reason and tts_service and tts_service.enabled:
            tts_utterance = _build_tts_utterance(
                tts_service,
                request,
                request_id,
                npc_id,
                result.text,
            )
            if tts_utterance:
                arguments.update(
                    {
                        "presentation_mode": "tts",
                        "utterance_id": tts_utterance.utterance_id,
                        "conversation_id": tts_utterance.conversation_id,
                    }
                )
        LOGGER.info(
            "NPC response received from provider npc=%s request=%s provider=%s model=%s text=%s",
            npc_id,
            request_id,
            provider_name,
            model_name,
            preview(result.text),
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
        preview(arguments.get("response_text")),
        preview(arguments.get("error")),
    )
    await client.call(NAMESPACE, "deliverChat", arguments, state.runtime_id or "")
    if tts_utterance and tts_service:
        try:
            accepted = await tts_service.enqueue(
                tts_utterance,
                on_started=_speech_started_callback(client, state, request_id),
                on_finished=_speech_finished_callback(client, state, request_id),
                on_failed=_speech_failed_callback(client, state, request_id),
            )
        except Exception as error:
            tts_service.last_error = f"TTS queue failure: {error}"[:500]
            LOGGER.warning("TTS queue failed; requesting text-only fallback: %s", error)
            accepted = False
        if not accepted:
            await _publish_speech_fallback(
                client,
                state,
                request_id,
                tts_utterance,
                tts_service.last_error or "TTS queue rejected the utterance",
            )
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


def request_message(request: dict[str, Any]) -> str:
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


def preview(value: object, limit: int = 1200) -> str:
    rendered = " ".join(str(value or "").split())
    if not rendered:
        return "<empty>"
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def semantic_tool_calls_for(
    tool_calls: list[dict[str, Any]] | None,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Forward only tools Project Hoomans explicitly exposed for this turn."""
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
                "arguments": {
                    str(key): value for key, value in list(raw_arguments.items())[:16]
                },
            }
        )
    return normalized

