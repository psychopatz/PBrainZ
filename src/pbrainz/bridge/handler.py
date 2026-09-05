"""Complete one game request and deliver its response and presentation state."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace
from json import JSONDecodeError
from typing import Any

from pbrainz.api.models import ChatCompletionRequest
from pbrainz.conversation_runtime import Utterance
from pbrainz.conversation_service import ConversationRequest, ConversationService
from pbrainz.exceptions import ProviderError
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.semantic_tool_protocol import (
    ensure_identity_intent,
    ensure_social_intent,
    extract_text_tool_calls,
    is_provider_scaffold,
    is_social_insult,
    strip_provider_scaffold,
)
from pbrainz.tts import TTSService

from .client import BridgeClient
from .delivery import (
    _build_tts_utterance,
    _publish_speech_fallback,
    _speech_failed_callback,
    _speech_finished_callback,
    _speech_started_callback,
)
from .protocol import NAMESPACE
from .state import BridgeState
from .voice import voice_channel_available

LOGGER = logging.getLogger(__name__)

_NPC_META_RESPONSE_RE = re.compile(
    r"(?:\b(?:i am|i'm)\s+(?:an?\s+)?ai(?:\s+assistant|\s+chatbot)?\b|"
    r"\bas an ai\b|\b(?:language model|large language model)\b|"
    r"\b(?:openai|chatgpt)\b|"
    r"\b(?:i do not|i don't) have (?:a )?(?:personal )?(?:identity|name)\b|"
    r"\b(?:i cannot|i can't) engage with .*?(?:hostile|demeaning)\b|"
    r"\bself[- ]correction\s+check\b|"
    r"\b(?:the\s+)?last\s+turn['’]s\s+instructions\b|"
    r"\b(?:actual\s+)?prompt\s+for\s+the\s+final\s+response\b|"
    r"\bwill\s+assume\s+the\s+player['’]s\s+last\s+message\b|"
    r"\bplayer['’]s\s+last\s+message\s*:|"
    r"\brequired\s+action\s*:)",
    re.IGNORECASE,
)

_AMBIENT_SENTENCE_RE = re.compile(
    r"(.+?[.!?](?:[\"')\]]+)?)(?:\s+|$)",
    re.DOTALL,
)
_AMBIENT_WORD_RE = re.compile(r"\S+")


class _AmbientSpeechStream:
    """Turn safe ambient response deltas into ordered local TTS utterances."""

    MAX_CHUNK_CHARS = 320
    START_WAIT_SECONDS = 4.0

    def __init__(
        self,
        tts_service: TTSService,
        request: dict[str, Any],
        request_id: str,
        npc_id: str,
    ) -> None:
        self.tts_service = tts_service
        self.request = request
        self.request_id = request_id
        self.npc_id = npc_id
        self._buffer = ""
        self._chunk_index = 0
        self.started = False
        self.accepted_chunks = 0
        self.disabled = False
        self.failed = False
        self._started_event = asyncio.Event()
        self.conversation_ids: set[str] = set()

    async def push(self, delta: str) -> None:
        if self.disabled or not delta:
            return
        self._buffer += str(delta)
        await self._flush_ready()

    async def finish(self) -> None:
        if self.disabled:
            self._buffer = ""
            return
        while self._buffer.strip():
            chunk = self._take_sentence()
            if chunk is None:
                chunk = self._buffer.strip()
                self._buffer = ""
            if chunk is None:
                break
            await self._enqueue(chunk)

    def abort(self) -> None:
        self._buffer = ""
        self.disabled = True
        self.failed = True
        self._started_event.set()

    async def cancel_local_audio(self) -> None:
        """Remove chunks already queued when the game rejected the response."""

        self.abort()
        cancel = getattr(self.tts_service, "cancel_conversation", None)
        if not callable(cancel):
            return
        for conversation_id in tuple(self.conversation_ids):
            try:
                await cancel(conversation_id)
            except Exception as error:
                LOGGER.warning(
                    "ambient streaming TTS cancellation failed npc=%s request=%s: %s",
                    self.npc_id,
                    self.request_id,
                    error,
                )

    async def wait_for_playback_start(self) -> bool:
        """Only claim Core's voice channel after local audio really starts."""

        if self.started:
            return True
        if not self.accepted_chunks or self.failed:
            return False
        try:
            await asyncio.wait_for(
                self._started_event.wait(), timeout=self.START_WAIT_SECONDS
            )
        except TimeoutError:
            return False
        return self.started and not self.failed

    async def _flush_ready(self) -> None:
        while not self.disabled:
            chunk = self._take_sentence()
            if chunk is None:
                return
            await self._enqueue(chunk)

    def _take_sentence(self) -> str | None:
        match = _AMBIENT_SENTENCE_RE.search(self._buffer)
        if not match:
            return None
        raw_chunk = match.group(1).strip()
        if not raw_chunk:
            return None
        remainder = self._buffer[match.end() :].lstrip()
        if len(raw_chunk) > self.MAX_CHUNK_CHARS:
            split_at = raw_chunk.rfind(" ", 0, self.MAX_CHUNK_CHARS + 1)
            split_at = split_at if split_at > 0 else self.MAX_CHUNK_CHARS
            chunk = raw_chunk[:split_at].strip()
            tail = raw_chunk[split_at:].lstrip()
            self._buffer = tail + ((" " + remainder) if remainder else "")
            return chunk
        self._buffer = remainder
        return raw_chunk

    async def _enqueue(self, text: str) -> None:
        if self.disabled or not text.strip():
            return
        utterance = _build_tts_utterance(
            self.tts_service,
            self.request,
            f"{self.request_id}:chunk:{self._chunk_index}",
            self.npc_id,
            text,
        )
        if utterance is None:
            self.disabled = True
            return
        utterance = replace(
            utterance,
            utterance_id=f"{utterance.utterance_id}:ambient:{self._chunk_index}",
            turn=utterance.turn + self._chunk_index,
        )
        self.conversation_ids.add(utterance.conversation_id)
        try:
            accepted = await self.tts_service.enqueue(
                utterance,
                on_started=self._on_started,
                on_failed=self._on_failed,
                wait_for_capacity=True,
            )
        except Exception as error:
            self.disabled = True
            LOGGER.warning(
                "ambient streaming TTS stopped npc=%s request=%s: %s",
                self.npc_id,
                self.request_id,
                error,
            )
            return
        if not accepted:
            self.disabled = True
            self.failed = True
            self._started_event.set()
            LOGGER.info(
                "ambient streaming TTS unavailable npc=%s request=%s reason=%s",
                self.npc_id,
                self.request_id,
                self.tts_service.last_error or "queue_rejected",
            )
            return
        self.accepted_chunks += 1
        self._chunk_index += 1
        LOGGER.info(
            "ambient streaming TTS chunk queued npc=%s request=%s chunk=%s words=%s",
            self.npc_id,
            self.request_id,
            self._chunk_index,
            len(_AMBIENT_WORD_RE.findall(text)),
        )

    async def _on_started(self, _utterance: Utterance) -> None:
        self.started = True
        self._started_event.set()

    async def _on_failed(self, _utterance: Utterance, error: Exception) -> None:
        self.failed = True
        self._started_event.set()
        LOGGER.warning(
            "ambient streaming TTS chunk failed npc=%s request=%s: %s",
            self.npc_id,
            self.request_id,
            error,
        )


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
    if conversation_service and conversation_service.debug_trace_enabled():
        conversation_service.record_debug_trace(
            "bridge.request",
            request,
            request_id=request_id,
            npc_id=npc_id,
            session_id=str(
                (request.get("conversation_context") or request.get("context") or {}).get(
                    "session_id", ""
                )
                if isinstance(request.get("conversation_context") or request.get("context"), dict)
                else ""
            ),
            source="project-hoomans.bridge",
        )
    tts_utterance: Utterance | None = None
    ambient_tts: _AmbientSpeechStream | None = None
    tts_managed = False
    structured = bool(
        request.get("conversation_context")
        or request.get("world_uuid")
        or request.get("worldUUID")
    )
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
        if structured:
            if conversation_service is None:
                raise ValueError("conversation service is unavailable")
            conversation_request = ConversationRequest.from_mapping(request)
            if (
                tts_service
                and tts_service.enabled
                and _is_ambient_request(request)
            ):
                ambient_tts = _AmbientSpeechStream(
                    tts_service,
                    request,
                    request_id,
                    npc_id,
                )
            conversation_result = await conversation_service.complete(
                conversation_request,
                stream_consumer=ambient_tts.push if ambient_tts else None,
            )
            if ambient_tts:
                await ambient_tts.finish()
                tts_managed = await ambient_tts.wait_for_playback_start()
                if ambient_tts.accepted_chunks and not tts_managed:
                    await ambient_tts.cancel_local_audio()
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
        response_text = str(result.text or "").strip()
        semantic_tool_calls = (
            semantic_tool_calls_for(result.tool_calls, request) if structured else []
        )
        if structured:
            response_text, text_tool_calls = extract_text_tool_calls(
                response_text, request
            )
            semantic_tool_calls.extend(text_tool_calls)
            semantic_tool_calls = ensure_social_intent(
                semantic_tool_calls, request
            )
            semantic_tool_calls = ensure_identity_intent(
                semantic_tool_calls, request
            )
            response_text = sanitize_npc_response(
                response_text,
                request,
                semantic_tool_calls,
            )
        presentation_reason: str | None = None
        if (
            structured
            and conversation_service
            and (
                conversation_result.diagnostics.get("empty_response_retry")
                == "text_only_authorized_tools"
                or conversation_result.diagnostics.get("contextual_response_retry")
                == "explicit_social_subtype"
            )
            and response_text
        ):
            presentation_reason = "llm_tool_response_repair"
        if not response_text and semantic_tool_calls:
            # Providers commonly return a tool-only assistant turn.  The game
            # will execute the semantic call, but it still needs one shared
            # piece of dialogue for the conversation log, nameplate, and TTS.
            # Keep this deliberately non-committal: acceptance is decided by
            # the game after delivery, so this must not claim that the action
            # already succeeded.
            response_text = tool_ack_text(request, semantic_tool_calls)
            presentation_reason = "tool_ack"
        arguments: dict[str, Any] = {
            "request_id": request_id,
            "npc_id": npc_id,
            # Dialogue is bounded by the bridge response packet size, not by
            # an arbitrary per-field character limit.  The latter used to
            # silently cut valid NPC replies before Project Hoomans saw them.
            "response_text": response_text,
            "finish_reason": str(result.finish_reason or "unknown")[:128],
            "tool_call_count": len(result.tool_calls or []),
        }
        if presentation_reason:
            arguments["presentation_reason"] = presentation_reason
            arguments["tool_result_pending"] = True
        if structured and conversation_service and conversation_service.settings.llm_diagnostics:
            arguments["diagnostics"] = conversation_result.diagnostics
        if semantic_tool_calls:
            arguments["semantic_tool_calls"] = semantic_tool_calls
        if tts_managed:
            arguments["tts_managed"] = True
        if not response_text and not semantic_tool_calls:
            failure_reason = "provider_empty_response"
            arguments["error"] = (
                "LLM provider returned an empty response "
                f"(finish_reason={str(result.finish_reason or 'unknown')[:128]}, "
                f"tool_calls={len(result.tool_calls or [])})."
            )
        # Core's generic voice channel now owns presentation for all
        # canonical NPC messages, including authored dialogue. Keep the old
        # request-scoped TTS path only for older game/Core runtimes that do
        # not advertise that channel.
        if (
            not failure_reason
            and tts_service
            and tts_service.enabled
            and not tts_managed
            and not voice_channel_available(state)
        ):
            tts_utterance = _build_tts_utterance(
                tts_service,
                request,
                request_id,
                npc_id,
                response_text,
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
            "NPC response received from provider npc=%s request=%s provider=%s model=%s "
            "finish_reason=%s provider_tool_calls=%s semantic_tool_calls=%s text=%s",
            npc_id,
            request_id,
            provider_name,
            model_name,
            result.finish_reason or "unknown",
            len(result.tool_calls or []),
            len(semantic_tool_calls),
            preview(response_text),
        )
    except ProviderError as error:
        if ambient_tts:
            await ambient_tts.cancel_local_audio()
        failure_reason = error.code
        arguments = {
            "request_id": request_id,
            "npc_id": npc_id,
            "error": error.message[:1024],
        }
        if structured:
            fallback_calls = ensure_social_intent(
                [], request, reason="provider_request_failed"
            )
            fallback_calls = ensure_identity_intent(
                fallback_calls, request, reason="provider_request_failed"
            )
            if fallback_calls:
                arguments["semantic_tool_calls"] = fallback_calls
            arguments["response_text"] = npc_fallback_response(
                request, fallback_calls
            )
            arguments["provider_failure"] = True
            arguments["context_eligible"] = False
    except Exception as error:
        if ambient_tts:
            await ambient_tts.cancel_local_audio()
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
    if conversation_service and conversation_service.debug_trace_enabled():
        conversation_service.record_debug_trace(
            "bridge.delivery",
            {
                "provider": provider_name,
                "model": model_name,
                "failure_reason": failure_reason,
                "arguments": arguments,
            },
            request_id=request_id,
            npc_id=npc_id,
            session_id=(
                conversation_result.session_id
                if structured and "conversation_result" in locals()
                else ""
            ),
            source="project-hoomans.bridge",
        )
    try:
        delivery = await client.call(NAMESPACE, "deliverChat", arguments, state.runtime_id or "")
    except Exception:
        if ambient_tts:
            await ambient_tts.cancel_local_audio()
        raise
    if ambient_tts and delivery.get("accepted") is False:
        LOGGER.info(
            "ambient response was rejected by Project Hoomans; canceling local TTS "
            "npc=%s request=%s",
            npc_id,
            request_id,
        )
        await ambient_tts.cancel_local_audio()
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
        LOGGER.info(
            "NPC TTS enqueue %s npc=%s request=%s utterance=%s reason=%s",
            "accepted" if accepted else "rejected",
            npc_id,
            request_id,
            tts_utterance.utterance_id,
            "ready" if accepted else (tts_service.last_error or "unknown"),
        )
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


def _is_ambient_request(request: dict[str, Any]) -> bool:
    context = request.get("conversation_context") or request.get("context")
    if not isinstance(context, dict):
        return False
    metadata = context.get("metadata")
    return (
        isinstance(metadata, dict)
        and str(metadata.get("mode") or "").strip().casefold()
        == "ambient_social"
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


def tool_ack_text(
    request: dict[str, Any],
    semantic_tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """Return safe dialogue for a tool-only turn before game validation."""
    context = request.get("conversation_context") or request.get("context")
    if isinstance(context, dict):
        for key in ("tool_ack_text", "toolAckText"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    names = {
        str(call.get("name") or "").strip()
        for call in semantic_tool_calls or []
        if isinstance(call, dict)
    }
    if "ask_name" in names:
        return "Sure. Let me introduce myself."
    if "social_react" in names:
        return "I hear you."
    if any(name.startswith("order_") for name in names):
        return "All right."
    return "Give me a moment."


def sanitize_npc_response(
    response: str,
    request: dict[str, Any],
    semantic_tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """Keep provider identity/policy boilerplate out of NPC dialogue."""
    response = str(response or "").strip()
    has_meta = bool(response and _NPC_META_RESPONSE_RE.search(response))
    has_scaffold = is_provider_scaffold(response)
    if not response or not (has_meta or has_scaffold):
        return response
    fallback = npc_fallback_response(request, semantic_tool_calls)
    if has_scaffold and not has_meta:
        cleaned = strip_provider_scaffold(response)
        if cleaned:
            return cleaned
    LOGGER.warning(
        "NPC provider response replaced meta boilerplate npc=%s request=%s reaction=%s",
        str(request.get("npc_id") or "unknown"),
        str(request.get("request_id") or "unknown"),
        reaction_from_tool_calls(semantic_tool_calls) or "none",
    )
    return fallback


def reaction_from_tool_calls(
    semantic_tool_calls: list[dict[str, Any]] | None,
) -> str:
    for call in semantic_tool_calls or []:
        if not isinstance(call, dict) or call.get("name") != "social_react":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            return str(
                arguments.get("kind") or arguments.get("reaction") or ""
            ).strip()
    return ""


def npc_fallback_response(
    request: dict[str, Any],
    semantic_tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    reaction = reaction_from_tool_calls(semantic_tool_calls)
    if not reaction and is_social_insult(request_message(request)):
        reaction = "insult"
    fallback = {
        "insult": "Watch your mouth.",
        "praise": "I appreciate that.",
        "admire": "That means something coming from you.",
        "comfort": "Thanks. I needed that.",
        "apologize": "All right. Let's move on.",
        "apology": "Fine. Just don't do it again.",
        "flirt": "Careful. You might get the wrong idea.",
        "greeting": "Hello.",
        "farewell": "Stay safe.",
    }.get(reaction, "Give me a moment.")
    return fallback


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
        exposed_tools = []
    exposed: set[str] = set()
    for tool in exposed_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if name:
            exposed.add(str(name))
    tool_ids = context.get("available_tool_ids") or []
    if not isinstance(tool_ids, list):
        tool_ids = []
    for tool_id in tool_ids:
        if not isinstance(tool_id, str):
            continue
        prefix = "projecthoomans.llm:"
        if tool_id.startswith(prefix):
            name = tool_id[len(prefix) :].strip()
            if name:
                exposed.add(name)
    normalized: list[dict[str, Any]] = []
    rejected: list[str] = []
    for call in tool_calls[:8]:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or "").strip()
        if not name or name not in exposed:
            rejected.append(name or "<missing>")
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
    if rejected:
        LOGGER.warning(
            "NPC provider tool calls rejected npc=%s request=%s rejected=%s exposed=%s",
            str(request.get("npc_id") or "unknown"),
            str(request.get("request_id") or "unknown"),
            ",".join(rejected[:8]),
            ",".join(sorted(exposed)[:16]) or "<none>",
        )
    return normalized


def add_fallback_social_tool_call(
    calls: list[dict[str, Any]],
    request: dict[str, Any],
    *,
    reason: str = "provider_no_native_tool",
) -> list[dict[str, Any]]:
    """Compatibility name for the provider-neutral social intent stage."""
    return ensure_social_intent(calls, request, reason=reason)
