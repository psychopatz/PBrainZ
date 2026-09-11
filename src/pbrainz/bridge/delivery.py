"""TTS lifecycle event delivery for bridge-backed NPC conversations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pbrainz.conversation_runtime import AudioPresentation, Utterance, VoiceBinding
from pbrainz.tts import TTSService
from pbrainz.tts.text import normalize_tts_text

from .client import BridgeClient
from .protocol import NAMESPACE
from .state import BridgeState

LOGGER = logging.getLogger(__name__)


def _build_tts_utterance(
    tts_service: TTSService,
    request: dict[str, object],
    request_id: str,
    npc_id: str,
    text: str,
) -> Utterance | None:
    """Build a presentation candidate without allowing TTS to affect LLM flow."""

    speech_text = normalize_tts_text(text)
    if not speech_text:
        return None
    try:
        context = request.get("conversation_context") or request.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        conversation_id = str(
            context.get("conversation_id")
            or context.get("conversationID")
            or context.get("session_id")
            or request_id
        )[:256]
        binding = tts_service.resolve_voice_binding(
            conversation_id,
            npc_id,
            VoiceBinding.from_mapping(
                context.get("voice_binding") or context.get("voiceBinding")
            ),
        )
        if binding is None:
            LOGGER.warning(
                "NPC TTS skipped npc=%s request=%s conversation=%s reason=voice_binding_missing",
                npc_id,
                request_id,
                conversation_id,
            )
            return None
        if not tts_service.can_synthesize(binding):
            LOGGER.warning(
                "NPC TTS skipped npc=%s request=%s conversation=%s reason=tts_unavailable "
                "slot=%s",
                npc_id,
                request_id,
                conversation_id,
                binding.slot,
            )
            return None
        audio_presentation = AudioPresentation.from_mapping(
            context.get("audio_presentation")
            or context.get("audioPresentation")
            or context.get("audio_context")
            or context.get("audioContext")
            or context.get("speech")
            or context.get("speech_policy")
            or context
        )
        utterance = Utterance(
            utterance_id=f"{conversation_id}:{request_id}",
            conversation_id=conversation_id,
            turn=int(context.get("turn") or 0),
            speaker_npc_uuid=npc_id,
            text=speech_text,
            voice_binding=binding,
            audio_presentation=audio_presentation,
        )
        LOGGER.info(
            "NPC TTS utterance prepared npc=%s request=%s conversation=%s slot=%s chars=%s",
            npc_id,
            request_id,
            conversation_id,
            binding.slot,
            len(utterance.text),
        )
        return utterance
    except Exception as error:
        tts_service.last_error = f"TTS preparation failed: {error}"[:500]
        LOGGER.warning("TTS preparation failed; keeping the response text-only: %s", error)
        return None


def _speech_started_callback(
    client: BridgeClient, state: BridgeState, request_id: str
) -> Callable[[Utterance], Awaitable[None]]:
    async def publish(utterance: Utterance) -> None:
        delivered = await _publish_speech_event(
            client,
            state,
            "speechStarted",
            {"request_id": request_id, **utterance.as_event()},
        )
        if not delivered:
            await _deliver_text_fallback(
                client, state, request_id, utterance, "speech start event unavailable"
            )

    return publish


def _speech_finished_callback(
    client: BridgeClient, state: BridgeState, request_id: str
) -> Callable[[Utterance], Awaitable[None]]:
    async def publish(utterance: Utterance) -> None:
        await _publish_speech_event(
            client,
            state,
            "speechFinished",
            {
                "request_id": request_id,
                "conversation_id": utterance.conversation_id,
                "utterance_id": utterance.utterance_id,
                "npc_uuid": utterance.speaker_npc_uuid,
            },
        )

    return publish


def _speech_failed_callback(
    client: BridgeClient, state: BridgeState, request_id: str
) -> Callable[[Utterance, Exception], Awaitable[None]]:
    async def publish(utterance: Utterance, error: Exception) -> None:
        await _publish_speech_fallback(client, state, request_id, utterance, str(error))

    return publish


async def _publish_speech_fallback(
    client: BridgeClient,
    state: BridgeState,
    request_id: str,
    utterance: Utterance,
    error: str,
) -> None:
    delivered = await _publish_speech_event(
        client,
        state,
        "speechFallback",
        {
            "request_id": request_id,
            "conversation_id": utterance.conversation_id,
            "utterance_id": utterance.utterance_id,
            "npc_uuid": utterance.speaker_npc_uuid,
            "text": utterance.text,
            "error": error[:500],
        },
    )
    if not delivered:
        await _deliver_text_fallback(client, state, request_id, utterance, error)


async def _publish_speech_event(
    client: BridgeClient,
    state: BridgeState,
    command: str,
    arguments: dict[str, object],
) -> bool:
    try:
        result = await client.call(NAMESPACE, command, arguments, state.runtime_id or "")
        return result.get("accepted") is not False
    except Exception as error:
        # Audio must not be coupled to game availability.  The local playback
        # scheduler continues even if the subtitle event cannot be delivered.
        LOGGER.warning("speech event %s could not reach Project Hoomans: %s", command, error)
        return False


async def _deliver_text_fallback(
    client: BridgeClient,
    state: BridgeState,
    request_id: str,
    utterance: Utterance,
    reason: str,
) -> None:
    """Release a TTS-pending game conversation if lifecycle events are absent."""

    try:
        await client.call(
            NAMESPACE,
            "deliverChat",
            {
                "request_id": request_id,
                "npc_id": utterance.speaker_npc_uuid,
                "response_text": utterance.text,
                "presentation_mode": "text_only",
                "error": reason[:500],
            },
            state.runtime_id or "",
        )
    except Exception as error:
        LOGGER.warning("text-only fallback could not reach Project Hoomans: %s", error)
