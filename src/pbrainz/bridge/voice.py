"""Consume Core's generic voice packets and enqueue them into local TTS."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

from pbrainz.conversation_runtime import (
    AudioPresentation,
    SpeechMode,
    Utterance,
    VoiceBinding,
)
from pbrainz.tts import TTSService
from pbrainz.tts.text import normalize_tts_text

from .client import BridgeClient
from .protocol import BridgeClientError
from .state import BridgeState
from .streams import PacketStreamClient

LOGGER = logging.getLogger(__name__)

VOICE_NAMESPACE = "psychopatzcore.voice"
VOICE_CHANNEL = "utterances"
VOICE_EVENT_TYPE = "speech.enqueue"
VOICE_LIFECYCLE_COMMANDS = frozenset(
    {"speechStarted", "speechFinished", "speechFailed"}
)
MAX_VOICE_EVENTS_PER_POLL = 32
MAX_RECENT_UTTERANCES = 512
MAX_LIFECYCLE_VALUE = 256
DEFAULT_EXPIRY_MS = 60_000
MAX_RETRY_PACKETS = 64
MAX_RETRY_ATTEMPTS = 60


def voice_channel_available(state: BridgeState) -> bool:
    """Return whether the current Core runtime advertises the voice channel."""

    return any(
        isinstance(channel, dict)
        and channel.get("namespace") == VOICE_NAMESPACE
        and channel.get("channel") == VOICE_CHANNEL
        for channel in state.packet_channels
    )


def _value(packet: dict[str, Any], *names: str) -> object:
    for name in names:
        if name in packet:
            return packet[name]
    return None


def _text(value: object, limit: int = MAX_LIFECYCLE_VALUE) -> str:
    return str(value or "").strip()[:limit]


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (OverflowError, TypeError, ValueError):
        return default


def _boolean(value: object) -> bool:
    return value is True


def _speech_mapping(packet: dict[str, Any]) -> dict[str, Any]:
    speech = packet.get("speech")
    return speech if isinstance(speech, dict) else {}


def _is_expired(packet: dict[str, Any], now_ms: int) -> bool:
    created = _integer(
        _value(packet, "created_real_time_ms", "createdRealTimeMs"), 0
    )
    if created <= 0 or now_ms <= created:
        return False
    speech = _speech_mapping(packet)
    expiry = _integer(
        _value(packet, "expires_after_ms", "expiresAfterMs")
        or _value(speech, "expires_after_ms", "expiresAfterMs"),
        DEFAULT_EXPIRY_MS,
    )
    expiry = max(1000, min(expiry, 10 * 60 * 1000))
    return now_ms - created > expiry


def utterance_from_packet(
    packet: object,
    tts_service: TTSService,
    *,
    now_ms: int | None = None,
) -> Utterance | None:
    """Validate one wire packet and convert it to a scheduler utterance."""

    if not isinstance(packet, dict):
        raise ValueError("voice packet is not an object")
    if packet.get("schema_version") != 1:
        raise ValueError("voice packet uses an unsupported schema")
    if packet.get("event_type") != VOICE_EVENT_TYPE:
        raise ValueError("voice packet has an unsupported event type")
    if _is_expired(packet, int(time.time() * 1000) if now_ms is None else now_ms):
        return None
    raw_text = _text(_value(packet, "text"), 12000)
    text = normalize_tts_text(raw_text)
    conversation_id = _text(_value(packet, "conversation_id", "conversationID"))
    speaker_id = _text(
        _value(packet, "speaker_id", "speakerID", "npc_uuid", "npcUUID")
    )
    speaker_kind = _text(
        _value(packet, "speaker_kind", "speakerKind") or "npc"
    ).lower()
    if speaker_kind not in {"npc", "player"}:
        raise ValueError("voice packet has an unsupported speaker kind")
    utterance_id = _text(
        _value(packet, "utterance_id", "utteranceID", "message_id", "messageID")
    )
    if not raw_text or not conversation_id or not speaker_id or not utterance_id:
        raise ValueError("voice packet is missing utterance identity or text")
    if not text:
        return None

    raw_binding = _value(packet, "voice_binding", "voiceBinding")
    binding = VoiceBinding.from_mapping(raw_binding)
    if speaker_kind == "player":
        resolved_binding = tts_service.resolve_voice_binding(
            conversation_id, speaker_id, binding, "player"
        )
    else:
        # Keep the legacy three-argument call path for older test doubles and
        # external adapters while the default remains NPC-compatible.
        resolved_binding = tts_service.resolve_voice_binding(
            conversation_id, speaker_id, binding
        )
    if not isinstance(resolved_binding, VoiceBinding):
        resolved_binding = VoiceBinding.from_mapping(resolved_binding)
    if resolved_binding is None:
        return None

    speech = _speech_mapping(packet)
    mode_value = str(
        _value(speech, "mode", "speech_mode") or SpeechMode.RESPONSE.value
    ).upper()
    try:
        mode = SpeechMode(mode_value)
    except ValueError:
        mode = SpeechMode.RESPONSE
    audio_presentation = AudioPresentation.from_mapping(speech)
    return Utterance(
        utterance_id=utterance_id,
        conversation_id=conversation_id,
        turn=max(0, _integer(_value(packet, "sequence"), 0)),
        speaker_npc_uuid=speaker_id,
        text=text,
        speaker_kind=speaker_kind,
        speech_mode=mode,
        allow_overlap=_boolean(
            _value(speech, "allow_overlap", "allowOverlap")
        ),
        can_interrupt=_boolean(
            _value(speech, "can_interrupt", "canInterrupt")
        ),
        voice_binding=resolved_binding,
        audio_presentation=audio_presentation,
        created_at=time.monotonic(),
    )


class VoicePacketConsumer:
    """Bounded, de-duplicating adapter from Core packets to local TTS."""

    def __init__(
        self,
        tts_service: TTSService,
        stream_client: PacketStreamClient | None = None,
    ) -> None:
        self.tts_service = tts_service
        self.stream_client = stream_client or PacketStreamClient()
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._retry: OrderedDict[str, tuple[dict[str, Any], int, float]] = OrderedDict()

    def reset(self) -> None:
        self.stream_client.reset()
        self._recent.clear()
        self._retry.clear()

    async def poll(self, client: BridgeClient, state: BridgeState) -> int:
        if not self.tts_service.enabled or not voice_channel_available(state):
            return 0
        result = await self.stream_client.poll(
            client,
            state,
            [
                {
                    "namespace": VOICE_NAMESPACE,
                    "channel": VOICE_CHANNEL,
                    "limit": MAX_VOICE_EVENTS_PER_POLL,
                }
            ],
        )
        return await self.consume(client, state, result)

    async def consume(
        self,
        client: BridgeClient,
        state: BridgeState,
        result: object,
    ) -> int:
        if not isinstance(result, dict):
            raise BridgeClientError("voice packet response is invalid")
        streams = result.get("streams")
        if not isinstance(streams, list):
            raise BridgeClientError("voice packet response has no streams")
        accepted = 0
        retry_rows = list(self._retry.values())
        self._retry.clear()
        for packet, attempts, next_at in retry_rows:
            if next_at <= time.monotonic():
                result_value = await self._enqueue_packet(
                    client, state, packet, attempts
                )
                if result_value is True:
                    accepted += 1
            else:
                self._retry[self._packet_id(packet, {})] = (
                    packet, attempts, next_at
                )
        for stream in streams:
            if not isinstance(stream, dict) or stream.get("error"):
                continue
            events = stream.get("events")
            if not isinstance(events, list):
                raise BridgeClientError("voice packet stream has invalid events")
            for row in events[:MAX_VOICE_EVENTS_PER_POLL]:
                if not isinstance(row, dict):
                    continue
                packet = row.get("packet")
                packet_id = self._packet_id(packet, row)
                if not packet_id or packet_id in self._recent:
                    continue
                self._remember(packet_id)
                result_value = await self._enqueue_packet(
                    client, state, packet, 0
                )
                if result_value is True:
                    accepted += 1
        return accepted

    async def _enqueue_packet(
        self,
        client: BridgeClient,
        state: BridgeState,
        packet: object,
        attempts: int,
    ) -> bool | None:
        try:
            utterance = utterance_from_packet(packet, self.tts_service)
            if utterance is None:
                return None
            metadata = self._metadata(packet, utterance)
            queued = await self.tts_service.enqueue(
                utterance,
                on_started=self._lifecycle_callback(
                    client, state, "speechStarted", metadata
                ),
                on_finished=self._lifecycle_callback(
                    client, state, "speechFinished", metadata
                ),
                on_failed=self._failure_callback(
                    client, state, metadata
                ),
            )
            if queued:
                return True
            if attempts < MAX_RETRY_ATTEMPTS:
                packet_id = self._packet_id(packet, {})
                if packet_id and len(self._retry) < MAX_RETRY_PACKETS:
                    delay = min(2.0, 0.25 * (attempts + 1))
                    self._retry[packet_id] = (
                        packet if isinstance(packet, dict) else {},
                        attempts + 1,
                        time.monotonic() + delay,
                    )
            LOGGER.debug(
                "Core voice packet is waiting for TTS capacity utterance=%s attempt=%s",
                utterance.utterance_id,
                attempts + 1,
            )
            return False
        except (ValueError, TypeError, OverflowError) as error:
            LOGGER.warning("Invalid Core voice packet: %s", error)
            return None

    @staticmethod
    def _packet_id(packet: object, row: dict[str, Any]) -> str:
        if isinstance(packet, dict):
            return _text(
                _value(packet, "utterance_id", "utteranceID", "message_id", "messageID")
            )
        return _text(row.get("sequence"))

    def _remember(self, packet_id: str) -> None:
        self._recent[packet_id] = None
        self._recent.move_to_end(packet_id)
        while len(self._recent) > MAX_RECENT_UTTERANCES:
            self._recent.popitem(last=False)

    @staticmethod
    def _metadata(packet: dict[str, Any], utterance: Utterance) -> dict[str, str]:
        return {
            "utterance_id": utterance.utterance_id,
            "message_id": _text(_value(packet, "message_id", "messageID")),
            "conversation_id": utterance.conversation_id,
            "npc_uuid": utterance.speaker_npc_uuid,
            "speaker_id": utterance.speaker_id,
            "speaker_kind": utterance.speaker_kind,
            "source_mod": _text(_value(packet, "source_mod", "sourceMod")),
        }

    @staticmethod
    def _lifecycle_callback(
        client: BridgeClient,
        state: BridgeState,
        command: str,
        metadata: dict[str, str],
    ):
        async def callback(utterance: Utterance) -> None:
            arguments = dict(metadata)
            arguments["duration_ms"] = str(
                utterance.estimated_or_actual_duration_ms or 0
            )
            await VoicePacketConsumer._send_lifecycle(
                client, state, command, arguments
            )

        return callback

    @staticmethod
    def _failure_callback(
        client: BridgeClient,
        state: BridgeState,
        metadata: dict[str, str],
    ):
        async def callback(utterance: Utterance, error: Exception) -> None:
            arguments = dict(metadata)
            arguments["error"] = _text(error)
            await VoicePacketConsumer._send_lifecycle(
                client, state, "speechFailed", arguments
            )

        return callback

    @staticmethod
    async def _send_lifecycle(
        client: BridgeClient,
        state: BridgeState,
        command: str,
        arguments: dict[str, str],
    ) -> None:
        if command not in VOICE_LIFECYCLE_COMMANDS or not state.runtime_id:
            return
        try:
            await client.call(
                VOICE_NAMESPACE,
                command,
                arguments,
                state.runtime_id,
            )
        except BridgeClientError as error:
            # Lifecycle acknowledgement is diagnostic/presentation metadata;
            # a missing callback must never stop local audio playback.
            LOGGER.debug("Core voice lifecycle callback failed: %s", error)
