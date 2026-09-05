from __future__ import annotations

import pytest

from pbrainz.bridge import BridgeState
from pbrainz.bridge.handler import _AmbientSpeechStream, complete_and_deliver
from pbrainz.bridge.voice import (
    VOICE_CHANNEL,
    VOICE_NAMESPACE,
    VoicePacketConsumer,
    utterance_from_packet,
    voice_channel_available,
)
from pbrainz.conversation_runtime import VoiceBinding


class FakeTTS:
    enabled = True

    def __init__(self) -> None:
        self.enqueued = []

    def resolve_voice_binding(
        self, _conversation_id, _speaker_id, binding, _speaker_kind="npc"
    ):
        return binding

    def can_synthesize(self, binding) -> bool:
        return isinstance(binding, VoiceBinding)

    async def enqueue(
        self,
        utterance,
        *,
        on_started=None,
        on_finished=None,
        on_failed=None,
        wait_for_capacity=False,
    ):
        self.enqueued.append(utterance)
        if on_started:
            await on_started(utterance)
        return True


class DelayedTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__()
        self.rejects = 1

    async def enqueue(
        self,
        utterance,
        *,
        on_started=None,
        on_finished=None,
        on_failed=None,
        wait_for_capacity=False,
    ):
        if self.rejects:
            self.rejects -= 1
            return False
        return await super().enqueue(
            utterance,
            on_started=on_started,
            on_finished=on_finished,
            on_failed=on_failed,
        )


class LifecycleClient:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, namespace, command, arguments, runtime_id):
        self.calls.append((namespace, command, arguments, runtime_id))
        return {"accepted": True}


class TextProvider:
    def resolve(self, _provider, model):
        return "custom", model

    async def complete(self, _provider, request):
        from pbrainz.providers.base import CompletionResult

        return CompletionResult(request.model, "A shared voice response.")


class AmbientStreamingProvider:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def stream_events(self, _provider, _request):
        from pbrainz.providers.base import StreamEvent

        yield StreamEvent(text="Stay close and watch the road ahead with me")
        yield StreamEvent(text=".")


class AmbientStreamingTTS(FakeTTS):
    last_error = None


@pytest.mark.asyncio
async def test_ambient_speech_stream_queues_before_provider_finishes() -> None:
    tts = AmbientStreamingTTS()
    request = {
        "conversation_context": {
            "session_id": "ambient-session",
            "voice_binding": {
                "npc_uuid": "npc-one",
                "slot": "VoiceFemale:2",
            },
        }
    }
    stream = _AmbientSpeechStream(tts, request, "ambient-request", "npc-one")

    await stream.push("Stay close and watch the road ahead")

    assert stream.started is False
    assert len(tts.enqueued) == 0

    await stream.push(" with me")
    assert stream.started is False
    assert len(tts.enqueued) == 0

    await stream.push(".")
    await stream.finish()
    assert stream.started is True
    assert len(tts.enqueued) == 1
    assert tts.enqueued[0].text == "Stay close and watch the road ahead with me."


def ready_state() -> BridgeState:
    return BridgeState(
        available=True,
        enabled=True,
        ready=True,
        runtime_id="runtime-one",
        packet_channels=({"namespace": VOICE_NAMESPACE, "channel": VOICE_CHANNEL},),
    )


def packet() -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_type": "speech.enqueue",
        "utterance_id": "voice:1",
        "message_id": "message:1",
        "source_mod": "TestMod",
        "conversation_id": "conversation-one",
        "sequence": 1,
        "speaker_id": "npc-one",
        "npc_uuid": "npc-one",
        "text": "What is your name?",
        "voice_binding": {
            "npc_uuid": "npc-one",
            "slot": "VoiceFemale:2",
            "pitch": 4,
        },
        "speech": {"mode": "RESPONSE", "allow_overlap": False},
    }


def test_voice_channel_capability_is_explicit() -> None:
    assert voice_channel_available(ready_state())
    assert not voice_channel_available(BridgeState(available=True))


def test_voice_packet_maps_to_existing_utterance_contract() -> None:
    utterance = utterance_from_packet(packet(), FakeTTS())
    assert utterance is not None
    assert utterance.text == "What is your name?"
    assert utterance.speaker_npc_uuid == "npc-one"
    assert utterance.voice_binding is not None
    assert utterance.voice_binding.slot == "VoiceFemale:2"


def test_player_voice_packet_uses_player_identity_and_binding() -> None:
    value = packet()
    value.update(
        {
            "utterance_id": "voice:player:1",
            "message_id": "message:player:1",
            "speaker_id": "player-one",
            "speaker_kind": "player",
            "player_uuid": "player-one",
            "voice_binding": {
                "speaker_id": "player-one",
                "speaker_kind": "player",
                "player_uuid": "player-one",
                "slot": "VoiceFemale:1",
                "pitch": -7,
            },
        }
    )
    utterance = utterance_from_packet(value, FakeTTS())
    assert utterance is not None
    assert utterance.speaker_id == "player-one"
    assert utterance.speaker_kind == "player"
    assert utterance.speaker_key == "player:player-one"
    assert utterance.voice_binding is not None
    assert utterance.voice_binding.speaker_kind == "player"
    assert utterance.voice_binding.pitch == -7


def test_expired_voice_packet_is_discarded() -> None:
    value = packet()
    value["created_real_time_ms"] = 1000
    value["speech"] = {"expires_after_ms": 1000}
    assert utterance_from_packet(value, FakeTTS(), now_ms=2501) is None


@pytest.mark.asyncio
async def test_consumer_enqueues_once_and_reports_lifecycle() -> None:
    tts = FakeTTS()
    client = LifecycleClient()
    consumer = VoicePacketConsumer(tts)
    result = {"streams": [{"events": [{"sequence": 1, "packet": packet()}]}]}

    assert await consumer.consume(client, ready_state(), result) == 1
    assert await consumer.consume(client, ready_state(), result) == 0
    assert len(tts.enqueued) == 1
    assert client.calls[0][0:2] == (VOICE_NAMESPACE, "speechStarted")
    assert client.calls[0][2]["message_id"] == "message:1"


@pytest.mark.asyncio
async def test_consumer_retries_while_tts_is_loading() -> None:
    tts = DelayedTTS()
    consumer = VoicePacketConsumer(tts)
    client = LifecycleClient()
    state = ready_state()
    result = {"streams": [{"events": [{"sequence": 1, "packet": packet()}]}]}

    assert await consumer.consume(client, state, result) == 0
    assert not tts.enqueued
    consumer._retry["voice:1"] = (packet(), 1, 0.0)
    assert await consumer.consume(client, state, {"streams": []}) == 1
    assert len(tts.enqueued) == 1


@pytest.mark.asyncio
async def test_llm_handler_uses_generic_channel_without_legacy_tts_duplicate(tmp_path) -> None:
    providers = TextProvider()
    client = LifecycleClient()
    tts = FakeTTS()
    request = {
        "request_id": "generic-voice-1",
        "npc_id": "npc-one",
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Hello."}],
        "context": {
            "conversation_id": "conversation-one",
            "voice_binding": {
                "npc_uuid": "npc-one",
                "slot": "VoiceFemale:2",
                "pitch": 7,
            },
        },
    }
    state = ready_state()
    await complete_and_deliver(providers, client, state, request, tts_service=tts)
    assert len(tts.enqueued) == 0
    assert client.calls[0][1] == "deliverChat"
    assert "presentation_mode" not in client.calls[0][2]


@pytest.mark.asyncio
async def test_ambient_llm_streams_local_tts_and_marks_final_message_managed(tmp_path) -> None:
    from pbrainz.config import Settings
    from pbrainz.conversation_service import ConversationService

    providers = AmbientStreamingProvider()
    tts = AmbientStreamingTTS()
    service = ConversationService(
        Settings(
            database_path=str(tmp_path / "settings.db"),
            bridge_required=False,
        ),
        providers,
    )
    client = LifecycleClient()
    request = {
        "request_id": "ambient-stream-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "ambient-session",
            "message": "The player killed a zombie.",
            "metadata": {"mode": "ambient_social"},
            "voice_binding": {
                "npc_uuid": "npc-one",
                "slot": "VoiceFemale:2",
            },
        },
    }

    await complete_and_deliver(
        providers,
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
        tts_service=tts,
    )

    assert len(tts.enqueued) == 1
    delivery = next(call[2] for call in client.calls if call[1] == "deliverChat")
    assert delivery["response_text"] == "Stay close and watch the road ahead with me."
    assert delivery["tts_managed"] is True
