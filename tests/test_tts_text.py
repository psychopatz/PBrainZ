from __future__ import annotations

import pytest

from pbrainz.bridge.voice import utterance_from_packet
from pbrainz.conversation_runtime import Utterance, VoiceBinding
from pbrainz.tts.service import TTSService
from pbrainz.tts.text import normalize_tts_text


class FakeTTS:
    enabled = True

    def resolve_voice_binding(self, _conversation_id, _npc_id, binding):
        return binding


class RecordingScheduler:
    def __init__(self) -> None:
        self.utterance = None

    async def enqueue(self, utterance, **_callbacks):
        self.utterance = utterance
        return True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("I **really** like you. *chuckle*", "I really like you. Ha ha!"),
        ("That was close. * gasp*", "That was close. Huh!"),
        ("She [waves](https://example.invalid) at me.", "She waves at me."),
        ("I *really* mean it.", "I really mean it."),
        ("*waves* Hello.", "waves Hello."),
    ],
)
def test_normalize_tts_text_preserves_dialogue_and_replaces_known_cues(
    source: str, expected: str
) -> None:
    assert normalize_tts_text(source) == expected


def test_normalize_tts_text_handles_nested_cue_delimiters() -> None:
    assert normalize_tts_text("(*chuckle*) Well, that worked.") == (
        "Ha ha! Well, that worked."
    )


def test_normalize_tts_text_does_not_speak_markdown_fence_or_list_syntax() -> None:
    assert normalize_tts_text("# Status\n- **Ready**") == "Status Ready"


def test_normalize_tts_text_removes_residual_markdown_and_spoken_punctuation() -> None:
    assert normalize_tts_text(
        "**Ready:** _now_ - *gasp!*\n1. [go](https://example.invalid)"
    ) == "Ready. now Huh! go"


def test_normalize_tts_text_preserves_words_with_malformed_formatting() -> None:
    assert normalize_tts_text("*waves at you: **hello**") == "waves at you. hello"


def test_voice_packet_uses_sanitized_text_but_keeps_packet_contract() -> None:
    packet = {
        "schema_version": 1,
        "event_type": "speech.enqueue",
        "utterance_id": "voice:markdown",
        "conversation_id": "conversation-one",
        "speaker_id": "npc-one",
        "text": "I **agree**. *chuckles*",
        "voice_binding": {
            "npc_uuid": "npc-one",
            "slot": "VoiceFemale:2",
        },
    }

    utterance = utterance_from_packet(packet, FakeTTS())

    assert utterance is not None
    assert utterance.text == "I agree. Ha ha!"
    assert utterance.voice_binding == VoiceBinding("npc-one", "VoiceFemale:2")


@pytest.mark.asyncio
async def test_tts_service_is_a_final_normalization_boundary() -> None:
    service = object.__new__(TTSService)
    service.scheduler = RecordingScheduler()
    service.last_error = None
    utterance = Utterance(
        utterance_id="voice:service",
        conversation_id="conversation-one",
        turn=1,
        speaker_npc_uuid="npc-one",
        text="That was close. * gasp*",
        voice_binding=VoiceBinding("npc-one", "VoiceFemale:2"),
    )

    assert await service.enqueue(utterance)
    assert service.scheduler.utterance.text == "That was close. Huh!"
