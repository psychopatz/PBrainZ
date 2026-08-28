import pytest

from pbrainz.conversation_runtime import (
    ConversationRuntime,
    GossipScene,
    SpeakerCandidate,
    SpeakerCoordinator,
    SpeechMode,
    Utterance,
    UtteranceState,
    VoiceBinding,
    VoiceBindingCache,
    validate_gossip_batch,
)


def _utterance(
    utterance_id: str,
    speaker: str,
    turn: int,
    *,
    mode: SpeechMode = SpeechMode.RESPONSE,
    overlap: bool = False,
) -> Utterance:
    return Utterance(
        utterance_id=utterance_id,
        conversation_id="conversation-one",
        turn=turn,
        speaker_npc_uuid=speaker,
        text=f"line {utterance_id}",
        speech_mode=mode,
        allow_overlap=overlap,
        voice_binding=VoiceBinding(speaker, "VoiceMale:0"),
    )


def test_runtime_reserves_floor_during_natural_gap_and_preserves_speaker_fifo() -> None:
    runtime = ConversationRuntime(
        "conversation-one",
        participants=("npc-a", "npc-b"),
        max_generated_ahead=4,
        max_tts_ready_ahead=4,
    )
    first = _utterance("one", "npc-a", 0)
    second = _utterance("two", "npc-a", 1)
    other = _utterance("three", "npc-b", 2)
    for utterance in (first, second, other):
        runtime.enqueue_generated(utterance)
        runtime.mark_tts_pending(utterance.utterance_id)
        runtime.mark_tts_ready(utterance.utterance_id, 500)

    assert runtime.next_ready() is first
    runtime.mark_waiting_for_floor(first.utterance_id)
    assert runtime.current_floor_owner == first.utterance_id
    assert not runtime.can_start(other)
    assert not runtime.can_start(second)

    runtime.mark_playing(first.utterance_id)
    runtime.mark_complete(first.utterance_id)
    assert runtime.next_ready() is second
    assert runtime.can_start(second)


def test_only_explicit_reactions_can_overlap_and_reactor_limit_is_bounded() -> None:
    runtime = ConversationRuntime(
        "conversation-one", participants=("npc-a", "npc-b", "npc-c"), max_tts_ready_ahead=4
    )
    main = _utterance("main", "npc-a", 0)
    reaction = _utterance("reaction", "npc-b", 1, mode=SpeechMode.REACTION, overlap=True)
    ordinary = _utterance("ordinary", "npc-b", 2)
    for utterance in (main, reaction, ordinary):
        runtime.enqueue_generated(utterance)
        runtime.mark_tts_pending(utterance.utterance_id)
        runtime.mark_tts_ready(utterance.utterance_id, 500)

    runtime.mark_playing(main.utterance_id)
    assert runtime.can_start(reaction)
    assert not runtime.can_start(ordinary)
    second_reaction = _utterance(
        "second-reaction", "npc-c", 3, mode=SpeechMode.REACTION, overlap=True
    )
    runtime.enqueue_generated(second_reaction)
    runtime.mark_tts_pending(second_reaction.utterance_id)
    runtime.mark_tts_ready(second_reaction.utterance_id, 500)
    runtime.mark_playing(reaction.utterance_id)
    assert not runtime.can_start(second_reaction)


def test_gossip_batch_is_bounded_to_active_participants() -> None:
    scene = GossipScene("gossip-one", ("npc-a", "npc-b"), "npc-a")
    rows = [
        {"speaker_npc_uuid": "npc-a", "text": "Did you hear?", "speech_mode": "BANTER"},
        {"speaker_npc_uuid": "npc-b", "text": "I heard something."},
    ]

    batch = validate_gossip_batch(rows, scene)

    assert [item.speaker_npc_uuid for item in batch] == ["npc-a", "npc-b"]
    assert all(item.speech_mode is SpeechMode.BANTER for item in batch)
    with pytest.raises(ValueError, match="outside the scene"):
        validate_gossip_batch(rows + [{"speaker": "npc-c", "text": "No."}], scene)


def test_speaker_selection_is_deterministic_and_caps_group_size() -> None:
    candidates = [
        SpeakerCandidate("npc-a", direct_relevance=0.9, turns_since_last_speech=1),
        SpeakerCandidate("npc-b", topic_knowledge=1.0, turns_since_last_speech=1),
        SpeakerCandidate("npc-c", willing=False),
    ]
    coordinator = SpeakerCoordinator()

    first = coordinator.select_speakers(
        candidates, mode=SpeechMode.RESPONSE, max_speakers=9, conversation_id="gossip-one"
    )
    second = coordinator.select_speakers(
        candidates, mode=SpeechMode.RESPONSE, max_speakers=9, conversation_id="gossip-one"
    )

    assert [item.npc_uuid for item in first] == [item.npc_uuid for item in second]
    assert len(first) == 2
    assert first[0].npc_uuid == "npc-a"


def test_voice_binding_rejects_unresolved_piper_identity() -> None:
    assert VoiceBinding.from_mapping({"npc_uuid": "npc-a", "slot": "VoiceFemale:1"})
    assert VoiceBinding.from_mapping({"npc_uuid": "npc-a", "piper_model_id": "secret"}) is None
    assert UtteranceState.GENERATED.value == "GENERATED"


def test_voice_binding_cache_is_session_scoped_and_bounded() -> None:
    cache = VoiceBindingCache(max_entries=1)
    binding = VoiceBinding("npc-a", "VoiceFemale:2", pitch=7)

    cache.remember("session-a", binding)

    assert cache.get("session-a", "npc-a") == binding
    assert cache.get("session-b", "npc-a") is None
    cache.remember("session-b", VoiceBinding("npc-b", "VoiceMale:0"))
    assert len(cache) == 1
    assert cache.get("session-a", "npc-a") is None
