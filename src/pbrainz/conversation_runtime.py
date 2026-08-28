"""Ephemeral conversation and speech-coordination primitives.

The runtime is deliberately separate from the save-scoped memory store.  It
owns only active-session queues and presentation state; Project Hoomans keeps
authority over gameplay and the LLM remains responsible for dialogue text.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class ConversationKind(StrEnum):
    PLAYER_NPC = "PLAYER_NPC"
    GROUP_PLAYER_NPC = "GROUP_PLAYER_NPC"
    NPC_GOSSIP = "NPC_GOSSIP"


class SpeechMode(StrEnum):
    EMERGENCY = "EMERGENCY"
    DIRECT = "DIRECT"
    INTERRUPT = "INTERRUPT"
    RESPONSE = "RESPONSE"
    REACTION = "REACTION"
    BANTER = "BANTER"


class UtteranceState(StrEnum):
    GENERATED = "GENERATED"
    TTS_PENDING = "TTS_PENDING"
    TTS_READY = "TTS_READY"
    WAITING_FOR_FLOOR = "WAITING_FOR_FLOOR"
    PLAYING = "PLAYING"
    DISPLAYED = "DISPLAYED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class VoiceBinding:
    """Compact voice identity resolved by Project Hoomans' Lua side."""

    npc_uuid: str
    slot: str
    pitch: int = 0

    def __post_init__(self) -> None:
        if not str(self.npc_uuid).strip():
            raise ValueError("voice binding requires npc_uuid")
        if not str(self.slot).strip() or ":" not in self.slot:
            raise ValueError("voice binding requires a prefix:voiceType slot")

    @classmethod
    def from_mapping(cls, value: object) -> VoiceBinding | None:
        if not isinstance(value, dict):
            return None
        npc_uuid = str(value.get("npc_uuid") or value.get("npcUUID") or "").strip()
        slot = str(value.get("slot") or "").strip()
        if not npc_uuid or not slot:
            return None
        try:
            pitch = int(float(value.get("pitch") or 0))
            return cls(
                npc_uuid=npc_uuid[:256],
                slot=slot[:128],
                pitch=max(-48, min(48, pitch)),
            )
        except (OverflowError, TypeError, ValueError):
            return None

    def as_dict(self) -> dict[str, object]:
        return {"npc_uuid": self.npc_uuid, "slot": self.slot, "pitch": self.pitch}


@dataclass(slots=True)
class VoiceBindingCache:
    """Bounded per-session cache of Lua-resolved voice identities."""

    max_entries: int = 256
    _items: OrderedDict[tuple[str, str], VoiceBinding] = field(
        default_factory=OrderedDict, repr=False
    )

    def __post_init__(self) -> None:
        self.max_entries = max(1, min(int(self.max_entries), 2048))

    def remember(self, conversation_id: str, binding: VoiceBinding) -> VoiceBinding:
        key = (str(conversation_id)[:256], binding.npc_uuid)
        self._items.pop(key, None)
        self._items[key] = binding
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        return binding

    def get(self, conversation_id: str, npc_uuid: str) -> VoiceBinding | None:
        key = (str(conversation_id)[:256], str(npc_uuid)[:256])
        binding = self._items.get(key)
        if binding is not None:
            self._items.move_to_end(key)
        return binding

    def clear(self, conversation_id: str | None = None) -> None:
        if conversation_id is None:
            self._items.clear()
            return
        prefix = str(conversation_id)[:256]
        for key in tuple(self._items):
            if key[0] == prefix:
                self._items.pop(key, None)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(slots=True)
class Utterance:
    utterance_id: str
    conversation_id: str
    turn: int
    speaker_npc_uuid: str
    text: str
    speech_mode: SpeechMode = SpeechMode.RESPONSE
    allow_overlap: bool = False
    can_interrupt: bool = False
    voice_binding: VoiceBinding | None = None
    generation_state: UtteranceState = UtteranceState.GENERATED
    tts_state: UtteranceState = UtteranceState.GENERATED
    playback_state: UtteranceState = UtteranceState.GENERATED
    estimated_or_actual_duration_ms: int | None = None
    created_at: float = 0.0

    def __post_init__(self) -> None:
        self.utterance_id = str(self.utterance_id).strip()[:256]
        self.conversation_id = str(self.conversation_id).strip()[:256]
        self.speaker_npc_uuid = str(self.speaker_npc_uuid).strip()[:256]
        self.text = str(self.text or "").strip()[:12000]
        if not self.utterance_id or not self.conversation_id or not self.speaker_npc_uuid:
            raise ValueError("utterance identity is required")
        if not self.text:
            raise ValueError("utterance text is required")
        if self.voice_binding and self.voice_binding.npc_uuid != self.speaker_npc_uuid:
            raise ValueError("voice binding does not match utterance speaker")

    @property
    def is_overlap(self) -> bool:
        return self.allow_overlap and self.speech_mode in {
            SpeechMode.INTERRUPT,
            SpeechMode.REACTION,
            SpeechMode.EMERGENCY,
        }

    def as_event(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "utterance_id": self.utterance_id,
            "npc_uuid": self.speaker_npc_uuid,
            "text": self.text,
            "duration_ms": self.estimated_or_actual_duration_ms or 0,
        }


@dataclass(slots=True)
class ConversationRuntime:
    """Ephemeral state for one active player/group/gossip scene."""

    conversation_id: str
    kind: ConversationKind = ConversationKind.PLAYER_NPC
    participants: tuple[str, ...] = ()
    state: str = "active"
    current_floor_owner: str | None = None
    active_utterances: dict[str, Utterance] = field(default_factory=dict)
    generated_queue: deque[str] = field(default_factory=deque)
    tts_ready_queue: deque[str] = field(default_factory=deque)
    next_turn: int = 0
    generation_target: str | None = None
    max_generated_ahead: int = 3
    max_tts_ready_ahead: int = 1
    _utterances: dict[str, Utterance] = field(default_factory=dict, repr=False)
    _speaker_queues: dict[str, deque[str]] = field(
        default_factory=lambda: defaultdict(deque), repr=False
    )

    def __post_init__(self) -> None:
        self.conversation_id = str(self.conversation_id).strip()[:256]
        self.participants = tuple(
            dict.fromkeys(str(item) for item in self.participants if str(item))
        )
        self.max_generated_ahead = max(1, min(int(self.max_generated_ahead), 16))
        self.max_tts_ready_ahead = max(1, min(int(self.max_tts_ready_ahead), 8))
        if not self.conversation_id:
            raise ValueError("conversation_id is required")

    @property
    def utterances(self) -> tuple[Utterance, ...]:
        return tuple(self._utterances.values())

    @property
    def generated_ahead_count(self) -> int:
        return sum(
            1
            for utterance in self._utterances.values()
            if utterance.playback_state
            in {UtteranceState.GENERATED, UtteranceState.TTS_PENDING, UtteranceState.TTS_READY}
        )

    @property
    def tts_ready_ahead_count(self) -> int:
        return sum(
            1
            for utterance in self._utterances.values()
            if utterance.playback_state
            in {UtteranceState.TTS_READY, UtteranceState.WAITING_FOR_FLOOR}
        )

    def can_generate(self) -> bool:
        return self.state == "active" and self.generated_ahead_count < self.max_generated_ahead

    def add_participant(self, npc_uuid: str) -> None:
        npc_uuid = str(npc_uuid).strip()[:256]
        if npc_uuid and npc_uuid not in self.participants:
            self.participants = (*self.participants, npc_uuid)

    def enqueue_generated(self, utterance: Utterance) -> None:
        if utterance.conversation_id != self.conversation_id:
            raise ValueError("utterance belongs to a different conversation")
        if self.participants and utterance.speaker_npc_uuid not in self.participants:
            raise ValueError("utterance speaker is not an active participant")
        if utterance.utterance_id in self._utterances:
            raise ValueError("duplicate utterance_id")
        self._utterances[utterance.utterance_id] = utterance
        self._speaker_queues[utterance.speaker_npc_uuid].append(utterance.utterance_id)
        self.generated_queue.append(utterance.utterance_id)
        self.next_turn = max(self.next_turn, utterance.turn + 1)

    def mark_tts_pending(self, utterance_id: str) -> Utterance:
        utterance = self._get(utterance_id)
        self._remove(self.generated_queue, utterance_id)
        utterance.tts_state = UtteranceState.TTS_PENDING
        utterance.playback_state = UtteranceState.TTS_PENDING
        return utterance

    def mark_tts_ready(self, utterance_id: str, duration_ms: int) -> Utterance:
        utterance = self._get(utterance_id)
        utterance.estimated_or_actual_duration_ms = max(0, int(duration_ms))
        utterance.tts_state = UtteranceState.TTS_READY
        utterance.playback_state = UtteranceState.TTS_READY
        if utterance_id not in self.tts_ready_queue:
            self.tts_ready_queue.append(utterance_id)
        return utterance

    def mark_waiting_for_floor(self, utterance_id: str) -> Utterance:
        utterance = self._get(utterance_id)
        utterance.playback_state = UtteranceState.WAITING_FOR_FLOOR
        if not utterance.is_overlap and self.current_floor_owner is None:
            # Reserve the conversational floor during the natural inter-line
            # gap so another normal line cannot start before this one does.
            self.current_floor_owner = utterance_id
        return utterance

    def mark_playing(self, utterance_id: str) -> Utterance:
        utterance = self._get(utterance_id)
        self._remove(self.tts_ready_queue, utterance_id)
        utterance.playback_state = UtteranceState.PLAYING
        self.active_utterances[utterance_id] = utterance
        if not utterance.is_overlap:
            self.current_floor_owner = utterance_id
        return utterance

    def mark_displayed(self, utterance_id: str) -> Utterance:
        utterance = self._get(utterance_id)
        utterance.playback_state = UtteranceState.DISPLAYED
        return utterance

    def mark_complete(self, utterance_id: str, *, failed: bool = False) -> Utterance:
        utterance = self._get(utterance_id)
        utterance.playback_state = UtteranceState.FAILED if failed else UtteranceState.COMPLETE
        self.active_utterances.pop(utterance_id, None)
        if self.current_floor_owner == utterance_id:
            self.current_floor_owner = None
        self._remove(self.generated_queue, utterance_id)
        self._remove(self.tts_ready_queue, utterance_id)
        return utterance

    def next_ready(self) -> Utterance | None:
        for utterance_id in tuple(self.tts_ready_queue):
            utterance = self._get(utterance_id)
            if utterance.playback_state is UtteranceState.TTS_READY and self._speaker_is_fifo_ready(
                utterance
            ):
                return utterance
        return None

    def can_start(self, utterance: Utterance, *, reactor_limit: int = 1) -> bool:
        if not self._speaker_is_fifo_ready(utterance):
            return False
        if not utterance.is_overlap:
            return self.current_floor_owner is None
        active_reactors = sum(1 for item in self.active_utterances.values() if item.is_overlap)
        return active_reactors < max(1, reactor_limit)

    def close(self) -> None:
        self.state = "closed"
        self.generated_queue.clear()
        self.tts_ready_queue.clear()
        self.active_utterances.clear()
        self.current_floor_owner = None
        self._utterances.clear()
        self._speaker_queues.clear()

    def _speaker_is_fifo_ready(self, utterance: Utterance) -> bool:
        queue = self._speaker_queues.get(utterance.speaker_npc_uuid, ())
        for earlier_id in queue:
            if earlier_id == utterance.utterance_id:
                return True
            earlier = self._utterances.get(earlier_id)
            if earlier and earlier.playback_state not in {
                UtteranceState.COMPLETE,
                UtteranceState.FAILED,
            }:
                return False
        return False

    def _get(self, utterance_id: str) -> Utterance:
        try:
            return self._utterances[utterance_id]
        except KeyError as error:
            raise KeyError(f"unknown utterance: {utterance_id}") from error

    @staticmethod
    def _remove(queue: deque[str], value: str) -> None:
        try:
            queue.remove(value)
        except ValueError:
            pass


@dataclass(frozen=True, slots=True)
class SpeakerCandidate:
    npc_uuid: str
    direct_relevance: float = 0.0
    relationship_relevance: float = 0.0
    topic_knowledge: float = 0.0
    talkativeness: float = 0.5
    turns_since_last_speech: int = 99
    identity_seed: int = 0
    willing: bool = True


class SpeakerCoordinator:
    """Cheap, deterministic speaker selection before any LLM request."""

    PRECEDENCE = {
        SpeechMode.EMERGENCY: 600,
        SpeechMode.DIRECT: 500,
        SpeechMode.INTERRUPT: 400,
        SpeechMode.RESPONSE: 300,
        SpeechMode.REACTION: 200,
        SpeechMode.BANTER: 100,
    }

    def select_speakers(
        self,
        candidates: Iterable[SpeakerCandidate],
        *,
        mode: SpeechMode = SpeechMode.RESPONSE,
        max_speakers: int = 1,
        conversation_id: str = "",
    ) -> list[SpeakerCandidate]:
        """Return selected candidates; no provider call is made here."""

        eligible = [candidate for candidate in candidates if candidate.willing]
        eligible.sort(
            key=lambda candidate: (
                self.PRECEDENCE[mode]
                + max(0.0, min(1.0, candidate.direct_relevance)) * 80
                + max(0.0, min(1.0, candidate.relationship_relevance)) * 35
                + max(0.0, min(1.0, candidate.topic_knowledge)) * 30
                + max(0.0, min(1.0, candidate.talkativeness)) * 12
                + min(12, max(0, candidate.turns_since_last_speech)) * 8,
                self._tie_break(conversation_id, candidate),
            ),
            reverse=True,
        )
        return eligible[: max(1, min(int(max_speakers), 2))]

    @staticmethod
    def _tie_break(conversation_id: str, candidate: SpeakerCandidate) -> int:
        digest = hashlib.sha256(
            f"{conversation_id}:{candidate.npc_uuid}:{candidate.identity_seed}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big")


@dataclass(slots=True)
class GossipScene:
    conversation_id: str
    participants: tuple[str, ...]
    initiator: str
    topic_refs: tuple[str, ...] = ()
    turns_completed: int = 0
    max_turns: int = 12
    state: str = "active"

    def is_valid_for_batch(self) -> bool:
        return (
            self.state == "active"
            and self.turns_completed < max(1, self.max_turns)
            and bool(self.participants)
            and self.initiator in self.participants
        )


def validate_gossip_batch(
    rows: object,
    scene: GossipScene,
    *,
    conversation_id: str | None = None,
    max_batch: int = 4,
) -> list[Utterance]:
    """Validate provider-returned mini-batch rows without trusting speakers."""

    if not scene.is_valid_for_batch():
        raise ValueError("gossip scene is no longer valid")
    if not isinstance(rows, list) or not rows:
        raise ValueError("gossip batch must be a non-empty list")
    if len(rows) > max(1, min(int(max_batch), 4)):
        raise ValueError("gossip batch exceeds the bounded batch size")
    allowed = set(scene.participants)
    output: list[Utterance] = []
    active_conversation_id = conversation_id or scene.conversation_id
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("gossip batch rows must be objects")
        speaker = str(row.get("speaker_npc_uuid") or row.get("speaker") or "").strip()
        text = str(row.get("text") or row.get("content") or "").strip()
        if speaker not in allowed:
            raise ValueError("gossip batch contains a speaker outside the scene")
        if not text:
            raise ValueError("gossip batch contains empty dialogue")
        mode_value = str(row.get("speech_mode") or SpeechMode.BANTER.value).upper()
        try:
            mode = SpeechMode(mode_value)
        except ValueError:
            mode = SpeechMode.BANTER
        output.append(
            Utterance(
                utterance_id=f"{active_conversation_id}:turn-{scene.turns_completed + index + 1}",
                conversation_id=active_conversation_id,
                turn=scene.turns_completed + index,
                speaker_npc_uuid=speaker,
                text=text,
                speech_mode=mode,
                allow_overlap=bool(row.get("allow_overlap") is True and mode != SpeechMode.BANTER),
                can_interrupt=bool(row.get("can_interrupt") is True),
                voice_binding=VoiceBinding.from_mapping(row.get("voice_binding")),
            )
        )
    return output
