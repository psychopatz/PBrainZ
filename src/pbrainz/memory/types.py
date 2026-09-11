"""Provider-independent types for NPC memory and conversation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class MemoryType(StrEnum):
    FACT = "FACT"
    OBSERVED_FACT = "OBSERVED_FACT"
    CLAIM = "CLAIM"
    HEARSAY = "HEARSAY"
    SOCIAL_EVENT = "SOCIAL_EVENT"
    COMMITMENT = "COMMITMENT"
    PERSONAL_EVENT = "PERSONAL_EVENT"
    OPINION = "OPINION"
    DISCOVERY = "DISCOVERY"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"
    EPISODE = "EPISODE"


class MemoryVisibility(StrEnum):
    """Who may receive a memory during actor-specific prompt assembly."""

    PUBLIC = "PUBLIC"
    OBSERVED = "OBSERVED"
    TOLD = "TOLD"
    PRIVATE = "PRIVATE"
    SYSTEM = "SYSTEM"


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """The only scope in which NPC conversation memory may be retrieved."""

    world_uuid: str
    player_uuid: str
    npc_uuid: str

    def __post_init__(self) -> None:
        for name, value in (
            ("world_uuid", self.world_uuid),
            ("player_uuid", self.player_uuid),
            ("npc_uuid", self.npc_uuid),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} is required")


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: str
    content: str
    created_at: str = ""
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str | None = None
    game_day: int | None = None
    world_age_hours: float | None = None
    speaker_uuid: str | None = None
    speaker_name: str | None = None
    speaker_kind: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Bounded, actor-aware retrieval request.

    The actor is part of the query rather than inferred from a free-form
    prompt. Implementations must apply access filtering before relevance
    ranking.
    """

    scope: MemoryScope
    actor_id: str
    current_message: str = ""
    conversation_id: str | None = None
    current_day: int | None = None
    participants: tuple[str, ...] = ()
    mentioned_entities: tuple[str, ...] = ()
    current_topic: str | None = None
    requested_kinds: tuple[MemoryType, ...] = ()
    max_results: int = 6
    token_budget: int = 700
    requested_tags: tuple[str, ...] = ()
    token_expansions: tuple[tuple[str, tuple[str, ...]], ...] | None = None


@dataclass(frozen=True, slots=True)
class TurnWriteResult:
    """The durable, idempotent result of recording one canonical message."""

    turn: ConversationTurn
    duplicate: bool = False
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class ConversationSession:
    session_id: str
    scope: MemoryScope
    started_at: str
    last_activity_at: str
    ended_at: str | None = None
    turn_count: int = 0
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    scope: MemoryScope
    memory_type: MemoryType
    content: str
    tags: tuple[str, ...] = ()
    importance: float = 0.5
    state: str = "active"
    provenance: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_recalled_at: str | None = None
    active: bool = True
    visibility: MemoryVisibility = MemoryVisibility.PRIVATE
    game_day: int | None = None
    participants: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    transcript_ref: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryEpisode:
    """A coherent conversation/event unit indexed instead of every utterance."""

    episode_id: str
    scope: MemoryScope
    conversation_id: str
    game_day: int | None
    participants: tuple[str, ...] = ()
    witnesses: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    summary: str = ""
    key_facts: tuple[str, ...] = ()
    emotional_tone: str | None = None
    importance: float = 0.5
    visibility: MemoryVisibility = MemoryVisibility.PUBLIC
    transcript_ref: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class DaySynopsis:
    scope: MemoryScope
    game_day: int
    synopsis: str = ""
    commitments: tuple[str, ...] = ()
    plans: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    unresolved_topics: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class StructuredFact:
    fact_id: str
    scope: MemoryScope
    kind: str
    content: str
    subject_uuid: str | None = None
    source_uuid: str | None = None
    truth_status: str = "unverified"
    visibility: MemoryVisibility = MemoryVisibility.PUBLIC
    game_day: int | None = None
    status: str = "active"
    importance: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    active: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    memory: MemoryRecord
    score: float
    reasons: tuple[str, ...] = ()

    def as_diagnostic(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.memory_id,
            "type": self.memory.memory_type.value,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


class MemoryStore(Protocol):
    """Persistence boundary used by the conversation service."""

    def ensure_session(
        self,
        session_id: str,
        scope: MemoryScope,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession: ...

    def add_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn: ...

    def record_turn(
        self,
        session_id: str,
        scope: MemoryScope,
        role: str,
        content: str,
        *,
        message_id: str,
        metadata: dict[str, Any] | None = None,
        game_day: int | None = None,
        world_age_hours: float | None = None,
        speaker_uuid: str | None = None,
        speaker_name: str | None = None,
        speaker_kind: str | None = None,
    ) -> TurnWriteResult: ...

    def recent_turns(
        self,
        session_id: str,
        scope: MemoryScope,
        limit: int = 8,
    ) -> list[ConversationTurn]: ...

    def search_turns(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 4,
    ) -> list[ConversationTurn]: ...

    def retrieve_query(self, query: MemoryQuery) -> list[RetrievalMatch]: ...

    def get_day_synopsis(
        self, scope: MemoryScope, game_day: int
    ) -> DaySynopsis | None: ...

    def list_structured_facts(
        self, query: MemoryQuery, limit: int = 12
    ) -> list[StructuredFact]: ...

    def remember(self, memory: MemoryRecord) -> MemoryRecord: ...


class MemoryRetriever(Protocol):
    """Retrieval boundary reserved for FTS5, sqlite-vec, or another backend."""

    def retrieve(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 6,
    ) -> list[RetrievalMatch]: ...
