"""Provider-independent types for NPC memory and conversation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class MemoryType(StrEnum):
    FACT = "FACT"
    SOCIAL_EVENT = "SOCIAL_EVENT"
    COMMITMENT = "COMMITMENT"
    PERSONAL_EVENT = "PERSONAL_EVENT"
    OPINION = "OPINION"
    DISCOVERY = "DISCOVERY"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"


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

    def recent_turns(
        self,
        session_id: str,
        scope: MemoryScope,
        limit: int = 8,
    ) -> list[ConversationTurn]: ...

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
