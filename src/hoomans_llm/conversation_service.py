"""Conversation orchestration for Project Hoomans NPC dialogue."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.config import Settings
from hoomans_llm.context_builder import ContextBuilder, ContextInput
from hoomans_llm.memory import (
    ConversationTurn,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievalMatch,
    SQLiteMemoryStore,
)
from hoomans_llm.providers.base import CompletionResult
from hoomans_llm.providers.registry import ProviderRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    """Structured request sent by the Project Hoomans data tunnel."""

    request_id: str
    scope: MemoryScope
    session_id: str
    message: str
    npc_name: str = "the survivor"
    player_name: str = "the player"
    character_card: dict[str, Any] = field(default_factory=dict)
    relationship_snapshot: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    recent_conversation: tuple[dict[str, str], ...] = ()
    available_tools: tuple[dict[str, Any], ...] = ()
    provider: str | None = None
    model: str = "default"
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    end_session: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ConversationRequest:
        context = value.get("conversation_context") or value.get("context") or value
        if not isinstance(context, dict):
            context = value
        world_uuid = context.get("world_uuid") or context.get("worldUUID")
        player_uuid = context.get("player_uuid") or context.get("playerUUID")
        npc_uuid = context.get("npc_uuid") or context.get("npcUUID") or value.get("npc_id")
        message = context.get("message") or context.get("current_player_message")
        required = (world_uuid, player_uuid, npc_uuid, message)
        if not all(str(item or "").strip() for item in required):
            raise ValueError(
                "structured conversation requests require world, player, NPC, and message"
            )
        request_id = str(
            value.get("request_id") or context.get("request_id") or uuid.uuid4().hex
        )
        session_id = str(
            context.get("session_id")
            or context.get("sessionID")
            or f"session-{request_id}"
        )
        return cls(
            request_id=request_id,
            scope=MemoryScope(str(world_uuid), str(player_uuid), str(npc_uuid)),
            session_id=session_id,
            message=str(message).strip()[:4000],
            npc_name=str(context.get("npc_name") or context.get("npcName") or "the survivor"),
            player_name=str(
                context.get("player_name")
                or context.get("playerName")
                or "the player"
            ),
            character_card=_mapping(context.get("character_card") or context.get("characterCard")),
            relationship_snapshot=_mapping(
                context.get("relationship_snapshot")
                or context.get("relationshipSnapshot")
            ),
            preferences=_mapping(context.get("preferences")),
            current_state=_mapping(context.get("current_state") or context.get("currentState")),
            recent_conversation=tuple(
                {
                    "role": str(item.get("role") or "assistant"),
                    "content": str(item.get("content") or "")[:4000],
                }
                for item in (
                    context.get("recent_conversation")
                    or context.get("recentConversation")
                    or []
                )
                if isinstance(item, dict) and str(item.get("content") or "").strip()
            )[:16],
            available_tools=tuple(
                item
                for item in (
                    context.get("available_tools")
                    or context.get("availableTools")
                    or []
                )
                if isinstance(item, dict)
            )[:12],
            provider=_optional_text(context.get("provider")),
            model=str(context.get("model") or "default"),
            temperature=context.get("temperature"),
            max_tokens=context.get("max_tokens"),
            metadata=_mapping(context.get("metadata")),
            end_session=context.get("end_session") is True or context.get("endSession") is True,
        )


@dataclass(frozen=True, slots=True)
class ConversationResult:
    completion: CompletionResult
    session_id: str
    retrieved_memories: tuple[RetrievalMatch, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    summary: str
    memories: tuple[MemoryRecord, ...] = ()


class ConversationConsolidator(Protocol):
    def consolidate(
        self,
        scope: MemoryScope,
        session_id: str,
        turns: list[ConversationTurn],
    ) -> ConsolidationResult: ...


class HeuristicConsolidator:
    """Small deterministic baseline; an LLM extractor can replace this hook."""

    _commitment = re.compile(r"\b(i will|we will|let's|lets|i promise|we should)\b", re.I)
    _preference = re.compile(r"\b(i like|i love|i hate|i prefer|my favorite)\b", re.I)
    _identity = re.compile(r"\b(my name is|call me|i am)\b", re.I)

    def consolidate(
        self,
        scope: MemoryScope,
        session_id: str,
        turns: list[ConversationTurn],
    ) -> ConsolidationResult:
        usable = [turn for turn in turns if turn.content.strip()]
        summary = "\n".join(
            f"{turn.role.title()}: {turn.content[:420]}" for turn in usable[-8:]
        )[:2400]
        memories: list[MemoryRecord] = []
        for turn in usable[-12:]:
            content = turn.content.strip()
            memory_type: MemoryType | None = None
            tags: tuple[str, ...] = ()
            importance = 0.55
            if self._commitment.search(content):
                memory_type, tags, importance = MemoryType.COMMITMENT, ("commitment",), 0.82
            elif self._identity.search(content):
                memory_type, tags, importance = MemoryType.FACT, ("identity",), 0.8
            elif self._preference.search(content):
                memory_type, tags, importance = MemoryType.OPINION, ("preference",), 0.68
            if memory_type is not None:
                memories.append(
                    MemoryRecord(
                        memory_id=uuid.uuid4().hex,
                        scope=scope,
                        memory_type=memory_type,
                        content=content[:1200],
                        tags=tags,
                        importance=importance,
                        provenance={
                            "source": "heuristic_consolidator",
                            "session_id": session_id,
                            "turn_index": turn.turn_index,
                        },
                        session_id=session_id,
                    )
                )
        return ConsolidationResult(summary=summary, memories=tuple(memories))


class ConversationService:
    """Coordinates memory and provider calls without owning gameplay state."""

    def __init__(
        self,
        settings: Settings,
        providers: ProviderRegistry,
        *,
        consolidator: ConversationConsolidator | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.context_builder = ContextBuilder(
            max_chars=settings.context_max_chars,
            recent_turn_limit=settings.memory_recent_turns,
            memory_limit=settings.memory_retrieval_limit,
        )
        self.consolidator = consolidator or HeuristicConsolidator()
        self._stores: dict[str, SQLiteMemoryStore] = {}

    async def complete(self, request: ConversationRequest) -> ConversationResult:
        store = self._store(request.scope.world_uuid)
        diagnostics: dict[str, Any] = {
            "world_uuid": request.scope.world_uuid,
            "player_uuid": request.scope.player_uuid,
            "npc_uuid": request.scope.npc_uuid,
            "session_id": request.session_id,
            "memory_enabled": True,
        }
        recent: list[ConversationTurn] = []
        matches: list[RetrievalMatch] = []
        session_turn_count = 0
        try:
            session = store.ensure_session(request.session_id, request.scope, request.metadata)
            session_turn_count = session.turn_count
            recent = store.recent_turns(
                request.session_id,
                request.scope,
                settings_limit(self.settings.memory_recent_turns),
            )
            if not recent:
                recent = [
                    ConversationTurn(role=item["role"], content=item["content"])
                    for item in request.recent_conversation
                ]
            store.add_turn(
                request.session_id,
                request.scope,
                "user",
                request.message,
                {"source": "project-hoomans", "request_id": request.request_id},
            )
            matches = store.retrieve(
                request.scope,
                request.message,
                limit=settings_limit(self.settings.memory_retrieval_limit),
            )
            diagnostics.update(
                {
                    "memory_path": str(store.path),
                    "recent_turn_count": len(recent),
                    "memory_count": store.stats().get("memory_count", 0),
                    "retrieved_memories": [match.as_diagnostic() for match in matches],
                }
            )
        except Exception as error:  # SQLite is an optional enhancement to dialogue.
            diagnostics.update({"memory_enabled": False, "memory_error": type(error).__name__})
            LOGGER.warning("NPC memory unavailable; continuing without it: %s", error)

        built = self.context_builder.build(
            ContextInput(
                npc_name=request.npc_name,
                player_name=request.player_name,
                character_card=request.character_card,
                relationship_snapshot=request.relationship_snapshot,
                preferences=request.preferences,
                current_state=request.current_state,
                retrieved_memories=tuple(matches),
                recent_turns=tuple(recent),
                available_tools=request.available_tools,
                current_message=request.message,
            )
        )
        provider_name, model_name = self.providers.resolve(request.provider, request.model)
        provider_request = ChatCompletionRequest(
            model=model_name,
            provider=provider_name,
            messages=built.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=built.tools or None,
            metadata={"source": "project-hoomans", "npc_uuid": request.scope.npc_uuid},
        )
        result = await self.providers.complete(provider_name, provider_request)
        diagnostics.update(
            {
                "provider": provider_name,
                "model": model_name,
                "context": built.diagnostics,
                "context_message_count": len(built.messages),
            }
        )

        try:
            store.add_turn(
                request.session_id,
                request.scope,
                "assistant",
                result.text,
                {"source": "provider", "provider": provider_name, "model": model_name},
            )
            current_count = session_turn_count + 2
            should_consolidate = request.end_session or (
                current_count >= self.settings.memory_consolidation_turns
                and current_count % self.settings.memory_consolidation_turns == 0
            )
            if should_consolidate:
                self._consolidate(store, request)
                diagnostics["consolidated"] = True
            else:
                diagnostics["consolidated"] = False
        except Exception as error:  # The response must not depend on persistence.
            diagnostics["memory_write_error"] = type(error).__name__
            LOGGER.warning("NPC memory write failed after completion: %s", error)
        return ConversationResult(result, request.session_id, tuple(matches), diagnostics)

    def _store(self, world_uuid: str) -> SQLiteMemoryStore:
        if world_uuid not in self._stores:
            root = self.settings.memory_root
            if root:
                memory_root = Path(root).expanduser()
            else:
                database_path = self.settings.database_path
                if database_path:
                    memory_root = Path(database_path).expanduser().parent / "memory"
                else:
                    memory_root = Path.home() / ".config" / "HoomansLLM" / "memory"
            self._stores[world_uuid] = SQLiteMemoryStore(memory_root, world_uuid)
        return self._stores[world_uuid]

    def _consolidate(self, store: SQLiteMemoryStore, request: ConversationRequest) -> None:
        turns = store.recent_turns(request.session_id, request.scope, limit=24)
        result = self.consolidator.consolidate(request.scope, request.session_id, turns)
        if result.summary:
            store.set_summary(request.session_id, request.scope, result.summary)
            store.remember(
                MemoryRecord(
                    memory_id=f"summary-{request.session_id}-{len(turns)}",
                    scope=request.scope,
                    memory_type=MemoryType.CONVERSATION_SUMMARY,
                    content=result.summary,
                    tags=("conversation", "summary"),
                    importance=0.6,
                    provenance={
                        "source": "conversation_consolidation",
                        "session_id": request.session_id,
                    },
                    session_id=request.session_id,
                )
            )
        for memory in result.memories:
            if memory.memory_type is MemoryType.COMMITMENT:
                store.add_commitment(
                    request.scope,
                    memory.content,
                    session_id=request.session_id,
                    importance=memory.importance,
                    provenance=memory.provenance,
                    commitment_id=memory.memory_id,
                )
            else:
                store.remember(memory)
        if request.end_session:
            store.end_session(request.session_id, request.scope, result.summary)


def settings_limit(value: int) -> int:
    return max(1, min(int(value), 64))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
