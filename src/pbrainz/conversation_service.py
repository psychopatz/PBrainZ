"""Conversation orchestration for Project Hoomans NPC dialogue."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pbrainz.api.models import ChatCompletionRequest, ChatMessage
from pbrainz.config import Settings
from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.memory import (
    ConversationTurn,
    DaySynopsis,
    MemoryEpisode,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    MemoryVisibility,
    RetrievalMatch,
    SQLiteMemoryStore,
    StructuredFact,
    TurnWriteResult,
    is_context_eligible,
    memory_root_for_settings,
)
from pbrainz.providers.base import CompletionResult
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.semantic_tool_protocol import (
    ensure_identity_intent,
    ensure_social_intent,
    extract_text_tool_calls,
    infer_social_intent,
    is_provider_scaffold,
    social_reply_repair_needed,
    strip_provider_scaffold,
)
from pbrainz.template_profiles import (
    TemplateProfile,
    active_template_profile,
    template_profile_for_provider,
)

LOGGER = logging.getLogger(__name__)
TraceWriter = Callable[..., None]


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    """Structured request sent by the Project Hoomans data tunnel."""

    request_id: str
    scope: MemoryScope
    session_id: str
    message: str
    message_id: str | None = None
    npc_name: str = "the survivor"
    player_name: str = "the player"
    character_card: dict[str, Any] = field(default_factory=dict)
    relationship_snapshot: dict[str, Any] = field(default_factory=dict)
    relationship_capabilities: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)
    participants: tuple[dict[str, Any], ...] = ()
    current_topic: str | None = None
    mentioned_entities: tuple[str, ...] = ()
    game_day: int | None = None
    world_age_hours: float | None = None
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
        message_id = _optional_text(
            value.get("message_id")
            or value.get("messageID")
            or context.get("message_id")
            or context.get("messageID")
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
            message_id=message_id,
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
            relationship_capabilities=_mapping(
                context.get("relationship_capabilities")
                or context.get("relationshipCapabilities")
            ),
            preferences=_mapping(context.get("preferences")),
            current_state=_mapping(context.get("current_state") or context.get("currentState")),
            scene=_mapping(context.get("scene") or context.get("scene_context")),
            participants=_participants(context.get("participants")),
            current_topic=_optional_text(
                context.get("current_topic") or context.get("currentTopic")
            ),
            mentioned_entities=tuple(
                str(item).strip()[:128]
                for item in (
                    context.get("mentioned_entities")
                    or context.get("mentionedEntities")
                    or []
                )
                if str(item).strip()
            )[:16],
            game_day=_optional_int(
                context,
                "game_day",
                "gameDay",
            ),
            world_age_hours=_optional_float(
                context,
                "world_age_hours",
                "worldAgeHours",
            ),
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
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.template_profile = active_template_profile(
            settings.template_profiles_json,
            settings.active_template_profile_id,
        )
        self.context_builder = ContextBuilder(
            max_chars=settings.context_max_chars,
            recent_turn_limit=settings.memory_recent_turns,
            memory_limit=settings.memory_retrieval_limit,
            tool_rag_enabled=settings.tool_rag_enabled,
            tool_limit=settings.tool_retrieval_limit,
            tool_budget_chars=settings.tool_budget_chars,
            template_profile=self.template_profile,
        )
        self.consolidator = consolidator or HeuristicConsolidator()
        self._stores: dict[str, SQLiteMemoryStore] = {}
        self._trace_writer = trace_writer

    def set_template_profile(self, profile: TemplateProfile) -> None:
        """Switch the live prompt profile without restarting the bridge."""

        self.template_profile = profile
        self.context_builder.set_template_profile(profile)

    def debug_trace_enabled(self) -> bool:
        """Return whether full diagnostic payload construction is active."""
        return self.settings.llm_trace_capture and self._trace_writer is not None

    def record_debug_trace(
        self,
        phase: str,
        payload: object,
        *,
        request_id: str = "",
        npc_id: str = "",
        session_id: str = "",
        source: str = "pbrainz",
    ) -> None:
        """Write opt-in diagnostics without serializing payloads when disabled."""
        if not self.debug_trace_enabled():
            return
        try:
            self._trace_writer(
                source=source,
                phase=phase,
                request_id=request_id,
                npc_id=npc_id,
                session_id=session_id,
                payload=payload,
            )
        except Exception as error:  # Diagnostics must never break gameplay.
            LOGGER.warning("LLM trace write failed: %s", error)

    async def complete(self, request: ConversationRequest) -> ConversationResult:
        if self.debug_trace_enabled():
            self.record_debug_trace(
                "conversation.input",
                {
                    "scope": {
                        "world_uuid": request.scope.world_uuid,
                        "player_uuid": request.scope.player_uuid,
                        "npc_uuid": request.scope.npc_uuid,
                    },
                    "session_id": request.session_id,
                    "message": request.message,
                    "npc_name": request.npc_name,
                    "player_name": request.player_name,
                    "character_card": request.character_card,
                    "relationship_snapshot": request.relationship_snapshot,
                    "relationship_capabilities": request.relationship_capabilities,
                    "preferences": request.preferences,
                    "current_state": request.current_state,
                    "scene": request.scene,
                    "participants": request.participants,
                    "current_topic": request.current_topic,
                    "mentioned_entities": request.mentioned_entities,
                    "game_day": request.game_day,
                    "world_age_hours": request.world_age_hours,
                    "recent_conversation": request.recent_conversation,
                    "available_tools": request.available_tools,
                    "provider": request.provider,
                    "model": request.model,
                },
                request_id=request.request_id,
                npc_id=request.scope.npc_uuid,
                session_id=request.session_id,
                source="pbrainz.conversation",
            )
        store = self._store(request.scope.world_uuid)
        diagnostics: dict[str, Any] = {
            "world_uuid": request.scope.world_uuid,
            "player_uuid": request.scope.player_uuid,
            "npc_uuid": request.scope.npc_uuid,
            "session_id": request.session_id,
            "memory_enabled": True,
        }
        recent: list[ConversationTurn] = []
        recalled: list[ConversationTurn] = []
        matches: list[RetrievalMatch] = []
        day_synopsis: DaySynopsis | None = None
        structured_facts: list[StructuredFact] = []
        session_turn_count = 0
        retrieval_needed = (
            self.settings.memory_rag_enabled and retrieval_needed_for(request.message)
        )
        try:
            session = store.ensure_session(request.session_id, request.scope, request.metadata)
            session_turn_count = session.turn_count
            recent = store.recent_turns(
                request.session_id,
                request.scope,
                settings_limit(self.settings.memory_recent_turns),
            )
            if retrieval_needed:
                recalled = store.search_turns(
                    request.scope,
                    request.message,
                    limit=settings_limit(self.settings.memory_recent_turns // 2),
                )
                recent_ids = {
                    turn.message_id for turn in recent if turn.message_id is not None
                }
                recalled = [
                    turn
                    for turn in recalled
                    if turn.message_id is None or turn.message_id not in recent_ids
                ]
            if not recent:
                recent = [
                    ConversationTurn(role=item["role"], content=item["content"])
                    for item in request.recent_conversation
                ]
            input_write = store.record_turn(
                request.session_id,
                request.scope,
                "user",
                request.message,
                message_id=request.message_id or f"llm-input:{request.request_id}",
                metadata={
                    "source": "project-hoomans",
                    "request_id": request.request_id,
                    "canonical_message_id": request.message_id,
                    "visibility": MemoryVisibility.PUBLIC.value,
                    "participants": [
                        str(item.get("id"))
                        for item in request.participants
                        if item.get("id")
                    ],
                },
                game_day=request.game_day,
                world_age_hours=request.world_age_hours,
                speaker_uuid=request.scope.player_uuid,
                speaker_name=request.player_name,
                speaker_kind="player",
            )
            LOGGER.info(
                "NPC memory turn saved phase=input npc=%s request=%s message=%s "
                "duplicate=%s skipped=%s",
                request.scope.npc_uuid,
                request.request_id,
                request.message_id or f"llm-input:{request.request_id}",
                input_write.duplicate,
                input_write.skipped,
            )
            query = MemoryQuery(
                scope=request.scope,
                actor_id=request.scope.npc_uuid,
                current_message=request.message,
                conversation_id=request.session_id,
                current_day=request.game_day,
                participants=tuple(
                    str(item.get("id"))
                    for item in request.participants
                    if item.get("id")
                ),
                mentioned_entities=request.mentioned_entities,
                current_topic=request.current_topic,
                max_results=settings_limit(self.settings.memory_retrieval_limit),
                token_budget=max(200, self.settings.memory_retrieval_limit * 120),
            )
            if retrieval_needed:
                matches = store.retrieve_query(query)
            if request.game_day is not None:
                day_synopsis = store.get_day_synopsis(request.scope, request.game_day)
            structured_facts = store.list_structured_facts(query, limit=12)
            diagnostics.update(
                {
                    "memory_path": str(store.path),
                    "recent_turn_count": len(recent),
                    "recalled_turn_count": len(recalled),
                    "retrieval_needed": retrieval_needed,
                    "retrieval_skipped": not retrieval_needed,
                    "day_synopsis_available": day_synopsis is not None,
                    "structured_fact_count": len(structured_facts),
                    "memory_count": store.stats().get("memory_count", 0),
                    "retrieved_memories": [match.as_diagnostic() for match in matches],
                }
            )
        except Exception as error:  # SQLite is an optional enhancement to dialogue.
            diagnostics.update({"memory_enabled": False, "memory_error": type(error).__name__})
            LOGGER.warning("NPC memory unavailable; continuing without it: %s", error)

        provider_name, model_name = self.providers.resolve(request.provider, request.model)
        configured_profile = active_template_profile(
            self.settings.template_profiles_json,
            self.settings.active_template_profile_id,
        )
        template_profile = template_profile_for_provider(
            self.settings.template_profiles_json,
            self.settings.active_template_profile_id,
            provider_name,
        )
        profile_auto_selected = template_profile.id != configured_profile.id
        context_started = time.perf_counter() if self.debug_trace_enabled() else None
        built = self.context_builder.build(
            ContextInput(
                npc_name=request.npc_name,
                player_name=request.player_name,
                character_card=request.character_card,
                relationship_snapshot=request.relationship_snapshot,
                relationship_capabilities=request.relationship_capabilities,
                preferences=request.preferences,
                current_state=request.current_state,
                scene=request.scene,
                day_synopsis=day_synopsis.synopsis if day_synopsis else "",
                structured_facts=tuple(
                    {
                        "kind": fact.kind,
                        "content": fact.content,
                        "truth_status": fact.truth_status,
                        "game_day": fact.game_day,
                    }
                    for fact in structured_facts
                ),
                retrieved_memories=tuple(matches),
                recalled_turns=tuple(recalled),
                recent_turns=tuple(recent),
                available_tools=request.available_tools,
                current_message=request.message,
            ),
            template_profile=template_profile,
        )
        context_build_ms = _elapsed_ms(context_started)
        provider_request = ChatCompletionRequest(
            model=model_name,
            provider=provider_name,
            messages=built.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=built.tools or None,
            stop=list(template_profile.stop_sequences) or None,
            metadata={
                "source": "project-hoomans",
                "npc_uuid": request.scope.npc_uuid,
                "template_profile_id": template_profile.id,
                "template_profile_requested_id": configured_profile.id,
                "template_profile_auto_selected": profile_auto_selected,
            },
        )
        if self.debug_trace_enabled():
            self.record_debug_trace(
                "provider.request",
                {
                    "attempt": 1,
                    "provider": provider_name,
                    "model": model_name,
                    "messages": [
                        message.model_dump(exclude_none=True) for message in built.messages
                    ],
                    "tools": built.tools,
                    "context_diagnostics": built.diagnostics,
                    "context_build_ms": context_build_ms,
                },
                request_id=request.request_id,
                npc_id=request.scope.npc_uuid,
                session_id=request.session_id,
                source="pbrainz.provider",
            )
        provider_started = time.perf_counter() if self.debug_trace_enabled() else None
        try:
            result = await self.providers.complete(provider_name, provider_request)
        except Exception as error:
            if self.debug_trace_enabled():
                self.record_debug_trace(
                    "provider.error",
                    {
                        "attempt": 1,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "latency_ms": _elapsed_ms(provider_started),
                    },
                    request_id=request.request_id,
                    npc_id=request.scope.npc_uuid,
                    session_id=request.session_id,
                    source="pbrainz.provider",
                )
            raise
        if self.debug_trace_enabled():
            self.record_debug_trace(
                "provider.response",
                _completion_trace_payload(
                    result,
                    attempt=1,
                    latency_ms=_elapsed_ms(provider_started),
                    context_build_ms=context_build_ms,
                ),
                request_id=request.request_id,
                npc_id=request.scope.npc_uuid,
                session_id=request.session_id,
                source="pbrainz.provider",
            )
        # Some providers return a structurally valid tool-only candidate. Make
        # one bounded text-only repair pass so the game gets natural dialogue as
        # well as the semantic action. The original authorized calls are kept
        # separately and remain the only calls that can reach the game. This
        # costs a second provider request only for the otherwise underwhelming
        # tool-only case; ordinary turns stay single-pass.
        initial_tool_calls = list(result.tool_calls or [])
        initial_authorized_tool_calls = _authorized_tool_call_count(
            initial_tool_calls, built.tools
        )
        initial_result = result
        result_text = strip_provider_scaffold(result.text)
        if result.text and is_provider_scaffold(result.text):
            diagnostics["provider_scaffold_filtered"] = True
        should_retry_empty = not result_text and bool(built.tools)
        should_retry_contextual = (
            bool(built.tools)
            and social_reply_repair_needed(request.message, result_text)
        )
        if should_retry_empty or should_retry_contextual:
            retry_system = (built.messages[0].content or "").rstrip()
            retry_system += (
                "\n\nReturn only the NPC's short spoken reply for this turn. This is "
                "a text repair after the game selected any needed tool calls. Do "
                "not call tools or emit action markup; write one or two concise "
                "in-world sentences as plain dialogue text. Do not say 'I'll "
                "check that now', 'I will take care of that', or 'I understand'. "
                "For an ask_name action, do not invent a name; use a natural "
                "introduction lead-in and let the game provide the authoritative "
                "name."
            )
            if should_retry_contextual:
                intent = infer_social_intent(request.message) or {}
                subtype = str(intent.get("subtype") or "social action")
                if subtype == "sexual_advance":
                    retry_system += (
                        " The player made an explicit sexual advance. Respond "
                        "directly and in character; if it is unwelcome, set a "
                        "clear boundary. Do not claim that consent, sex, or a "
                        "relationship change occurred, and do not mention tools "
                        "or this repair."
                    )
                else:
                    retry_system += (
                        " The player used hostile abuse. Respond directly and in "
                        "character with a boundary, warning, or anger. Do not "
                        "mention tools or this repair."
                    )
            retry_messages = list(built.messages)
            retry_messages[0] = ChatMessage(role="system", content=retry_system)
            retry_max_tokens = (
                120
                if provider_request.max_tokens is None
                else min(provider_request.max_tokens, 120)
            )
            retry_request = provider_request.model_copy(
                update={
                    "messages": retry_messages,
                    "tools": None,
                    "max_tokens": retry_max_tokens,
                }
            )
            if should_retry_empty:
                diagnostics["empty_response_retry"] = (
                    "text_only_authorized_tools"
                    if initial_authorized_tool_calls
                    else "text_only_unrecognized_tools"
                    if initial_tool_calls
                    else "text_only"
                )
            else:
                diagnostics["contextual_response_retry"] = "explicit_social_subtype"
            repair_reason = diagnostics.get("empty_response_retry") or diagnostics.get(
                "contextual_response_retry"
            )
            if self.debug_trace_enabled():
                self.record_debug_trace(
                    "provider.request",
                    {
                        "attempt": 2,
                        "reason": repair_reason,
                        "provider": provider_name,
                        "model": model_name,
                        "messages": [
                            message.model_dump(exclude_none=True)
                            for message in retry_messages
                        ],
                        "tools": None,
                    },
                    request_id=request.request_id,
                    npc_id=request.scope.npc_uuid,
                    session_id=request.session_id,
                    source="pbrainz.provider",
                )
            retry_started = time.perf_counter() if self.debug_trace_enabled() else None
            try:
                result = await self.providers.complete(provider_name, retry_request)
            except Exception as error:
                if self.debug_trace_enabled():
                    self.record_debug_trace(
                        "provider.error",
                        {
                            "attempt": 2,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "latency_ms": _elapsed_ms(retry_started),
                        },
                        request_id=request.request_id,
                        npc_id=request.scope.npc_uuid,
                        session_id=request.session_id,
                        source="pbrainz.provider",
                    )
                if initial_authorized_tool_calls:
                    # A repair failure must not discard an already-valid game
                    # action. The bridge will select its bounded fallback
                    # dialogue while still executing these original calls.
                    diagnostics["tool_response_repair_error"] = type(error).__name__
                    LOGGER.warning(
                        "NPC tool-response repair failed; preserving authorized "
                        "tool calls npc=%s request=%s: %s",
                        request.scope.npc_uuid,
                        request.request_id,
                        error,
                    )
                    result = initial_result
                else:
                    raise
            if self.debug_trace_enabled():
                self.record_debug_trace(
                    "provider.response",
                    _completion_trace_payload(
                        result,
                        attempt=2,
                        latency_ms=_elapsed_ms(retry_started),
                        context_build_ms=context_build_ms,
                    ),
                    request_id=request.request_id,
                    npc_id=request.scope.npc_uuid,
                    session_id=request.session_id,
                    source="pbrainz.provider",
                )
            result_text = strip_provider_scaffold(result.text)
            if result.text and is_provider_scaffold(result.text):
                diagnostics["provider_scaffold_filtered"] = True

            if initial_authorized_tool_calls:
                # Never allow the text-only repair to replace or invent the
                # authoritative tool decision. A provider that ignores the
                # tools=None repair request cannot add a second action.
                authorized_names = {
                    _tool_name(tool)
                    for tool in built.tools
                    if _tool_name(tool)
                }
                preserved_calls = [
                    call
                    for call in initial_tool_calls
                    if _tool_name(call) in authorized_names
                ]
                result = CompletionResult(
                    model=result.model,
                    text=result.text,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    tool_calls=preserved_calls or None,
                    reasoning=result.reasoning,
                )
        protocol_request = {
            "request_id": request.request_id,
            "npc_id": request.scope.npc_uuid,
            "conversation_context": {
                "message": request.message,
                "available_tools": list(request.available_tools),
            },
        }
        result_text, text_tool_calls = extract_text_tool_calls(
            result_text, protocol_request
        )
        if result_text:
            result_text = strip_provider_scaffold(result_text)
        normalized_tool_calls = list(result.tool_calls or [])
        normalized_tool_calls.extend(text_tool_calls)
        normalized_tool_calls = ensure_social_intent(
            normalized_tool_calls, protocol_request
        )
        normalized_tool_calls = ensure_identity_intent(
            normalized_tool_calls, protocol_request
        )
        result = CompletionResult(
            model=result.model,
            text=result_text,
            finish_reason=result.finish_reason,
            usage=result.usage,
            tool_calls=normalized_tool_calls or None,
            reasoning=result.reasoning,
        )
        diagnostics.update(
            {
                "provider": provider_name,
                "model": model_name,
                "context": built.diagnostics,
                "context_message_count": len(built.messages),
                "finish_reason": result.finish_reason or "unknown",
                "tool_call_count": len(result.tool_calls or []),
                "provider_tool_call_count": len(initial_tool_calls),
                "provider_authorized_tool_call_count": initial_authorized_tool_calls,
                "reasoning_available": bool(result.reasoning),
            }
        )
        if result.usage and result.usage.as_dict():
            diagnostics["usage"] = result.usage.as_dict()

        try:
            if result_text:
                output_write = store.record_turn(
                    request.session_id,
                    request.scope,
                    "assistant",
                    result_text,
                    message_id=f"llm-response:{request.request_id}",
                    metadata={
                        "source": "provider",
                        "provider": provider_name,
                        "model": model_name,
                        "visibility": MemoryVisibility.PUBLIC.value,
                        "participants": [
                            str(item.get("id"))
                            for item in request.participants
                            if item.get("id")
                        ],
                    },
                    game_day=request.game_day,
                    world_age_hours=request.world_age_hours,
                    speaker_uuid=request.scope.npc_uuid,
                    speaker_name=request.npc_name,
                    speaker_kind="npc",
                )
                LOGGER.info(
                    "NPC memory turn saved phase=output npc=%s request=%s message=%s "
                    "duplicate=%s skipped=%s",
                    request.scope.npc_uuid,
                    request.request_id,
                    f"llm-response:{request.request_id}",
                    output_write.duplicate,
                    output_write.skipped,
                )
                current_count = session_turn_count + 2
                consolidation_threshold = max(
                    1, int(self.settings.memory_consolidation_turns)
                )
                should_consolidate = request.end_session or (
                    current_count >= consolidation_threshold
                    and session_turn_count // consolidation_threshold
                    < current_count // consolidation_threshold
                )
                if should_consolidate:
                    self._consolidate(store, request)
                    diagnostics["consolidated"] = True
                    stats = store.stats()
                    for key in (
                        "memory_count",
                        "episode_count",
                        "fact_count",
                    ):
                        diagnostics[key] = stats[key]
                else:
                    diagnostics["consolidated"] = False
            else:
                diagnostics["empty_response"] = True
                LOGGER.warning(
                    "NPC provider returned no dialogue text provider=%s model=%s "
                    "finish_reason=%s tool_calls=%s",
                    provider_name,
                    model_name,
                    result.finish_reason or "unknown",
                    len(result.tool_calls or []),
                )
        except Exception as error:  # The response must not depend on persistence.
            diagnostics["memory_write_error"] = type(error).__name__
            LOGGER.warning("NPC memory write failed after completion: %s", error)
        return ConversationResult(result, request.session_id, tuple(matches), diagnostics)

    def record_message(self, message: dict[str, Any]) -> TurnWriteResult:
        """Persist one canonical game message and make retries harmless."""
        if not isinstance(message, dict):
            raise ValueError("conversation sync message must be an object")
        world_uuid = _text_value(message, "world_uuid", "worldUUID", "save_uuid", "saveUUID")
        player_uuid = _text_value(message, "player_uuid", "playerUUID")
        npc_uuid = _text_value(message, "npc_uuid", "npcUUID")
        message_id = _text_value(message, "message_id", "messageID", "event_id", "eventID")
        conversation_id = _text_value(
            message, "conversation_id", "conversationID", "session_id", "sessionID"
        )
        content = _text_value(message, "text", "content")
        if not all((world_uuid, player_uuid, npc_uuid, message_id, conversation_id, content)):
            raise ValueError(
                "conversation sync messages require world, player, NPC, message ID, "
                "conversation ID, and text"
            )
        scope = MemoryScope(world_uuid, player_uuid, npc_uuid)
        source = message.get("source")
        metadata: dict[str, Any] = {
            "source": "project-hoomans-sync",
            "conversation_id": conversation_id,
        }
        namespace = _optional_text(message.get("namespace"))
        if namespace:
            metadata["namespace"] = namespace
        if isinstance(source, dict):
            metadata["event_source"] = source
        provenance = message.get("provenance")
        if isinstance(provenance, dict):
            metadata["provenance"] = provenance
        participants = message.get("participants")
        if isinstance(participants, list):
            metadata["participants"] = participants[:16]
        speaker_kind = _text_value(message, "speaker_kind", "speakerKind")
        role = "user" if speaker_kind.casefold() == "player" else "assistant"
        if not is_context_eligible(
            content,
            role=role,
            metadata=metadata,
        ):
            result = TurnWriteResult(
                ConversationTurn(
                    role=role,
                    content=content,
                    message_id=message_id,
                    metadata=metadata,
                    speaker_kind=speaker_kind,
                ),
                skipped=True,
            )
            LOGGER.info(
                "NPC memory sync skipped message=%s reason=llm_context_excluded",
                message_id,
            )
            return result
        store = self._store(world_uuid)
        store.ensure_session(
            conversation_id,
            scope,
            {"source": "project-hoomans-sync", "conversation_id": conversation_id},
        )
        result = store.record_turn(
            conversation_id,
            scope,
            role,
            content,
            message_id=message_id,
            metadata=metadata,
            game_day=_optional_int(message, "game_day", "gameDay"),
            world_age_hours=_optional_float(message, "world_age_hours", "worldAgeHours"),
            speaker_uuid=_optional_text(
                message.get("speaker_uuid") or message.get("speakerID")
            ),
            speaker_name=_optional_text(
                message.get("speaker_name") or message.get("speakerName")
            ),
            speaker_kind=speaker_kind,
        )
        LOGGER.info(
            "NPC memory sync saved npc=%s conversation=%s message=%s role=%s "
            "duplicate=%s skipped=%s",
            npc_uuid,
            conversation_id,
            message_id,
            role,
            result.duplicate,
            result.skipped,
        )
        return result

    def record_message_batch(self, batch: dict[str, Any]) -> tuple[str, ...]:
        """Record valid outbox entries and return only IDs safe to acknowledge."""
        if not isinstance(batch, dict):
            raise ValueError("conversation sync batch must be an object")
        messages = batch.get("messages")
        if not isinstance(messages, list):
            raise ValueError("conversation sync batch messages must be a list")
        acknowledged: list[str] = []
        for message in messages:
            try:
                result = self.record_message(message)
            except ValueError as error:
                LOGGER.warning("Skipping invalid conversation sync message: %s", error)
                continue
            acknowledged.append(result.turn.message_id or "")
        LOGGER.info(
            "NPC memory sync batch processed records=%s acknowledged=%s",
            len(messages),
            len(acknowledged),
        )
        return tuple(message_id for message_id in acknowledged if message_id)

    def _store(self, world_uuid: str) -> SQLiteMemoryStore:
        if world_uuid not in self._stores:
            self._stores[world_uuid] = SQLiteMemoryStore(
                memory_root_for_settings(self.settings), world_uuid
            )
        return self._stores[world_uuid]

    def _consolidate(self, store: SQLiteMemoryStore, request: ConversationRequest) -> None:
        turns = store.recent_turns(request.session_id, request.scope, limit=24)
        result = self.consolidator.consolidate(request.scope, request.session_id, turns)
        participant_ids = _participant_ids(request)
        game_day = request.game_day
        summary_saved = bool(result.summary)
        memories_saved = len(result.memories)
        if game_day is None:
            dated_turns = [turn.game_day for turn in turns if turn.game_day is not None]
            game_day = dated_turns[-1] if dated_turns else None
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
                        "participants": participant_ids,
                    },
                    session_id=request.session_id,
                    visibility=MemoryVisibility.PUBLIC,
                    game_day=game_day,
                    participants=participant_ids,
                    topic_tags=((request.current_topic,) if request.current_topic else ()),
                    transcript_ref=request.session_id,
                )
            )
            store.remember_episode(
                MemoryEpisode(
                    episode_id=f"episode:{request.session_id}:{game_day or 0}",
                    scope=request.scope,
                    conversation_id=request.session_id,
                    game_day=game_day,
                    participants=participant_ids,
                    witnesses=participant_ids,
                    topic_tags=((request.current_topic,) if request.current_topic else ()),
                    summary=result.summary,
                    key_facts=tuple(memory.content for memory in result.memories[:8]),
                    importance=0.65,
                    visibility=MemoryVisibility.PUBLIC,
                    transcript_ref=request.session_id,
                )
            )
        for memory in result.memories:
            memory = MemoryRecord(
                memory_id=memory.memory_id,
                scope=memory.scope,
                memory_type=memory.memory_type,
                content=memory.content,
                tags=memory.tags,
                importance=memory.importance,
                state=memory.state,
                provenance={
                    **memory.provenance,
                    "participants": participant_ids,
                    "truth_status": (
                        "unverified"
                        if memory.memory_type in {MemoryType.CLAIM, MemoryType.HEARSAY}
                        else "stated"
                    ),
                },
                session_id=memory.session_id,
                game_day=game_day,
                visibility=MemoryVisibility.PUBLIC,
                participants=participant_ids,
                transcript_ref=request.session_id,
            )
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
            store.save_structured_fact(
                StructuredFact(
                    fact_id=f"fact:{request.session_id}:{memory.memory_id}",
                    scope=request.scope,
                    kind=memory.memory_type.value,
                    content=memory.content,
                    source_uuid=request.scope.player_uuid,
                    truth_status=memory.provenance.get("truth_status", "stated"),
                    visibility=MemoryVisibility.PUBLIC,
                    game_day=game_day,
                    importance=memory.importance,
                    provenance=memory.provenance,
                    conversation_id=request.session_id,
                )
            )
        if result.summary and game_day is not None:
            previous = store.get_day_synopsis(request.scope, game_day)
            summary_lines = [
                line.strip() for line in result.summary.splitlines() if line.strip()
            ]
            prior_lines = previous.synopsis.splitlines() if previous else []
            merged_lines = list(dict.fromkeys(prior_lines + summary_lines))[-12:]
            commitments = list(previous.commitments if previous else ())
            claims = list(previous.claims if previous else ())
            for memory in result.memories:
                if memory.memory_type is MemoryType.COMMITMENT:
                    commitments.append(memory.content)
                elif memory.memory_type in {MemoryType.CLAIM, MemoryType.HEARSAY}:
                    claims.append(memory.content)
            store.save_day_synopsis(
                DaySynopsis(
                    scope=request.scope,
                    game_day=game_day,
                    synopsis="\n".join(dict.fromkeys(merged_lines))[-2400:],
                    commitments=tuple(dict.fromkeys(commitments))[-16:],
                    claims=tuple(dict.fromkeys(claims))[-16:],
                    unresolved_topics=tuple(
                        [request.current_topic] if request.current_topic else ()
                    ),
                )
            )
        if request.end_session:
            store.end_session(request.session_id, request.scope, result.summary)
        stats = store.stats(request.scope)
        LOGGER.info(
            "NPC memory consolidation saved npc=%s session=%s turns=%s "
            "summary_saved=%s memories_saved=%s end_session=%s "
            "totals_memory=%s totals_episode=%s totals_fact=%s",
            request.scope.npc_uuid,
            request.session_id,
            len(turns),
            summary_saved,
            memories_saved,
            request.end_session,
            stats.get("memory_count", 0),
            stats.get("episode_count", 0),
            stats.get("fact_count", 0),
        )


_HISTORICAL_CUES = re.compile(
    r"\b(remember|yesterday|earlier|before|last|morning|afternoon|said|told|"
    r"happened|trust|trusted|promise|promised|agreed|again|what did|why do you)\b",
    re.I,
)


def retrieval_needed_for(message: str) -> bool:
    """Cheap gate: keep historical RAG off the path for ordinary chatter."""
    normalized = " ".join(str(message or "").split())
    if not normalized:
        return False
    return bool(_HISTORICAL_CUES.search(normalized))


def _tool_name(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    function = value.get("function")
    if not isinstance(function, dict):
        function = value
    return str(function.get("name") or "").strip()


def _authorized_tool_call_count(
    tool_calls: list[dict[str, Any]],
    exposed_tools: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> int:
    exposed = {_tool_name(tool) for tool in exposed_tools}
    exposed.discard("")
    return sum(1 for call in tool_calls if _tool_name(call) in exposed)


def _elapsed_ms(start: float | None) -> float | None:
    if start is None:
        return None
    return round((time.perf_counter() - start) * 1000, 2)


def _completion_trace_payload(
    result: CompletionResult,
    *,
    attempt: int,
    latency_ms: float | None = None,
    context_build_ms: float | None = None,
) -> dict[str, Any]:
    """Expose provider-returned diagnostics without requesting hidden reasoning."""
    return {
        "attempt": attempt,
        "model": result.model,
        "text": result.text,
        "finish_reason": result.finish_reason,
        "tool_calls": result.tool_calls,
        "reasoning": result.reasoning,
        "usage": result.usage.as_dict() if result.usage else None,
        "latency_ms": latency_ms,
        "context_build_ms": context_build_ms,
    }


def _participants(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value[:16]:
        if isinstance(item, dict):
            participant = {
                key: item[key]
                for key in ("id", "name", "kind", "role", "active")
                if item.get(key) is not None
            }
        else:
            participant = {"id": str(item)}
        participant_id = _optional_text(
            participant.get("id") or participant.get("speakerID")
        )
        if not participant_id or participant_id in seen:
            continue
        participant["id"] = participant_id
        seen.add(participant_id)
        output.append(participant)
    return tuple(output)


def _participant_ids(request: ConversationRequest) -> tuple[str, ...]:
    values = [request.scope.player_uuid, request.scope.npc_uuid]
    values.extend(
        str(item.get("id")) for item in request.participants if item.get("id")
    )
    return tuple(dict.fromkeys(value for value in values if value))


def settings_limit(value: int) -> int:
    return max(1, min(int(value), 64))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _text_value(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        result = _optional_text(value.get(key))
        if result:
            return result
    return ""


def _optional_int(value: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if value.get(key) is not None:
            try:
                return int(value[key])
            except (TypeError, ValueError):
                return None
    return None


def _optional_float(value: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if value.get(key) is not None:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    return None
