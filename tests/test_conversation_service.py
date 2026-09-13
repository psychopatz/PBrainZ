import pytest

from pbrainz.config import Settings
from pbrainz.conversation_service import (
    ConversationRequest,
    ConversationService,
    retrieval_needed_for,
)
from pbrainz.memory import MemoryIdentity, MemoryRecord, MemoryScope, MemoryType, MemoryVisibility
from pbrainz.providers.base import CompletionResult, StreamEvent


class FakeProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        return CompletionResult(request.model, "I remember that.")


class StreamingProviders(FakeProviders):
    async def stream_events(self, provider, request):
        self.requests.append((provider, request))
        yield StreamEvent(text="First streamed part ")
        yield StreamEvent(text="and the final part.", finish_reason="stop")


class HordeTemplateProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return "horde", model if model not in {"default", "auto"} else "horde-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        return CompletionResult(request.model, "I am here.")


class EmptyThenTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        if len(self.requests) == 1:
            return CompletionResult(request.model, "", finish_reason="stop")
        return CompletionResult(request.model, "I am here.")


class TruncatedThenTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        if len(self.requests) == 1:
            return CompletionResult(request.model, "I was just", finish_reason="length")
        return CompletionResult(request.model, "I was just thinking about resting.")


class AuthorizedToolOnlyThenTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        if len(self.requests) == 1:
            return CompletionResult(
                request.model,
                "",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "social-call",
                        "function": {
                            "name": "social_react",
                            "arguments": '{"kind":"admire"}',
                        },
                    }
                ],
            )
        return CompletionResult(request.model, "You have my respect.")


class CombinedTextAndToolProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        return CompletionResult(
            request.model,
            "You have my respect.",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "social-call",
                    "function": {
                        "name": "social_react",
                        "arguments": '{"kind":"admire"}',
                    },
                }
            ],
        )


class ExplicitSocialGenericThenTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        return CompletionResult(request.model, "No. Back off.")


class UnselectedTextActionProviders:
    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        return CompletionResult(
            request.model,
            'I live nearby. <projecthoomans-action>{"name":"ask_name",'
            '"arguments":{}}</projecthoomans-action>',
        )


class UnknownToolThenTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        if len(self.requests) == 1:
            return CompletionResult(
                request.model,
                "",
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "unknown-call",
                        "function": {"name": "unknown_tool", "arguments": "{}"},
                    }
                ],
            )
        return CompletionResult(request.model, "I can help with that.")


class TextEmptyTextProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        if len(self.requests) == 2:
            return CompletionResult(request.model, "", finish_reason="stop")
        return CompletionResult(request.model, "The plan still stands.")


class HordeScaffoldProviders:
    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        return CompletionResult(
            request.model,
            '"Ugh, you look terrible."\n\nInstruction:\n'
            "dude you look terrible\n\nResponse",
        )


def _memory_identity(world_uuid: str) -> MemoryIdentity:
    return MemoryIdentity.from_mapping(
        world_uuid,
        {
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": world_uuid,
        },
    )


def _conversation_request(
    request_id: str,
    world_uuid: str,
    player_uuid: str,
    npc_uuid: str,
    session_id: str,
    message: str,
    **kwargs,
) -> ConversationRequest:
    identity = _memory_identity(world_uuid)
    return ConversationRequest(
        request_id=request_id,
        scope=MemoryScope(identity.world_uuid, player_uuid, npc_uuid),
        session_id=session_id,
        message=message,
        memory_identity=identity,
        **kwargs,
    )


@pytest.mark.parametrize(
    "message",
    (
        "When did we first meet exactly?",
        "Do you remember our first meeting?",
        "Have we met before?",
    ),
)
def test_relationship_history_questions_enable_memory_retrieval(message: str) -> None:
    assert retrieval_needed_for(message) is True


@pytest.mark.asyncio
async def test_first_meeting_memory_is_added_to_rag_context(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        context_max_chars=8000,
    )
    service = ConversationService(settings, providers)
    identity = _memory_identity("world-one")
    scope = MemoryScope(identity.world_uuid, "player-one", "npc-one")
    service._store(identity).remember(
        MemoryRecord(
            "first-meeting",
            scope,
            MemoryType.PERSONAL_EVENT,
            "The player first met Alice on July 9, 1993 (Day 0, 15:39).",
            tags=("primitive", "first_meeting"),
            importance=0.8,
            visibility=MemoryVisibility.PUBLIC,
            participants=("player-one", "npc-one"),
            entity_refs=("npc-one",),
        )
    )

    result = await service.complete(
        _conversation_request(
            "first-meeting-question",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "When did we first meet exactly?",
        )
    )

    assert result.diagnostics["retrieval_needed"] is True
    assert result.retrieved_memories
    prompt = "\n".join(message.content or "" for message in providers.requests[0][1].messages)
    assert "The player first met Alice on July 9, 1993" in prompt


@pytest.mark.asyncio
async def test_conversation_leaves_default_generation_to_selected_model(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)

    await service.complete(
        _conversation_request(
            "bounded-output",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "Hello there.",
        )
    )

    assert providers.requests[0][1].max_tokens is None
    assert providers.requests[0][1].reasoning_effort is None


@pytest.mark.asyncio
async def test_bridge_history_replaces_partial_provider_session_history(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    identity = _memory_identity("world-one")
    scope = MemoryScope(identity.world_uuid, "player-one", "npc-one")
    store = service._store(identity)
    store.ensure_session("session-one", scope)
    store.add_turn("session-one", scope, "assistant", "Partial stored response.")

    result = await service.complete(
        _conversation_request(
            "bridge-history",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "Good.",
            recent_conversation=(
                {"role": "assistant", "content": "Older orphan response."},
                {"role": "user", "content": "Are you still there?"},
                {"role": "assistant", "content": "I am here."},
            ),
        )
    )

    request = providers.requests[0][1]
    assert [message.role for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    rendered = "\n".join(message.content or "" for message in request.messages)
    assert "Partial stored response." not in rendered
    assert "Older orphan response." in rendered
    assert "Conversation history begins with an NPC reply" in rendered
    assert result.diagnostics["history_projection"] == {
        "source": "bridge",
        "bridge_turns": 3,
        "stored_turns": 1,
        "selected_turns": 4,
        "dropped_leading_assistant_turns": 0,
        "anchored_leading_assistant_turns": 1,
    }
    assert request.max_tokens is None
    assert request.reasoning_effort is None


@pytest.mark.asyncio
async def test_provider_history_input_projection_does_not_collide_with_canonical_id(
    tmp_path,
) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    canonical = {
        "messageID": "conversation-one:player-1",
        "saveUUID": "world-one",
        "worldMode": "multiplayer",
        "serverInstanceId": "test-server",
        "serverWorldGeneration": "world-one",
        "conversationID": "conversation-one",
        "playerUUID": "player-one",
        "npcUUID": "npc-one",
        "speakerID": "player-one",
        "speakerName": "Alex",
        "speakerKind": "player",
        "text": "Hello there.",
    }
    service.record_message(canonical)

    identity = _memory_identity("world-one")
    scope = MemoryScope(identity.world_uuid, "player-one", "npc-one")
    await service.complete(
        _conversation_request(
            "provider-projection",
            "world-one",
            "player-one",
            "npc-one",
            "pnc-session-one",
            "Hello there.",
            message_id=canonical["messageID"],
        )
    )

    turns = service._store(identity).recent_turns("pnc-session-one", scope, 8)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].message_id == (
        "llm-input:pnc-session-one:conversation-one:player-1"
    )
    assert turns[0].metadata["canonical_message_id"] == canonical["messageID"]


@pytest.mark.asyncio
async def test_truncated_dialogue_retries_same_model_without_imposed_limits(tmp_path) -> None:
    providers = TruncatedThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)

    result = await service.complete(
        _conversation_request(
            "truncated-dialogue",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "What do you mean?",
        )
    )

    assert result.completion.text == "I was just thinking about resting."
    assert result.diagnostics["truncated_response_retry"] == "same_model_text_recovery"
    assert len(providers.requests) == 2
    assert providers.requests[0][1].model == providers.requests[1][1].model == "fake-model"
    assert providers.requests[0][1].max_tokens is None
    assert providers.requests[1][1].max_tokens is None
    assert providers.requests[0][1].reasoning_effort is None
    assert providers.requests[1][1].reasoning_effort is None


@pytest.mark.asyncio
async def test_structured_conversation_owns_history_and_consolidates(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        memory_consolidation_turns=2,
        context_max_chars=4000,
    )
    service = ConversationService(settings, providers)
    request = _conversation_request(
        "request-one",
        "world-one",
        "player-one",
        "npc-one",
        "session-one",
        "My name is Alex.",
        npc_name="Harley",
        player_name="Alex",
    )

    first = await service.complete(request)
    second = await service.complete(
        _conversation_request(
            "request-two",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "What do you remember about me?",
        )
    )

    assert first.completion.text == "I remember that."
    assert len(providers.requests) == 2
    messages = providers.requests[1][1].messages
    assert any(message.content == "I remember that." for message in messages)
    assert second.diagnostics["consolidated"] is True
    assert second.diagnostics["memory_count"] >= 1


@pytest.mark.asyncio
async def test_optional_provider_stream_reconstructs_text_and_forwards_deltas(tmp_path) -> None:
    providers = StreamingProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        context_max_chars=4000,
    )
    service = ConversationService(settings, providers)
    chunks: list[str] = []

    async def consume(chunk: str) -> None:
        chunks.append(chunk)

    result = await service.complete(
        _conversation_request(
            "stream-request",
            "world-stream",
            "player-stream",
            "npc-stream",
            "session-stream",
            "What happened?",
        ),
        stream_consumer=consume,
    )

    assert chunks == ["First streamed part ", "and the final part."]
    assert result.completion.text == "First streamed part and the final part."
    assert result.diagnostics["provider_streaming"] is True
    assert len(providers.requests) == 1


@pytest.mark.asyncio
async def test_horde_request_uses_dedicated_instruct_profile(tmp_path) -> None:
    providers = HordeTemplateProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="horde",
        context_max_chars=4000,
    )
    service = ConversationService(settings, providers)

    result = await service.complete(
        _conversation_request(
            "horde-request",
            "world-horde",
            "player-horde",
            "npc-horde",
            "horde-session",
            "What's your name?",
            npc_name="Harley",
            player_name="Alex",
            provider="horde",
        )
    )

    assert result.completion.text == "I am here."
    assert len(providers.requests) == 1
    provider, request = providers.requests[0]
    assert provider == "horde"
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert "User: What's your name?" in (request.messages[0].content or "")
    assert request.stop == ["\nUser: ", "\n### Instruction:"]
    assert request.metadata["template_profile_id"] == "instruct-text"
    assert request.metadata["template_profile_requested_id"] == "native-chat"
    assert request.metadata["template_profile_auto_selected"] is True


def test_structured_request_rejects_missing_scope() -> None:
    with pytest.raises(ValueError):
        ConversationRequest.from_mapping(
            {"request_id": "one", "conversation_context": {"message": "hello"}}
        )


@pytest.mark.asyncio
async def test_horde_prompt_scaffold_is_filtered_before_memory_write(tmp_path) -> None:
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, HordeScaffoldProviders())
    identity = _memory_identity("world-one")
    scope = MemoryScope(identity.world_uuid, "player-one", "npc-one")
    request = _conversation_request(
        "horde-scaffold",
        "world-one",
        "player-one",
        "npc-one",
        "session-one",
        "dude you look terrible",
        npc_name="Emilio",
        player_name="Alex",
    )

    result = await service.complete(request)

    assert result.completion.text == '"Ugh, you look terrible."'
    assert result.diagnostics["provider_scaffold_filtered"] is True
    turns = service._store(identity).recent_turns("session-one", scope, 8)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert "Instruction:" not in turns[-1].content


@pytest.mark.asyncio
async def test_empty_provider_candidate_retries_without_tools(tmp_path) -> None:
    providers = EmptyThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    request = _conversation_request(
        "retry-one",
        "world-one",
        "player-one",
        "npc-one",
        "session-one",
        "Where are you?",
        available_tools=(
            {
                "type": "function",
                "function": {
                    "name": "social_react",
                    "description": "React socially.",
                    "parameters": {"type": "object"},
                },
            },
        ),
    )

    result = await service.complete(request)

    assert result.completion.text == "I am here."
    assert result.diagnostics["empty_response_retry"] == "text_only"
    assert len(providers.requests) == 2
    assert providers.requests[0][1].tools
    assert providers.requests[1][1].tools is None


@pytest.mark.asyncio
async def test_unselected_text_action_is_not_replayed(tmp_path) -> None:
    providers = UnselectedTextActionProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    result = await service.complete(
        _conversation_request(
            "unselected-text-action",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "Where do you live?",
            available_tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "ask_name",
                        "description": "Ask the NPC to say their name.",
                    },
                },
            ),
        )
    )

    assert result.completion.text == "I live nearby."
    assert result.completion.tool_calls is None
    assert result.diagnostics["context"]["tool_routing"]["sent"] == 0


@pytest.mark.asyncio
async def test_authorized_tool_only_candidate_gets_dialogue_repair(
    tmp_path,
) -> None:
    providers = AuthorizedToolOnlyThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    result = await service.complete(
        _conversation_request(
            "repair-authorized-tool",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "Where are you?",
            available_tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "social_react",
                        "description": "React socially.",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        )
    )

    assert result.completion.text == "You have my respect."
    assert result.completion.tool_calls[0]["function"]["name"] == "social_react"
    assert result.diagnostics["initial_response_shape"] == "tool_only"
    assert result.diagnostics["empty_response_retry"] == "text_only_authorized_tools"
    assert "tool_only_fallback" not in result.diagnostics
    assert len(providers.requests) == 2
    assert providers.requests[0][1].max_tokens is None
    assert providers.requests[0][1].reasoning_effort is None
    assert providers.requests[1][1].tools is None
    assert providers.requests[1][1].max_tokens is None
    assert providers.requests[1][1].reasoning_effort is None


@pytest.mark.asyncio
async def test_native_provider_can_return_dialogue_and_tool_in_one_turn(tmp_path) -> None:
    providers = CombinedTextAndToolProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    result = await service.complete(
        _conversation_request(
            "combined-tool-turn",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "I admire you.",
            available_tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "social_react",
                        "description": "React socially.",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        )
    )

    assert result.completion.text == "You have my respect."
    assert result.completion.tool_calls[0]["function"]["name"] == "social_react"
    assert result.diagnostics["initial_response_shape"] == "text_and_tool"
    assert result.diagnostics["response_shape"] == "text_and_tool"
    assert len(providers.requests) == 1


@pytest.mark.asyncio
async def test_explicit_social_uses_dialogue_only_and_semantic_action(
    tmp_path,
) -> None:
    providers = ExplicitSocialGenericThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    result = await service.complete(
        _conversation_request(
            "repair-sexual-social",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "I want to have sex with you.",
            available_tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "social_react",
                        "description": "React socially.",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        )
    )

    assert result.completion.text == "No. Back off."
    assert result.completion.tool_calls[0]["name"] == "social_react"
    assert result.diagnostics["semantic_action_lane"] == "social_deterministic"
    assert "contextual_response_retry" not in result.diagnostics
    assert len(providers.requests) == 1
    assert providers.requests[0][1].tools is None


@pytest.mark.asyncio
async def test_unrecognized_tool_candidate_retries_as_dialogue(tmp_path) -> None:
    providers = UnknownToolThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    request = _conversation_request(
        "retry-tool-one",
        "world-one",
        "player-one",
        "npc-one",
        "session-one",
        "What should we do?",
        available_tools=(
            {
                "type": "function",
                "function": {
                    "name": "social_react",
                    "description": "React socially.",
                    "parameters": {"type": "object"},
                },
            },
        ),
    )

    result = await service.complete(request)

    assert result.completion.text == "I can help with that."
    assert result.diagnostics["empty_response_retry"] == "text_only_unrecognized_tools"
    assert result.diagnostics["provider_tool_call_count"] == 1
    assert result.diagnostics["provider_authorized_tool_call_count"] == 0
    assert providers.requests[1][1].tools is None


@pytest.mark.asyncio
async def test_consolidation_runs_when_failed_turn_crosses_boundary(tmp_path) -> None:
    providers = TextEmptyTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        memory_consolidation_turns=4,
    )
    service = ConversationService(settings, providers)
    identity = _memory_identity("world-one")

    first = await service.complete(
        _conversation_request(
            "boundary-one",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "We should stay together.",
        )
    )
    failed = await service.complete(
        _conversation_request(
            "boundary-two",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "Are you still there?",
        )
    )
    recovered = await service.complete(
        _conversation_request(
            "boundary-three",
            "world-one",
            "player-one",
            "npc-one",
            "session-one",
            "What was our plan?",
        )
    )

    assert first.completion.text == "The plan still stands."
    assert failed.completion.text == "The plan still stands."
    assert failed.diagnostics["empty_response_retry"] == "text_only"
    assert failed.diagnostics["consolidated"] is True
    assert recovered.diagnostics["consolidated"] is False
    assert service._store(identity).stats()["memory_count"] >= 1


def test_conversation_sync_owns_canonical_message_identity(tmp_path) -> None:
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, FakeProviders())
    message = {
        "version": 1,
        "messageID": "conversation-one:2",
        "saveUUID": "world-one",
        "worldMode": "multiplayer",
        "serverInstanceId": "test-server",
        "serverWorldGeneration": "world-one",
        "conversationID": "conversation-one",
        "playerUUID": "player-one",
        "npcUUID": "npc-one",
        "speakerID": "npc-one",
        "speakerName": "Harley",
        "speakerKind": "npc",
        "text": "The shelter is north.",
        "gameDay": 3,
        "worldAgeHours": 73.0,
        "source": {"kind": "conversation", "channel": "choice"},
    }

    first = service.record_message(message)
    second = service.record_message(message)

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.turn.message_id == "conversation-one:2"
    assert second.turn.game_day == 3
    identity = _memory_identity("world-one")
    assert service._store(identity).stats()["turn_count"] == 1


def test_conversation_sync_excludes_provider_failures_but_keeps_tool_ack(tmp_path) -> None:
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, FakeProviders())
    failure = {
        "messageID": "conversation-one:failure",
        "saveUUID": "world-one",
        "worldMode": "multiplayer",
        "serverInstanceId": "test-server",
        "serverWorldGeneration": "world-one",
        "conversationID": "conversation-one",
        "playerUUID": "player-one",
        "npcUUID": "npc-one",
        "speakerID": "npc-one",
        "speakerName": "Harley",
        "speakerKind": "npc",
        "text": "I cannot answer right now. (OpenAI-compatible provider request failed.)",
        "source": {
            "kind": "llm",
            "channel": "response",
            "providerFailure": True,
            "contextEligible": False,
        },
    }
    tool_ack = {
        **failure,
        "messageID": "conversation-one:tool-ack",
        "text": "I will check that now.",
        "source": {
            "kind": "llm",
            "channel": "response",
            "providerFailure": False,
            "contextEligible": True,
        },
    }

    skipped = service.record_message(failure)
    recorded = service.record_message(tool_ack)
    acknowledged = service.record_message_batch({"messages": [failure, tool_ack]})

    assert skipped.skipped is True
    assert skipped.turn.message_id == failure["messageID"]
    assert recorded.skipped is False
    assert acknowledged == (failure["messageID"], tool_ack["messageID"])
    identity = _memory_identity("world-one")
    turns = service._store(identity).recent_turns(
        "conversation-one",
        MemoryScope(identity.world_uuid, "player-one", "npc-one"),
    )
    assert [turn.content for turn in turns] == [tool_ack["text"]]


@pytest.mark.asyncio
async def test_conversation_recall_finds_dated_turns_from_a_previous_session(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        context_max_chars=8000,
    )
    service = ConversationService(settings, providers)
    service.record_message(
        {
            "messageID": "yesterday:1",
            "saveUUID": "world-one",
            "worldMode": "multiplayer",
            "serverInstanceId": "test-server",
            "serverWorldGeneration": "world-one",
            "conversationID": "yesterday",
            "playerUUID": "player-one",
            "npcUUID": "npc-one",
            "speakerID": "npc-one",
            "speakerName": "Harley",
            "speakerKind": "npc",
            "text": "I hid the medicine in the northern shed.",
            "gameDay": 12,
            "worldAgeHours": 289.0,
        }
    )

    result = await service.complete(
        _conversation_request(
            "today-request",
            "world-one",
            "player-one",
            "npc-one",
            "today",
            "Do you remember the medicine?",
        )
    )

    system = "\n".join(message.content or "" for message in providers.requests[0][1].messages)
    assert result.diagnostics["recalled_turn_count"] == 1
    assert "Relevant Conversation Recall" in system
    assert "game day 12" in system


@pytest.mark.asyncio
async def test_conversation_builds_dated_layers_and_skips_rag_for_greeting(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        memory_consolidation_turns=2,
        context_max_chars=8000,
    )
    service = ConversationService(settings, providers)
    identity = _memory_identity("world-one")
    scope = MemoryScope(identity.world_uuid, "player-one", "npc-alice")
    participants = (
        {"id": "player-one", "name": "Alex", "kind": "player"},
        {"id": "npc-alice", "name": "Alice", "kind": "npc"},
        {"id": "npc-bob", "name": "Bob", "kind": "npc"},
    )

    first = await service.complete(
        _conversation_request(
            "layer-one",
            "world-one",
            "player-one",
            "npc-alice",
            "scene-one",
            "Hello.",
            game_day=6,
            world_age_hours=145.0,
            participants=participants,
            scene={"active_participants": ["player-one", "npc-alice", "npc-bob"]},
        )
    )
    assert first.diagnostics["retrieval_skipped"] is True

    second = await service.complete(
        _conversation_request(
            "layer-two",
            "world-one",
            "player-one",
            "npc-alice",
            "scene-one",
            "I will check the Riverside shed tomorrow.",
            game_day=6,
            world_age_hours=146.0,
            participants=participants,
            scene={"active_participants": ["player-one", "npc-alice", "npc-bob"]},
            current_topic="Riverside",
            end_session=True,
        )
    )

    assert second.diagnostics["consolidated"] is True
    assert second.diagnostics["day_synopsis_available"] is True
    store = service._store(identity)
    assert store.get_day_synopsis(scope, 6) is not None
    assert store.stats()["episode_count"] == 1
    assert store.stats()["fact_count"] >= 1
    system = "\n".join(message.content or "" for message in providers.requests[-1][1].messages)
    assert "Conversation Scene" in system
    assert "Today So Far" in system
