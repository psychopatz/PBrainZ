import pytest

from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationRequest, ConversationService
from pbrainz.memory import MemoryScope
from pbrainz.providers.base import CompletionResult


class FakeProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, provider, request):
        self.requests.append((provider, request))
        return CompletionResult(request.model, "I remember that.")


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


@pytest.mark.asyncio
async def test_structured_conversation_owns_history_and_consolidates(tmp_path) -> None:
    providers = FakeProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        memory_consolidation_turns=2,
        context_max_chars=4000,
    )
    service = ConversationService(settings, providers)
    scope = MemoryScope("world-one", "player-one", "npc-one")
    request = ConversationRequest(
        request_id="request-one",
        scope=scope,
        session_id="session-one",
        message="My name is Alex.",
        npc_name="Harley",
        player_name="Alex",
    )

    first = await service.complete(request)
    second = await service.complete(
        ConversationRequest(
            request_id="request-two",
            scope=scope,
            session_id="session-one",
            message="What do you remember about me?",
        )
    )

    assert first.completion.text == "I remember that."
    assert len(providers.requests) == 2
    messages = providers.requests[1][1].messages
    assert any(message.content == "I remember that." for message in messages)
    assert second.diagnostics["consolidated"] is True
    assert second.diagnostics["memory_count"] >= 1


def test_structured_request_rejects_missing_scope() -> None:
    with pytest.raises(ValueError):
        ConversationRequest.from_mapping(
            {"request_id": "one", "conversation_context": {"message": "hello"}}
        )


@pytest.mark.asyncio
async def test_empty_provider_candidate_retries_without_tools(tmp_path) -> None:
    providers = EmptyThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    request = ConversationRequest(
        request_id="retry-one",
        scope=MemoryScope("world-one", "player-one", "npc-one"),
        session_id="session-one",
        message="Where are you?",
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
async def test_unrecognized_tool_candidate_retries_as_dialogue(tmp_path) -> None:
    providers = UnknownToolThenTextProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    request = ConversationRequest(
        request_id="retry-tool-one",
        scope=MemoryScope("world-one", "player-one", "npc-one"),
        session_id="session-one",
        message="What should we do?",
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
    scope = MemoryScope("world-one", "player-one", "npc-one")

    first = await service.complete(
        ConversationRequest(
            request_id="boundary-one",
            scope=scope,
            session_id="session-one",
            message="We should stay together.",
        )
    )
    failed = await service.complete(
        ConversationRequest(
            request_id="boundary-two",
            scope=scope,
            session_id="session-one",
            message="Are you still there?",
        )
    )
    recovered = await service.complete(
        ConversationRequest(
            request_id="boundary-three",
            scope=scope,
            session_id="session-one",
            message="What was our plan?",
        )
    )

    assert first.completion.text == "The plan still stands."
    assert failed.diagnostics["empty_response"] is True
    assert recovered.diagnostics["consolidated"] is True
    assert service._store("world-one").stats()["memory_count"] >= 1


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
    assert service._store("world-one").stats()["turn_count"] == 1


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
        ConversationRequest(
            request_id="today-request",
            scope=MemoryScope("world-one", "player-one", "npc-one"),
            session_id="today",
            message="Do you remember the medicine?",
        )
    )

    system = providers.requests[0][1].messages[0].content
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
    scope = MemoryScope("world-one", "player-one", "npc-alice")
    participants = (
        {"id": "player-one", "name": "Alex", "kind": "player"},
        {"id": "npc-alice", "name": "Alice", "kind": "npc"},
        {"id": "npc-bob", "name": "Bob", "kind": "npc"},
    )

    first = await service.complete(
        ConversationRequest(
            request_id="layer-one",
            scope=scope,
            session_id="scene-one",
            message="Hello.",
            game_day=6,
            world_age_hours=145.0,
            participants=participants,
            scene={"active_participants": ["player-one", "npc-alice", "npc-bob"]},
        )
    )
    assert first.diagnostics["retrieval_skipped"] is True

    second = await service.complete(
        ConversationRequest(
            request_id="layer-two",
            scope=scope,
            session_id="scene-one",
            message="I will check the Riverside shed tomorrow.",
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
    store = service._store("world-one")
    assert store.get_day_synopsis(scope, 6) is not None
    assert store.stats()["episode_count"] == 1
    assert store.stats()["fact_count"] >= 1
    system = providers.requests[-1][1].messages[0].content
    assert "Conversation Scene" in system
    assert "Today So Far" in system
