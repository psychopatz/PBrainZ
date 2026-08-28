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
