from types import SimpleNamespace

import pytest

from pbrainz.api.control_routes import (
    mock_chat,
    seed_mock_memories,
    ui_delete_memory,
    ui_memory,
)
from pbrainz.api.models import MemoryDeleteRequest, MockChatRequest, MockChatSeedRequest
from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationService
from pbrainz.providers.base import CompletionResult


class FakeProviders:
    def resolve(self, provider, model):
        return provider or "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "I remember the Riverside shelter.")


def _request_context(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        memory_root=str(tmp_path / "memory"),
        enabled_providers="custom",
        custom_base_url="http://mock-provider",
        custom_models="fake-model",
    )
    service = ConversationService(settings, FakeProviders())
    app = SimpleNamespace(
        state=SimpleNamespace(settings=settings, conversation_service=service)
    )
    return SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_mock_chat_uses_seeded_memory_pipeline_and_browser(tmp_path) -> None:
    request = _request_context(tmp_path)
    seed = await seed_mock_memories(
        request,
        MockChatSeedRequest(
            world_uuid="mock-world",
            player_uuid="mock-player",
            npc_uuid="mock-npc",
            game_day=4,
        ),
    )
    assert seed["seeded"] is True

    result = await mock_chat(
        request,
        MockChatRequest(
            provider="custom",
            model="fake-model",
            message="What do you remember about Riverside?",
            world_uuid="mock-world",
            player_uuid="mock-player",
            npc_uuid="mock-npc",
            session_id="mock-test-session",
            game_day=4,
        ),
    )

    assert result["response_text"] == "I remember the Riverside shelter."
    assert result["retrieved_memories"]
    assert result["diagnostics"]["retrieval_needed"] is True
    assert result["diagnostics"]["world_uuid"] == "mock-world"

    browser = await ui_memory(request, world_uuid="mock-world", search="Riverside")
    assert browser["total"] >= 4
    assert {item["record_kind"] for item in browser["items"]} >= {
        "memory",
        "episode",
        "fact",
        "day_synopsis",
    }

    deleted = await ui_delete_memory(
        request,
        MemoryDeleteRequest(
            world_uuid="mock-world",
            record_kind="memory",
            record_id="mock-memory-riverside",
            player_uuid="mock-player",
            npc_uuid="mock-npc",
        ),
    )
    assert deleted["deleted"] is True
    remaining = await ui_memory(request, world_uuid="mock-world", search="Riverside")
    assert all(
        item["record_id"] != "mock-memory-riverside" for item in remaining["items"]
    )
