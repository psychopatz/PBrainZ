from types import SimpleNamespace

import pytest

from pbrainz.api.control_routes import (
    add_template_model,
    mock_chat,
    seed_mock_memories,
    ui_delete_memory,
    ui_memory,
    ui_memory_active,
)
from pbrainz.api.models import (
    MemoryDeleteRequest,
    MockChatRequest,
    MockChatSeedRequest,
    UITemplateModelAddRequest,
)
from pbrainz.bridge.memory_context import ActiveMemoryContextCache
from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationService
from pbrainz.database import SettingsDatabase
from pbrainz.providers.base import CompletionResult


class FakeProviders:
    def resolve(self, provider, model):
        return provider or "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "I remember the Riverside shelter.")


class FakeRegistry:
    provider_names = ("custom",)


def _template_model_request(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://mock-provider",
    )
    database = SettingsDatabase(settings.database_path)
    database.initialize()
    app = SimpleNamespace(
        title="PBrainZ",
        state=SimpleNamespace(
            settings=settings,
            providers=FakeRegistry(),
            database=database,
            bridge_controller=SimpleNamespace(
                as_dict=lambda: {
                    "bridge": {
                        "available": False,
                        "enabled": False,
                        "ready": False,
                        "message": "test",
                    },
                    "worker_enabled": False,
                    "worker_running": False,
                }
            ),
            game_bridge_settings=SimpleNamespace(
                read=lambda: SimpleNamespace(enabled=False)
            ),
        ),
    )
    return SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_template_model_can_add_and_persist_a_provider_model(tmp_path) -> None:
    request = _template_model_request(tmp_path)

    result = await add_template_model(
        request,
        UITemplateModelAddRequest(provider="custom", model="npc-template-v2"),
    )

    assert result.providers[0].configured_models == ["npc-template-v2"]
    assert request.app.state.settings.custom_models == "npc-template-v2"
    assert request.app.state.database.load_settings()["custom_models"] == "npc-template-v2"


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
            world_mode="multiplayer",
            server_instance_id="pbrainz-mock-server",
            server_world_generation="mock-world",
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
            world_mode="multiplayer",
            server_instance_id="pbrainz-mock-server",
            server_world_generation="mock-world",
            player_uuid="mock-player",
            npc_uuid="mock-npc",
            session_id="mock-test-session",
            game_day=4,
        ),
    )

    assert result["response_text"] == "I remember the Riverside shelter."
    assert result["retrieved_memories"]
    assert result["diagnostics"]["retrieval_needed"] is True
    mock_world = "mp-v1|pbrainz-mock-server|mock-world"
    assert result["diagnostics"]["world_uuid"] == mock_world

    browser = await ui_memory(request, world_uuid=mock_world, search="Riverside")
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
            world_uuid=mock_world,
            record_kind="memory",
            record_id="mock-memory-riverside",
            player_uuid="mock-player",
            npc_uuid="mock-npc",
        ),
    )
    assert deleted["deleted"] is True
    remaining = await ui_memory(request, world_uuid=mock_world, search="Riverside")
    assert all(
        item["record_id"] != "mock-memory-riverside" for item in remaining["items"]
    )


@pytest.mark.asyncio
async def test_memory_browser_defaults_to_the_fresh_active_save(tmp_path) -> None:
    request = _request_context(tmp_path)
    cache = ActiveMemoryContextCache()
    assert cache.update(
        {
            "world_mode": "multiplayer",
            "server_instance_id": "server-one",
            "server_world_generation": "wipe-one",
            "player_uuid": "player-one",
        },
        "runtime-one",
    )
    request.app.state.active_memory_context = cache

    active = await ui_memory_active(request)
    browser = await ui_memory(request)

    assert active["active"]["world_uuid"] == (
        "mp-v1|server-one|wipe-one"
    )
    assert active["active"]["exists"] is False
    assert browser["world_uuid"] == "mp-v1|server-one|wipe-one"
    assert browser["items"] == []
