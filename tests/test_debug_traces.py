from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationRequest, ConversationService
from pbrainz.database import SettingsDatabase
from pbrainz.memory import MemoryIdentity, MemoryScope
from pbrainz.providers.base import CompletionResult


def test_llm_trace_store_is_bounded_and_clearable(tmp_path) -> None:
    database = SettingsDatabase(tmp_path / "pbrainz.db")
    database.initialize()

    for index in range(305):
        database.add_llm_trace(
            source="test",
            phase="provider.response",
            request_id=f"request-{index}",
            payload={"index": index, "prompt": "hello"},
        )

    traces = database.recent_llm_traces(500)
    assert len(traces) == 300
    assert traces[0]["request_id"] == "request-5"
    assert traces[-1]["payload"]["index"] == 304
    assert database.recent_llm_traces(20, search="request-30")
    assert database.clear_llm_traces() == 300
    assert database.recent_llm_traces(10) == []


def test_llm_trace_capture_setting_is_persisted(tmp_path) -> None:
    database = SettingsDatabase(tmp_path / "pbrainz.db")
    database.initialize()
    database.save_settings({"llm_trace_capture": True})
    assert database.load_settings()["llm_trace_capture"] is True


class _Providers:
    def resolve(self, provider, model):
        return provider or "custom", model if model not in {"default", "auto"} else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "Hello.", reasoning="public reasoning")


def _identity(world_uuid: str) -> MemoryIdentity:
    return MemoryIdentity.from_mapping(
        world_uuid,
        {
            "world_mode": "multiplayer",
            "server_instance_id": "trace-server",
            "server_world_generation": world_uuid,
        },
    )


def test_conversation_trace_capture_has_input_prompt_and_response(tmp_path) -> None:
    events: list[dict] = []

    def writer(**event):
        events.append(event)

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        llm_trace_capture=True,
    )
    service = ConversationService(settings, _Providers(), trace_writer=writer)
    identity = _identity("world")

    import asyncio

    asyncio.run(
        service.complete(
            ConversationRequest(
                request_id="trace-request",
                scope=MemoryScope(identity.world_uuid, "player", "npc"),
                session_id="session",
                message="What do you see?",
                memory_identity=identity,
                scene={"weather": "rain"},
            )
        )
    )

    phases = [event["phase"] for event in events]
    assert phases == ["conversation.input", "provider.request", "provider.response"]
    provider_request = events[1]["payload"]
    assert provider_request["messages"][0]["role"] == "system"
    assert "What do you see?" in provider_request["messages"][-1]["content"]
    assert events[2]["payload"]["reasoning"] == "public reasoning"


def test_conversation_trace_capture_disabled_does_not_call_writer(tmp_path) -> None:
    events: list[dict] = []
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        bridge_required=False,
        llm_trace_capture=False,
    )
    service = ConversationService(
        settings,
        _Providers(),
        trace_writer=lambda **event: events.append(event),
    )
    identity = _identity("world")

    import asyncio

    asyncio.run(
        service.complete(
            ConversationRequest(
                request_id="no-trace",
                scope=MemoryScope(identity.world_uuid, "player", "npc"),
                session_id="session",
                message="Hello",
                memory_identity=identity,
            )
        )
    )
    assert events == []
