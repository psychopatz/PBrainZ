import asyncio
import json
from pathlib import Path

import pytest

from pbrainz.bridge import BridgeRequest, BridgeRuntimeMonitor, BridgeState
from pbrainz.bridge.handler import complete_and_deliver, semantic_tool_calls_for
from pbrainz.bridge.pump import run_bridge_pump
from pbrainz.bridge.transport import FileBridgeTransport
from pbrainz.config import Settings
from pbrainz.conversation_runtime import Utterance
from pbrainz.memory import MemoryScope, SQLiteMemoryStore
from pbrainz.providers.base import CompletionResult


def test_file_transport_matches_psychopatzcore_slot_protocol(tmp_path) -> None:
    transport = FileBridgeTransport(tmp_path)
    request = BridgeRequest.create(
        "projecthoomans.llm",
        "pollChat",
        {},
        "runtime-123",
    )

    slot = transport.write_request(request)
    request_path = tmp_path / "requests" / f"slot-{slot:02d}.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload["message_type"] == "request"
    assert payload["protocol_version"] == 1
    assert payload["target_runtime_id"] == "runtime-123"

    response = {
        "message_type": "response",
        "protocol_version": 1,
        "request_id": request.request_id,
        "runtime_id": "runtime-123",
        "status": "ok",
        "request_state": "COMPLETE",
        "result": {"status": "idle"},
        "error": None,
    }
    response_dir = tmp_path / "responses"
    (response_dir / f"slot-{slot:02d}.json").write_text(
        json.dumps(response),
        encoding="utf-8",
    )
    (response_dir / f"slot-{slot:02d}.ready.txt").write_text(
        request.request_id,
        encoding="utf-8",
    )

    received = transport.read_response(slot, request.request_id)
    assert received is not None
    assert received.result == {"status": "idle"}
    transport.release(slot)
    assert not request_path.exists()


def test_file_transport_does_not_overwrite_busy_slots(tmp_path) -> None:
    transport = FileBridgeTransport(tmp_path)
    requests = [
        BridgeRequest.create("projecthoomans.llm", "pollChat", {}, "runtime-123")
        for _ in range(16)
    ]
    slots = [transport.write_request(request) for request in requests]

    try:
        try:
            transport.write_request(
                BridgeRequest.create(
                    "projecthoomans.llm",
                    "pollChat",
                    {},
                    "runtime-123",
                )
            )
        except RuntimeError as error:
            assert "slots" in str(error)
        else:
            raise AssertionError("expected all slots to be busy")
    finally:
        for slot in slots:
            transport.release(slot)


def test_file_transport_recovers_only_slots_from_previous_runtime(tmp_path) -> None:
    transport = FileBridgeTransport(tmp_path)
    stale = BridgeRequest.create(
        "projecthoomans.llm", "pollChat", {}, "old-runtime"
    )
    current = BridgeRequest.create(
        "projecthoomans.llm", "pollChat", {}, "current-runtime"
    )
    stale_slot = transport.write_request(stale)
    current_slot = transport.write_request(current)
    stale_response = {
        "message_type": "response",
        "protocol_version": 1,
        "request_id": stale.request_id,
        "runtime_id": "old-runtime",
        "status": "ok",
        "result": {},
    }
    response_path = tmp_path / "responses" / f"slot-{stale_slot:02d}.json"
    response_path.write_text(json.dumps(stale_response), encoding="utf-8")

    assert transport.recover_stale_slots("current-runtime") == 1
    assert not (tmp_path / "requests" / f"slot-{stale_slot:02d}.json").exists()
    assert (tmp_path / "requests" / f"slot-{current_slot:02d}.json").exists()
    transport.release(current_slot)


def test_semantic_tool_calls_are_limited_to_tools_exposed_by_the_game() -> None:
    request = {
        "conversation_context": {
            "available_tools": [
                {"type": "function", "function": {"name": "order_follow"}},
            ]
        }
    }
    calls = [
        {
            "id": "call-1",
            "function": {
                "name": "order_follow",
                "arguments": '{"command_id":"follow"}',
            },
        },
        {
            "id": "call-2",
            "function": {"name": "order_unknown", "arguments": "{}"},
        },
    ]

    assert semantic_tool_calls_for(calls, request) == [
        {
            "id": "call-1",
            "name": "order_follow",
            "arguments": {"command_id": "follow"},
        }
    ]


class StructuredProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        self.requests.append((_provider, request))
        return CompletionResult(
            request.model,
            "",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call-follow",
                    "function": {
                        "name": "order_follow",
                        "arguments": '{"command_id":"follow"}',
                    },
                }
            ],
        )


class EmptyResponseProviders:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "", finish_reason="stop")


class DeliveryClient:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, namespace, command, arguments, runtime_id):
        self.calls.append((namespace, command, arguments, runtime_id))
        return {"accepted": True}


@pytest.mark.asyncio
async def test_structured_bridge_delivers_authorized_semantic_tool_calls(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    providers = StructuredProviders()
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    service = ConversationService(settings, providers)
    client = DeliveryClient()
    request = {
        "request_id": "pnc-structured-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "Follow me.",
            "available_tools": [
                {"type": "function", "function": {"name": "order_follow"}},
            ],
        },
    }

    await complete_and_deliver(
        providers,
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    assert client.calls[0][1] == "deliverChat"
    assert client.calls[0][2]["semantic_tool_calls"][0]["name"] == "order_follow"


@pytest.mark.asyncio
async def test_tool_only_turn_uses_shared_ack_for_delivery_and_tts(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    service = ConversationService(settings, StructuredProviders())
    client = DeliveryClient()
    tts = FakeTTSService()
    request = {
        "request_id": "pnc-tool-tts-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "Follow me.",
            "tool_ack_text": "I will check that now.",
            "voice_binding": {
                "npc_uuid": "npc-one",
                "slot": "VoiceFemale:2",
                "pitch": 7,
            },
            "available_tools": [
                {"type": "function", "function": {"name": "order_follow"}},
            ],
        },
    }

    await complete_and_deliver(
        StructuredProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
        tts_service=tts,
    )

    delivery = client.calls[0][2]
    assert delivery["response_text"] == "I will check that now."
    assert delivery["presentation_reason"] == "tool_ack"
    assert delivery["tool_result_pending"] is True
    assert delivery["presentation_mode"] == "tts"
    assert len(tts.enqueued) == 1
    assert tts.enqueued[0].text == delivery["response_text"]


@pytest.mark.asyncio
async def test_empty_provider_response_is_explained_and_not_saved_as_memory(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    service = ConversationService(settings, EmptyResponseProviders())
    client = DeliveryClient()
    request = {
        "request_id": "pnc-empty-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "Why did you leave?",
        },
    }

    await complete_and_deliver(
        EmptyResponseProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    delivery = client.calls[0][2]
    assert delivery["response_text"] == ""
    assert delivery["finish_reason"] == "stop"
    assert "empty response" in delivery["error"]
    assert service._store("world-one").stats()["turn_count"] == 1


class FakeProviders:
    def resolve(self, _provider: str | None, model: str) -> tuple[str, str]:
        return "gemini", model

    async def complete(self, _provider: str, request) -> CompletionResult:
        return CompletionResult(model=request.model, text="Stay close to the shelter.")


class FakeTTSService:
    def __init__(self) -> None:
        self.enabled = True
        self.last_error = None
        self.enqueued: list[Utterance] = []

    def resolve_voice_binding(self, _conversation_id, _npc_uuid, binding):
        return binding

    def can_synthesize(self, binding) -> bool:
        return binding is not None

    async def enqueue(self, utterance, *, on_started=None, on_finished=None, on_failed=None):
        self.enqueued.append(utterance)
        if on_started:
            await on_started(utterance)
        return True


@pytest.mark.asyncio
async def test_tts_bridge_sends_only_compact_start_event_and_keeps_text_payload() -> None:
    client = DeliveryClient()
    tts = FakeTTSService()
    request = {
        "request_id": "pnc-tts-1",
        "npc_id": "npc-one",
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Hello."}],
        "context": {
            "conversation_id": "conversation-one",
            "voice_binding": {
                "npc_uuid": "npc-one",
                "slot": "VoiceFemale:2",
                "pitch": 7,
            },
        },
    }

    await complete_and_deliver(
        FakeProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        tts_service=tts,
    )

    delivery = client.calls[0][2]
    speech_start = client.calls[1][2]
    assert delivery["presentation_mode"] == "tts"
    assert delivery["response_text"] == "Stay close to the shelter."
    assert speech_start == {
        "request_id": "pnc-tts-1",
        "conversation_id": "conversation-one",
        "utterance_id": "conversation-one:pnc-tts-1",
        "npc_uuid": "npc-one",
        "text": "Stay close to the shelter.",
        "duration_ms": 0,
    }
    assert "audio" not in speech_start
    assert "model_path" not in speech_start


async def _fake_game(root: Path, delivered: asyncio.Event) -> None:
    requests = root / "requests"
    responses = root / "responses"
    handled: set[str] = set()
    while not delivered.is_set():
        for request_path in requests.glob("slot-*.json"):
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request_id = request["request_id"]
            if request_id in handled:
                continue
            handled.add(request_id)
            command = request["command"]
            if command == "pollChat":
                result = {
                    "status": "pending",
                    "request_id": "pnc_llm_1",
                    "npc_id": "npc-1",
                    "messages": [
                        {"role": "system", "content": "You are an NPC."},
                        {"role": "user", "content": "Where should we go?"},
                    ],
                }
            elif command == "pollConversationSync":
                result = {"status": "idle", "messages": [], "pendingCount": 0}
            elif command == "ackConversationSync":
                result = {"acknowledged": 0, "pendingCount": 0}
            else:
                result = {"accepted": True}
                assert request["arguments"]["response_text"] == (
                    "Stay close to the shelter."
                )
                delivered.set()
            slot = request_path.stem.rsplit("-", 1)[-1]
            response = {
                "message_type": "response",
                "protocol_version": 1,
                "request_id": request_id,
                "runtime_id": "runtime-123",
                "status": "ok",
                "request_state": "COMPLETE",
                "result": result,
                "error": None,
            }
            (responses / f"slot-{slot}.json").write_text(
                json.dumps(response),
                encoding="utf-8",
            )
            (responses / f"slot-{slot}.ready.txt").write_text(
                request_id,
                encoding="utf-8",
            )
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_bridge_pump_completes_and_delivers_an_npc_message(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    runtime = {
        "protocol_version": 1,
        "runtime_id": "runtime-123",
        "enabled": True,
        "lifecycle": "READY",
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-123", encoding="utf-8")
    delivered = asyncio.Event()
    trace_events: list[dict[str, object]] = []
    settings = Settings(
        bridge_required=True,
        bridge_root=str(tmp_path),
        bridge_poll_interval=0.06,
        request_timeout=1,
        llm_trace_capture=True,
    )
    pump = asyncio.create_task(
        run_bridge_pump(
            settings,
            FakeProviders(),
            BridgeRuntimeMonitor(tmp_path),
            trace_writer=lambda **event: trace_events.append(event),
        )
    )
    game = asyncio.create_task(_fake_game(tmp_path, delivered))
    try:
        await asyncio.wait_for(delivered.wait(), timeout=2)
    finally:
        pump.cancel()
        game.cancel()
        await asyncio.gather(pump, game, return_exceptions=True)
    assert [event["phase"] for event in trace_events] == [
        "bridge.request",
        "bridge.delivery",
    ]


async def _fake_sync_game(root: Path, acknowledged: asyncio.Event) -> None:
    requests = root / "requests"
    responses = root / "responses"
    handled: set[str] = set()
    while not acknowledged.is_set():
        for request_path in requests.glob("slot-*.json"):
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request_id = request["request_id"]
            if request_id in handled:
                continue
            handled.add(request_id)
            command = request["command"]
            if command == "pollConversationSync":
                result = {
                    "status": "pending",
                    "messages": [
                        {
                            "messageID": "conversation-sync:1",
                            "saveUUID": "world-sync",
                            "conversationID": "conversation-sync",
                            "playerUUID": "player-one",
                            "npcUUID": "npc-one",
                            "speakerID": "npc-one",
                            "speakerName": "Harley",
                            "speakerKind": "npc",
                            "text": "The shelter is north.",
                            "gameDay": 5,
                            "worldAgeHours": 121.0,
                        }
                    ],
                    "pendingCount": 1,
                }
            elif command == "ackConversationSync":
                assert request["arguments"]["message_ids"] == ["conversation-sync:1"]
                result = {"acknowledged": 1, "pendingCount": 0}
                acknowledged.set()
            else:
                result = {"status": "idle"}
            slot = request_path.stem.rsplit("-", 1)[-1]
            response = {
                "message_type": "response",
                "protocol_version": 1,
                "request_id": request_id,
                "runtime_id": "runtime-sync",
                "status": "ok",
                "request_state": "COMPLETE",
                "result": result,
                "error": None,
            }
            (responses / f"slot-{slot}.json").write_text(
                json.dumps(response),
                encoding="utf-8",
            )
            (responses / f"slot-{slot}.ready.txt").write_text(
                request_id,
                encoding="utf-8",
            )
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_bridge_pump_ingests_and_acknowledges_canonical_messages(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    runtime = {
        "protocol_version": 1,
        "runtime_id": "runtime-sync",
        "enabled": True,
        "lifecycle": "READY",
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-sync", encoding="utf-8")
    acknowledged = asyncio.Event()
    settings = Settings(
        bridge_required=True,
        bridge_root=str(tmp_path),
        database_path=str(tmp_path / "settings.db"),
        bridge_poll_interval=0.06,
        request_timeout=1,
    )
    pump = asyncio.create_task(
        run_bridge_pump(settings, FakeProviders(), BridgeRuntimeMonitor(tmp_path))
    )
    game = asyncio.create_task(_fake_sync_game(tmp_path, acknowledged))
    try:
        await asyncio.wait_for(acknowledged.wait(), timeout=2)
    finally:
        pump.cancel()
        game.cancel()
        await asyncio.gather(pump, game, return_exceptions=True)

    store = SQLiteMemoryStore(tmp_path / "memory", "world-sync")
    turns = store.recent_turns(
        "conversation-sync",
        MemoryScope("world-sync", "player-one", "npc-one"),
    )
    assert len(turns) == 1
    assert turns[0].message_id == "conversation-sync:1"
    assert turns[0].game_day == 5
