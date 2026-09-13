import asyncio
import json
from pathlib import Path

import pytest

from pbrainz.bridge import BridgeRequest, BridgeRuntimeMonitor, BridgeState
from pbrainz.bridge.handler import (
    add_fallback_social_tool_call,
    complete_and_deliver,
    sanitize_npc_response,
    semantic_tool_calls_for,
)
from pbrainz.bridge.pump import _CycleFailureReporter, run_bridge_pump
from pbrainz.bridge.transport import FileBridgeTransport
from pbrainz.config import Settings
from pbrainz.conversation_runtime import Utterance
from pbrainz.exceptions import ProviderError
from pbrainz.memory import MemoryIdentity, MemoryScope, SQLiteMemoryStore
from pbrainz.providers.base import CompletionResult
from pbrainz.semantic_tool_protocol import (
    ensure_identity_intent,
    ensure_social_intent,
    extract_text_tool_calls,
    infer_social_intent,
    is_provider_scaffold,
    social_reply_repair_needed,
    strip_provider_scaffold,
)


def _test_memory_identity(world_uuid: str) -> MemoryIdentity:
    return MemoryIdentity.from_mapping(
        world_uuid,
        {
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": world_uuid,
        },
    )


def test_repeating_bridge_cycle_failure_is_rate_limited() -> None:
    reporter = _CycleFailureReporter(repeat_interval=300)
    error = RuntimeError("bridge command 'pollChat' timed out")

    assert reporter.message("runtime-1", error, now=100.0) == str(error)
    assert reporter.message("runtime-1", error, now=120.0) is None
    assert reporter.message("runtime-1", error, now=399.0) is None
    assert "repeated 3 times" in reporter.message("runtime-1", error, now=400.0)
    assert reporter.recovered() == 0


def test_bridge_cycle_failure_reporter_distinguishes_runtime_or_error_changes() -> None:
    reporter = _CycleFailureReporter(repeat_interval=300)
    first = RuntimeError("pollChat timed out")
    second = RuntimeError("bridge response was malformed")

    assert reporter.message("runtime-1", first, now=10.0) == str(first)
    assert reporter.message("runtime-1", first, now=11.0) is None
    changed = reporter.message("runtime-2", second, now=12.0)
    assert changed == str(second) + "; suppressed 1 identical repeats"


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


def test_catalog_tool_ids_authorize_semantic_calls_without_full_schemas() -> None:
    request = {
        "conversation_context": {
            "available_tool_ids": ["projecthoomans.llm:social_react"],
        }
    }
    calls = [
        {
            "id": "call-insult",
            "function": {
                "name": "social_react",
                "arguments": '{"kind":"insult"}',
            },
        }
    ]

    assert semantic_tool_calls_for(calls, request)[0]["name"] == "social_react"


def test_horde_style_text_turn_adds_bounded_insult_tool_call() -> None:
    request = {
        "request_id": "request-insult",
        "npc_id": "npc-one",
        "conversation_context": {
            "message": "You are an idiot. Shut up.",
            "available_tool_ids": ["projecthoomans.llm:social_react"],
        },
    }

    result = add_fallback_social_tool_call([], request)

    assert result == [
        {
            "id": "fallback-insult:request-insult",
            "name": "social_react",
            "arguments": {
                "kind": "insult",
                "intensity": "normal",
                "subtype": "hostile_abuse",
                "explicit": False,
            },
            "origin": "provider_neutral_social_fallback",
        }
    ]


@pytest.mark.parametrize(
    ("message", "kind", "subtype"),
    [
        ("I admire you.", "admire", "compliment"),
        ("You are incredibly attractive.", "flirt", "romantic_interest"),
        ("I'm here for you.", "comfort", None),
        ("I'm sorry about that.", "apologize", None),
        ("Good job out there.", "praise", "compliment"),
    ],
)
def test_provider_text_turn_adds_explicit_positive_social_tool_call(
    message: str,
    kind: str,
    subtype: str | None,
) -> None:
    request = {
        "request_id": f"request-{kind}",
        "npc_id": "npc-one",
        "conversation_context": {
            "message": message,
            "available_tool_ids": ["projecthoomans.llm:social_react"],
        },
    }

    result = add_fallback_social_tool_call([], request)

    assert result[0]["id"] == f"fallback-{kind}:request-{kind}"
    assert result[0]["name"] == "social_react"
    assert result[0]["arguments"]["kind"] == kind
    assert result[0]["arguments"]["intensity"] == "normal"
    if subtype:
        assert result[0]["arguments"]["subtype"] == subtype
    else:
        assert "subtype" not in result[0]["arguments"]


@pytest.mark.parametrize(
    ("message", "reaction", "subtype"),
    [
        ("I want to have sex with you.", "flirt", "sexual_advance"),
        ("wanna fuck babe", "flirt", "sexual_advance"),
        ("Fuck you.", "insult", "hostile_abuse"),
        ("You are beautiful.", "flirt", "romantic_interest"),
    ],
)
def test_social_language_gets_distinct_safe_subtypes(
    message: str,
    reaction: str,
    subtype: str,
) -> None:
    intent = infer_social_intent(message)
    assert intent == {
        "reaction": reaction,
        "subtype": subtype,
        **({"explicit": True} if subtype == "sexual_advance" else {
            "explicit": False,
        } if subtype == "hostile_abuse" else {}),
    }


def test_provider_cannot_route_an_explicit_advance_into_insult_channel() -> None:
    request = {
        "request_id": "request-sexual-advance",
        "npc_id": "npc-one",
        "conversation_context": {
            "message": "wanna fuck babe",
            "available_tool_ids": ["projecthoomans.llm:social_react"],
        },
    }
    calls = [{
        "id": "provider-misclassified",
        "name": "social_react",
        "arguments": {"kind": "insult", "intensity": "normal"},
    }]

    normalized = ensure_social_intent(calls, request)

    assert normalized[0]["arguments"]["kind"] == "flirt"
    assert normalized[0]["arguments"]["subtype"] == "sexual_advance"
    assert normalized[0]["arguments"]["explicit"] is True


def test_explicit_social_reply_repair_only_targets_generic_acknowledgements() -> None:
    assert social_reply_repair_needed(
        "I want to have sex with you.", "I'll check that now."
    ) is True
    assert social_reply_repair_needed(
        "I want to have sex with you.", "No. Back off."
    ) is False
    assert social_reply_repair_needed("You are an idiot.", "Watch your mouth.") is False


def test_name_question_adds_authoritative_identity_tool_call() -> None:
    request = {
        "request_id": "request-name",
        "npc_id": "npc-one",
        "conversation_context": {
            "message": "What's your name?",
            "available_tool_ids": ["projecthoomans.llm:ask_name"],
        },
    }

    result = ensure_identity_intent([], request)

    assert result == [
        {
            "id": "fallback-ask-name:request-name",
            "name": "ask_name",
            "arguments": {},
            "origin": "provider_neutral_identity_fallback",
        }
    ]


def test_provider_text_action_uses_the_same_canonical_tool_shape() -> None:
    request = {
        "request_id": "request-text-action",
        "npc_id": "npc-one",
        "conversation_context": {
            "available_tools": [
                {"type": "function", "function": {"name": "social_react"}},
            ],
        },
    }

    text, calls = extract_text_tool_calls(
        'Watch your mouth. <projecthoomans-action>{"name":"social_react",'
        '"arguments":{"kind":"insult"}}</projecthoomans-action>',
        request,
    )

    assert text == "Watch your mouth."
    assert calls[0]["name"] == "social_react"
    assert calls[0]["arguments"] == {"kind": "insult"}


def test_provider_text_action_requires_a_tool_selected_for_this_request() -> None:
    request = {
        "request_id": "request-unselected-action",
        "npc_id": "npc-one",
        "conversation_context": {
            "available_tools": [
                {"type": "function", "function": {"name": "ask_name"}},
            ],
            "provider_tool_names": [],
        },
    }

    text, calls = extract_text_tool_calls(
        'I am here. <projecthoomans-action>{"name":"ask_name",'
        '"arguments":{}}</projecthoomans-action>',
        request,
    )

    assert text == "I am here."
    assert calls == []


def test_truncated_provider_action_is_removed_from_npc_dialogue() -> None:
    request = {
        "request_id": "request-truncated-action",
        "npc_id": "npc-one",
        "conversation_context": {
            "message": "You are an idiot.",
            "available_tools": [
                {"type": "function", "function": {"name": "social_react"}},
            ],
        },
    }

    text, calls = extract_text_tool_calls(
        'Watch your mouth. <projecthoomans-action>{"name":"social_react",'
        '"arguments',
        request,
    )

    assert text == "Watch your mouth."
    assert calls == []
    recovered = ensure_social_intent(calls, request)
    assert recovered[0]["name"] == "social_react"
    assert recovered[0]["arguments"]["kind"] == "insult"


def test_horde_prompt_scaffold_is_not_npc_dialogue() -> None:
    leaked = (
        '"Ugh, you look terrible."'
        "\n\nInstruction:\n"
        "dude you look terrible\n\nResponse"
    )

    assert is_provider_scaffold(leaked) is True
    assert strip_provider_scaffold(leaked) == '"Ugh, you look terrible."'
    assert strip_provider_scaffold("Instruction:\ndude you look terrible") == ""
    assert strip_provider_scaffold("Watch your mouth.") == "Watch your mouth."

    leaked_review = (
        "Hmm? Join you? I'm just wandering, truth be told.\n\n"
        "Self-Correction: That was commentary, not the final output.\n"
        "Final Check: respond directly.\n"
        "New attempt: Hmm? Join you?"
    )
    assert is_provider_scaffold(leaked_review) is True
    assert strip_provider_scaffold(leaked_review) == (
        "Hmm? Join you? I'm just wandering, truth be told."
    )
    assert is_provider_scaffold("Final Check: respond directly") is True
    assert strip_provider_scaffold("New attempt: Hmm?") == ""


def test_provider_identity_boilerplate_becomes_in_world_npc_dialogue() -> None:
    request = {
        "request_id": "request-meta",
        "npc_id": "npc-one",
        "conversation_context": {"message": "Are you a bitch?"},
    }
    calls = add_fallback_social_tool_call(
        [],
        {
            **request,
            "conversation_context": {
                **request["conversation_context"],
                "available_tools": [
                    {"type": "function", "function": {"name": "social_react"}}
                ],
            },
        },
    )

    assert sanitize_npc_response(
        "I am an AI assistant and I don't have a personal identity.",
        request,
        calls,
    ) == "Watch your mouth."
    leaked_scaffold = (
        '"Ugh, you always seem to have the most demanding things to say!"\n\n'
        "---\n\n"
        "Self-Correction Check: The last turn's instructions are missing the "
        "actual prompt for the final response. Will assume the player's last "
        "message was \"You are a jerk.\".\n\n"
        "Player's last message: You are a jerk.\n\n"
        "Emilio's required action: Use"
    )
    assert sanitize_npc_response(leaked_scaffold, request, calls) == "Watch your mouth."
    assert sanitize_npc_response("Watch the road.", request, calls) == "Watch the road."


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


class TextActionProviders:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(
            request.model,
            'Watch your mouth. <projecthoomans-action>{"name":"social_react",'
            '"arguments":{"kind":"insult"}}</projecthoomans-action>',
        )


class PlainNameProviders:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "I'm Harley.")


class NameToolOnlyThenDialogueProviders:
    def __init__(self) -> None:
        self.requests = []

    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return CompletionResult(
                request.model,
                "",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-name",
                        "function": {"name": "ask_name", "arguments": "{}"},
                    }
                ],
            )
        return CompletionResult(request.model, "Of course. Let me introduce myself.")


class EmptyResponseProviders:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "", finish_reason="stop")


class LongResponseProviders:
    def resolve(self, _provider, model):
        return "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "R" * 5000)


class FailedProviders:
    def resolve(self, _provider, model):
        return "horde", model if model != "default" else "horde-model"

    async def complete(self, _provider, _request):
        raise ProviderError("Horde request failed", code="provider_server_error")


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
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
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
async def test_text_action_enters_the_same_delivery_pipeline(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    service = ConversationService(settings, TextActionProviders())
    client = DeliveryClient()
    request = {
        "request_id": "pnc-text-action-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "You are an idiot.",
            "available_tools": [
                {"type": "function", "function": {"name": "social_react"}},
            ],
        },
    }

    await complete_and_deliver(
        TextActionProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    delivery = client.calls[0][2]
    assert delivery["response_text"] == "Watch your mouth."
    assert delivery["semantic_tool_calls"][0]["name"] == "social_react"
    assert delivery["semantic_tool_calls"][0]["arguments"]["kind"] == "insult"
    identity = _test_memory_identity("world-one")
    stored = service._store(identity).recent_turns(
        "session-one",
        MemoryScope(identity.world_uuid, "player-one", "npc-one"),
        8,
    )
    assert all("<projecthoomans-action>" not in turn.content for turn in stored)


@pytest.mark.asyncio
async def test_plain_name_turn_enters_the_authoritative_identity_pipeline(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    service = ConversationService(settings, PlainNameProviders())
    client = DeliveryClient()
    request = {
        "request_id": "pnc-name-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "What's your name?",
            "available_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_name",
                        "description": "Ask the NPC to say their name.",
                    },
                },
            ],
        },
    }

    await complete_and_deliver(
        PlainNameProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    delivery = client.calls[0][2]
    assert delivery["semantic_tool_calls"][0]["name"] == "ask_name"
    assert delivery["semantic_tool_calls"][0]["arguments"] == {}


@pytest.mark.asyncio
async def test_name_tool_only_turn_gets_dialogue_repair_and_tool(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://127.0.0.1:1/v1",
        bridge_required=False,
    )
    providers = NameToolOnlyThenDialogueProviders()
    service = ConversationService(settings, providers)
    client = DeliveryClient()
    request = {
        "request_id": "pnc-name-repair-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "What's your name?",
            "available_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_name",
                        "description": "Ask the NPC to say their name.",
                    },
                },
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

    delivery = client.calls[0][2]
    assert delivery["response_text"] == "Of course. Let me introduce myself."
    assert delivery["presentation_reason"] == "llm_tool_response_repair"
    assert delivery["semantic_tool_calls"][0]["name"] == "ask_name"
    assert len(providers.requests) == 2


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
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
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
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
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
    assert service._store(_test_memory_identity("world-one")).stats()["turn_count"] == 1


@pytest.mark.asyncio
async def test_delivery_preserves_long_response_text() -> None:
    client = DeliveryClient()
    request = {
        "request_id": "pnc-long-response-1",
        "npc_id": "npc-one",
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Tell me more."}],
    }

    await complete_and_deliver(
        LongResponseProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
    )

    assert client.calls[0][2]["response_text"] == "R" * 5000


@pytest.mark.asyncio
async def test_provider_error_defers_semantic_reaction_reply_to_game(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="horde",
        horde_api_key="test-key",
        bridge_required=False,
    )
    service = ConversationService(settings, FailedProviders())
    client = DeliveryClient()
    request = {
        "request_id": "pnc-provider-error-insult-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "You are an idiot.",
            "available_tools": [
                {"type": "function", "function": {"name": "social_react"}},
            ],
        },
    }

    await complete_and_deliver(
        FailedProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    delivery = client.calls[0][2]
    assert delivery["semantic_tool_calls"][0]["name"] == "social_react"
    assert delivery["semantic_tool_calls"][0]["arguments"]["kind"] == "insult"
    assert delivery["response_text"] == ""
    assert delivery["provider_failure"] is True
    assert delivery["context_eligible"] is False
    assert delivery["error"] == "Horde request failed"


@pytest.mark.asyncio
async def test_provider_error_defers_authoritative_name_reply_to_game(tmp_path) -> None:
    from pbrainz.bridge import BridgeState
    from pbrainz.conversation_service import ConversationService

    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="horde",
        horde_api_key="test-key",
        bridge_required=False,
    )
    service = ConversationService(settings, FailedProviders())
    client = DeliveryClient()
    request = {
        "request_id": "pnc-provider-error-name-1",
        "npc_id": "npc-one",
        "conversation_context": {
            "world_uuid": "world-one",
            "world_mode": "multiplayer",
            "server_instance_id": "test-server",
            "server_world_generation": "world-one",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "session_id": "session-one",
            "message": "What is your name?",
            "available_tools": [
                {"type": "function", "function": {"name": "ask_name"}},
            ],
        },
    }

    await complete_and_deliver(
        FailedProviders(),
        client,
        BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-one"),
        request,
        conversation_service=service,
    )

    delivery = client.calls[0][2]
    assert delivery["semantic_tool_calls"][0]["name"] == "ask_name"
    assert delivery["response_text"] == ""
    assert delivery["provider_failure"] is True
    assert delivery["context_eligible"] is False


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
        "speaker_id": "npc-one",
        "speaker_kind": "npc",
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
                            "worldMode": "multiplayer",
                            "serverInstanceId": "test-server",
                            "serverWorldGeneration": "world-sync",
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

    identity = _test_memory_identity("world-sync")
    store = SQLiteMemoryStore(tmp_path / "memory", identity.world_uuid)
    turns = store.recent_turns(
        "conversation-sync",
        MemoryScope(identity.world_uuid, "player-one", "npc-one"),
    )
    assert len(turns) == 1
    assert turns[0].message_id == "conversation-sync:1"
    assert turns[0].game_day == 5
