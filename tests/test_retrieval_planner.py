from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.memory import MemoryType
from pbrainz.retrieval_planner import plan_retrieval


def _tool(name: str, description: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_first_meeting_uses_exact_memory_lane_without_raw_recall() -> None:
    plan = plan_retrieval("When did we first meet exactly?")

    assert plan.memory_retrieval is True
    assert plan.conversation_recall is False
    assert plan.first_meeting is True
    assert plan.requested_memory_kinds == (MemoryType.PERSONAL_EVENT,)
    assert plan.requested_memory_tags == ("first_meeting",)


def test_action_request_enables_tool_lane_but_small_talk_does_not() -> None:
    assert plan_retrieval("Follow me to the shelter.").tool_retrieval is True
    assert plan_retrieval("Tell me about the weather.").tool_retrieval is False


def test_context_does_not_send_unrelated_tool_schemas_or_duplicate_descriptions() -> None:
    result = ContextBuilder().build(
        ContextInput(
            npc_name="Alice",
            player_name="Alex",
            current_message="Tell me about the weather.",
            available_tools=(
                _tool("order_follow", "Request the companion follow the player."),
                _tool("social_react", "Express a social reaction intent."),
            ),
        )
    )

    assert result.tools == []
    prompt = "\n".join(message.content or "" for message in result.messages)
    assert "Request the companion follow" not in prompt


def test_context_sends_relevant_tool_schema_without_prompt_duplication() -> None:
    result = ContextBuilder().build(
        ContextInput(
            npc_name="Alice",
            player_name="Alex",
            current_message="Follow me to the shelter.",
            available_tools=(
                _tool("order_follow", "Request the companion follow the player."),
                _tool("social_react", "Express a social reaction intent."),
            ),
        )
    )

    assert [tool["function"]["name"] for tool in result.tools] == ["order_follow"]
    prompt = "\n".join(message.content or "" for message in result.messages)
    assert "Tool Response Contract" in prompt
    assert "- order_follow" not in prompt
    assert "Request the companion follow" not in prompt
