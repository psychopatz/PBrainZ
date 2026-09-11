from pbrainz.tool_routing import ToolRouter


def _tool(name: str, description: str, **metadata):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
        "metadata": metadata,
    }


def test_tool_router_filters_before_relevance_and_preserves_schema() -> None:
    social = _tool("social_react", "Express a social reaction intent.")
    follow = _tool("order_follow", "Request the companion follow the player.", tags=["movement"])
    internal = _tool("inventory_scan", "Inspect inventory internals.", internal=True)
    ineligible = _tool("combat_attack", "Attack a target.", eligible=False)

    selection = ToolRouter(max_results=2).select(
        (social, follow, internal, ineligible),
        "Follow me to the shelter.",
    )

    assert selection.registered == 4
    assert selection.eligible == 2
    assert selection.diagnostics["rejected"][0]["reason"] == "internal_primitive"
    assert selection.diagnostics["rejected"][1]["reason"] == "game_ineligible"
    assert follow in selection.selected
    assert all(tool is social or tool is follow for tool in selection.selected)


def test_tool_router_keeps_safe_fallback_for_small_talk() -> None:
    selection = ToolRouter(max_results=4).select(
        (
            _tool("social_react", "Express a social reaction intent."),
            _tool("order_follow", "Request the companion follow the player."),
        ),
        "Hello.",
    )

    assert [tool["function"]["name"] for tool in selection.selected] == ["social_react"]


def test_tool_router_selects_identity_tool_for_name_question() -> None:
    selection = ToolRouter(max_results=4).select(
        (
            _tool("social_react", "Express a social reaction intent."),
            _tool(
                "ask_name",
                "Ask the NPC to say their name through authoritative identity disclosure.",
            ),
        ),
        "What's your name?",
    )

    assert [tool["function"]["name"] for tool in selection.selected] == ["ask_name"]
