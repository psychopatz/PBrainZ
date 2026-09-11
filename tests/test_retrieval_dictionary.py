from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.memory import MemoryQuery, MemoryScope, MemoryType
from pbrainz.memory.sqlite import SQLiteMemoryStore
from pbrainz.retrieval_dictionary import (
    DEFAULT_HISTORICAL_TERMS,
    default_retrieval_dictionary,
    load_retrieval_dictionary,
)
from pbrainz.retrieval_planner import plan_retrieval


def test_empty_dictionary_uses_shipped_defaults() -> None:
    dictionary = load_retrieval_dictionary({})

    assert dictionary.active_locale == "en"
    assert dictionary.locale().historical == DEFAULT_HISTORICAL_TERMS


def test_custom_locale_changes_planner_cues() -> None:
    dictionary = load_retrieval_dictionary(
        {
            "active_locale": "es",
            "locales": {
                "es": {
                    "historical": ["recuerdas"],
                    "first_meeting": ["primera vez"],
                    "tool": ["sígueme"],
                    "stop_words": ["el", "la"],
                    "token_expansions": {"conocer": ["conocimos"]},
                }
            },
        }
    )

    plan = plan_retrieval("¿Recuerdas nuestra primera vez?", dictionary)

    assert plan.memory_retrieval is True
    assert plan.first_meeting is True
    assert plan.requested_memory_kinds == (MemoryType.PERSONAL_EVENT,)
    assert plan.requested_memory_tags == ("first_meeting",)
    assert dictionary.token_expansion_items() == (("conocer", ("conocimos",)),)


def test_locale_round_trip_preserves_unicode_and_intentional_empty_groups() -> None:
    dictionary = load_retrieval_dictionary(
        {
            "active_locale": "ja",
            "locales": {
                "ja": {
                    "historical": [],
                    "first_meeting": ["初めて会った"],
                    "tool": [],
                    "stop_words": ["は"],
                    "token_expansions": {},
                }
            },
        }
    )

    restored = load_retrieval_dictionary(dictionary.as_json())

    assert restored.active_locale == "ja"
    assert restored.locale().first_meeting == ("初めて会った",)
    assert restored.locale().historical == ()
    assert restored.locale().tool == ()
    assert restored.locale().token_expansions == ()


def test_default_dictionary_serializes_as_a_complete_editable_profile() -> None:
    payload = default_retrieval_dictionary().as_dict()

    assert payload["version"] == 1
    assert payload["active_locale"] == "en"
    assert set(payload["locales"]["en"]) == {
        "historical",
        "first_meeting",
        "tool",
        "stop_words",
        "token_expansions",
    }


def test_localized_token_expansion_reaches_tool_matching() -> None:
    dictionary = load_retrieval_dictionary(
        {
            "active_locale": "es",
            "locales": {
                "es": {
                    "tool": ["sígueme"],
                    "token_expansions": {"sígueme": ["follow"]},
                }
            },
        }
    )
    result = ContextBuilder(retrieval_dictionary=dictionary).build(
        ContextInput(
            npc_name="Alice",
            player_name="Alex",
            current_message="Sígueme.",
            available_tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "order_follow",
                        "description": "Request the companion follow the player.",
                    },
                },
            ),
        )
    )

    assert [tool["function"]["name"] for tool in result.tools] == ["order_follow"]


def test_empty_custom_expansions_disable_shipped_memory_aliases() -> None:
    query = MemoryQuery(
        scope=MemoryScope("world", "player", "npc"),
        actor_id="npc",
        current_message="When did we meet?",
        token_expansions=(),
    )

    assert "met" not in SQLiteMemoryStore._query_tokens(query)
