import pytest

from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.memory import (
    ConversationTurn,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    SQLiteMemoryStore,
)


def test_sqlite_memory_is_partitioned_by_world_and_pair(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    scope = MemoryScope("world-one", "player-one", "npc-one")
    other_scope = MemoryScope("world-one", "player-two", "npc-one")
    store.ensure_session("session-one", scope)
    store.add_turn("session-one", scope, "user", "hello")
    store.remember(
        MemoryRecord(
            "memory-one",
            scope,
            MemoryType.FACT,
            "The shelter is north",
            ("shelter",),
            0.9,
        )
    )
    store.add_commitment(scope, "We will meet at the shelter.")

    assert store.retrieve(scope, "shelter")
    assert store.retrieve(other_scope, "shelter") == []
    assert store.recent_turns("session-one", scope, 4)[0].content == "hello"
    assert store.stats()["memory_count"] == 2

    with pytest.raises(ValueError):
        store.ensure_session("session-one", other_scope)
    with pytest.raises(ValueError):
        store.remember(
            MemoryRecord(
                "memory-one",
                other_scope,
                MemoryType.FACT,
                "This must not retarget the original memory",
            )
        )

    tables = {
        row[0]
        for row in store._connect().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "memories" in tables
    assert "memory-one" not in tables
    assert store.fts_enabled in {True, False}


def test_context_builder_omits_normal_optional_sections_and_enforces_budget() -> None:
    builder = ContextBuilder(max_chars=2000, recent_turn_limit=2)
    result = builder.build(
        ContextInput(
            npc_name="Harley",
            player_name="Alex",
            character_card={"traits": {"bravery": 0.8}, "personality": {"loyalty": 0.7}},
            relationship_snapshot={"state": "neutral", "approval": 0},
            current_state={"activeBehavior": "idle", "healthState": "normal"},
            recent_turns=(
                ConversationTurn("user", "x" * 1800, turn_index=1),
            ),
            current_message="Where is the shelter?",
        )
    )
    total = sum(len(message.content or "") for message in result.messages)
    assert total <= 2000
    assert "Core NPC Rules" in (result.messages[0].content or "")
    assert "Relationship Snapshot" not in (result.messages[0].content or "")
    assert result.messages[-1].content == "Where is the shelter?"


def test_hearsay_memory_keeps_claim_provenance_without_making_a_fact(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    scope = MemoryScope("world-one", "player-one", "npc-bob")

    memory = store.remember_hearsay(
        scope,
        "Alice claimed Sarah stole medicine.",
        source_npc_uuid="npc-alice",
        subject_npc_uuid="npc-sarah",
        session_id="gossip-one",
    )

    assert memory.memory_type is MemoryType.HEARSAY
    assert memory.provenance["source_npc_uuid"] == "npc-alice"
    assert memory.provenance["subject_npc_uuid"] == "npc-sarah"
    assert store.retrieve(scope, "medicine")[0].memory.memory_type is MemoryType.HEARSAY
