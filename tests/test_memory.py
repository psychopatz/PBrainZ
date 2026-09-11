import sqlite3

import pytest

from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.memory import (
    ConversationTurn,
    DaySynopsis,
    MemoryEpisode,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    MemoryVisibility,
    RetrievalMatch,
    SQLiteMemoryStore,
    StructuredFact,
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


def test_tagged_primitive_retrieval_uses_narrow_scope(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    scope = MemoryScope("world-one", "player-one", "npc-one")
    store.remember(
        MemoryRecord(
            "first-meeting",
            scope,
            MemoryType.PERSONAL_EVENT,
            "The player first met Alice on July 9, 1993 (Day 0, 15:39).",
            tags=("primitive", "first_meeting"),
            importance=0.8,
            visibility=MemoryVisibility.PUBLIC,
            participants=("player-one", "npc-one"),
        )
    )
    store.remember(
        MemoryRecord(
            "unrelated-event",
            scope,
            MemoryType.PERSONAL_EVENT,
            "The player found a backpack near the shelter.",
            tags=("primitive", "discovery"),
            importance=1.0,
            visibility=MemoryVisibility.PUBLIC,
            participants=("player-one", "npc-one"),
        )
    )

    matches = store.retrieve_query(
        MemoryQuery(
            scope=scope,
            actor_id="npc-one",
            current_message="When did we first meet?",
            requested_kinds=(MemoryType.PERSONAL_EVENT,),
            requested_tags=("first_meeting",),
        )
    )

    assert [match.memory.memory_id for match in matches] == ["first-meeting"]


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


def test_context_builder_puts_compact_horde_output_contract_first() -> None:
    assert len(ContextBuilder.CORE_RULES) <= 1450

    result = ContextBuilder(max_chars=2000).build(
        ContextInput(
            npc_name="Emilio",
            player_name="Alex",
            current_message="dude you look terrible",
        )
    )
    system = result.messages[0].content or ""
    assert "Output only 1-2 natural" in system
    assert "Game context is authoritative facts" in system
    assert "Final Check:" not in system
    assert "Self-Correction:" not in system


def test_context_builder_projects_live_state_and_keeps_internal_metadata_out() -> None:
    result = ContextBuilder().build(
        ContextInput(
            npc_name="Hassan patz",
            player_name="Psycho",
            character_card={
                "archetype": "General Survivor",
                "archetype_id": "General",
                "personality": {
                    "aggression": 0.6,
                    "compassion": 0.67,
                    "generatedFromSeed": True,
                    "schemaVersion": 1,
                    "foodPreference": "spicy",
                },
                "skills": {
                    "LongBlade": 5,
                    "Maintenance": 4,
                    "Cooking": 1,
                },
            },
            relationship_snapshot={
                "state": "friend",
                "approval": 78.8,
                "respect": 81.1,
                "familiarity": 100,
                "identityDiagnostics": {"revision": 28},
            },
            relationship_capabilities={
                "available_reactions": ["praise", "comfort"],
                "positive_action_cooldown_remaining_hours": 0,
                "revision": 28,
                "server_authoritative": True,
            },
            current_state={
                "activeBehavior": "FollowOwner:moving",
                "needs": {
                    "hunger": 0.8,
                    "hunger_level": "CRITICAL",
                    "revision": 2,
                    "sampled_at": 45.2,
                },
                "weaponStatus": "melee_ready",
            },
            current_message="How are you holding up?",
        )
    )
    system = result.messages[0].content or ""
    rendered = "\n".join(message.content or "" for message in result.messages)
    assert "interactionJournal" not in rendered
    assert "identityDiagnostics" not in rendered
    assert "generatedFromSeed" not in rendered
    assert "schemaVersion" not in rendered
    assert "revision=28" not in rendered
    assert "approval: high" in rendered
    assert "familiarity: very close" in rendered
    assert "hunger=critical" in rendered
    assert "weaponStatus" not in system
    assert result.diagnostics["dynamic_context_chars"] > 0


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


def test_canonical_turn_write_is_idempotent_and_date_aware(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    scope = MemoryScope("world-one", "player-one", "npc-one")
    store.ensure_session("conversation-one", scope)

    first = store.record_turn(
        "conversation-one",
        scope,
        "assistant",
        "The shelter is north.",
        message_id="conversation-one:1",
        game_day=4,
        world_age_hours=97.5,
        speaker_uuid="npc-one",
        speaker_name="Harley",
        speaker_kind="npc",
    )
    duplicate = store.record_turn(
        "conversation-one",
        scope,
        "assistant",
        "This retry must not create another turn.",
        message_id="conversation-one:1",
        game_day=4,
        world_age_hours=97.5,
        speaker_uuid="npc-one",
        speaker_name="Harley",
        speaker_kind="npc",
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.turn.content == first.turn.content
    assert duplicate.turn.game_day == 4
    assert duplicate.turn.speaker_uuid == "npc-one"
    assert store.stats()["turn_count"] == 1


def test_context_filters_legacy_failures_from_turns_memories_and_prompt(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    scope = MemoryScope("world-one", "player-one", "npc-one")
    store.ensure_session("conversation-one", scope)
    store.add_turn(
        "conversation-one",
        scope,
        "assistant",
        "I cannot answer right now. (provider request failed)",
    )
    store.add_turn("conversation-one", scope, "assistant", "The shelter is north.")
    assert [turn.content for turn in store.recent_turns("conversation-one", scope)] == [
        "The shelter is north."
    ]

    store.remember(
        MemoryRecord(
            "provider-failure-memory",
            scope,
            MemoryType.FACT,
            "I cannot answer right now. (provider request failed) The shelter is north.",
            visibility=MemoryVisibility.PUBLIC,
            participants=("npc-one",),
        )
    )
    store.remember(
        MemoryRecord(
            "clean-memory",
            scope,
            MemoryType.FACT,
            "The shelter is north.",
            visibility=MemoryVisibility.PUBLIC,
            participants=("npc-one",),
        )
    )
    matches = store.retrieve(scope, "shelter", limit=8)
    assert [match.memory.memory_id for match in matches] == ["clean-memory"]

    result = ContextBuilder().build(
        ContextInput(
            npc_name="Harley",
            player_name="Alex",
            retrieved_memories=tuple(matches)
            + (
                RetrievalMatch(
                    MemoryRecord(
                        "direct-failure",
                        scope,
                        MemoryType.FACT,
                        "I cannot answer right now. (provider request failed)",
                    ),
                    1.0,
                ),
            ),
            recent_turns=(
                ConversationTurn("assistant", "I cannot answer right now."),
                ConversationTurn("assistant", "I will check that now."),
            ),
            current_message="Where is the shelter?",
        )
    )
    rendered = "\n".join(message.content or "" for message in result.messages)
    assert "I cannot answer right now." not in rendered
    assert "I will check that now." in rendered
    assert result.diagnostics["retrieved_memories"] == 1
    assert result.diagnostics["recent_turns"] == 1


def test_context_filters_provider_identity_leaks() -> None:
    from pbrainz.memory import is_context_eligible

    assert not is_context_eligible(
        "I am an AI assistant and I don't have a personal identity."
    )


def test_canonical_turn_columns_migrate_existing_database(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(store.path) as connection:
        connection.executescript(
            """
            CREATE TABLE world_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO world_metadata(key, value) VALUES ('world_uuid', 'world-one');
            CREATE TABLE conversation_sessions (
                session_id TEXT PRIMARY KEY,
                world_uuid TEXT NOT NULL,
                player_uuid TEXT NOT NULL,
                npc_uuid TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                ended_at TEXT,
                turn_count INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO conversation_sessions(
                session_id, world_uuid, player_uuid, npc_uuid,
                started_at, last_activity_at
            ) VALUES ('conversation-one', 'world-one', 'player-one', 'npc-one', 'now', 'now');
            CREATE TABLE conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                world_uuid TEXT NOT NULL,
                player_uuid TEXT NOT NULL,
                npc_uuid TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )

    result = store.record_turn(
        "conversation-one",
        MemoryScope("world-one", "player-one", "npc-one"),
        "assistant",
        "Migrated safely.",
        message_id="conversation-one:1",
        game_day=1,
    )

    assert result.duplicate is False
    assert result.turn.message_id == "conversation-one:1"
    assert store.stats()["turn_count"] == 1


def test_actor_visibility_filters_private_memory_and_shares_public_episode(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    alice = MemoryScope("world-one", "player-one", "npc-alice")
    bob = MemoryScope("world-one", "player-one", "npc-bob")
    store.remember(
        MemoryRecord(
            "alice-private",
            alice,
            MemoryType.OPINION,
            "Alice privately distrusts Bob.",
            importance=0.9,
            visibility=MemoryVisibility.PRIVATE,
        )
    )
    store.remember_episode(
        MemoryEpisode(
            "episode-riverside",
            alice,
            "conversation-riverside",
            8,
            participants=("player-one", "npc-alice", "npc-bob"),
            summary="The group discussed the Riverside route.",
            topic_tags=("riverside",),
        )
    )

    matches = store.retrieve_query(
        MemoryQuery(
            scope=bob,
            actor_id="npc-bob",
            current_message="What happened at Riverside?",
            participants=("player-one", "npc-alice", "npc-bob"),
        )
    )

    assert any(match.memory.memory_type is MemoryType.EPISODE for match in matches)
    assert all(match.memory.memory_id != "alice-private" for match in matches)


def test_day_synopsis_and_structured_facts_are_dated_and_actor_filtered(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-one")
    alice = MemoryScope("world-one", "player-one", "npc-alice")
    bob = MemoryScope("world-one", "player-one", "npc-bob")
    store.save_day_synopsis(
        DaySynopsis(
            scope=alice,
            game_day=4,
            synopsis="- Alice agreed to check the shed.",
            commitments=("Check the shed",),
        )
    )
    store.save_structured_fact(
        StructuredFact(
            fact_id="alice-private-plan",
            scope=alice,
            kind="PLAN",
            content="Alice plans to leave before dawn.",
            visibility=MemoryVisibility.PRIVATE,
            game_day=4,
        )
    )

    synopsis = store.get_day_synopsis(alice, 4)
    assert synopsis is not None
    assert synopsis.commitments == ("Check the shed",)
    assert store.get_day_synopsis(alice, 5) is None
    assert store.list_structured_facts(
        MemoryQuery(scope=bob, actor_id="npc-bob", current_message="plans")
    ) == []


def test_memory_browser_lists_layers_and_deletes_selected_records(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path, "world-browser")
    scope = MemoryScope("world-browser", "player-one", "npc-one")
    store.remember(
        MemoryRecord(
            "browser-memory",
            scope,
            MemoryType.FACT,
            "The Riverside shelter is north.",
            importance=0.8,
            visibility=MemoryVisibility.PUBLIC,
            game_day=3,
        )
    )
    store.remember_episode(
        MemoryEpisode(
            "browser-episode",
            scope,
            "browser-session",
            3,
            summary="We discussed the Riverside route.",
            key_facts=("The shelter is north.",),
        )
    )
    store.save_structured_fact(
        StructuredFact(
            "browser-fact",
            scope,
            "LOCATION",
            "Riverside is north.",
            game_day=3,
        )
    )
    store.save_day_synopsis(
        DaySynopsis(scope, 3, synopsis="Discussed Riverside.", commitments=("Travel north",))
    )

    listed = store.list_saved_memories(search="riverside")
    assert listed["total"] == 4
    assert {item["record_kind"] for item in listed["items"]} == {
        "memory",
        "episode",
        "fact",
        "day_synopsis",
    }
    assert listed["items"][0]["preview"]

    assert store.delete_saved_memory("memory", "browser-memory") is True
    assert all(
        match.memory.memory_id != "browser-memory" for match in store.retrieve(scope, "Riverside")
    )
    assert store.delete_saved_memory(
        "day_synopsis",
        "player-one:npc-one:3",
        player_uuid="player-one",
        npc_uuid="npc-one",
        game_day=3,
    ) is True
    assert store.list_saved_memories(record_kind="day_synopsis")["total"] == 0
    assert store.delete_saved_memory("memory", "missing") is False
