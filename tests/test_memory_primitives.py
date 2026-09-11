from pbrainz.config import Settings
from pbrainz.memory import MemoryPrimitiveService, SQLiteMemoryStore


def _settings(tmp_path):
    zomboid = tmp_path / "Zomboid"
    (zomboid / "Saves" / "Apocalypse" / "active").mkdir(parents=True)
    return Settings(
        database_path=str(tmp_path / "settings.db"),
        zomboid_path=str(zomboid),
        bridge_required=False,
    )


def _first_meeting_event(event_id="meeting-one"):
    return {
        "event_id": event_id,
        "primitive_type": "first_meeting",
        "world": {
            "world_mode": "singleplayer",
            "save_relative_path": "Apocalypse/active",
        },
        "player_uuid": "player-one",
        "player_name": "Alex",
        "npc_uuid": "npc-one",
        "npc_name": "Harley",
        "event_time": {
            "kind": "in_world_calendar",
            "game_day": 3,
            "world_age_hours": 73.5,
            "year": 1993,
            "month": 7,
            "day": 4,
            "hour": 9,
            "minute": 30,
        },
        "source": "identity_disclosure",
        "authoritative": True,
    }


def test_first_meeting_is_idempotent_and_keeps_names_and_calendar(tmp_path) -> None:
    settings = _settings(tmp_path)
    service = MemoryPrimitiveService(settings)

    acknowledged = service.record_batch({"memory_primitives": [_first_meeting_event()]})
    duplicate = service.record_batch(
        {"memory_primitives": [_first_meeting_event("meeting-retry")]}
    )

    assert acknowledged == ("meeting-one",)
    assert duplicate == ("meeting-retry",)
    store = SQLiteMemoryStore(
        tmp_path / "Zomboid/Saves/Apocalypse/active/PBrainZ/memory",
        "sp-v1|Apocalypse/active",
    )
    browser = store.list_saved_memories()
    assert browser["total"] == 1
    record = browser["items"][0]
    assert record["npc_name"] == "Harley"
    assert record["player_name"] == "Alex"
    assert record["primitive_type"] == "first_meeting"
    assert record["event_time"]["game_day"] == 3
    assert "Harley" in record["content"]
    assert store.list_saved_memories(search="Harley")["total"] == 1


def test_pre_outbreak_relationship_uses_a_typed_event_time(tmp_path) -> None:
    settings = _settings(tmp_path)
    service = MemoryPrimitiveService(settings)

    record = service.record(
        {
            "event_id": "relationship-one",
            "primitive_type": "pre_outbreak_relationship",
            "world_mode": "singleplayer",
            "save_relative_path": "Apocalypse/active",
            "player_uuid": "player-one",
            "npc_uuid": "npc-one",
            "npc_name": "Harley",
            "relationship_kind": "lover",
            "variant_key": "lover-01",
            "event_time": {"kind": "pre_outbreak"},
        }
    )

    assert record.game_day is None
    assert record.provenance["event_time"]["kind"] == "pre_outbreak"
    assert record.provenance["relationship_kind"] == "lover"
    assert "romantic partners" in record.content


def test_in_world_time_accepts_a_day_only_fallback_from_early_game_loading() -> None:
    from pbrainz.memory import normalize_event_time

    normalized = normalize_event_time(
        {
            "kind": "in_world_calendar",
            "game_day": 3,
            "world_age_hours": 73.5,
            "label": "In-world day 3",
        }
    )

    assert normalized["year"] is None
    assert normalized["label"] == "In-world day 3"
