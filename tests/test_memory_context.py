from types import SimpleNamespace

from pbrainz.bridge.memory_context import ActiveMemoryContextCache


def test_active_context_cache_returns_canonical_singleplayer_identity() -> None:
    cache = ActiveMemoryContextCache(ttl_seconds=5)

    assert cache.update(
        {
            "world_mode": "singleplayer",
            "save_relative_path": r"Apocalypse\Save One",
            "player_uuid": "player-one",
            "player_name": "Alex",
        },
        "runtime-one",
        now=100,
    )

    assert cache.as_dict(now=102) == {
        "status": "active",
        "world_uuid": "sp-v1|Apocalypse/Save One",
        "world_mode": "singleplayer",
        "save_relative_path": "Apocalypse/Save One",
        "server_instance_id": None,
        "server_world_generation": None,
        "player_uuid": "player-one",
        "player_name": "Alex",
        "runtime_id": "runtime-one",
        "age_seconds": 2.0,
    }


def test_active_context_cache_expires_without_a_new_game_poll() -> None:
    cache = ActiveMemoryContextCache(ttl_seconds=2)
    assert cache.update(
        {
            "world_mode": "multiplayer",
            "server_instance_id": "server-one",
            "server_world_generation": "wipe-one",
        },
        "runtime-one",
        now=10,
    )

    expired = cache.as_dict(now=12.1)

    assert expired["status"] == "unavailable"
    assert expired["reason"] == "game_context_stale"


def test_active_context_cache_rejects_unlocatable_context() -> None:
    cache = ActiveMemoryContextCache()

    assert not cache.update({"world_mode": "singleplayer"}, "runtime-one", now=1)
    assert cache.as_dict(now=1)["reason"] == "no_game_context"


def test_active_context_cache_does_not_require_an_http_request_object() -> None:
    # This guards the cache's small bridge-facing contract from accidental
    # coupling to FastAPI request state.
    cache = ActiveMemoryContextCache()
    assert cache.update(
        SimpleNamespace(),  # rejected cleanly because it is not a mapping
        "runtime-one",
    ) is False
