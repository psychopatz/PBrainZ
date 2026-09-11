from pathlib import Path

import pytest

from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationRequest, ConversationService
from pbrainz.memory import (
    MemoryIdentity,
    MemoryLocationError,
    SQLiteMemoryStore,
    list_memory_worlds,
    normalize_save_relative_path,
)
from pbrainz.providers.base import CompletionResult


class _Providers:
    provider_names = ("custom",)

    def resolve(self, provider, model):
        return provider or "custom", model if model != "default" else "fake-model"

    async def complete(self, _provider, request):
        return CompletionResult(request.model, "A remembered reply.")


def test_save_relative_path_normalization_rejects_escape_inputs() -> None:
    assert normalize_save_relative_path(r"Apocalypse\2026-09-06_10-21-49") == (
        "Apocalypse/2026-09-06_10-21-49"
    )
    assert normalize_save_relative_path("/tmp/not-a-save") == ""
    assert normalize_save_relative_path("Apocalypse/../other") == ""
    assert normalize_save_relative_path("C:/Users/player/save") == ""


def test_singleplayer_identity_targets_the_exact_save_directory(tmp_path) -> None:
    zomboid = tmp_path / "Zomboid"
    save = zomboid / "Saves" / "Apocalypse" / "2026-09-06_10-21-49"
    save.mkdir(parents=True)
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        zomboid_path=str(zomboid),
    )
    identity = MemoryIdentity.from_mapping(
        "save-label",
        {
            "world_mode": "singleplayer",
            "save_relative_path": r"Apocalypse\2026-09-06_10-21-49",
        },
    )

    root = identity.root_for(settings)
    store = SQLiteMemoryStore(root, identity.world_uuid)
    store.initialize()

    assert root == save / "PBrainZ" / "memory"
    assert store.path.parent == root
    assert store.path.is_file()
    assert store.path.parent.parent.parent == save


def test_singleplayer_identity_requires_existing_save_and_rejects_symlink_escape(tmp_path) -> None:
    zomboid = tmp_path / "Zomboid"
    save = zomboid / "Saves" / "Apocalypse" / "active"
    save.mkdir(parents=True)
    settings = Settings(database_path=str(tmp_path / "settings.db"), zomboid_path=str(zomboid))

    missing = MemoryIdentity.from_mapping(
        "world",
        {"world_mode": "singleplayer", "save_relative_path": "Apocalypse/missing"},
    )
    with pytest.raises(MemoryLocationError, match="does not exist"):
        missing.root_for(settings)

    outside = tmp_path / "outside"
    outside.mkdir()
    (save / "PBrainZ").symlink_to(outside, target_is_directory=True)
    identity = MemoryIdentity.from_mapping(
        "world",
        {"world_mode": "singleplayer", "save_relative_path": "Apocalypse/active"},
    )
    with pytest.raises(MemoryLocationError, match="escapes"):
        identity.root_for(settings)


def test_multiplayer_identity_is_external_and_generation_scoped(tmp_path) -> None:
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        zomboid_path=str(tmp_path / "Zomboid"),
    )
    first = MemoryIdentity.from_mapping(
        "same-local-save-label",
        {
            "world_mode": "multiplayer",
            "server_instance_id": "server.example:16261",
            "server_world_generation": "wipe-001",
        },
    )
    second = MemoryIdentity.from_mapping(
        "same-local-save-label",
        {
            "world_mode": "multiplayer",
            "server_instance_id": "server.example:16261",
            "server_world_generation": "wipe-002",
        },
    )

    assert first.world_uuid.startswith("mp-v1|")
    assert first.world_uuid != second.world_uuid
    assert first.root_for(settings) == Path(tmp_path / "memory")
    assert SQLiteMemoryStore(first.root_for(settings), first.world_uuid).path != (
        SQLiteMemoryStore(second.root_for(settings), second.world_uuid).path
    )


def test_identity_requires_explicit_mode_and_locator() -> None:
    with pytest.raises(MemoryLocationError, match="world_mode"):
        MemoryIdentity.from_mapping("world-one", {})
    with pytest.raises(MemoryLocationError, match="server_instance_id"):
        MemoryIdentity.from_mapping("world-one", {"world_mode": "multiplayer"})

    request = ConversationRequest.from_mapping(
        {
            "request_id": "request-one",
            "conversation_context": {
                "world_uuid": "old-label",
                "world_mode": "singleplayer",
                "save_relative_path": "Apocalypse/active",
                "player_uuid": "player-one",
                "npc_uuid": "npc-one",
                "message": "Hello",
            },
        }
    )
    assert request.scope.world_uuid == "sp-v1|Apocalypse/active"
    assert request.memory_identity is not None


@pytest.mark.asyncio
async def test_service_writes_singleplayer_memory_inside_save_folder(tmp_path) -> None:
    zomboid = tmp_path / "Zomboid"
    save = zomboid / "Saves" / "Apocalypse" / "active"
    save.mkdir(parents=True)
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        zomboid_path=str(zomboid),
        bridge_required=False,
    )
    service = ConversationService(settings, _Providers())
    request = ConversationRequest.from_mapping(
        {
            "request_id": "request-one",
            "conversation_context": {
                "world_uuid": "save-label",
                "world_mode": "singleplayer",
                "save_relative_path": "Apocalypse/active",
                "player_uuid": "player-one",
                "npc_uuid": "npc-one",
                "message": "Remember this.",
            },
        }
    )

    await service.complete(request)

    expected_root = save / "PBrainZ" / "memory"
    assert list(expected_root.glob("world-*.db"))
    assert not list((tmp_path / "memory").glob("world-*.db"))


def test_memory_browser_discovers_external_and_save_local_worlds(tmp_path) -> None:
    zomboid = tmp_path / "Zomboid"
    save = zomboid / "Saves" / "Apocalypse" / "active"
    save.mkdir(parents=True)
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        zomboid_path=str(zomboid),
        memory_root=str(tmp_path / "external-memory"),
    )
    external_identity = MemoryIdentity.from_mapping(
        "external-world",
        {
            "world_mode": "multiplayer",
            "server_instance_id": "external-server",
            "server_world_generation": "generation-1",
        },
    )
    SQLiteMemoryStore(settings.memory_root, external_identity.world_uuid).initialize()
    local_root = save / "PBrainZ" / "memory"
    SQLiteMemoryStore(local_root, "sp-v1|Apocalypse/active").initialize()

    worlds = list_memory_worlds(settings)

    assert {(item["world_uuid"], item["storage_kind"]) for item in worlds} == {
        (external_identity.world_uuid, "external"),
        ("sp-v1|Apocalypse/active", "save_local"),
    }
    assert all("_root" in item for item in worlds)
