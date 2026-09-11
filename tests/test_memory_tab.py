from pbrainz.gui.memory_tab import MemoryTab, _storage_label, _world_label


def test_memory_world_labels_show_save_path_and_disambiguate_duplicates() -> None:
    worlds = [
        {
            "world_uuid": "sp-v1|one",
            "save_relative_path": "Apocalypse/Save One",
            "storage_kind": "save_local",
        },
        {
            "world_uuid": "sp-v1|two",
            "save_relative_path": "Apocalypse/Save One",
            "storage_kind": "save_local",
        },
    ]

    labels = MemoryTab._build_world_labels(worlds)

    assert labels == {
        "sp-v1|one": "Save: Apocalypse/Save One",
        "sp-v1|two": "Save: Apocalypse/Save One (2)",
    }


def test_memory_world_labels_explain_external_storage() -> None:
    world = {"world_uuid": "mp-v1|server", "storage_kind": "external"}

    assert _world_label(world) == "Multiplayer client memory"
    assert _storage_label(world) == "Stored in PBrainZ client storage (multiplayer)"
