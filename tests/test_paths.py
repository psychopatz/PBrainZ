import sys
from pathlib import Path

from pbrainz.database import application_root, legacy_database_candidates
from pbrainz.game_bridge_settings import GameBridgeSettings
from pbrainz.paths import bridge_root_for, default_zomboid_path


def test_default_zomboid_path_honors_portable_environment_override(
    tmp_path, monkeypatch
) -> None:
    configured = tmp_path / "custom-zomboid"
    monkeypatch.setenv("ZOMBOID_PATH", str(configured))

    assert default_zomboid_path() == configured


def test_default_zomboid_path_uses_existing_cross_platform_candidate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("ZOMBOID_PATH", raising=False)
    monkeypatch.delenv("ZOMBOID_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    candidate = tmp_path / "Documents" / "Zomboid"
    candidate.mkdir(parents=True)

    assert default_zomboid_path() == candidate


def test_bridge_root_and_game_toggle_follow_selected_zomboid_path(tmp_path) -> None:
    zomboid_path = tmp_path / "custom-zomboid"
    expected_bridge = zomboid_path / "Lua" / "PsychopatzBridge"
    expected_config = zomboid_path / "Lua" / "PsychopatzCore_Bridge.txt"

    assert bridge_root_for(zomboid_path) == expected_bridge
    settings = GameBridgeSettings(zomboid_path=zomboid_path)
    assert settings.path == expected_config

    settings.set_zomboid_path(tmp_path / "moved-zomboid")
    assert settings.path == tmp_path / "moved-zomboid" / "Lua" / "PsychopatzCore_Bridge.txt"


def test_explicit_bridge_toggle_path_remains_authoritative(tmp_path) -> None:
    explicit = tmp_path / "bridge" / "PsychopatzCore_Bridge.txt"
    settings = GameBridgeSettings(explicit, zomboid_path=tmp_path / "zomboid")

    settings.set_zomboid_path(tmp_path / "moved-zomboid")

    assert settings.path == explicit


def test_application_root_honors_explicit_portable_root(tmp_path, monkeypatch) -> None:
    configured = tmp_path / "portable-copy"
    monkeypatch.setenv("PBRAINZ_PORTABLE_ROOT", str(configured))
    monkeypatch.delenv("APPIMAGE", raising=False)

    assert application_root() == configured


def test_application_root_uses_appimage_location(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "release" / "PBrainZ.AppImage"
    artifact.parent.mkdir()
    monkeypatch.delenv("PBRAINZ_PORTABLE_ROOT", raising=False)
    monkeypatch.setenv("APPIMAGE", str(artifact))

    assert application_root() == artifact.parent


def test_application_root_uses_frozen_executable_location(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "PBrainZ.exe"
    monkeypatch.delenv("PBRAINZ_PORTABLE_ROOT", raising=False)
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert application_root() == executable.parent


def test_legacy_candidates_include_database_beside_portable_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PBRAINZ_PORTABLE_ROOT", str(tmp_path))

    assert tmp_path / "data" / "hoomansllm.db" in legacy_database_candidates()
