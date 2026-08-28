from pbrainz.game_bridge_settings import GameBridgeSettings, parse_config


def test_parse_config_matches_core_defaults() -> None:
    config = parse_config(
        "config_version=1\nbridge_enabled=YES\n"
        "bridge_transport=unexpected\nbridge_poll_interval_ms=99999\n"
    )

    assert config.enabled is True
    assert config.transport == "file"
    assert config.poll_interval_ms == 5000


def test_set_enabled_preserves_bridge_transport_settings(tmp_path) -> None:
    path = tmp_path / "PsychopatzCore_Bridge.txt"
    path.write_text(
        "config_version=1\nbridge_enabled=true\n"
        "bridge_transport=file\nbridge_poll_interval_ms=400\n",
        encoding="utf-8",
    )

    settings = GameBridgeSettings(path)
    config = settings.set_enabled(False)

    assert config.enabled is False
    assert settings.read().poll_interval_ms == 400
    assert "bridge_enabled=false" in path.read_text(encoding="utf-8")
