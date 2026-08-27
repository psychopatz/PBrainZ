import json

from hoomans_llm.database import SettingsDatabase


def test_sqlite_settings_and_logs_are_persistent(tmp_path) -> None:
    path = tmp_path / "hoomansllm.db"
    first = SettingsDatabase(path)
    first.initialize()
    first.save_settings({"default_provider": "gemini", "gemini_api_key": "secret"})
    first.replace_model_catalog("gemini", ["gemini-2.5-flash", "gemini-2.0-flash"])
    first.add_log("INFO", "test", "settings saved")

    second = SettingsDatabase(path)
    assert second.load_settings()["default_provider"] == "gemini"
    assert second.load_settings()["gemini_api_key"] == "secret"
    assert [row["model_id"] for row in second.model_catalog("gemini")] == [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]
    assert second.recent_logs()[0]["message"] == "settings saved"
    assert json.loads(json.dumps(second.load_settings()))["gemini_api_key"] == "secret"


def test_legacy_database_import_populates_an_unconfigured_store(tmp_path) -> None:
    legacy_path = tmp_path / "legacy" / "hoomansllm.db"
    target_path = tmp_path / "config" / "hoomansllm.db"

    legacy = SettingsDatabase(legacy_path)
    legacy.initialize()
    legacy.save_settings(
        {
            "default_provider": "gemini",
            "default_model": "gemini-2.5-flash",
            "gemini_api_key": "legacy-secret",
        }
    )
    legacy.replace_model_catalog("gemini", ["gemini-2.5-flash"])

    target = SettingsDatabase(target_path)
    target.initialize()
    target.save_settings(
        {
            "default_provider": "openai",
            "default_model": None,
            "gemini_api_key": None,
        }
    )

    assert target.import_legacy_if_unconfigured(legacy_path) is True
    assert target.load_settings()["gemini_api_key"] == "legacy-secret"
    assert target.model_catalog("gemini")[0]["model_id"] == "gemini-2.5-flash"


def test_legacy_database_import_does_not_overwrite_configured_store(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.db"
    target_path = tmp_path / "target.db"

    legacy = SettingsDatabase(legacy_path)
    legacy.initialize()
    legacy.save_settings({"gemini_api_key": "legacy-secret"})

    target = SettingsDatabase(target_path)
    target.initialize()
    target.save_settings({"gemini_api_key": "current-secret"})

    assert target.import_legacy_if_unconfigured(legacy_path) is False
    assert target.load_settings()["gemini_api_key"] == "current-secret"
