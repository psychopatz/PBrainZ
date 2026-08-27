from hoomans_llm.config import get_settings
from hoomans_llm.database import SettingsDatabase


def test_explicit_environment_values_override_legacy_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "PORT=8000\nBRIDGE_REQUIRED=true\nGEMINI_API_KEY=legacy-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PORT", "8765")
    monkeypatch.setenv("BRIDGE_REQUIRED", "false")
    monkeypatch.setenv("HOOMANSLLM_DB", str(tmp_path / "settings.db"))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.port == 8765
    assert settings.bridge_required is False
    assert settings.gemini_api_key == "legacy-key"
    assert SettingsDatabase(tmp_path / "settings.db").load_settings()["port"] == 8765
    get_settings.cache_clear()


def test_get_settings_loads_persistent_sqlite_values(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "settings.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    database.save_settings(
        {
            "default_provider": "gemini",
            "default_model": "gemini-2.5-flash",
            "gemini_api_key": "persisted-key",
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOOMANSLLM_DB", str(database_path))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.default_provider == "gemini"
    assert settings.default_model == "gemini-2.5-flash"
    assert settings.gemini_api_key == "persisted-key"
    get_settings.cache_clear()


def test_get_settings_migrates_old_provider_list_for_local_profiles(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "settings.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    database.save_settings({"enabled_providers": "openai,gemini"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOOMANSLLM_DB", str(database_path))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.provider_names == ("openai", "ollama", "lmstudio", "custom", "gemini")
    assert SettingsDatabase(database_path).load_settings()["enabled_providers"] == (
        "openai,ollama,lmstudio,custom,gemini"
    )
    get_settings.cache_clear()


def test_explicit_database_override_does_not_import_working_directory_database(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy = SettingsDatabase(tmp_path / "hoomansllm.db")
    legacy.initialize()
    legacy.save_settings({"gemini_api_key": "legacy-secret"})
    database_path = tmp_path / "explicit.db"
    monkeypatch.setenv("HOOMANSLLM_DB", str(database_path))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.gemini_api_key is None
    assert SettingsDatabase(database_path).load_settings()["gemini_api_key"] is None
    get_settings.cache_clear()
