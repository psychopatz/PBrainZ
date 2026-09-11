from pbrainz.config import get_settings
from pbrainz.database import SettingsDatabase


def _create_database(path, values, catalog=None) -> None:
    database = SettingsDatabase(path)
    database.initialize()
    database.save_settings(values)
    if catalog:
        for provider, model_ids in catalog.items():
            database.replace_model_catalog(provider, model_ids)


def test_legacy_database_is_migrated_from_the_adjacent_data_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PBRAINZ_PORTABLE_ROOT", str(tmp_path))
    monkeypatch.delenv("PBRAINZ_DB", raising=False)
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    legacy_path = tmp_path / "data" / "hoomansllm.db"
    _create_database(
        legacy_path,
        {
            "app_name": "HoomansLLM",
            "default_provider": "gemini",
            "default_model": "legacy-model",
            "gemini_api_key": "legacy-secret",
        },
        {"gemini": ["legacy-model"]},
    )

    get_settings.cache_clear()
    try:
        settings = get_settings()
        canonical_path = tmp_path / "data" / "pbrainz.db"

        assert settings.database_path == str(canonical_path)
        assert settings.gemini_api_key == "legacy-secret"
        assert settings.default_model == "legacy-model"
        assert SettingsDatabase(canonical_path).model_catalog("gemini")
        assert legacy_path.is_file()
    finally:
        get_settings.cache_clear()


def test_legacy_migration_fills_missing_values_without_overwriting_current_ones(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PBRAINZ_PORTABLE_ROOT", str(tmp_path))
    monkeypatch.delenv("PBRAINZ_DB", raising=False)
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    canonical_path = tmp_path / "data" / "pbrainz.db"
    legacy_path = tmp_path / "data" / "hoomansllm.db"
    _create_database(
        canonical_path,
        {"default_model": "current-model", "ui_theme": "dark", "gemini_api_key": None},
    )
    _create_database(
        legacy_path,
        {"default_model": "legacy-model", "ui_theme": "light", "gemini_api_key": "secret"},
    )

    get_settings.cache_clear()
    try:
        settings = get_settings()

        assert settings.default_model == "current-model"
        assert settings.ui_theme == "dark"
        assert settings.gemini_api_key == "secret"
    finally:
        get_settings.cache_clear()


def test_explicit_database_override_does_not_import_adjacent_legacy_data(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PBRAINZ_PORTABLE_ROOT", str(tmp_path / "portable"))
    explicit_path = tmp_path / "explicit" / "settings.db"
    legacy_path = tmp_path / "portable" / "data" / "hoomansllm.db"
    _create_database(legacy_path, {"gemini_api_key": "must-not-be-imported"})
    monkeypatch.setenv("PBRAINZ_DB", str(explicit_path))
    monkeypatch.delenv("DATABASE_PATH", raising=False)

    get_settings.cache_clear()
    try:
        settings = get_settings()

        assert settings.database_path == str(explicit_path)
        assert settings.gemini_api_key is None
        assert SettingsDatabase(explicit_path).load_settings()["gemini_api_key"] is None
    finally:
        get_settings.cache_clear()
