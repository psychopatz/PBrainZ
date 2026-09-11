"""Application configuration loaded from SQLite and process environment."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pbrainz.branding import DATABASE_ENV, PRODUCT_NAME
from pbrainz.paths import default_zomboid_path

OPENAI_COMPATIBLE_PROVIDERS = ("openai", "ollama", "lmstudio", "custom", "horde")
NO_API_KEY_PROVIDERS = frozenset({"ollama", "lmstudio", "horde"})
PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "custom": "",
    "horde": "https://oai.aihorde.net/v1",
}


class Settings(BaseSettings):
    """Runtime settings for the gateway.

    Provider credentials deliberately remain optional so the health endpoint and
    API documentation can be used before a provider is configured.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = PRODUCT_NAME
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    database_path: str | None = None

    default_provider: str = "gemini"
    default_model: str | None = None
    enabled_providers: str = "openai,ollama,lmstudio,custom,horde,gemini"
    request_timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    bridge_required: bool = True
    bridge_root: str | None = None
    bridge_config_path: str | None = None
    zomboid_path: str = Field(default_factory=lambda: str(default_zomboid_path()))
    bridge_poll_interval: float = Field(default=0.5, gt=0.05, le=10)
    open_gui: bool = True
    auto_refresh_models: bool = False
    model_refresh_interval: float = Field(default=21600.0, gt=60, le=604800)
    ui_theme: str = "light"
    memory_root: str | None = None
    context_max_chars: int = Field(default=12000, ge=2000, le=100000)
    memory_recent_turns: int = Field(default=8, ge=1, le=32)
    memory_retrieval_limit: int = Field(default=6, ge=1, le=16)
    memory_consolidation_turns: int = Field(default=12, ge=2, le=100)
    memory_rag_enabled: bool = True
    tool_rag_enabled: bool = True
    tool_retrieval_limit: int = Field(default=8, ge=1, le=32)
    tool_budget_chars: int = Field(default=2600, ge=400, le=20000)
    template_profiles_json: str = "{}"
    active_template_profile_id: str = "native-chat"
    llm_diagnostics: bool = False
    llm_trace_capture: bool = False

    # TTS is a local presentation enhancement.  It is deliberately disabled
    # by default and has no bearing on provider, memory, or gameplay calls.
    tts_enabled: bool = False
    tts_piper_executable: str = "piper"
    tts_model_root: str | None = None
    tts_metadata_path: str | None = None
    tts_voice_catalog_url: str = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json?download=true"
    )
    tts_voice_catalog_ttl_seconds: int = Field(default=86400, ge=60, le=2592000)
    tts_voice_catalog_language: str = Field(default="English", max_length=64)
    tts_output_device: str = ""
    tts_master_volume: float = Field(default=1.0, ge=0, le=1)
    tts_synthesis_workers: int = Field(default=1, ge=1, le=4)
    tts_model_cache_size: int = Field(default=2, ge=1, le=16)
    tts_max_simultaneous_playback: int = Field(default=4, ge=1, le=8)
    tts_max_generated_ahead: int = Field(default=3, ge=1, le=16)
    tts_max_tts_ready_ahead: int = Field(default=1, ge=1, le=8)
    tts_natural_gap_ms: int = Field(default=180, ge=0, le=2000)
    tts_synthesis_timeout: float = Field(default=45.0, gt=0, le=300)
    tts_audio_buffer_ms: int = Field(default=50, ge=0, le=2000)
    tts_voice_presets_json: str = "{}"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_models: str = "gpt-4o-mini"

    ollama_api_key: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_models: str = ""

    lmstudio_api_key: str | None = None
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_models: str = ""

    custom_api_key: str | None = None
    custom_base_url: str = ""
    custom_models: str = ""

    # AI Horde's OpenAI-compatible gateway accepts the documented anonymous
    # key when no personal key is configured. A personal key is optional and
    # only changes request priority/quota behavior.
    horde_api_key: str | None = None
    horde_base_url: str = "https://oai.aihorde.net/v1"
    horde_models: str = ""

    gemini_api_key: str | None = None
    gemini_models: str = "gemini-2.5-flash"

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Return normalized, de-duplicated provider names in config order."""
        names = [name.strip().lower() for name in self.enabled_providers.split(",")]
        return tuple(dict.fromkeys(name for name in names if name))

    def models_for(self, provider_name: str) -> tuple[str, ...]:
        """Return configured model IDs for a provider."""
        configured = getattr(self, f"{provider_name.lower()}_models", "")
        models = (model.strip() for model in configured.split(",") if model.strip())
        return tuple(dict.fromkeys(models))

    def base_url_for(self, provider_name: str) -> str:
        """Return the endpoint configured for an OpenAI-compatible provider."""
        return getattr(
            self,
            f"{provider_name.strip().lower()}_base_url",
            PROVIDER_DEFAULT_BASE_URLS.get(provider_name.strip().lower(), ""),
        )

    def provider_configured(self, provider_name: str) -> bool:
        """Return whether a provider has enough configuration to receive requests."""
        name = provider_name.strip().lower()
        if name in OPENAI_COMPATIBLE_PROVIDERS:
            api_key = getattr(self, f"{name}_api_key", None)
            base_url = self.base_url_for(name).rstrip("/")
            default_url = PROVIDER_DEFAULT_BASE_URLS[name]
            # Ollama, LM Studio, and Horde can operate without a personal API
            # key; their configured endpoints are enough to make the profile usable.
            return bool(api_key or name in NO_API_KEY_PROVIDERS or base_url != default_url)
        if name == "gemini":
            return bool(self.gemini_api_key)
        return False

    def provider_explicitly_configured(self, provider_name: str) -> bool:
        """Return whether a provider has credentials or a non-default endpoint."""
        name = provider_name.strip().lower()
        if not self.provider_configured(name):
            return False
        if name in NO_API_KEY_PROVIDERS:
            return bool(
                getattr(self, f"{name}_api_key", None)
                or self.base_url_for(name).rstrip("/") != PROVIDER_DEFAULT_BASE_URLS[name]
            )
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the process-wide settings from environment and the local database."""
    from pbrainz.database import SettingsDatabase, legacy_database_candidates

    environment_settings = Settings()
    database = SettingsDatabase(environment_settings.database_path)
    database.initialize()
    if environment_settings.database_path is None and not os.getenv(DATABASE_ENV):
        database.import_legacy_if_needed(legacy_database_candidates())
    stored = database.load_settings()
    values = environment_settings.model_dump()
    values.update(stored)
    stored_providers = stored.get("enabled_providers")
    providers_migrated = (
        isinstance(stored_providers, str)
        and "horde" not in {
            item.strip().casefold()
            for item in stored_providers.split(",")
            if item.strip()
        }
    )
    if providers_migrated:
        # Existing installations predate the built-in Horde profile. Preserve
        # the user's other providers while making the new no-key profile
        # visible after the normal settings migration.
        values["enabled_providers"] = (
            f"{stored_providers.strip().strip(',')},horde"
            if stored_providers.strip().strip(",")
            else "horde"
        )
    # Branding is part of the executable, not a user-configurable setting.
    # Never resurrect the pre-rename HoomansLLM service name from an old local
    # settings row.
    values["app_name"] = PRODUCT_NAME
    if stored.get("app_name") != PRODUCT_NAME:
        database.save_settings({"app_name": PRODUCT_NAME})
    values["database_path"] = str(database.path)
    settings = Settings(**values)
    if not stored:
        database.save_settings(settings.model_dump())
    elif "zomboid_path" not in stored:
        # Backfill the new path setting for databases created by older builds.
        database.save_settings({"zomboid_path": settings.zomboid_path})
    if providers_migrated:
        database.save_settings({"enabled_providers": settings.enabled_providers})
    return settings
