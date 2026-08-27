"""Application configuration loaded from SQLite and process environment."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

OPENAI_COMPATIBLE_PROVIDERS = ("openai", "ollama", "lmstudio", "custom")
PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "custom": "",
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

    app_name: str = "HoomansLLM"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    database_path: str | None = None

    default_provider: str = "openai"
    default_model: str | None = None
    enabled_providers: str = "openai,ollama,lmstudio,custom,gemini"
    request_timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    bridge_required: bool = True
    bridge_root: str | None = None
    bridge_poll_interval: float = Field(default=0.5, gt=0.05, le=10)
    open_gui: bool = True
    auto_refresh_models: bool = False
    model_refresh_interval: float = Field(default=21600.0, gt=60, le=604800)
    ui_theme: str = "light"

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
            # Ollama and LM Studio normally have no API key; their configured
            # localhost endpoints are enough to make the profile usable.
            return bool(api_key or name in {"ollama", "lmstudio"} or base_url != default_url)
        if name == "gemini":
            return bool(self.gemini_api_key)
        return False

    def provider_explicitly_configured(self, provider_name: str) -> bool:
        """Return whether a provider has credentials or a non-default endpoint."""
        name = provider_name.strip().lower()
        if not self.provider_configured(name):
            return False
        if name in {"ollama", "lmstudio"}:
            return bool(
                getattr(self, f"{name}_api_key", None)
                or self.base_url_for(name).rstrip("/") != PROVIDER_DEFAULT_BASE_URLS[name]
            )
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the process-wide settings, migrating old dotenv values once if needed."""
    from hoomans_llm.database import SettingsDatabase, legacy_database_candidates

    environment_settings = Settings()
    database = SettingsDatabase(environment_settings.database_path)
    database.initialize()
    if environment_settings.database_path is None and not os.getenv("HOOMANSLLM_DB"):
        for legacy_path in legacy_database_candidates():
            if database.import_legacy_if_unconfigured(legacy_path):
                break
    stored = database.load_settings()
    values = environment_settings.model_dump()
    if not stored:
        values.update(database.read_legacy_dotenv())
        values.update(
            {
                key: value
                for key, value in environment_settings.model_dump().items()
                if key in environment_settings.model_fields_set
            }
        )
    values.update(stored)
    # Older databases used the two-provider default. Keep that configuration
    # working while making the local OpenAI-compatible profiles available.
    stored_enabled = str(stored.get("enabled_providers", "")).strip().lower()
    stored_provider_names = tuple(
        name.strip() for name in stored_enabled.split(",") if name.strip()
    )
    if set(stored_provider_names) == {"openai", "gemini"}:
        values["enabled_providers"] = "openai,ollama,lmstudio,custom,gemini"
    values["database_path"] = str(database.path)
    settings = Settings(**values)
    if not stored:
        database.save_settings(settings.model_dump())
    elif set(stored_provider_names) == {"openai", "gemini"}:
        database.save_settings({"enabled_providers": settings.enabled_providers})
    return settings
