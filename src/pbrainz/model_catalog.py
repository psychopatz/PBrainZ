"""Provider model discovery and periodic catalog refresh."""

from __future__ import annotations

import logging

from pbrainz.config import Settings
from pbrainz.database import SettingsDatabase
from pbrainz.exceptions import ProviderError
from pbrainz.providers.registry import ProviderRegistry

LOGGER = logging.getLogger(__name__)


class ModelCatalogManager:
    """Keep provider model lists current while retaining the last good catalog."""

    def __init__(
        self,
        settings: Settings,
        providers: ProviderRegistry,
        database: SettingsDatabase,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.database = database

    async def refresh_provider(self, provider_name: str) -> list[str]:
        if not self.settings.provider_configured(provider_name):
            raise ProviderError(
                f"Provider '{provider_name}' has no configured API key or endpoint.",
                status_code=400,
                code="provider_not_configured",
            )
        models = await self.providers.list_models(provider_name)
        if not models:
            raise ProviderError(
                f"Provider '{provider_name}' returned no supported chat models.",
                status_code=502,
                code="model_catalog_unavailable",
            )
        models = self.database.replace_model_catalog(provider_name, models)
        self._set_models(provider_name, models)
        LOGGER.info("model catalog refreshed provider=%s count=%d", provider_name, len(models))
        return models

    def _set_models(self, provider_name: str, models: list[str]) -> None:
        model_value = ",".join(models)
        setting_name = f"{provider_name}_models"
        if hasattr(self.settings, setting_name):
            setattr(self.settings, setting_name, model_value)
        if self.settings.default_provider == provider_name:
            if self.settings.default_model not in models:
                self.settings.default_model = models[0]
        self.database.save_settings(
            {
                setting_name: model_value,
                "default_model": self.settings.default_model,
            }
        )
