"""FastAPI application factory and process lifecycle."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hoomans_llm.api.routes import router
from hoomans_llm.bridge import BridgeController, BridgeRuntimeMonitor
from hoomans_llm.config import Settings, get_settings
from hoomans_llm.database import SettingsDatabase
from hoomans_llm.exceptions import ProviderError
from hoomans_llm.game_bridge_settings import GameBridgeSettings
from hoomans_llm.logging_setup import configure_logging
from hoomans_llm.model_catalog import ModelCatalogManager
from hoomans_llm.providers.registry import ProviderRegistry
from hoomans_llm.tts import TTSService

LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application instance, useful both in production and tests."""
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = SettingsDatabase(app_settings.database_path)
        database.initialize()
        configure_logging(database, app_settings.log_level)
        providers = ProviderRegistry(app_settings)
        if not app_settings.provider_configured(app_settings.default_provider):
            app_settings.default_provider = next(
                (
                    provider_name
                    for provider_name in providers.provider_names
                    if app_settings.provider_explicitly_configured(provider_name)
                ),
                next(
                    (
                        provider_name
                        for provider_name in providers.provider_names
                        if app_settings.provider_configured(provider_name)
                    ),
                    app_settings.default_provider,
                ),
            )
        bridge = BridgeRuntimeMonitor(app_settings.bridge_root)
        game_bridge_settings = GameBridgeSettings(app_settings.bridge_config_path)
        app.state.settings = app_settings
        app.state.database = database
        app.state.providers = providers
        app.state.bridge = bridge
        app.state.game_bridge_settings = game_bridge_settings
        catalog = ModelCatalogManager(app_settings, providers, database)
        app.state.model_catalog = catalog
        tts = TTSService(app_settings)
        app.state.tts = tts
        await tts.start()
        # Model catalogs are loaded from SQLite synchronously by the status
        # route. Network discovery is deliberately manual so startup never
        # waits on unavailable local providers.
        app.state.model_refresh_task = None
        controller = BridgeController(app_settings, providers, bridge, tts)
        app.state.bridge_controller = controller
        app.state.bridge_pump = None
        if controller.enabled:
            await controller.start()
            app.state.bridge_pump = controller.task
        LOGGER.info("HoomansLLM started database=%s", database.path)
        try:
            yield
        finally:
            await controller.stop()
            app.state.bridge_pump = None
            await tts.stop()
            await providers.close()
            LOGGER.info("HoomansLLM stopped")

    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="A provider-agnostic LLM gateway for Project Hoomans NPCs.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.code,
                    "param": None,
                    "code": exc.code,
                }
            },
        )

    return app


app = create_app()
