"""Native control-panel routes for settings, bridge control, and diagnostics."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from hoomans_llm.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    UIBridgeRequest,
    UILogResponse,
    UIModelRefreshRequest,
    UISettingsRequest,
    UIStatus,
)
from hoomans_llm.config import OPENAI_COMPATIBLE_PROVIDERS
from hoomans_llm.database import DEFAULT_ACTIVITY_LIMIT, SettingsDatabase
from hoomans_llm.providers.registry import ProviderRegistry

from .response_format import _completion_response
from .route_support import (
    _available_models,
    _provider_configured_with_updates,
    _registry,
    _ui_status,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.get("/api/settings", response_model=UIStatus, tags=["control-panel"])
async def ui_status(request: Request) -> UIStatus:
    return _ui_status(request)


@router.post("/api/settings", response_model=UIStatus, tags=["control-panel"])
async def update_ui_settings(request: Request, body: UISettingsRequest) -> UIStatus:
    settings = request.app.state.settings
    registry: ProviderRegistry = _registry(request)
    database: SettingsDatabase = request.app.state.database
    provider_name = (body.default_provider or settings.default_provider).strip().lower()
    if provider_name not in registry.provider_names:
        enabled = ", ".join(registry.provider_names) or "none"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or disabled provider '{provider_name}'. Enabled: {enabled}.",
        )

    credential_updates: dict[str, object] = {}
    changed_providers: set[str] = set()
    for provider_name_for_key in (*OPENAI_COMPATIBLE_PROVIDERS, "gemini"):
        field_name = f"{provider_name_for_key}_api_key"
        clear_field = f"clear_{field_name}"
        supplied_key = getattr(body, field_name)
        if getattr(body, clear_field):
            credential_updates[field_name] = None
            changed_providers.add(provider_name_for_key)
        elif supplied_key and supplied_key.strip():
            credential_updates[field_name] = supplied_key.strip()
            changed_providers.add(provider_name_for_key)

    for provider_name_for_url in OPENAI_COMPATIBLE_PROVIDERS:
        base_url = getattr(body, f"{provider_name_for_url}_base_url")
        if base_url is not None:
            credential_updates[f"{provider_name_for_url}_base_url"] = (
                base_url.strip().rstrip("/")
            )
            changed_providers.add(provider_name_for_url)

    if not _provider_configured_with_updates(settings, provider_name, credential_updates):
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider_name}' has no configured API key or endpoint.",
        )

    configured_models = _available_models(request, provider_name)
    model_name = body.default_model.strip() if body.default_model else None
    if model_name and model_name not in configured_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not configured for provider '{provider_name}'.",
        )
    selected_model = model_name or settings.default_model
    if selected_model not in configured_models:
        selected_model = configured_models[0] if configured_models else None
    values: dict[str, object] = {
        "default_provider": provider_name,
        "default_model": selected_model,
        "request_timeout": (
            body.request_timeout
            if body.request_timeout is not None
            else settings.request_timeout
        ),
        "bridge_poll_interval": (
            body.bridge_poll_interval
            if body.bridge_poll_interval is not None
            else settings.bridge_poll_interval
        ),
        "ui_theme": body.ui_theme or settings.ui_theme,
    }
    values.update(credential_updates)
    try:
        database.save_settings(values)
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save local settings: {error}",
        ) from error

    settings.default_provider = provider_name
    settings.default_model = selected_model
    settings.request_timeout = values["request_timeout"]
    settings.bridge_poll_interval = values["bridge_poll_interval"]
    settings.ui_theme = values["ui_theme"]
    for key, value in credential_updates.items():
        setattr(settings, key, value)
    await registry.invalidate(changed_providers)
    LOGGER.info(
        "settings updated provider=%s api_keys_changed=%s",
        provider_name,
        ",".join(sorted(changed_providers)) or "none",
    )
    return _ui_status(request)


@router.post("/api/bridge", response_model=UIStatus, tags=["control-panel"])
async def update_ui_bridge(request: Request, body: UIBridgeRequest) -> UIStatus:
    controller = request.app.state.bridge_controller
    try:
        request.app.state.game_bridge_settings.set_enabled(body.enabled)
        request.app.state.database.save_settings({"bridge_required": body.enabled})
    except (OSError, PermissionError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save game bridge setting: {error}",
        ) from error
    await controller.set_enabled(body.enabled)
    LOGGER.info("bridge worker %s", "enabled" if body.enabled else "disabled")
    return _ui_status(request)


@router.post("/api/models/refresh", response_model=UIStatus, tags=["control-panel"])
async def refresh_ui_models(request: Request, body: UIModelRefreshRequest) -> UIStatus:
    settings = request.app.state.settings
    registry = _registry(request)
    provider_name = (body.provider or settings.default_provider).strip().lower()
    if provider_name not in registry.provider_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or disabled provider '{provider_name}'.",
        )
    await request.app.state.model_catalog.refresh_provider(provider_name)
    return _ui_status(request)


@router.get("/api/logs", response_model=UILogResponse, tags=["control-panel"])
async def ui_logs(
    request: Request, limit: int = DEFAULT_ACTIVITY_LIMIT
) -> UILogResponse:
    return UILogResponse(entries=request.app.state.database.recent_logs(limit))


@router.post(
    "/api/chat",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    tags=["control-panel"],
)
async def control_panel_chat(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Run a direct provider chat for the native control panel.

    This intentionally bypasses the Project Hoomans bridge requirement so the
    provider can be tested before the game is running. Game traffic continues
    to use the bridge-gated ``/v1/chat/completions`` route.
    """

    registry = _registry(request)
    provider_name, model_name = registry.resolve(body.provider, body.model)
    provider_request = body.model_copy(update={"model": model_name, "stream": False})
    LOGGER.info("control panel chat provider=%s model=%s", provider_name, model_name)
    result = await registry.complete(provider_name, provider_request)
    return _completion_response(result, model_name)
