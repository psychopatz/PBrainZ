"""Shared request-state access and control-panel status serialization."""

from __future__ import annotations

from fastapi import Request

from pbrainz.api.models import BridgeStatus, UIProviderStatus, UIStatus
from pbrainz.config import (
    NO_API_KEY_PROVIDERS,
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_DEFAULT_BASE_URLS,
    Settings,
)
from pbrainz.database import SettingsDatabase
from pbrainz.providers.registry import ProviderRegistry


def _registry(request: Request) -> ProviderRegistry:
    return request.app.state.providers


def _bridge_status(request: Request) -> BridgeStatus:
    return BridgeStatus(**request.app.state.bridge.read().as_dict())


def _ui_status(request: Request) -> UIStatus:
    settings = request.app.state.settings
    registry = _registry(request)
    database: SettingsDatabase = request.app.state.database
    controller_status = request.app.state.bridge_controller.as_dict()
    bridge = BridgeStatus(**controller_status["bridge"])
    game_bridge_setting_enabled = request.app.state.game_bridge_settings.read().enabled
    default_provider = settings.default_provider.strip().lower()
    configured_models = _available_models(request, default_provider)
    provider_statuses = [
        UIProviderStatus(
            name=provider_name,
            configured=settings.provider_configured(provider_name),
            base_url=(
                settings.base_url_for(provider_name)
                if provider_name in OPENAI_COMPATIBLE_PROVIDERS
                else None
            ),
            api_key=_api_key(settings, provider_name),
            api_key_hint=_api_key_hint(settings, provider_name),
            # The control panel displays only persisted discovery results. This
            # keeps an unrefreshed provider blank instead of showing another
            # provider's configured/default model.
            models=list(_cached_models(request, provider_name)),
            model_source=_model_source(database, provider_name),
            models_updated_at=_models_updated_at(database, provider_name),
            selected=provider_name == default_provider,
        )
        for provider_name in registry.provider_names
    ]
    return UIStatus(
        service=request.app.title,
        host=settings.host,
        port=settings.port,
        default_provider=default_provider,
        default_model=(
            settings.default_model
            if settings.default_model in configured_models
            else configured_models[0] if configured_models else None
        ),
        request_timeout=settings.request_timeout,
        bridge_poll_interval=settings.bridge_poll_interval,
        zomboid_path=settings.zomboid_path,
        ui_theme=settings.ui_theme,
        providers=provider_statuses,
        openai_base_url=settings.openai_base_url,
        bridge=bridge,
        game_bridge_setting_enabled=game_bridge_setting_enabled,
        bridge_worker_enabled=bool(controller_status["worker_enabled"]),
        bridge_worker_running=bool(controller_status["worker_running"]),
        tts_synthesis_workers=settings.tts_synthesis_workers,
        tts_model_cache_size=settings.tts_model_cache_size,
        tts_max_simultaneous_playback=settings.tts_max_simultaneous_playback,
        tts_max_generated_ahead=settings.tts_max_generated_ahead,
        tts_max_tts_ready_ahead=settings.tts_max_tts_ready_ahead,
        tts_natural_gap_ms=settings.tts_natural_gap_ms,
        tts_synthesis_timeout=settings.tts_synthesis_timeout,
        tts_audio_buffer_ms=settings.tts_audio_buffer_ms,
    )


def _available_models(request: Request, provider_name: str) -> tuple[str, ...]:
    settings = request.app.state.settings
    catalog = _catalog_rows(request, provider_name)
    if catalog:
        return tuple(row["model_id"] for row in catalog)
    return settings.models_for(provider_name)


def _cached_models(request: Request, provider_name: str) -> tuple[str, ...]:
    """Return only models discovered and persisted for this provider."""

    return tuple(row["model_id"] for row in _catalog_rows(request, provider_name))


def _catalog_rows(request: Request, provider_name: str) -> list[dict[str, str]]:
    return request.app.state.database.model_catalog(provider_name)


def _provider_configured_with_updates(
    settings: Settings,
    provider_name: str,
    updates: dict[str, object],
) -> bool:
    if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
        api_key = updates.get(
            f"{provider_name}_api_key",
            getattr(settings, f"{provider_name}_api_key"),
        )
        base_url = updates.get(
            f"{provider_name}_base_url",
            settings.base_url_for(provider_name),
        )
        if provider_name in NO_API_KEY_PROVIDERS:
            return bool(str(base_url).strip())
        return bool(
            api_key
            or str(base_url).rstrip("/") != PROVIDER_DEFAULT_BASE_URLS[provider_name]
        )
    if provider_name == "gemini":
        return bool(updates.get("gemini_api_key", settings.gemini_api_key))
    return False


def _api_key_hint(settings: Settings, provider_name: str) -> str | None:
    value = _api_key(settings, provider_name)
    if not value:
        return None
    return f"••••{value[-4:]}" if len(value) >= 4 else "configured"


def _api_key(settings: Settings, provider_name: str) -> str | None:
    value = getattr(settings, f"{provider_name}_api_key", None)
    return value if isinstance(value, str) and value else None


def _model_source(database: SettingsDatabase, provider_name: str) -> str:
    catalog = database.model_catalog(provider_name)
    return catalog[0]["source"] if catalog else "configuration"


def _models_updated_at(database: SettingsDatabase, provider_name: str) -> str | None:
    catalog = database.model_catalog(provider_name)
    timestamps = [row["updated_at"] for row in catalog if row.get("updated_at")]
    return max(timestamps) if timestamps else None
