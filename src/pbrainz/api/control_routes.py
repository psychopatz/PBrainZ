"""Native control-panel routes for settings, bridge control, and diagnostics."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from pbrainz.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    UIBridgeRequest,
    UILogResponse,
    UIModelRefreshRequest,
    UISettingsRequest,
    UIStatus,
    UITTSSettingsRequest,
    UITTSTestRequest,
    UITTSVoiceInstallRequest,
    UITTSVoicePreviewRequest,
    UITTSVoiceUninstallRequest,
)
from pbrainz.config import OPENAI_COMPATIBLE_PROVIDERS
from pbrainz.database import DEFAULT_ACTIVITY_LIMIT, SettingsDatabase
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.tts import TTSException, TTSService

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
    tts_fields = (
        "tts_synthesis_workers",
        "tts_model_cache_size",
        "tts_max_simultaneous_playback",
        "tts_max_generated_ahead",
        "tts_max_tts_ready_ahead",
        "tts_natural_gap_ms",
        "tts_synthesis_timeout",
        "tts_audio_buffer_ms",
    )
    tts_reconfigure_required = False
    for field_name in tts_fields:
        value = getattr(body, field_name)
        if value is not None:
            values[field_name] = value
            setattr(settings, field_name, value)
            tts_reconfigure_required = True
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
    if tts_reconfigure_required:
        await request.app.state.tts.reconfigure()
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


@router.get("/api/tts", tags=["tts"])
async def tts_status(request: Request, refresh_catalog: bool = False) -> dict[str, object]:
    """Return local Piper/catalog/playback diagnostics for the native panel."""

    service: TTSService = request.app.state.tts
    await service.refresh_catalog(force=refresh_catalog)
    return {"status": "ok", **service.status()}


@router.post("/api/tts/settings", tags=["tts"])
async def update_tts_settings(request: Request, body: UITTSSettingsRequest) -> dict[str, object]:
    settings = request.app.state.settings
    database: SettingsDatabase = request.app.state.database
    service: TTSService = request.app.state.tts
    field_map = {
        "enabled": "tts_enabled",
        "piper_executable": "tts_piper_executable",
        "model_root": "tts_model_root",
        "metadata_path": "tts_metadata_path",
        "voice_catalog_url": "tts_voice_catalog_url",
        "voice_catalog_ttl_seconds": "tts_voice_catalog_ttl_seconds",
        "catalog_language": "tts_voice_catalog_language",
        "output_device": "tts_output_device",
        "master_volume": "tts_master_volume",
        "synthesis_workers": "tts_synthesis_workers",
        "model_cache_size": "tts_model_cache_size",
        "max_simultaneous_playback": "tts_max_simultaneous_playback",
        "max_generated_ahead": "tts_max_generated_ahead",
        "max_tts_ready_ahead": "tts_max_tts_ready_ahead",
        "natural_gap_ms": "tts_natural_gap_ms",
        "synthesis_timeout": "tts_synthesis_timeout",
        "audio_buffer_ms": "tts_audio_buffer_ms",
    }
    values: dict[str, object] = {}
    supplied = body.model_dump(exclude_none=True)
    reconfigure_required = False
    for api_name, setting_name in field_map.items():
        if api_name in supplied:
            values[setting_name] = supplied[api_name]
            setattr(settings, setting_name, supplied[api_name])
            reconfigure_required = True
    if body.voice_presets is not None:
        service.presets.replace(preset.model_dump() for preset in body.voice_presets)
        values["tts_voice_presets_json"] = settings.tts_voice_presets_json
    try:
        database.save_settings(values)
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500, detail=f"Could not save TTS settings: {error}"
        ) from error
    if reconfigure_required or body.voice_presets is not None:
        await service.reconfigure()
    LOGGER.info("TTS settings updated enabled=%s", settings.tts_enabled)
    return {"status": "ok", **service.status()}


@router.post("/api/tts/voices/install", tags=["tts"])
async def install_tts_voice(request: Request, body: UITTSVoiceInstallRequest) -> dict[str, object]:
    """Start one allow-listed voice installation from the configured catalog."""

    service: TTSService = request.app.state.tts
    await service.refresh_catalog()
    try:
        install = service.start_voice_install(body.voice_model_id)
    except TTSException as error:
        status_code = 404 if "not in the catalog" in str(error) else 409
        service.last_error = str(error)[:500]
        raise HTTPException(status_code=status_code, detail=service.last_error) from error
    return {
        "status": "ok",
        "accepted": install.get("state") in {"queued", "installing", "complete"},
        "installed": install.get("state") == "complete",
        "install": install,
        "message": (
            "Voice installation already in progress."
            if install.get("already_running")
            else "Voice installation started."
        ),
        **service.status(),
    }


@router.get("/api/tts/voices/install/{job_id}", tags=["tts"])
async def tts_voice_install_status(request: Request, job_id: str) -> dict[str, object]:
    """Return progress for one background Piper voice installation."""

    service: TTSService = request.app.state.tts
    try:
        install = service.install_status(job_id)
    except TTSException as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"status": "ok", "install": install, **service.status()}


@router.post("/api/tts/voices/uninstall", tags=["tts"])
async def uninstall_tts_voice(
    request: Request, body: UITTSVoiceUninstallRequest
) -> dict[str, object]:
    """Remove one installed Piper voice and clear any preset references."""

    service: TTSService = request.app.state.tts
    database: SettingsDatabase = request.app.state.database
    await service.refresh_catalog()
    try:
        result = await service.uninstall_voice(body.voice_model_id)
        database.save_settings(
            {"tts_voice_presets_json": service.settings.tts_voice_presets_json}
        )
    except TTSException as error:
        status_code = 404 if "not in the catalog" in str(error) else 409
        service.last_error = str(error)[:500]
        raise HTTPException(status_code=status_code, detail=service.last_error) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500, detail=f"Could not save cleared TTS presets: {error}"
        ) from error
    cleared_slots = result["cleared_preset_slots"]
    return {
        "status": "ok",
        "uninstalled": True,
        "voice": result["voice"],
        "cleared_preset_slots": cleared_slots,
        "message": (
            f"Uninstalled {result['voice'].get('display_name') or body.voice_model_id}."
            + (f" Cleared presets: {', '.join(cleared_slots)}." if cleared_slots else "")
        ),
        **service.status(),
    }


@router.post("/api/tts/voices/preview", tags=["tts"])
async def preview_tts_voice(request: Request, body: UITTSVoicePreviewRequest) -> dict[str, object]:
    """Play Piper's pre-generated sample without installing the model."""

    service: TTSService = request.app.state.tts
    await service.refresh_catalog()
    try:
        model = await service.preview_voice(body.voice_model_id)
    except TTSException as error:
        status_code = 404 if "not in the catalog" in str(error) else 409
        service.last_error = str(error)[:500]
        raise HTTPException(status_code=status_code, detail=service.last_error) from error
    return {
        "status": "ok",
        "accepted": True,
        "voice": model.as_dict(include_download=True),
        "message": f"Playing the {model.display_name} voice sample.",
    }


@router.post("/api/tts/test", tags=["tts"])
async def test_tts(request: Request, body: UITTSTestRequest) -> dict[str, object]:
    service: TTSService = request.app.state.tts
    accepted = await service.test_voice(body.slot, body.text)
    if not accepted:
        raise HTTPException(
            status_code=409, detail=service.last_error or "TTS test was not accepted"
        )
    return {"status": "ok", "accepted": True, "message": "Voice test queued."}


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
    LOGGER.info(
        "control panel inference completed provider=%s model=%s response_chars=%s",
        provider_name,
        model_name,
        len(result.text),
    )
    tts: TTSService | None = getattr(request.app.state, "tts", None)
    if tts is not None and tts.enabled:
        try:
            accepted = await tts.speak_text(result.text, source="control-panel")
        except Exception as error:  # TTS must never make a completed inference fail.
            accepted = False
            LOGGER.warning("control panel TTS autoplay failed: %s", error)
        LOGGER.info(
            "control panel TTS autoplay %s response_chars=%s",
            "queued" if accepted else "skipped",
            len(result.text),
        )
    return _completion_response(result, model_name)
