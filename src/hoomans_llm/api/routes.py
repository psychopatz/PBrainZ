"""HTTP routes for the HoomansLLM gateway."""

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from hoomans_llm.api.models import (
    BridgeStatus,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    DeltaMessage,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    UIBridgeRequest,
    UILogResponse,
    UIModelRefreshRequest,
    UIProviderStatus,
    UISettingsRequest,
    UIStatus,
)
from hoomans_llm.config import (
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_DEFAULT_BASE_URLS,
    Settings,
)
from hoomans_llm.database import SettingsDatabase
from hoomans_llm.exceptions import ProviderError
from hoomans_llm.providers.base import TokenUsage
from hoomans_llm.providers.registry import ProviderRegistry

router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _registry(request: Request) -> ProviderRegistry:
    return request.app.state.providers


def _bridge_status(request: Request) -> BridgeStatus:
    return BridgeStatus(**request.app.state.bridge.read().as_dict())


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    registry = _registry(request)
    return HealthResponse(
        service=request.app.title,
        providers=list(registry.provider_names),
        bridge=_bridge_status(request),
    )


@router.get("/api/settings", response_model=UIStatus, tags=["control-panel"])
async def ui_status(request: Request) -> UIStatus:
    return _ui_status(request)


@router.post("/api/settings", response_model=UIStatus, tags=["control-panel"])
async def update_ui_settings(request: Request, body: UISettingsRequest) -> UIStatus:
    settings = request.app.state.settings
    registry = _registry(request)
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
        "request_timeout": body.request_timeout
        if body.request_timeout is not None
        else settings.request_timeout,
        "bridge_poll_interval": body.bridge_poll_interval
        if body.bridge_poll_interval is not None
        else settings.bridge_poll_interval,
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
        request.app.state.database.save_settings({"bridge_required": body.enabled})
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save bridge setting: {error}",
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
async def ui_logs(request: Request, limit: int = 100) -> UILogResponse:
    return UILogResponse(
        entries=request.app.state.database.recent_logs(limit),
    )


@router.get("/v1/models", response_model=ModelListResponse, tags=["models"])
async def list_models(request: Request) -> ModelListResponse:
    registry = _registry(request)
    return ModelListResponse(
        data=[
            ModelInfo(id=model_id, owned_by=provider_name)
            for provider_name, model_id in registry.model_ids()
        ]
    )


@router.post("/api/chat", response_model=ChatCompletionResponse, tags=["control-panel"])
async def control_panel_chat(request: Request, body: ChatCompletionRequest):
    """Run a direct provider chat for the native control panel.

    This intentionally bypasses the Project Hoomans bridge requirement so the
    provider can be tested before the game is running. Game traffic continues
    to use the bridge-gated ``/v1/chat/completions`` route below.
    """
    registry = _registry(request)
    provider_name, model_name = registry.resolve(body.provider, body.model)
    provider_request = body.model_copy(update={"model": model_name, "stream": False})
    LOGGER.info("control panel chat provider=%s model=%s", provider_name, model_name)
    result = await registry.complete(provider_name, provider_request)
    return _completion_response(result, model_name)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"])
async def create_chat_completion(request: Request, body: ChatCompletionRequest):
    registry = _registry(request)
    bridge = request.app.state.bridge.read()
    if request.app.state.settings.bridge_required and not bridge.ready:
        raise ProviderError(
            f"HoomansLLM requires a READY PsychopatzCore bridge: {bridge.message}.",
            status_code=503,
            code="bridge_unavailable",
        )
    claimed_runtime_id = (body.metadata or {}).get("bridge_runtime_id")
    if claimed_runtime_id and claimed_runtime_id != bridge.runtime_id:
        raise ProviderError(
            "The request targets a different Project Zomboid bridge runtime.",
            status_code=409,
            code="stale_runtime",
        )
    provider_name, model_name = registry.resolve(body.provider, body.model)
    provider_request = body.model_copy(update={"model": model_name})
    LOGGER.info(
        "chat request provider=%s model=%s stream=%s",
        provider_name,
        model_name,
        body.stream,
    )
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _stream_response(
                registry,
                provider_name,
                provider_request,
                completion_id=completion_id,
                created=created,
                response_model=model_name,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    result = await registry.complete(provider_name, provider_request)
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=result.model or model_name,
        choices=[
            CompletionChoice(
                index=0,
                message=CompletionMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=_completion_usage(result.usage),
    )


async def _stream_response(
    registry: ProviderRegistry,
    provider_name: str,
    request: ChatCompletionRequest,
    *,
    completion_id: str,
    created: int,
    response_model: str,
) -> AsyncIterator[str]:
    """Encode provider-neutral events as OpenAI-compatible SSE chunks."""
    yield _sse(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=response_model,
            choices=[ChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )
    )
    last_finish_reason: str | None = None
    last_usage: TokenUsage | None = None
    try:
        async for event in registry.stream_events(provider_name, request):
            last_finish_reason = event.finish_reason or last_finish_reason
            last_usage = event.usage or last_usage
            if event.text or event.role or event.finish_reason:
                yield _sse(
                    ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=response_model,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role=event.role if event.role == "assistant" else None,
                                    content=event.text or None,
                                ),
                                finish_reason=event.finish_reason,
                            )
                        ],
                        usage=_completion_usage(event.usage),
                    )
                )
    except Exception as exc:
        # The stream has already started, so the error must be represented as
        # an SSE data item instead of changing the HTTP status code.
        yield _sse({"error": {"message": str(exc), "type": "provider_error"}})
    if last_finish_reason is None:
        last_finish_reason = "stop"
    yield _sse(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=response_model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=last_finish_reason,
                )
            ],
            usage=_completion_usage(last_usage),
        )
    )
    yield "data: [DONE]\n\n"


def _sse(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    return f"data: {json.dumps(value, separators=(',', ':'))}\n\n"


def _completion_usage(usage: TokenUsage | None) -> CompletionUsage | None:
    values = usage.as_dict() if usage else None
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not values or not all(key in values for key in required):
        return None
    return CompletionUsage(**values)


def _completion_response(
    result: Any,
    model_name: str,
    *,
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=completion_id or f"chatcmpl_{uuid.uuid4().hex}",
        created=created or int(time.time()),
        model=result.model or model_name,
        choices=[
            CompletionChoice(
                index=0,
                message=CompletionMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=_completion_usage(result.usage),
    )


def _ui_status(request: Request) -> UIStatus:
    settings = request.app.state.settings
    registry = _registry(request)
    database: SettingsDatabase = request.app.state.database
    controller_status = request.app.state.bridge_controller.as_dict()
    bridge = BridgeStatus(**controller_status["bridge"])
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
        ui_theme=settings.ui_theme,
        providers=provider_statuses,
        openai_base_url=settings.openai_base_url,
        bridge=bridge,
        bridge_worker_enabled=bool(controller_status["worker_enabled"]),
        bridge_worker_running=bool(controller_status["worker_running"]),
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
        if provider_name in {"ollama", "lmstudio"}:
            return bool(str(base_url).strip())
        return bool(
            api_key
            or str(base_url).rstrip("/") != PROVIDER_DEFAULT_BASE_URLS[provider_name]
        )
    if provider_name == "gemini":
        return bool(updates.get("gemini_api_key", settings.gemini_api_key))
    return False


def _api_key_hint(settings: Settings, provider_name: str) -> str | None:
    value = getattr(settings, f"{provider_name}_api_key", None)
    if not value:
        return None
    return f"••••{value[-4:]}" if len(value) >= 4 else "configured"


def _model_source(database: SettingsDatabase, provider_name: str) -> str:
    catalog = database.model_catalog(provider_name)
    return catalog[0]["source"] if catalog else "configuration"


def _models_updated_at(database: SettingsDatabase, provider_name: str) -> str | None:
    catalog = database.model_catalog(provider_name)
    timestamps = [row["updated_at"] for row in catalog if row.get("updated_at")]
    return max(timestamps) if timestamps else None
