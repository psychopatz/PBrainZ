"""Native control-panel routes for settings, bridge control, and diagnostics."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from pbrainz.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    MemoryDeleteRequest,
    MockChatRequest,
    MockChatSeedRequest,
    UIBridgeRequest,
    UIDebugTraceSettingsRequest,
    UILogResponse,
    UIModelRefreshRequest,
    UIRetrievalDictionarySaveRequest,
    UISettingsRequest,
    UIStatus,
    UITemplateModelAddRequest,
    UITemplateProfileActionRequest,
    UITemplateProfileSaveRequest,
    UITTSSettingsRequest,
    UITTSTestRequest,
    UITTSVoiceInstallRequest,
    UITTSVoicePreviewRequest,
    UITTSVoiceUninstallRequest,
)
from pbrainz.config import OPENAI_COMPATIBLE_PROVIDERS
from pbrainz.conversation_runtime import AudioPresentation
from pbrainz.conversation_service import ConversationRequest
from pbrainz.database import (
    DEFAULT_ACTIVITY_LIMIT,
    DEFAULT_TRACE_LIMIT,
    SettingsDatabase,
)
from pbrainz.memory import (
    DaySynopsis,
    MemoryEpisode,
    MemoryIdentity,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    MemoryVisibility,
    SQLiteMemoryStore,
    StructuredFact,
    list_memory_worlds,
    public_memory_world,
)
from pbrainz.paths import normalize_zomboid_path
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.retrieval_dictionary import load_retrieval_dictionary
from pbrainz.template_profiles import (
    active_template_profile,
    delete_template_profile,
    load_template_profiles,
    normalize_profile,
    profile_by_id,
    reset_template_profile,
    serialize_template_profiles,
    upsert_template_profile,
)
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
        "memory_recent_turns": (
            body.memory_recent_turns
            if body.memory_recent_turns is not None
            else settings.memory_recent_turns
        ),
        "ui_theme": body.ui_theme or settings.ui_theme,
    }
    path_changed = False
    if body.zomboid_path is not None:
        normalized_path = str(normalize_zomboid_path(body.zomboid_path))
        values["zomboid_path"] = normalized_path
        path_changed = normalized_path != settings.zomboid_path
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
    settings.memory_recent_turns = values["memory_recent_turns"]
    conversation_service = getattr(request.app.state, "conversation_service", None)
    if conversation_service is not None:
        conversation_service.context_builder.set_recent_turn_limit(
            settings.memory_recent_turns
        )
    settings.ui_theme = values["ui_theme"]
    if path_changed:
        settings.zomboid_path = str(values["zomboid_path"])
        request.app.state.game_bridge_settings.set_zomboid_path(settings.zomboid_path)
        await request.app.state.bridge_controller.set_zomboid_path(settings.zomboid_path)
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


@router.post("/api/template-models", response_model=UIStatus, tags=["control-panel"])
async def add_template_model(
    request: Request, body: UITemplateModelAddRequest
) -> UIStatus:
    """Add a model ID to an enabled provider without contacting its API."""

    settings = request.app.state.settings
    registry: ProviderRegistry = _registry(request)
    database: SettingsDatabase = request.app.state.database
    provider_name = body.provider.strip().lower()
    model_name = body.model.strip()
    if provider_name not in registry.provider_names:
        enabled = ", ".join(registry.provider_names) or "none"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or disabled provider '{provider_name}'. Enabled: {enabled}.",
        )
    if not model_name or any(char in model_name for char in ",\r\n"):
        raise HTTPException(
            status_code=400,
            detail="Model ID must be one non-empty value without commas or line breaks.",
        )
    setting_name = f"{provider_name}_models"
    configured_models = list(settings.models_for(provider_name))
    if model_name not in configured_models:
        configured_models.append(model_name)
        model_value = ",".join(configured_models)
        setattr(settings, setting_name, model_value)
        try:
            database.save_settings({setting_name: model_value})
        except (OSError, sqlite3.Error) as error:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save template model: {error}",
            ) from error
        LOGGER.info("template model added provider=%s model=%s", provider_name, model_name)
    return _ui_status(request)


@router.post("/api/template-profiles", response_model=UIStatus, tags=["control-panel"])
async def save_ui_template_profile(
    request: Request, body: UITemplateProfileSaveRequest
) -> UIStatus:
    """Persist one editable NPC prompt-template profile."""

    settings = request.app.state.settings
    database: SettingsDatabase = request.app.state.database
    try:
        profile = normalize_profile(body.profile.model_dump())
        profiles = upsert_template_profile(
            load_template_profiles(settings.template_profiles_json), profile
        )
        serialized = serialize_template_profiles(profiles)
        database.save_settings({"template_profiles_json": serialized})
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not save template profile: {error}",
        ) from error
    settings.template_profiles_json = serialized
    _apply_active_template_profile(request)
    LOGGER.info("template profile saved id=%s mode=%s", profile.id, profile.mode)
    return _ui_status(request)


@router.post("/api/retrieval-dictionary", response_model=UIStatus, tags=["control-panel"])
async def save_ui_retrieval_dictionary(
    request: Request, body: UIRetrievalDictionarySaveRequest
) -> UIStatus:
    """Persist and immediately activate the player-editable retrieval dictionary."""

    settings = request.app.state.settings
    try:
        dictionary = load_retrieval_dictionary(body.dictionary)
        serialized = dictionary.as_json()
        request.app.state.database.save_settings(
            {"retrieval_dictionary_json": serialized}
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not save retrieval dictionary: {error}",
        ) from error
    settings.retrieval_dictionary_json = serialized
    conversation_service = getattr(request.app.state, "conversation_service", None)
    if conversation_service is not None and hasattr(
        conversation_service, "set_retrieval_dictionary"
    ):
        conversation_service.set_retrieval_dictionary(serialized)
    LOGGER.info(
        "retrieval dictionary saved active_locale=%s locales=%s",
        dictionary.active_locale,
        len(dictionary.locales),
    )
    return _ui_status(request)


@router.post("/api/template-profiles/active", response_model=UIStatus, tags=["control-panel"])
async def activate_ui_template_profile(
    request: Request, body: UITemplateProfileActionRequest
) -> UIStatus:
    """Select the prompt-template profile used by new NPC turns."""

    settings = request.app.state.settings
    profiles = load_template_profiles(settings.template_profiles_json)
    profile = profile_by_id(profiles, body.profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Template profile was not found.")
    try:
        request.app.state.database.save_settings(
            {"active_template_profile_id": profile.id}
        )
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not activate template profile: {error}",
        ) from error
    settings.active_template_profile_id = profile.id
    _apply_active_template_profile(request)
    LOGGER.info("template profile activated id=%s", profile.id)
    return _ui_status(request)


@router.post("/api/template-profiles/delete", response_model=UIStatus, tags=["control-panel"])
async def delete_ui_template_profile(
    request: Request, body: UITemplateProfileActionRequest
) -> UIStatus:
    """Delete one custom profile and fall back safely when it was active."""

    settings = request.app.state.settings
    profiles = load_template_profiles(settings.template_profiles_json)
    try:
        profiles = delete_template_profile(profiles, body.profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Template profile was not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    active_id = settings.active_template_profile_id
    if not profile_by_id(profiles, active_id):
        active_id = profiles[0].id
    serialized = serialize_template_profiles(profiles)
    try:
        request.app.state.database.save_settings(
            {
                "template_profiles_json": serialized,
                "active_template_profile_id": active_id,
            }
        )
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete template profile: {error}",
        ) from error
    settings.template_profiles_json = serialized
    settings.active_template_profile_id = active_id
    _apply_active_template_profile(request)
    LOGGER.info("template profile deleted id=%s", body.profile_id)
    return _ui_status(request)


@router.post("/api/template-profiles/reset", response_model=UIStatus, tags=["control-panel"])
async def reset_ui_template_profile(
    request: Request, body: UITemplateProfileActionRequest
) -> UIStatus:
    """Restore a shipped profile while keeping its active selection."""

    settings = request.app.state.settings
    profiles = load_template_profiles(settings.template_profiles_json)
    try:
        profiles = reset_template_profile(profiles, body.profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Template profile was not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    serialized = serialize_template_profiles(profiles)
    try:
        request.app.state.database.save_settings({"template_profiles_json": serialized})
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not reset template profile: {error}",
        ) from error
    settings.template_profiles_json = serialized
    _apply_active_template_profile(request)
    LOGGER.info("template profile reset id=%s", body.profile_id)
    return _ui_status(request)


def _apply_active_template_profile(request: Request) -> None:
    """Push a profile mutation into the live conversation service immediately."""

    service = getattr(request.app.state, "conversation_service", None)
    if service is None or not hasattr(service, "set_template_profile"):
        return
    settings = request.app.state.settings
    service.set_template_profile(
        active_template_profile(
            settings.template_profiles_json,
            settings.active_template_profile_id,
        )
    )


@router.get("/api/logs", response_model=UILogResponse, tags=["control-panel"])
async def ui_logs(
    request: Request, limit: int = DEFAULT_ACTIVITY_LIMIT
) -> UILogResponse:
    return UILogResponse(entries=request.app.state.database.recent_logs(limit))


@router.get("/api/debug/traces", tags=["debug"])
async def ui_debug_traces(
    request: Request,
    limit: int = DEFAULT_TRACE_LIMIT,
    request_id: str = "",
    search: str = "",
) -> dict[str, object]:
    """Return bounded local LLM traces; they are never save-scoped memory."""
    database: SettingsDatabase = request.app.state.database
    items = database.recent_llm_traces(limit, request_id=request_id, search=search)
    return {
        "status": "ok",
        "enabled": bool(request.app.state.settings.llm_trace_capture),
        "items": items,
        "total": len(items),
    }


@router.post("/api/debug/traces/settings", tags=["debug"])
async def update_ui_debug_trace_settings(
    request: Request, body: UIDebugTraceSettingsRequest
) -> dict[str, object]:
    """Toggle full prompt/context capture and clear retained data when disabling."""
    settings = request.app.state.settings
    database: SettingsDatabase = request.app.state.database
    settings.llm_trace_capture = body.enabled
    database.save_settings({"llm_trace_capture": body.enabled})
    if not body.enabled:
        database.clear_llm_traces()
    return {
        "status": "ok",
        "enabled": settings.llm_trace_capture,
        "items": database.recent_llm_traces(DEFAULT_TRACE_LIMIT),
    }


@router.post("/api/debug/traces/clear", tags=["debug"])
async def clear_ui_debug_traces(request: Request) -> dict[str, object]:
    deleted = request.app.state.database.clear_llm_traces()
    return {"status": "ok", "deleted": deleted}


def _memory_world_records(request: Request) -> list[dict[str, object]]:
    return list_memory_worlds(request.app.state.settings)


def _memory_worlds(request: Request) -> list[dict[str, object]]:
    return [public_memory_world(item) for item in _memory_world_records(request)]


def _memory_store(request: Request, world_uuid: str) -> SQLiteMemoryStore:
    item = next(
        (item for item in _memory_world_records(request) if item["world_uuid"] == world_uuid),
        None,
    )
    if item is None:
        raise HTTPException(status_code=404, detail=f"Memory world '{world_uuid}' was not found")
    return SQLiteMemoryStore(item["_root"], world_uuid)


def _active_memory_context(request: Request) -> dict[str, object]:
    cache = getattr(request.app.state, "active_memory_context", None)
    if cache is None or not hasattr(cache, "as_dict"):
        return {"status": "unavailable", "reason": "active_context_unconfigured"}
    active = dict(cache.as_dict())
    if active.get("status") != "active":
        return active
    world_uuid = str(active.get("world_uuid") or "")
    matching = next(
        (
            item
            for item in _memory_world_records(request)
            if str(item.get("world_uuid") or "") == world_uuid
        ),
        None,
    )
    active["exists"] = matching is not None
    active["storage_kind"] = (
        matching.get("storage_kind")
        if matching is not None
        else "save_local"
        if active.get("world_mode") == "singleplayer"
        else "external"
    )
    if matching is not None:
        for key in (
            "save_relative_path",
            "memory_count",
            "session_count",
            "turn_count",
            "episode_count",
            "fact_count",
        ):
            if matching.get(key) is not None:
                active[key] = matching[key]
    return active


@router.get("/api/memory/worlds", tags=["memory"])
async def ui_memory_worlds(request: Request) -> dict[str, object]:
    """List existing save-scoped memory databases without creating any."""

    return {"status": "ok", "worlds": _memory_worlds(request)}


@router.get("/api/memory/active", tags=["memory"])
async def ui_memory_active(request: Request) -> dict[str, object]:
    """Return the fresh save identity reported by the running game client."""

    return {"status": "ok", "active": _active_memory_context(request)}


@router.get("/api/memory", tags=["memory"])
async def ui_memory(
    request: Request,
    world_uuid: str | None = None,
    limit: int = 100,
    offset: int = 0,
    search: str = "",
    record_kind: Literal["all", "memory", "episode", "fact", "day_synopsis"] = "all",
) -> dict[str, object]:
    """Return a bounded, searchable view of reusable memory-layer records."""

    worlds = _memory_worlds(request)
    active = _active_memory_context(request)
    selected_world = world_uuid or (
        str(active["world_uuid"])
        if active.get("status") == "active" and active.get("world_uuid")
        else str(worlds[0]["world_uuid"]) if worlds else None
    )
    if selected_world is None:
        return {
            "status": "ok",
            "world_uuid": None,
            "items": [],
            "total": 0,
            "limit": max(1, min(int(limit), 200)),
            "offset": max(0, int(offset)),
            "worlds": worlds,
        }
    known_world = next(
        (item for item in worlds if str(item["world_uuid"]) == selected_world),
        None,
    )
    if known_world is None and active.get("status") == "active" and str(
        active.get("world_uuid")
    ) == selected_world:
        return {
            "status": "ok",
            "world_uuid": selected_world,
            "items": [],
            "total": 0,
            "limit": max(1, min(int(limit), 200)),
            "offset": max(0, int(offset)),
            "worlds": worlds,
        }
    store = _memory_store(request, selected_world)
    return {"status": "ok", **store.list_saved_memories(
        limit=limit,
        offset=offset,
        search=search,
        record_kind=record_kind,
    ), "worlds": worlds}


@router.post("/api/memory/delete", tags=["memory"])
async def ui_delete_memory(
    request: Request, body: MemoryDeleteRequest
) -> dict[str, object]:
    """Delete one explicitly selected durable memory-layer record."""

    store = _memory_store(request, body.world_uuid)
    try:
        deleted = store.delete_saved_memory(
            body.record_kind,
            body.record_id,
            player_uuid=body.player_uuid,
            npc_uuid=body.npc_uuid,
            game_day=body.game_day,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory record was not found")
    return {
        "status": "ok",
        "deleted": True,
        "record_kind": body.record_kind,
        "record_id": body.record_id,
        "worlds": _memory_worlds(request),
    }


@router.post("/api/mock-chat/seed", tags=["memory"])
async def seed_mock_memories(
    request: Request, body: MockChatSeedRequest
) -> dict[str, object]:
    """Create an idempotent fixture for testing the real memory retriever."""

    memory_identity = MemoryIdentity.from_mapping(body.world_uuid, body.model_dump())
    scope = MemoryScope(memory_identity.world_uuid, body.player_uuid, body.npc_uuid)
    store = SQLiteMemoryStore(
        memory_identity.root_for(request.app.state.settings),
        memory_identity.world_uuid,
    )
    participants = (body.player_uuid, body.npc_uuid)
    store.remember(
        MemoryRecord(
            memory_id="mock-memory-riverside",
            scope=scope,
            memory_type=MemoryType.FACT,
            content="The group agreed that the Riverside shelter is north of the gas station.",
            tags=("mock", "riverside", "shelter"),
            importance=0.9,
            provenance={"source": "mock_fixture"},
            visibility=MemoryVisibility.PUBLIC,
            game_day=body.game_day,
            participants=participants,
        )
    )
    store.remember(
        MemoryRecord(
            memory_id="mock-memory-gift",
            scope=scope,
            memory_type=MemoryType.SOCIAL_EVENT,
            content="The player gave Mock NPC a red scarf, and Mock NPC appreciated the gift.",
            tags=("mock", "gift", "scarf"),
            importance=0.75,
            provenance={"source": "mock_fixture"},
            visibility=MemoryVisibility.PUBLIC,
            game_day=body.game_day,
            participants=participants,
        )
    )
    store.remember_episode(
        MemoryEpisode(
            episode_id="mock-episode-riverside",
            scope=scope,
            conversation_id="mock-seed",
            game_day=body.game_day,
            participants=participants,
            witnesses=participants,
            topic_tags=("riverside", "shelter"),
            summary="The mock group discussed the Riverside shelter route.",
            key_facts=("The shelter is north of the gas station.",),
            importance=0.8,
            visibility=MemoryVisibility.PUBLIC,
        )
    )
    store.save_structured_fact(
        StructuredFact(
            fact_id="mock-fact-riverside",
            scope=scope,
            kind="LOCATION",
            content="Riverside shelter is north of the gas station.",
            source_uuid=body.player_uuid,
            truth_status="stated",
            visibility=MemoryVisibility.PUBLIC,
            game_day=body.game_day,
            importance=0.85,
            provenance={"source": "mock_fixture", "participants": list(participants)},
            conversation_id="mock-seed",
        )
    )
    store.save_day_synopsis(
        DaySynopsis(
            scope=scope,
            game_day=body.game_day,
            synopsis="The mock group discussed the Riverside shelter.",
            commitments=("Travel north to Riverside",),
        )
    )
    return {
        "status": "ok",
        "seeded": True,
        "world_uuid": memory_identity.world_uuid,
        "stats": store.stats(),
        "worlds": _memory_worlds(request),
    }


@router.post("/api/mock-chat", tags=["memory"])
async def mock_chat(request: Request, body: MockChatRequest) -> dict[str, object]:
    """Run an isolated panel chat through ConversationService and memory RAG."""

    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Conversation service is not ready")
    participant_ids = {body.player_uuid, body.npc_uuid}
    participants = [dict(item) for item in body.participants]
    participant_ids.update(str(item.get("id")) for item in participants if item.get("id"))
    participants.extend(
        {"id": identity, "name": name, "kind": kind}
        for identity, name, kind in (
            (body.player_uuid, body.player_name, "player"),
            (body.npc_uuid, body.npc_name, "npc"),
        )
        if not any(str(item.get("id")) == identity for item in participants)
    )
    session_id = body.session_id or (
        f"mock-session:{body.world_uuid}:{body.player_uuid}:{body.npc_uuid}"
    )
    memory_identity = MemoryIdentity.from_mapping(
        body.world_uuid,
        {
            "world_mode": body.world_mode,
            "save_relative_path": body.save_relative_path,
            "server_instance_id": body.server_instance_id,
            "server_world_generation": body.server_world_generation,
        },
    )
    conversation = ConversationRequest(
        request_id=f"mock-chat:{uuid.uuid4().hex}",
        scope=MemoryScope(
            memory_identity.world_uuid,
            body.player_uuid,
            body.npc_uuid,
        ),
        session_id=session_id,
        message=body.message.strip(),
        memory_identity=memory_identity,
        npc_name=body.npc_name,
        player_name=body.player_name,
        character_card=body.character_card,
        relationship_snapshot=body.relationship_snapshot,
        preferences=body.preferences,
        current_state=body.current_state,
        scene=body.scene,
        participants=tuple(participants[:16]),
        current_topic=body.current_topic,
        mentioned_entities=tuple(body.mentioned_entities[:16]),
        game_day=body.game_day,
        world_age_hours=body.world_age_hours,
        available_tools=tuple(body.available_tools[:12]),
        provider=body.provider,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        reasoning_effort=body.reasoning_effort,
        metadata={"source": "pbrainz-mock-chat", "participant_ids": sorted(participant_ids)},
        end_session=body.end_session,
    )
    result = await service.complete(conversation)
    completion = _completion_response(result.completion, result.completion.model)
    response = completion.model_dump(exclude_none=True)
    response.update(
        {
            "status": "ok",
            "session_id": result.session_id,
            "response_text": result.completion.text,
            "retrieved_memories": [match.as_diagnostic() for match in result.retrieved_memories],
            "diagnostics": result.diagnostics,
        }
    )
    return response


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
        "ambient_volume": "tts_ambient_volume",
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


@router.post("/api/tts/defaults/install", tags=["tts"])
async def install_default_tts_voices(request: Request) -> dict[str, object]:
    """Start the default voice setup after the panel receives user consent."""

    service: TTSService = request.app.state.tts
    await service.refresh_catalog()
    try:
        install = service.start_default_install()
    except TTSException as error:
        service.last_error = str(error)[:500]
        raise HTTPException(status_code=409, detail=service.last_error) from error
    return {
        "status": "ok",
        "accepted": install.get("state") in {"preparing", "queued", "installing"},
        "install": install,
        "message": "Default Piper voice setup started.",
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
        model = await service.preview_voice(
            body.voice_model_id,
            AudioPresentation.from_mapping(body.model_dump()),
            ambient_volume=body.ambient_volume,
        )
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
    accepted = await service.test_voice(
        body.slot,
        body.text,
        AudioPresentation.from_mapping(body.model_dump()),
        ambient_volume=body.ambient_volume,
    )
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
    trace_id = f"control-chat:{uuid.uuid4().hex}"
    if request.app.state.settings.llm_trace_capture:
        request.app.state.database.add_llm_trace(
            source="pbrainz.control",
            phase="provider.request",
            request_id=trace_id,
            payload={
                "provider": provider_name,
                "model": model_name,
                **provider_request.model_dump(exclude_none=True),
            },
        )
    LOGGER.info("control panel chat provider=%s model=%s", provider_name, model_name)
    result = await registry.complete(provider_name, provider_request)
    if request.app.state.settings.llm_trace_capture:
        request.app.state.database.add_llm_trace(
            source="pbrainz.control",
            phase="provider.response",
            request_id=trace_id,
            payload={
                "model": result.model,
                "text": result.text,
                "finish_reason": result.finish_reason,
                "tool_calls": result.tool_calls,
                "reasoning": result.reasoning,
                "usage": result.usage.as_dict() if result.usage else None,
            },
        )
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
