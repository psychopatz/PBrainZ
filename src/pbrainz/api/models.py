"""Pydantic models for the OpenAI-compatible HTTP surface."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pbrainz.branding import PRODUCT_NAME


class ChatMessage(BaseModel):
    """A chat message compatible with the core Chat Completions shape."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions``.

    ``provider`` is a P BrainZ extension. Existing OpenAI clients can still
    call this endpoint because unknown client-side fields are not required.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream_options: dict[str, Any] | None = None


class CompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class CompletionChoice(BaseModel):
    index: int
    message: CompletionMessage
    finish_reason: str | None = None
    logprobs: Any | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None


class MockChatRequest(BaseModel):
    """A panel-only request that exercises the structured NPC/RAG pipeline."""

    provider: str | None = Field(default=None, min_length=1)
    model: str = Field(default="default", min_length=1)
    message: str = Field(min_length=1, max_length=12000)
    world_uuid: str = Field(default="pbrainz-mock-world", min_length=1, max_length=256)
    player_uuid: str = Field(default="mock-player", min_length=1, max_length=256)
    npc_uuid: str = Field(default="mock-npc", min_length=1, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    npc_name: str = Field(default="Mock NPC", max_length=128)
    player_name: str = Field(default="Mock Player", max_length=128)
    game_day: int | None = Field(default=1, ge=0)
    world_age_hours: float | None = Field(default=None, ge=0)
    current_topic: str | None = Field(default=None, max_length=256)
    mentioned_entities: list[str] = Field(default_factory=list, max_length=16)
    scene: dict[str, Any] = Field(default_factory=dict)
    character_card: dict[str, Any] = Field(default_factory=dict)
    relationship_snapshot: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    participants: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    available_tools: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    end_session: bool = False


class MemoryDeleteRequest(BaseModel):
    """Identify one durable memory record for the local memory browser."""

    world_uuid: str = Field(min_length=1, max_length=256)
    record_kind: Literal["memory", "episode", "fact", "day_synopsis"]
    record_id: str = Field(min_length=1, max_length=256)
    player_uuid: str | None = Field(default=None, max_length=256)
    npc_uuid: str | None = Field(default=None, max_length=256)
    game_day: int | None = Field(default=None, ge=0)


class MockChatSeedRequest(BaseModel):
    """Identity fields for the panel's deterministic mock-memory fixture."""

    world_uuid: str = Field(default="pbrainz-mock-world", min_length=1, max_length=256)
    player_uuid: str = Field(default="mock-player", min_length=1, max_length=256)
    npc_uuid: str = Field(default="mock-npc", min_length=1, max_length=256)
    game_day: int = Field(default=1, ge=0)


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: CompletionUsage | None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class BridgeStatus(BaseModel):
    available: bool
    enabled: bool
    ready: bool
    runtime_id: str | None = None
    protocol_version: int | None = None
    lifecycle: str | None = None
    authority: str | None = None
    transport: str | None = None
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    providers: list[str]
    bridge: BridgeStatus


class UISettingsRequest(BaseModel):
    """Settings and credentials editable from the local control panel."""

    default_provider: str | None = Field(default=None, min_length=1)
    default_model: str | None = Field(default=None, min_length=1)
    request_timeout: float | None = Field(default=None, gt=0, le=600)
    bridge_poll_interval: float | None = Field(default=None, gt=0.05, le=10)
    zomboid_path: str | None = Field(default=None, min_length=1)
    ui_theme: Literal["light", "dark"] | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_base_url: str | None = Field(default=None, min_length=1)
    ollama_api_key: str | None = None
    ollama_base_url: str | None = Field(default=None, min_length=1)
    lmstudio_api_key: str | None = None
    lmstudio_base_url: str | None = Field(default=None, min_length=1)
    custom_api_key: str | None = None
    custom_base_url: str | None = Field(default=None, min_length=1)
    horde_api_key: str | None = None
    horde_base_url: str | None = Field(default=None, min_length=1)
    clear_openai_api_key: bool = False
    clear_ollama_api_key: bool = False
    clear_lmstudio_api_key: bool = False
    clear_custom_api_key: bool = False
    clear_horde_api_key: bool = False
    clear_gemini_api_key: bool = False
    tts_synthesis_workers: int | None = Field(default=None, ge=1, le=4)
    tts_model_cache_size: int | None = Field(default=None, ge=1, le=16)
    tts_max_simultaneous_playback: int | None = Field(default=None, ge=1, le=8)
    tts_max_generated_ahead: int | None = Field(default=None, ge=1, le=16)
    tts_max_tts_ready_ahead: int | None = Field(default=None, ge=1, le=8)
    tts_natural_gap_ms: int | None = Field(default=None, ge=0, le=2000)
    tts_synthesis_timeout: float | None = Field(default=None, gt=0, le=300)
    tts_audio_buffer_ms: int | None = Field(default=None, ge=0, le=2000)


class UIBridgeRequest(BaseModel):
    """Toggle the game bridge setting and the P BrainZ worker together."""

    enabled: bool


class UIProviderStatus(BaseModel):
    name: str
    configured: bool
    base_url: str | None = None
    api_key: str | None = None
    api_key_hint: str | None = None
    models: list[str]
    model_source: str
    models_updated_at: str | None = None
    selected: bool


class UIStatus(BaseModel):
    """Full status payload used by the lightweight local control panel."""

    status: Literal["ok"] = "ok"
    service: str
    host: str
    port: int
    default_provider: str
    default_model: str | None = None
    request_timeout: float
    bridge_poll_interval: float
    zomboid_path: str
    ui_theme: Literal["light", "dark"] = "light"
    providers: list[UIProviderStatus]
    openai_base_url: str
    bridge: BridgeStatus
    game_bridge_setting_enabled: bool
    bridge_worker_enabled: bool
    bridge_worker_running: bool
    tts_synthesis_workers: int = 1
    tts_model_cache_size: int = 2
    tts_max_simultaneous_playback: int = 4
    tts_max_generated_ahead: int = 3
    tts_max_tts_ready_ahead: int = 1
    tts_natural_gap_ms: int = 180
    tts_synthesis_timeout: float = 45.0
    tts_audio_buffer_ms: int = 50


class UITTSVoicePreset(BaseModel):
    slot: str = Field(min_length=1, max_length=128)
    voice_model_id: str = Field(min_length=1, max_length=256)
    optional_speaker_id: int | None = None


class UITTSSettingsRequest(BaseModel):
    """Local TTS controls kept separate from the provider settings tab."""

    enabled: bool | None = None
    piper_executable: str | None = Field(default=None, min_length=1)
    model_root: str | None = None
    metadata_path: str | None = None
    voice_catalog_url: str | None = Field(default=None, min_length=1)
    voice_catalog_ttl_seconds: int | None = Field(default=None, ge=60, le=2592000)
    catalog_language: str | None = Field(default=None, min_length=1, max_length=64)
    output_device: str | None = None
    master_volume: float | None = Field(default=None, ge=0, le=1)
    synthesis_workers: int | None = Field(default=None, ge=1, le=4)
    model_cache_size: int | None = Field(default=None, ge=1, le=16)
    max_simultaneous_playback: int | None = Field(default=None, ge=1, le=8)
    max_generated_ahead: int | None = Field(default=None, ge=1, le=16)
    max_tts_ready_ahead: int | None = Field(default=None, ge=1, le=8)
    natural_gap_ms: int | None = Field(default=None, ge=0, le=2000)
    synthesis_timeout: float | None = Field(default=None, gt=0, le=300)
    audio_buffer_ms: int | None = Field(default=None, ge=0, le=2000)
    voice_presets: list[UITTSVoicePreset] | None = None


class UITTSVoiceInstallRequest(BaseModel):
    voice_model_id: str = Field(min_length=1, max_length=256)


class UITTSVoiceUninstallRequest(BaseModel):
    voice_model_id: str = Field(min_length=1, max_length=256)


class UITTSVoicePreviewRequest(BaseModel):
    voice_model_id: str = Field(min_length=1, max_length=256)


class UITTSTestRequest(BaseModel):
    slot: str = Field(min_length=1, max_length=128)
    text: str = Field(
        default=f"This is a {PRODUCT_NAME} Piper voice test.", min_length=1, max_length=1200
    )


class UIModelRefreshRequest(BaseModel):
    provider: str | None = Field(default=None, min_length=1)


class UILogEntry(BaseModel):
    created_at: str
    level: str
    source: str
    message: str


class UILogResponse(BaseModel):
    entries: list[UILogEntry]


class UIDebugTraceSettingsRequest(BaseModel):
    """Explicit opt-in switch for retaining full LLM diagnostics locally."""

    enabled: bool
