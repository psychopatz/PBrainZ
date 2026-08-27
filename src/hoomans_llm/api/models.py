"""Pydantic models for the OpenAI-compatible HTTP surface."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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

    ``provider`` is a HoomansLLM extension. Existing OpenAI clients can still
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
    clear_openai_api_key: bool = False
    clear_ollama_api_key: bool = False
    clear_lmstudio_api_key: bool = False
    clear_custom_api_key: bool = False
    clear_gemini_api_key: bool = False


class UIBridgeRequest(BaseModel):
    """Toggle the game bridge setting and the HoomansLLM worker together."""

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
    ui_theme: Literal["light", "dark"] = "light"
    providers: list[UIProviderStatus]
    openai_base_url: str
    bridge: BridgeStatus
    game_bridge_setting_enabled: bool
    bridge_worker_enabled: bool
    bridge_worker_running: bool


class UIModelRefreshRequest(BaseModel):
    provider: str | None = Field(default=None, min_length=1)


class UILogEntry(BaseModel):
    created_at: str
    level: str
    source: str
    message: str


class UILogResponse(BaseModel):
    entries: list[UILogEntry]
