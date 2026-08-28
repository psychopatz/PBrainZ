"""Game-facing OpenAI-compatible chat completion routes."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from pbrainz.api.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    CompletionChoice,
    CompletionMessage,
    DeltaMessage,
)
from pbrainz.branding import PRODUCT_NAME
from pbrainz.exceptions import ProviderError
from pbrainz.providers.base import TokenUsage
from pbrainz.providers.registry import ProviderRegistry

from .response_format import _completion_usage, _sse
from .route_support import _registry

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    tags=["chat"],
)
async def create_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse | StreamingResponse:
    registry = _registry(request)
    bridge = request.app.state.bridge.read()
    if request.app.state.settings.bridge_required and not bridge.ready:
        raise ProviderError(
            f"{PRODUCT_NAME} requires a READY PsychopatzCore bridge: {bridge.message}.",
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
                message=CompletionMessage(content=result.text, tool_calls=result.tool_calls),
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
