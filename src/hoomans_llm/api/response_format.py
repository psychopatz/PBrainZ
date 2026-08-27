"""Provider-neutral response formatting for the HTTP API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from hoomans_llm.api.models import (
    ChatCompletionResponse,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
)
from hoomans_llm.providers.base import TokenUsage


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
                message=CompletionMessage(content=result.text, tool_calls=result.tool_calls),
                finish_reason=result.finish_reason,
            )
        ],
        usage=_completion_usage(result.usage),
    )
