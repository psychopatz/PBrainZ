from types import SimpleNamespace

import pytest

from pbrainz.api.models import ChatCompletionRequest, ChatMessage
from pbrainz.providers.gemini import GeminiProvider


def test_gemini_schema_replaces_legacy_bridge_depth_placeholders() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "[depth-limit]"},
            "intensity": {"type": "[unsupported]"},
        },
        "additionalProperties": False,
    }

    sanitized = GeminiProvider._sanitize_schema(schema)

    assert sanitized == {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "intensity": {"type": "string"},
        },
    }


@pytest.mark.asyncio
async def test_gemini_retries_retryable_model_failure_with_configured_fallback() -> None:
    calls: list[str] = []

    class ServerError(Exception):
        code = 500

    class Models:
        async def generate_content(self, *, model, contents, config):
            del contents, config
            calls.append(model)
            if model == "gemma-4-31b-it":
                raise ServerError("internal error")
            return SimpleNamespace(text="OK", candidates=[], usage_metadata=None)

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = SimpleNamespace(models=Models())
    provider._fallback_models = ("gemini-2.5-flash",)
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemma-4-31b-it",
        messages=[ChatMessage(role="user", content="Reply with OK")],
    )

    result = await provider.complete(request)

    assert result.text == "OK"
    assert result.model == "gemini-2.5-flash"
    assert calls == ["gemma-4-31b-it", "gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_gemini_retries_empty_model_response_with_configured_fallback() -> None:
    calls: list[str] = []

    class Models:
        async def generate_content(self, *, model, contents, config):
            del contents, config
            calls.append(model)
            if model == "gemma-4-31b-it":
                return SimpleNamespace(text="", candidates=[], usage_metadata=None)
            return SimpleNamespace(text="OK", candidates=[], usage_metadata=None)

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = SimpleNamespace(models=Models())
    provider._fallback_models = ("gemini-2.5-flash",)
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemma-4-31b-it",
        messages=[ChatMessage(role="user", content="Reply with OK")],
    )

    result = await provider.complete(request)

    assert result.text == "OK"
    assert result.model == "gemini-2.5-flash"
    assert calls == ["gemma-4-31b-it", "gemini-2.5-flash"]
