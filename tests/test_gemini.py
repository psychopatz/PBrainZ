from types import SimpleNamespace

import pytest

from pbrainz.api.models import ChatCompletionRequest, ChatMessage
from pbrainz.exceptions import ProviderError
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
async def test_gemini_does_not_switch_models_after_model_failure() -> None:
    calls: list[str] = []

    class ServerError(Exception):
        code = 500

    class Models:
        async def generate_content(self, *, model, contents, config):
            del contents, config
            calls.append(model)
            raise ServerError("internal error")

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = SimpleNamespace(models=Models())
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemma-4-31b-it",
        messages=[ChatMessage(role="user", content="Reply with OK")],
    )

    with pytest.raises(ProviderError, match="gemma-4-31b-it"):
        await provider.complete(request)

    assert calls == ["gemma-4-31b-it"]


@pytest.mark.asyncio
async def test_gemini_keeps_selected_model_after_empty_model_response() -> None:
    calls: list[str] = []

    class Models:
        async def generate_content(self, *, model, contents, config):
            del contents, config
            calls.append(model)
            return SimpleNamespace(text="", candidates=[], usage_metadata=None)

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = SimpleNamespace(models=Models())
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemma-4-31b-it",
        messages=[ChatMessage(role="user", content="Reply with OK")],
    )

    result = await provider.complete(request)

    assert result.text == ""
    assert result.model == "gemma-4-31b-it"
    assert calls == ["gemma-4-31b-it"]


@pytest.mark.asyncio
async def test_gemini_keeps_selected_model_for_tool_turn() -> None:
    calls: list[str] = []

    class Models:
        async def generate_content(self, *, model, contents, config):
            del contents, config
            calls.append(model)
            return SimpleNamespace(text="I can do that.", candidates=[], usage_metadata=None)

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = SimpleNamespace(models=Models())
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemma-4-31b-it",
        messages=[ChatMessage(role="user", content="Follow me")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "order_follow",
                    "description": "Follow the player.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    result = await provider.complete(request)

    assert result.model == "gemma-4-31b-it"
    assert calls == ["gemma-4-31b-it"]


def test_gemini_translates_reasoning_effort_for_flash_models() -> None:
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[ChatMessage(role="user", content="Reply briefly")],
        reasoning_effort="low",
    )

    _contents, config = GeminiProvider._contents_and_config(
        request, model=request.model
    )

    assert config.thinking_config.thinking_budget == 256


def test_gemini_does_not_invent_a_thinking_budget() -> None:
    request = ChatCompletionRequest(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[ChatMessage(role="user", content="Reply briefly")],
    )

    _contents, config = GeminiProvider._contents_and_config(
        request, model=request.model
    )

    assert config is None
