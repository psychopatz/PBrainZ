import sys
from types import ModuleType, SimpleNamespace

import pytest

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.config import Settings
from hoomans_llm.exceptions import UnsupportedMessageError
from hoomans_llm.providers.gemini import GeminiProvider
from hoomans_llm.providers.openai_compatible import OpenAICompatibleProvider
from hoomans_llm.providers.registry import ProviderRegistry


class FakeOpenAICompletions:
    async def create(self, **params):
        if params["stream"]:

            async def events():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(role="assistant", content="Hello"),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(role=None, content=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

            return events()
        return SimpleNamespace(
            model=params["model"],
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Hello"), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions())

    async def close(self):
        pass


def test_openai_request_translation() -> None:
    request = ChatCompletionRequest(
        model="gpt-test",
        messages=[
            {"role": "system", "content": "You are an NPC."},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.2,
        max_completion_tokens=32,
    )

    provider = object.__new__(OpenAICompatibleProvider)
    params = provider._request_params(request, stream=False)

    assert params["model"] == "gpt-test"
    assert params["messages"][1]["content"] == "Hello"
    assert params["max_completion_tokens"] == 32
    assert params["stream"] is False


def test_openai_tool_calls_are_normalized() -> None:
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(
                    name="order_follow",
                    arguments='{"command_id":"follow"}',
                ),
            )
        ]
    )

    assert OpenAICompatibleProvider._tool_calls(message) == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "order_follow",
                "arguments": '{"command_id":"follow"}',
            },
        }
    ]


def test_gemini_text_content_rejects_non_text() -> None:
    request = ChatCompletionRequest(
        model="gemini-test",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ],
    )

    with pytest.raises(UnsupportedMessageError, match="text-only"):
        GeminiProvider._text_content(request.messages[0])


def test_gemini_provider_error_keeps_actionable_sdk_detail() -> None:
    error = GeminiProvider._provider_exception(
        RuntimeError("400 INVALID_ARGUMENT: model does not support this request")
    )

    assert "RuntimeError" in error.message
    assert "model does not support this request" in error.message


def test_gemini_translates_function_tools_and_normalizes_calls() -> None:
    from google.genai import types

    request = ChatCompletionRequest(
        model="gemini-test",
        messages=[{"role": "user", "content": "Follow me."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "order_follow",
                    "description": "Request follow.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command_id": {"type": "string"}},
                    },
                },
            }
        ],
    )

    _, config = GeminiProvider._contents_and_config(request)
    assert config.tools[0].function_declarations[0].name == "order_follow"
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            functionCall=types.FunctionCall(
                                name="order_follow",
                                args={"command_id": "follow"},
                            )
                        )
                    ],
                )
            )
        ]
    )
    assert GeminiProvider._tool_calls(response)[0]["function"]["name"] == "order_follow"


def test_gemini_sanitizes_invalid_tool_schema_without_mutating_request() -> None:
    from google.genai import types

    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "depth-limit", "depth-limit": 4},
            "command_id": {"type": "string", "enum": ["follow"]},
        },
        "required": ["command_id", "unknown_field"],
        "additionalProperties": False,
    }
    request = ChatCompletionRequest(
        model="gemini-test",
        messages=[{"role": "user", "content": "Follow me."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "order_follow",
                    "parameters": parameters,
                },
            }
        ],
    )

    _, config = GeminiProvider._contents_and_config(request)
    schema = config.tools[0].function_declarations[0].parameters_json_schema

    assert schema["properties"]["kind"] == {"type": "string"}
    assert schema["properties"]["command_id"]["enum"] == ["follow"]
    assert schema["required"] == ["command_id"]
    assert "additionalProperties" not in schema
    assert "depth-limit" not in str(schema)
    assert parameters["properties"]["kind"]["type"] == "depth-limit"
    assert isinstance(config, types.GenerateContentConfig)


def test_default_model_uses_configured_gemini_when_openai_key_is_absent() -> None:
    registry = ProviderRegistry(
        Settings(
            default_provider="openai",
            enabled_providers="openai,gemini",
            openai_api_key=None,
            gemini_api_key="test-gemini",
        )
    )

    assert registry.resolve(None, "default") == ("gemini", "gemini-2.5-flash")


def test_openai_compatible_local_profiles_are_separate() -> None:
    settings = Settings(
        enabled_providers="openai,ollama,lmstudio,custom,gemini",
        openai_api_key="openai-key",
        ollama_models="llama3.2",
        lmstudio_models="qwen2.5",
        custom_base_url="http://127.0.0.1:9000/v1",
        custom_models="router-model",
    )
    registry = ProviderRegistry(settings)

    assert registry.provider_names == ("openai", "ollama", "lmstudio", "custom", "gemini")
    assert settings.models_for("ollama") == ("llama3.2",)
    assert settings.models_for("lmstudio") == ("qwen2.5",)
    assert settings.models_for("custom") == ("router-model",)
    assert settings.provider_configured("ollama") is True
    assert settings.provider_configured("lmstudio") is True
    assert settings.provider_configured("custom") is True
    assert registry.resolve("ollama", "llama3.2") == ("ollama", "llama3.2")


def test_default_request_prefers_hosted_credentials_over_idle_local_defaults() -> None:
    registry = ProviderRegistry(
        Settings(
            default_provider="openai",
            enabled_providers="openai,ollama,lmstudio,custom,gemini",
            gemini_api_key="test-gemini",
        )
    )

    assert registry.resolve(None, "default") == ("gemini", "gemini-2.5-flash")


def test_openai_compatible_adapter_reads_the_selected_profile(monkeypatch) -> None:
    class RecordingAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def close(self):
            pass

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = RecordingAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = OpenAICompatibleProvider(
        Settings(
            openai_api_key="cloud-key",
            ollama_api_key="ollama-key",
            ollama_base_url="http://localhost:11434/v1",
        ),
        provider_name="ollama",
    )

    assert provider.name == "ollama"
    assert provider._client.kwargs["api_key"] == "ollama-key"
    assert provider._client.kwargs["base_url"] == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_openai_adapter_normalizes_complete_and_streaming_responses() -> None:
    provider = object.__new__(OpenAICompatibleProvider)
    provider._client = FakeOpenAIClient()
    request = ChatCompletionRequest(
        model="gpt-test",
        messages=[{"role": "user", "content": "hello"}],
    )

    complete = await provider.complete(request)
    events = [event async for event in provider.stream(request)]

    assert complete.text == "Hello"
    assert complete.usage.as_dict() == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }
    assert "Hello" == "".join(event.text for event in events)
    assert events[0].finish_reason is None
    assert events[-1].finish_reason == "stop"
