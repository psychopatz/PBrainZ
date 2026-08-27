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
