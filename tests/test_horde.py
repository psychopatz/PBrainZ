from types import SimpleNamespace

import openai

from pbrainz.api.models import ChatCompletionRequest
from pbrainz.config import Settings, get_settings
from pbrainz.database import SettingsDatabase
from pbrainz.providers.openai_compatible import OpenAICompatibleProvider


def test_horde_is_enabled_and_works_without_a_personal_key() -> None:
    settings = Settings(enabled_providers="horde")

    assert settings.provider_names == ("horde",)
    assert settings.provider_configured("horde") is True
    assert settings.provider_explicitly_configured("horde") is False
    assert settings.base_url_for("horde") == "https://oai.aihorde.net/v1"


def test_horde_provider_uses_the_documented_anonymous_key(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    provider = OpenAICompatibleProvider(Settings(enabled_providers="horde"), "horde")

    assert provider.name == "horde"
    assert created["api_key"] == "0000000000"
    assert created["base_url"] == "https://oai.aihorde.net/v1"


def test_horde_model_listing_uses_the_openai_compatible_catalog(monkeypatch) -> None:
    class FakeModels:
        async def list(self):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id="aphrodite/uncensored-chat"),
                    SimpleNamespace(id="text-embedding-3-small"),
                ]
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    provider = OpenAICompatibleProvider(Settings(enabled_providers="horde"), "horde")

    import asyncio

    assert asyncio.run(provider.list_models()) == ["aphrodite/uncensored-chat"]


def test_horde_request_uses_only_the_gateway_supported_fields(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    provider = OpenAICompatibleProvider(Settings(enabled_providers="horde"), "horde")
    request = ChatCompletionRequest(
        model="aphrodite/uncensored-chat",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=240,
        stop="END",
        tools=[{"type": "function", "function": {"name": "unsupported"}}],
        metadata={"internal": True},
    )

    params = provider._request_params(request, stream=False)

    assert params["max_tokens"] == 240
    assert params["stop"] == ["END"]
    assert "max_completion_tokens" not in params
    assert "tools" not in params
    assert "metadata" not in params


def test_provider_error_preserves_the_upstream_reason() -> None:
    error = SimpleNamespace(
        status_code=400,
        body={"error": {"message": "model is not currently available"}},
    )

    result = OpenAICompatibleProvider._provider_exception(error)

    assert result.code == "provider_bad_request"
    assert "HTTP" not in result.message
    assert "model is not currently available" in result.message


def test_existing_saved_provider_list_is_migrated_to_include_horde(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "settings.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    database.save_settings({"enabled_providers": "openai,ollama,lmstudio,gemini"})
    monkeypatch.setenv("PBRAINZ_DB", str(database_path))
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert "horde" in settings.provider_names
        assert "horde" in database.load_settings()["enabled_providers"]
    finally:
        get_settings.cache_clear()
