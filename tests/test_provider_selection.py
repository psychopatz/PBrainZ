from pbrainz.config import Settings
from pbrainz.providers.registry import ProviderRegistry


def test_new_settings_default_to_gemini_template_profile() -> None:
    assert Settings().default_provider == "gemini"


def test_provider_registry_uses_live_provider_and_model_selection() -> None:
    settings = Settings(
        enabled_providers="gemini,horde",
        default_provider="gemini",
        default_model="gemini-2.5-flash",
        gemini_api_key="test-key",
        gemini_models="gemini-2.5-flash,gemini-2.0-flash",
        horde_models="aphrodite/uncensored-chat",
    )
    registry = ProviderRegistry(settings)

    assert registry.resolve(None, "default") == ("gemini", "gemini-2.5-flash")

    settings.default_provider = "horde"
    settings.default_model = "aphrodite/uncensored-chat"
    assert registry.resolve(None, "default") == ("horde", "aphrodite/uncensored-chat")
