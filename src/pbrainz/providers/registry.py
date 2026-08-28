"""Provider lifecycle and request routing."""

from collections.abc import AsyncIterator, Callable

from pbrainz.api.models import ChatCompletionRequest
from pbrainz.config import OPENAI_COMPATIBLE_PROVIDERS, Settings
from pbrainz.exceptions import ProviderError

from .base import CompletionResult, LLMProvider, StreamEvent
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    """Lazily constructs enabled providers and routes requests to one of them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        factories: dict[str, Callable[[], LLMProvider]] = {
            provider_name: lambda provider_name=provider_name: OpenAICompatibleProvider(
                settings, provider_name=provider_name
            )
            for provider_name in OPENAI_COMPATIBLE_PROVIDERS
        }
        factories["gemini"] = lambda: GeminiProvider(settings)
        self._factories = {
            name: factories[name] for name in settings.provider_names if name in factories
        }
        self._providers: dict[str, LLMProvider] = {}

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def resolve(self, requested_provider: str | None, model: str) -> tuple[str, str]:
        """Resolve a provider and strip an optional ``provider/model`` prefix."""
        if requested_provider:
            provider_name = requested_provider.lower()
            normalized_model = model
        else:
            provider_name, normalized_model = self._infer_from_model(model)
        if provider_name not in self._factories:
            if requested_provider is None and normalized_model.casefold() in {"default", "auto"}:
                provider_name = self._first_configured_provider()
            else:
                enabled = ", ".join(self._factories) or "none"
                raise ProviderError(
                    f"Unknown or disabled provider '{provider_name}'. "
                    f"Enabled providers: {enabled}.",
                    status_code=400,
                    code="unknown_provider",
                )
        if requested_provider is None and normalized_model.casefold() in {"default", "auto"}:
            provider_name = self._first_configured_provider(provider_name)
        if normalized_model.casefold() in {"default", "auto"}:
            configured_models = self.settings.models_for(provider_name)
            if configured_models:
                selected_model = self.settings.default_model
                normalized_model = (
                    selected_model if selected_model in configured_models else configured_models[0]
                )
        return provider_name, normalized_model

    async def complete(
        self, provider_name: str, request: ChatCompletionRequest
    ) -> CompletionResult:
        return await self._get(provider_name).complete(request)

    async def stream_events(
        self, provider_name: str, request: ChatCompletionRequest
    ) -> AsyncIterator[StreamEvent]:
        provider = self._get(provider_name)
        async for event in provider.stream(request):
            yield event

    def model_ids(self) -> list[tuple[str, str]]:
        """Return configured model IDs without making network calls."""
        return [
            (provider_name, model_id)
            for provider_name in self.provider_names
            for model_id in self.settings.models_for(provider_name)
        ]

    async def list_models(self, provider_name: str) -> list[str]:
        """Fetch a provider's current model catalog through its adapter."""
        return await self._get(provider_name).list_models()

    async def invalidate(self, provider_names: set[str]) -> None:
        """Close cached clients whose credentials or endpoint changed."""
        for provider_name in provider_names:
            provider = self._providers.pop(provider_name, None)
            if provider is not None:
                await provider.close()

    def _first_configured_provider(self, preferred: str | None = None) -> str:
        """Prefer the configured default, then any provider with credentials."""
        candidates = list(self._factories)
        if preferred in self._factories:
            if self.settings.provider_configured(preferred):
                return preferred
            candidates.remove(preferred)
            candidates.insert(0, preferred)
        for provider_name in candidates:
            if self.settings.provider_explicitly_configured(provider_name):
                return provider_name
        for provider_name in candidates:
            if self.settings.provider_configured(provider_name):
                return provider_name
        return preferred or (candidates[0] if candidates else "unknown")

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

    def _get(self, provider_name: str) -> LLMProvider:
        if provider_name not in self._factories:
            self.resolve(provider_name, "unused")
        if provider_name not in self._providers:
            self._providers[provider_name] = self._factories[provider_name]()
        return self._providers[provider_name]

    def _infer_from_model(self, model: str) -> tuple[str, str]:
        for provider_name in self._factories:
            for separator in ("/", ":"):
                prefix = f"{provider_name}{separator}"
                if model.lower().startswith(prefix):
                    return provider_name, model[len(prefix) :]
        return self.settings.default_provider.lower(), model
