"""Internal provider contract and provider-neutral completion values."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pbrainz.api.models import ChatCompletionRequest


@dataclass(slots=True)
class TokenUsage:
    """Provider-neutral token usage."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    thinking_tokens: int | None = None

    def as_dict(self) -> dict[str, int] | None:
        values = {
            key: value
            for key, value in {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "thinking_tokens": self.thinking_tokens,
            }.items()
            if value is not None
        }
        return values or None


@dataclass(slots=True)
class CompletionResult:
    """Normalized full completion returned by a provider."""

    model: str
    text: str
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Only provider-returned reasoning metadata is retained. The service never
    # asks a model to disclose hidden chain-of-thought.
    reasoning: str | None = None


@dataclass(slots=True)
class StreamEvent:
    """Normalized piece of a streaming provider response."""

    text: str = ""
    role: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class LLMProvider(ABC):
    """The only interface the HTTP/API layer needs from a provider."""

    name: str

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        """Generate a complete response."""

    @abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        """Generate a response as incremental events."""
        if False:  # pragma: no cover - keeps this abstract method an async generator.
            yield StreamEvent()

    async def list_models(self) -> list[str]:
        """Return models supported by the provider, when its API exposes a catalog."""
        return []

    async def close(self) -> None:
        """Release provider resources, if the adapter owns any."""
