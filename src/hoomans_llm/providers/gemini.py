"""Google Gemini provider adapter using the official ``google-genai`` SDK."""

from collections.abc import AsyncIterator
from typing import Any

from hoomans_llm.api.models import ChatCompletionRequest, ChatMessage
from hoomans_llm.config import Settings
from hoomans_llm.exceptions import (
    ProviderError,
    ProviderNotConfiguredError,
    UnsupportedMessageError,
)

from .base import CompletionResult, LLMProvider, StreamEvent, TokenUsage


class GeminiProvider(LLMProvider):
    """Translate the common chat contract to Gemini's content-generation API."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ProviderNotConfiguredError(self.name)

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared by the package.
            raise ProviderError(
                "The Gemini provider dependency is missing. "
                "Reinstall with the project dependencies.",
                status_code=503,
                code="missing_provider_dependency",
            ) from exc

        self._sync_client = genai.Client(api_key=settings.gemini_api_key)
        self._client = self._sync_client.aio

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        contents, config = self._contents_and_config(request)
        try:
            response = await self._client.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._provider_exception(exc) from exc

        return CompletionResult(
            model=request.model,
            text=getattr(response, "text", None) or "",
            finish_reason=self._finish_reason(response),
            usage=self._usage(getattr(response, "usage_metadata", None)),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        contents, config = self._contents_and_config(request)
        try:
            response_stream = await self._client.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config,
            )
            async for chunk in response_stream:
                yield StreamEvent(text=getattr(chunk, "text", None) or "")
        except Exception as exc:
            raise self._provider_exception(exc) from exc

    async def list_models(self) -> list[str]:
        """Read Gemini's current text-generation model catalog."""
        try:
            pager = await self._client.models.list(config={"page_size": 100})
            models: list[str] = []
            async for model in pager:
                actions = getattr(model, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                name = str(getattr(model, "name", "")).removeprefix("models/")
                if name:
                    models.append(name)
            return sorted(set(models))
        except Exception as exc:
            raise self._provider_exception(exc) from exc

    async def close(self) -> None:
        await self._client.aclose()
        self._sync_client.close()

    @classmethod
    def _contents_and_config(cls, request: ChatCompletionRequest) -> tuple[list[Any], Any]:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - checked during construction.
            raise ProviderError(
                "The Gemini provider dependency is missing.",
                status_code=503,
                code="missing_provider_dependency",
            ) from exc

        system_messages: list[str] = []
        contents: list[Any] = []
        for message in request.messages:
            text = cls._text_content(message)
            if message.role == "system":
                system_messages.append(text)
                continue
            if message.role not in {"user", "assistant"}:
                raise UnsupportedMessageError(
                    "Gemini adapter currently supports system, user, and assistant "
                    f"messages; got '{message.role}'."
                )
            role = "model" if message.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        config_values: dict[str, Any] = {}
        if system_messages:
            config_values["system_instruction"] = "\n\n".join(system_messages)
        for request_name, config_name in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stop", "stop_sequences"),
        ):
            value = getattr(request, request_name)
            if value is not None:
                if request_name == "stop" and isinstance(value, str):
                    value = [value]
                config_values[config_name] = value
        max_output_tokens = request.max_completion_tokens or request.max_tokens
        if max_output_tokens is not None:
            config_values["max_output_tokens"] = max_output_tokens
        config = types.GenerateContentConfig(**config_values) if config_values else None
        return contents, config

    @staticmethod
    def _text_content(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        if message.content is None:
            return ""
        parts: list[str] = []
        for part in message.content:
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise UnsupportedMessageError(
                    "Gemini adapter currently supports text-only message content."
                )
            parts.append(part["text"])
        return "".join(parts)

    @staticmethod
    def _usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    @staticmethod
    def _finish_reason(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "stop"
        reason = getattr(candidates[0], "finish_reason", None)
        value = getattr(reason, "name", None) or getattr(reason, "value", None) or str(reason)
        value = value.rsplit(".", 1)[-1].lower()
        return {"max_tokens": "length", "safety": "content_filter"}.get(value, value)

    @staticmethod
    def _provider_exception(exc: Exception) -> ProviderError:
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status_code == 401:
            return ProviderError(
                "Gemini rejected the API key.", status_code=502, code="provider_auth_error"
            )
        if status_code == 429:
            return ProviderError(
                "Gemini rate-limited the request.", status_code=429, code="provider_rate_limited"
            )
        return ProviderError(f"Gemini provider request failed: {type(exc).__name__}.")
