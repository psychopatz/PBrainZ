"""OpenAI and OpenAI-compatible provider adapter."""

from collections.abc import AsyncIterator, Mapping
from typing import Any

from pbrainz.api.models import ChatCompletionRequest, ChatMessage
from pbrainz.config import OPENAI_COMPATIBLE_PROVIDERS, Settings
from pbrainz.exceptions import (
    ProviderError,
    ProviderNotConfiguredError,
)

from .base import CompletionResult, LLMProvider, StreamEvent, TokenUsage


class OpenAICompatibleProvider(LLMProvider):
    """Use the official ``openai`` async client against OpenAI or a compatible URL."""

    name = "openai"

    def __init__(self, settings: Settings, provider_name: str = "openai") -> None:
        provider_name = provider_name.strip().lower()
        if provider_name not in OPENAI_COMPATIBLE_PROVIDERS:
            raise ProviderError(
                f"Unknown OpenAI-compatible provider '{provider_name}'.",
                status_code=400,
                code="unknown_provider",
            )
        self.name = provider_name
        base_url = settings.base_url_for(provider_name)
        api_key = getattr(settings, f"{provider_name}_api_key", None)
        if not settings.provider_configured(provider_name):
            raise ProviderNotConfiguredError(self.name)

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared by the package.
            raise ProviderError(
                "The OpenAI provider dependency is missing. "
                "Reinstall with the project dependencies.",
                status_code=503,
                code="missing_provider_dependency",
            ) from exc

        # OpenAI-compatible endpoints may not require authentication. The SDK
        # still requires a non-empty key. AI Horde documents 0000000000 as its
        # anonymous key; use it only for Horde when the user leaves the field
        # blank, and a harmless placeholder for local unauthenticated servers.
        if provider_name == "horde":
            api_key = api_key or "0000000000"
        else:
            api_key = api_key or "not-needed"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        params = self._request_params(request, stream=False)
        try:
            response = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise self._provider_exception(exc) from exc

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        return CompletionResult(
            model=response.model or request.model,
            text=(message.content if message and message.content else ""),
            finish_reason=self._finish_reason(choice.finish_reason if choice else None),
            usage=self._usage(getattr(response, "usage", None)),
            tool_calls=self._tool_calls(message),
            reasoning=self._reasoning(message),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        params = self._request_params(request, stream=True)
        try:
            response_stream = await self._client.chat.completions.create(**params)
            async for chunk in response_stream:
                usage = self._usage(getattr(chunk, "usage", None))
                if not chunk.choices:
                    if usage:
                        yield StreamEvent(usage=usage)
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = getattr(choice, "finish_reason", None)
                yield StreamEvent(
                    text=delta.content or "",
                    role=getattr(delta, "role", None),
                    finish_reason=(
                        self._finish_reason(finish_reason) if finish_reason is not None else None
                    ),
                    usage=usage,
                )
        except Exception as exc:
            raise self._provider_exception(exc) from exc

    async def list_models(self) -> list[str]:
        """Read the provider's current model catalog and omit obvious non-chat APIs."""
        try:
            response = await self._client.models.list()
        except Exception as exc:
            raise self._provider_exception(exc) from exc
        model_ids = [
            str(getattr(model, "id", ""))
            for model in getattr(response, "data", [])
            if getattr(model, "id", None)
        ]
        excluded = ("embedding", "moderation", "whisper", "tts", "dall-e", "transcription")
        return sorted(
            {
                model_id
                for model_id in model_ids
                if not any(marker in model_id.casefold() for marker in excluded)
            }
        )

    async def close(self) -> None:
        await self._client.close()

    def _request_params(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [self._message_param(message) for message in request.messages],
            "stream": stream,
        }
        # The Horde OpenAI gateway intentionally exposes a smaller request
        # schema than the full OpenAI API. In particular, it does not provide
        # native function/tool calling, so do not send fields it cannot use.
        field_names = (
            ("temperature", "top_p", "stop", "presence_penalty", "frequency_penalty")
            if self.name == "horde"
            else (
                "temperature",
                "top_p",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "seed",
                "user",
                "metadata",
                "response_format",
                "tools",
                "tool_choice",
            )
        )
        for field_name in field_names:
            value = getattr(request, field_name)
            if value is not None:
                if self.name == "horde" and field_name == "stop" and isinstance(value, str):
                    value = [value]
                params[field_name] = value

        if self.name == "horde":
            max_tokens = request.max_completion_tokens or request.max_tokens
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
        elif request.max_completion_tokens is not None:
            params["max_completion_tokens"] = request.max_completion_tokens
        elif request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if self.name != "horde" and request.stream_options is not None:
            params["stream_options"] = request.stream_options
        return params

    @staticmethod
    def _message_param(message: ChatMessage) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        for field_name in ("name", "tool_call_id", "tool_calls"):
            value = getattr(message, field_name)
            if value is not None:
                result[field_name] = value
        return result

    @staticmethod
    def _usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def _tool_calls(message: Any) -> list[dict[str, Any]] | None:
        """Normalize SDK tool-call objects without coupling the rest of the app to OpenAI."""
        calls = getattr(message, "tool_calls", None) if message is not None else None
        if not calls:
            return None
        normalized: list[dict[str, Any]] = []
        for call in calls[:16]:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if not name:
                continue
            normalized.append(
                {
                    "id": str(getattr(call, "id", "")),
                    "type": str(getattr(call, "type", "function") or "function"),
                    "function": {
                        "name": str(name),
                        "arguments": str(arguments or "{}"),
                    },
                }
            )
        return normalized or None

    @staticmethod
    def _reasoning(message: Any) -> str | None:
        """Read public reasoning fields used by compatible APIs, if present."""
        if message is None:
            return None
        for field_name in ("reasoning_content", "reasoning", "analysis", "thinking"):
            value = getattr(message, field_name, None)
            if value is None or value == "":
                continue
            if isinstance(value, str):
                return value[:12000]
            return str(value)[:12000]
        return None

    @staticmethod
    def _finish_reason(reason: Any) -> str:
        if reason is None:
            return "stop"
        value = getattr(reason, "value", reason)
        return str(value)

    @staticmethod
    def _provider_exception(exc: Exception) -> ProviderError:
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            return ProviderError(
                OpenAICompatibleProvider._error_message(
                    "The OpenAI-compatible provider rejected the API key.", exc
                ),
                status_code=502,
                code="provider_auth_error",
            )
        if status_code == 429:
            return ProviderError(
                OpenAICompatibleProvider._error_message(
                    "The OpenAI-compatible provider rate-limited the request.", exc
                ),
                status_code=429,
                code="provider_rate_limited",
            )
        if isinstance(status_code, int) and status_code >= 500:
            return ProviderError(
                OpenAICompatibleProvider._error_message(
                    "The OpenAI-compatible provider returned a server error.", exc
                ),
                status_code=502,
                code="provider_server_error",
            )
        return ProviderError(
            OpenAICompatibleProvider._error_message(
                f"OpenAI-compatible provider request failed ({type(exc).__name__}).", exc
            ),
            code="provider_bad_request" if isinstance(status_code, int) else "provider_error",
        )

    @staticmethod
    def _error_message(prefix: str, exc: Exception) -> str:
        """Keep a bounded, provider-supplied reason without exposing credentials."""
        detail: object = getattr(exc, "body", None)
        if isinstance(detail, Mapping):
            nested = detail.get("error")
            if isinstance(nested, Mapping):
                detail = nested.get("message") or nested.get("detail") or nested
            else:
                detail = detail.get("message") or detail.get("detail") or detail
        if detail is None:
            detail = getattr(exc, "message", None)
        rendered = " ".join(str(detail or "").split())
        if not rendered or rendered.casefold() in {"none", str(exc).casefold()}:
            if isinstance(getattr(exc, "status_code", None), int):
                rendered = f"HTTP {exc.status_code}"
            else:
                return prefix
        return f"{prefix}: {rendered[:500]}"
