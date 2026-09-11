"""Google Gemini provider adapter using the official ``google-genai`` SDK."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from pbrainz.api.models import ChatCompletionRequest, ChatMessage
from pbrainz.config import Settings
from pbrainz.exceptions import (
    ProviderError,
    ProviderNotConfiguredError,
    UnsupportedMessageError,
)

from .base import CompletionResult, LLMProvider, StreamEvent, TokenUsage

LOGGER = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Translate the common chat contract to Gemini's content-generation API."""

    name = "gemini"
    _FALLBACK_MODELS = ("gemini-2.5-flash",)
    _RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})
    _SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array"})
    _TRUNCATED_SCHEMA_VALUES = frozenset({"[depth-limit]", "[unsupported]"})

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

        from google.genai import types

        self._sync_client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=max(1000, int(settings.request_timeout * 1000)),
            ),
        )
        self._client = self._sync_client.aio
        configured_models = {
            model.casefold() for model in settings.models_for(self.name) if model.strip()
        }
        self._fallback_models = tuple(
            model
            for model in self._FALLBACK_MODELS
            if not configured_models or model.casefold() in configured_models
        )

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        contents, config = self._contents_and_config(request)
        response, model = await self._generate_content(
            request.model,
            contents,
            config,
        )

        return CompletionResult(
            model=model,
            text=getattr(response, "text", None) or "",
            finish_reason=self._finish_reason(response),
            usage=self._usage(getattr(response, "usage_metadata", None)),
            tool_calls=self._tool_calls(response),
            reasoning=self._reasoning(response),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        contents, config = self._contents_and_config(request)
        model = request.model
        yielded = False
        try:
            response_stream = await self._client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            async for chunk in response_stream:
                text = getattr(chunk, "text", None) or ""
                yielded = yielded or bool(str(text).strip())
                yield StreamEvent(text=text)
        except Exception as exc:
            fallback = self._fallback_model_for(model)
            if yielded or fallback is None or not self._is_retryable(exc):
                raise self._provider_exception(exc, model=model) from exc
            LOGGER.warning(
                "Gemini model request failed; retrying stream with fallback "
                "model=%s failed_model=%s error=%s",
                fallback,
                model,
                " ".join(str(exc).split())[:300],
            )
            try:
                response_stream = await self._client.models.generate_content_stream(
                    model=fallback,
                    contents=contents,
                    config=config,
                )
                async for chunk in response_stream:
                    yield StreamEvent(text=getattr(chunk, "text", None) or "")
            except Exception as fallback_exc:
                raise self._provider_exception(
                    fallback_exc,
                    model=fallback,
                    fallback_from=model,
                ) from fallback_exc
            return

        if yielded:
            return
        fallback = self._fallback_model_for(model)
        if fallback is None:
            return
        LOGGER.warning(
            "Gemini model returned no text; retrying stream with fallback "
            "model=%s failed_model=%s",
            fallback,
            model,
        )
        try:
            response_stream = await self._client.models.generate_content_stream(
                model=fallback,
                contents=contents,
                config=config,
            )
            async for chunk in response_stream:
                yield StreamEvent(text=getattr(chunk, "text", None) or "")
        except Exception as fallback_exc:
            raise self._provider_exception(
                fallback_exc,
                model=fallback,
                fallback_from=model,
            ) from fallback_exc

    async def _generate_content(
        self,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> tuple[Any, str]:
        try:
            response = await self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            fallback = self._fallback_model_for(model)
            if fallback is None or not self._is_retryable(exc):
                raise self._provider_exception(exc, model=model) from exc
            LOGGER.warning(
                "Gemini model request failed; retrying with fallback model=%s "
                "failed_model=%s error=%s",
                fallback,
                model,
                " ".join(str(exc).split())[:300],
            )
            return await self._fallback_content(fallback, model, contents, config)
        fallback = self._fallback_model_for(model)
        if self._has_usable_output(response) or fallback is None:
            return response, model
        LOGGER.warning(
            "Gemini model returned no usable text or tool call; retrying with "
            "fallback model=%s failed_model=%s",
            fallback,
            model,
        )
        return await self._fallback_content(fallback, model, contents, config)

    async def _fallback_content(
        self,
        fallback: str,
        failed_model: str,
        contents: list[Any],
        config: Any,
    ) -> tuple[Any, str]:
        try:
            response = await self._client.models.generate_content(
                model=fallback,
                contents=contents,
                config=config,
            )
            return response, fallback
        except Exception as fallback_exc:
            raise self._provider_exception(
                fallback_exc,
                model=fallback,
                fallback_from=failed_model,
            ) from fallback_exc

    @classmethod
    def _has_usable_output(cls, response: Any) -> bool:
        text = getattr(response, "text", None) or ""
        return bool(str(text).strip()) or bool(cls._tool_calls(response))

    def _fallback_model_for(self, model: str) -> str | None:
        if model.casefold() in {
            candidate.casefold() for candidate in self._fallback_models
        }:
            return None
        return self._fallback_models[0] if self._fallback_models else None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        for attribute in ("code", "status_code"):
            value = getattr(exc, attribute, None)
            try:
                if int(value) in cls._RETRYABLE_STATUS_CODES:
                    return True
            except (TypeError, ValueError):
                continue
        return type(exc).__name__ in {"ServerError", "InternalServerError"}

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
        tools = cls._tools(request.tools, types)
        if tools:
            config_values["tools"] = tools
        config = types.GenerateContentConfig(**config_values) if config_values else None
        return contents, config

    @classmethod
    def _tools(cls, tool_definitions: list[dict[str, Any]] | None, types: Any) -> list[Any]:
        """Translate the common function-tool shape to Gemini declarations."""
        declarations: list[Any] = []
        for index, tool in enumerate((tool_definitions or [])[:12]):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                continue
            declaration_values: dict[str, Any] = {
                "name": str(function["name"]),
                "description": str(function.get("description") or ""),
            }
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                declaration_values["parametersJsonSchema"] = cls._sanitize_schema(
                    parameters,
                    path=f"$.tools[{index}].function.parameters",
                )
            declarations.append(types.FunctionDeclaration(**declaration_values))
        return [types.Tool(functionDeclarations=declarations)] if declarations else []

    @classmethod
    def _sanitize_schema(cls, schema: dict[str, Any], *, path: str = "$") -> dict[str, Any]:
        """Convert the shared JSON-schema subset to Gemini's supported schema shape.

        Project Hoomans and OpenAI-compatible providers share tool definitions, but
        Gemini accepts only a smaller schema vocabulary.  In particular, custom
        schema metadata or an invalid ``type`` must not be sent to Gemini.  Keep
        this conversion provider-local so the common contract remains intact for
        OpenAI-compatible endpoints.
        """
        raw_type = schema.get("type")
        nullable = schema.get("nullable") is True
        schema_type: str | None = None

        if isinstance(raw_type, str) and raw_type in cls._TRUNCATED_SCHEMA_VALUES:
            # Older PsychopatzCore bridge builds serialized deeply nested tool
            # fields as placeholders. Keep those requests usable while the
            # game-side bridge is updated; an unknown scalar is the safest
            # representation for Gemini.
            schema_type = "string"
        elif isinstance(raw_type, (list, tuple)):
            raw_types = [str(value).lower() for value in raw_type]
            nullable = nullable or "null" in raw_types
            schema_type = next(
                (value for value in raw_types if value in cls._SCHEMA_TYPES),
                None,
            )
        elif raw_type is not None:
            candidate = str(raw_type).lower()
            if candidate in cls._SCHEMA_TYPES:
                schema_type = candidate

        if schema_type is None:
            if raw_type is not None and raw_type not in cls._TRUNCATED_SCHEMA_VALUES:
                LOGGER.warning(
                    "Normalizing unsupported Gemini tool schema type path=%s type=%r",
                    path,
                    raw_type,
                )
            if isinstance(schema.get("properties"), dict):
                schema_type = "object"
            elif isinstance(schema.get("items"), dict):
                schema_type = "array"
            else:
                # Tool arguments without a recognizable type are most commonly
                # scalar text values (for example, social_react.kind).
                schema_type = "string"

        output: dict[str, Any] = {"type": schema_type}
        if nullable:
            output["nullable"] = True

        description = schema.get("description")
        if isinstance(description, str) and description:
            output["description"] = description
        enum = schema.get("enum")
        if isinstance(enum, (list, tuple)):
            output["enum"] = list(enum)

        if schema_type == "object":
            properties = schema.get("properties")
            safe_properties: dict[str, Any] = {}
            if isinstance(properties, dict):
                for name, child in properties.items():
                    if isinstance(child, dict):
                        safe_name = str(name)
                        safe_properties[safe_name] = cls._sanitize_schema(
                            child,
                            path=f"{path}.properties.{safe_name}",
                        )
            if safe_properties:
                output["properties"] = safe_properties

            required = schema.get("required")
            if isinstance(required, (list, tuple)):
                safe_required = [
                    str(name) for name in required if str(name) in safe_properties
                ]
                if safe_required:
                    output["required"] = safe_required
        elif schema_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                output["items"] = cls._sanitize_schema(items, path=f"{path}.items")

        # Deliberately omit additionalProperties and all unknown/custom keywords.
        # They are not needed for the NPC command contract and can be rejected by
        # Gemini even when they are accepted by the OpenAI-compatible API shape.
        return output

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
    def _tool_calls(response: Any) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                function = getattr(part, "function_call", None)
                name = getattr(function, "name", None) if function else None
                if not name:
                    continue
                arguments = getattr(function, "args", None) or {}
                calls.append(
                    {
                        "id": str(getattr(function, "id", "") or ""),
                        "type": "function",
                        "function": {
                            "name": str(name),
                            "arguments": json.dumps(arguments, ensure_ascii=True),
                        },
                    }
                )
        return calls or None

    @staticmethod
    def _reasoning(response: Any) -> str | None:
        """Capture explicitly returned thought parts, when the provider exposes them."""
        parts: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "thought", False) is not True:
                    continue
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)[:12000] or None

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
    def _provider_exception(
        exc: Exception,
        *,
        model: str | None = None,
        fallback_from: str | None = None,
    ) -> ProviderError:
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status_code == 401:
            return ProviderError(
                "Gemini rejected the API key.", status_code=502, code="provider_auth_error"
            )
        if status_code == 429:
            return ProviderError(
                "Gemini rate-limited the request.", status_code=429, code="provider_rate_limited"
            )
        detail = " ".join(str(exc).split())[:600]
        if fallback_from:
            message = (
                f"Gemini provider request failed for fallback model '{model}' "
                f"after model '{fallback_from}' failed: {type(exc).__name__}"
            )
        elif model:
            message = (
                f"Gemini provider request failed for model '{model}': "
                f"{type(exc).__name__}"
            )
        else:
            message = f"Gemini provider request failed: {type(exc).__name__}"
        if detail:
            message += f" — {detail}"
        return ProviderError(message + ".")
