"""Shared eligibility rules for conversation prompt and memory ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_FAILURE_MARKERS = (
    "i cannot answer right now",
    "provider request failed",
    "provider returned an empty response",
    "openai-compatible provider",
    "llm completion failed",
    "i am an ai assistant",
    "as an ai",
    "language model",
    "large language model",
    "i don't have a personal identity",
    "i do not have a personal identity",
    "i don't have a name in the traditional sense",
    "i do not have a name in the traditional sense",
)
_USER_ROLES = {"user", "player"}


def _false_flag(value: Any) -> bool:
    return value is False or str(value or "").strip().casefold() in {
        "false",
        "0",
        "no",
        "off",
    }


def _true_flag(value: Any) -> bool:
    return value is True or str(value or "").strip().casefold() in {
        "true",
        "1",
        "yes",
        "on",
    }


def is_context_eligible(
    content: object,
    *,
    role: object = "",
    metadata: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a message may enter prompts or reusable NPC memory.

    Provider failures are presentation artifacts, not conversation turns. The
    explicit flags cover new messages; content markers cover older persisted
    records and bridge payloads created before those flags existed.
    """

    containers: list[Mapping[str, Any]] = []
    for value in (metadata, source):
        if isinstance(value, Mapping):
            containers.append(value)
            for nested_key in ("event_source", "source", "provenance"):
                nested = value.get(nested_key)
                if isinstance(nested, Mapping):
                    containers.append(nested)
    for container in containers:
        if _false_flag(container.get("contextEligible")):
            return False
        if _true_flag(container.get("providerFailure")):
            return False
        if _true_flag(container.get("excludeFromLLM")):
            return False

    normalized_role = str(role or "").strip().casefold()
    if normalized_role in _USER_ROLES:
        return True
    normalized_content = " ".join(str(content or "").split()).casefold()
    return not any(marker in normalized_content for marker in _FAILURE_MARKERS)
