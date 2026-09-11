"""Deterministic intent planning for memory and tool retrieval.

The planner does not decide whether a tool call is authorized.  Project
Hoomans remains the authority for that.  It only decides which retrieval lanes
are worth spending context on for the current request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pbrainz.memory.types import MemoryType
from pbrainz.retrieval_dictionary import (
    RetrievalDictionary,
    default_retrieval_dictionary,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """The bounded retrieval decisions for one conversation request."""

    memory_retrieval: bool = False
    conversation_recall: bool = False
    tool_retrieval: bool = False
    first_meeting: bool = False
    requested_memory_kinds: tuple[MemoryType, ...] = ()
    requested_memory_tags: tuple[str, ...] = ()
    reason: str = "ordinary_chat"

    def as_diagnostic(self) -> dict[str, object]:
        return {
            "memory_retrieval": self.memory_retrieval,
            "conversation_recall": self.conversation_recall,
            "tool_retrieval": self.tool_retrieval,
            "first_meeting": self.first_meeting,
            "requested_memory_kinds": [kind.value for kind in self.requested_memory_kinds],
            "requested_memory_tags": list(self.requested_memory_tags),
            "reason": self.reason,
        }


def plan_retrieval(
    message: str,
    dictionary: RetrievalDictionary | None = None,
) -> RetrievalPlan:
    """Classify a request without asking the provider or loading memory.

    Exact primitive questions use a narrow direct-memory lane and do not also
    spend context on a second raw conversation-recall lane.  Broader historical
    questions retain conversation recall for nuance.
    """

    normalized = " ".join(str(message or "").split())
    if not normalized:
        return RetrievalPlan()
    locale = (dictionary or default_retrieval_dictionary()).locale()
    first_meeting = _contains_any(normalized, locale.first_meeting)
    historical = _contains_any(normalized, locale.historical)
    tool_retrieval = _contains_any(normalized, locale.tool)
    if first_meeting:
        return RetrievalPlan(
            memory_retrieval=True,
            conversation_recall=False,
            tool_retrieval=tool_retrieval,
            first_meeting=True,
            requested_memory_kinds=(MemoryType.PERSONAL_EVENT,),
            requested_memory_tags=("first_meeting",),
            reason="first_meeting_primitive",
        )
    if historical:
        return RetrievalPlan(
            memory_retrieval=True,
            conversation_recall=True,
            tool_retrieval=tool_retrieval,
            reason="historical_question",
        )
    return RetrievalPlan(
        tool_retrieval=tool_retrieval,
        reason="tool_intent" if tool_retrieval else "ordinary_chat",
    )


def retrieval_needed_for(message: str) -> bool:
    """Compatibility helper for callers that only need the memory gate."""

    return plan_retrieval(message).memory_retrieval


def _contains_any(message: str, phrases: tuple[str, ...]) -> bool:
    message_tokens = tuple(_WORD_RE.findall(message.casefold()))
    if not message_tokens:
        return False
    for phrase in phrases:
        phrase_tokens = tuple(_WORD_RE.findall(phrase.casefold()))
        if not phrase_tokens:
            continue
        width = len(phrase_tokens)
        if any(
            message_tokens[index : index + width] == phrase_tokens
            for index in range(len(message_tokens) - width + 1)
        ):
            return True
    return False
