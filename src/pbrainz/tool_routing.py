"""Bounded canonical tool selection for structured NPC conversations.

Project Hoomans remains the source of truth for tool schemas and execution.
This module only normalizes the cards, applies deterministic eligibility, and
selects a small relevant subset for one provider request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pbrainz.retrieval_dictionary import DEFAULT_STOP_WORDS
from pbrainz.semantic_tool_protocol import infer_social_intent, is_name_question

_TOKEN_RE = re.compile(r"[\w-]{2,64}", re.UNICODE)
_SAFE_TOOL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_FALLBACK_NAMES = {"social_react"}


@dataclass(frozen=True, slots=True)
class ToolCard:
    """A canonical game-owned tool schema plus selection metadata."""

    schema: dict[str, Any]
    name: str
    description: str
    tags: tuple[str, ...] = ()
    eligible: bool = True
    reason: str = "eligible"


@dataclass(frozen=True, slots=True)
class ToolSelection:
    registered: int
    eligible: int
    selected: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


class ToolRouter:
    """Select canonical tools without treating relevance as authorization."""

    def __init__(
        self,
        *,
        max_results: int = 8,
        budget_chars: int = 2600,
        stop_words: tuple[str, ...] | frozenset[str] = DEFAULT_STOP_WORDS,
        token_expansions: tuple[tuple[str, tuple[str, ...]], ...] = (),
    ) -> None:
        self.max_results = max(1, min(int(max_results), 32))
        self.budget_chars = max(400, min(int(budget_chars), 20000))
        self.set_stop_words(stop_words)
        self.set_token_expansions(token_expansions)

    def set_stop_words(self, stop_words: tuple[str, ...] | frozenset[str]) -> None:
        self.stop_words = frozenset(
            str(word).casefold() for word in stop_words if str(word).strip()
        )

    def set_token_expansions(
        self, token_expansions: tuple[tuple[str, tuple[str, ...]], ...]
    ) -> None:
        self.token_expansions = {
            str(source).casefold(): tuple(str(value).casefold() for value in values)
            for source, values in token_expansions
            if str(source).strip()
        }

    def select(
        self,
        tools: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        query: str,
        *,
        current_topic: str | None = None,
        fallback_safe: bool = True,
    ) -> ToolSelection:
        cards = tuple(self._card(tool) for tool in tools if isinstance(tool, dict))
        eligible = tuple(card for card in cards if card.eligible)
        query_tokens = self._tokens(" ".join((query, current_topic or "")))
        name_question = is_name_question(query)
        social_intent = infer_social_intent(query) is not None
        scored = sorted(
            (
                (
                    self._score(
                        card,
                        query_tokens,
                        index,
                        name_question=name_question,
                        social_intent=social_intent,
                    ),
                    index,
                    card,
                )
                for index, card in enumerate(eligible)
            ),
            key=lambda value: (-value[0], value[1]),
        )
        selected: list[dict[str, Any]] = []
        selected_names: list[str] = []
        used_chars = 0
        for score, _, card in scored:
            serialized_size = len(repr(card.schema))
            if selected and used_chars + serialized_size > self.budget_chars:
                continue
            # Relevance and the safe fallback are separate concerns.  A safe
            # social card must not outrank or ride along with an unrelated
            # movement/action tool merely because it is a fallback.
            if score <= 0:
                continue
            selected.append(card.schema)
            selected_names.append(card.name)
            used_chars += serialized_size
            if len(selected) >= self.max_results:
                break
        if not selected and fallback_safe:
            # Keep a safe social-intent card available when supplied. It is an
            # intent only; Project Hoomans still decides whether it applies.
            for card in eligible:
                if card.name in _SAFE_FALLBACK_NAMES:
                    selected.append(card.schema)
                    selected_names.append(card.name)
                    break
        return ToolSelection(
            registered=len(cards),
            eligible=len(eligible),
            selected=tuple(selected),
            diagnostics={
                "registered": len(cards),
                "eligible": len(eligible),
                "selected": len(selected),
                "selected_names": selected_names,
                "rejected": [
                    {"name": card.name, "reason": card.reason}
                    for card in cards
                    if not card.eligible
                ],
                "tool_budget_chars": self.budget_chars,
            },
        )

    @classmethod
    def _card(cls, tool: dict[str, Any]) -> ToolCard:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(function.get("name") or function.get("id") or "").strip()
        description = str(function.get("description") or "").strip()[:500]
        metadata = tool.get("metadata") if isinstance(tool.get("metadata"), dict) else {}
        tags = tuple(
            str(tag).strip().casefold()
            for tag in metadata.get("tags", ())
            if str(tag).strip()
        )[:16]
        reason = "eligible"
        eligible = True
        if not _SAFE_TOOL_RE.match(name):
            eligible, reason = False, "invalid_name"
        elif metadata.get("eligible") is False or function.get("eligible") is False:
            eligible, reason = False, "game_ineligible"
        elif metadata.get("internal") is True or function.get("internal") is True:
            eligible, reason = False, "internal_primitive"
        elif metadata.get("clientOnly") is True or function.get("clientOnly") is True:
            eligible, reason = False, "client_only"
        return ToolCard(tool, name, description, tags, eligible, reason)

    def _tokens(self, value: str) -> set[str]:
        tokens: set[str] = set()
        for token in _TOKEN_RE.findall(value[:2000]):
            normalized = token.casefold()
            if len(normalized) < 3 or normalized in self.stop_words:
                continue
            tokens.add(normalized)
            for expansion in self.token_expansions.get(normalized, ()):
                for expanded_token in _TOKEN_RE.findall(expansion):
                    if (
                        len(expanded_token) >= 3
                        and expanded_token not in self.stop_words
                    ):
                        tokens.add(expanded_token)
        return tokens

    def _score(
        self,
        card: ToolCard,
        query_tokens: set[str],
        index: int,
        *,
        name_question: bool,
        social_intent: bool,
    ) -> float:
        # Identity and social actions have canonical classifiers downstream.
        # Do not let generic description words such as "you" expose ask_name
        # for unrelated small talk, or expose social_react merely because a
        # description mentions the player.  The fallback path still preserves
        # the standalone router's safe social behavior when explicitly asked.
        if card.name == "ask_name":
            return 100.0 if name_question else 0.0
        if card.name == "social_react":
            return 100.0 if social_intent else 0.0
        searchable = self._tokens(" ".join((card.name, card.description, *card.tags)))
        overlap = query_tokens.intersection(searchable)
        score = float(len(overlap))
        # Stable low-priority tie behavior; index is not a semantic signal.
        return score - index * 0.000001
