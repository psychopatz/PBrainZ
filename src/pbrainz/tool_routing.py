"""Bounded canonical tool selection for structured NPC conversations.

Project Hoomans remains the source of truth for tool schemas and execution.
This module only normalizes the cards, applies deterministic eligibility, and
selects a small relevant subset for one provider request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{2,64}")
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

    def __init__(self, *, max_results: int = 8, budget_chars: int = 2600) -> None:
        self.max_results = max(1, min(int(max_results), 32))
        self.budget_chars = max(400, min(int(budget_chars), 20000))

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
        scored = sorted(
            (
                (self._score(card, query_tokens, index), index, card)
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
            # Keep a safe social-intent card available when supplied. It is an
            # intent only; Project Hoomans still decides whether it applies.
            if score <= 0 and card.name not in _SAFE_FALLBACK_NAMES:
                continue
            selected.append(card.schema)
            selected_names.append(card.name)
            used_chars += serialized_size
            if len(selected) >= self.max_results:
                break
        if not selected and fallback_safe:
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

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _TOKEN_RE.findall(value[:2000])
            if len(token) >= 3
        }

    @classmethod
    def _score(cls, card: ToolCard, query_tokens: set[str], index: int) -> float:
        searchable = cls._tokens(" ".join((card.name, card.description, *card.tags)))
        overlap = query_tokens.intersection(searchable)
        score = float(len(overlap))
        if card.name in _SAFE_FALLBACK_NAMES:
            score += 0.1
        # Stable low-priority tie behavior; index is not a semantic signal.
        return score - index * 0.000001

