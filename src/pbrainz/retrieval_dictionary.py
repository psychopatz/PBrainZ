"""Editable, locale-aware dictionaries used by retrieval planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

DEFAULT_HISTORICAL_TERMS = (
    "remember",
    "recall",
    "yesterday",
    "earlier",
    "before",
    "last",
    "morning",
    "afternoon",
    "said",
    "told",
    "happened",
    "trust",
    "trusted",
    "promise",
    "promised",
    "agreed",
    "again",
    "what did",
    "why do you",
    "when did",
    "first met",
    "first meet",
    "have we met",
    "did we meet",
    "how did we meet",
    "where did we meet",
    "our first meeting",
    "since we met",
)
DEFAULT_FIRST_MEETING_TERMS = (
    "first time",
    "first encounter",
    "first meeting",
    "first meet",
    "first met",
    "when did we first meet",
    "have we met before",
    "how did we meet",
    "where did we meet",
)
DEFAULT_TOOL_TERMS = (
    "follow",
    "come",
    "go",
    "move",
    "stay",
    "trade",
    "give",
    "take",
    "join",
    "attack",
    "fight",
    "heal",
    "open",
    "close",
    "inspect",
    "look",
    "check",
    "use",
    "build",
    "craft",
    "admire",
    "praise",
    "comfort",
    "apology",
    "apologize",
    "apologise",
    "flirt",
    "insult",
    "sorry",
    "name",
)
DEFAULT_STOP_WORDS = (
    "about",
    "are",
    "can",
    "for",
    "from",
    "have",
    "how",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "with",
    "would",
    "you",
    "your",
)
DEFAULT_TOKEN_EXPANSIONS = {
    "meet": ("met", "meeting"),
    "met": ("meet", "meeting"),
    "meeting": ("meet", "met"),
    "remember": ("recall",),
    "recall": ("remember",),
}

MAX_TERMS = 256
MAX_TERM_CHARS = 96
MAX_LOCALES = 32


@dataclass(frozen=True, slots=True)
class DictionaryLocale:
    """The word groups consumed by one locale-aware planner."""

    historical: tuple[str, ...]
    first_meeting: tuple[str, ...]
    tool: tuple[str, ...]
    stop_words: tuple[str, ...]
    token_expansions: tuple[tuple[str, tuple[str, ...]], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "historical": list(self.historical),
            "first_meeting": list(self.first_meeting),
            "tool": list(self.tool),
            "stop_words": list(self.stop_words),
            "token_expansions": {
                key: list(values) for key, values in self.token_expansions
            },
        }


@dataclass(frozen=True, slots=True)
class RetrievalDictionary:
    """Versioned locale collection used by the live conversation service."""

    active_locale: str
    locales: tuple[tuple[str, DictionaryLocale], ...]

    def locale(self, locale: str | None = None) -> DictionaryLocale:
        selected = (locale or self.active_locale).casefold()
        for name, dictionary in self.locales:
            if name.casefold() == selected:
                return dictionary
        return self.locales[0][1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "active_locale": self.active_locale,
            "locales": {name: dictionary.as_dict() for name, dictionary in self.locales},
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))

    def token_expansion_items(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.locale().token_expansions


def default_retrieval_dictionary() -> RetrievalDictionary:
    return RetrievalDictionary(
        active_locale="en",
        locales=(
            (
                "en",
                DictionaryLocale(
                    historical=DEFAULT_HISTORICAL_TERMS,
                    first_meeting=DEFAULT_FIRST_MEETING_TERMS,
                    tool=DEFAULT_TOOL_TERMS,
                    stop_words=DEFAULT_STOP_WORDS,
                    token_expansions=tuple(DEFAULT_TOKEN_EXPANSIONS.items()),
                ),
            ),
        ),
    )


def load_retrieval_dictionary(raw: str | dict[str, Any] | None) -> RetrievalDictionary:
    """Parse persisted dictionary data and fill missing fields safely."""

    default = default_retrieval_dictionary()
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            value = {}
    elif isinstance(raw, dict):
        value = raw
    else:
        value = {}
    if not isinstance(value, dict):
        value = {}
    raw_locales = value.get("locales")
    if not isinstance(raw_locales, dict) or not raw_locales:
        raw_locales = {"en": {}}
    locales: list[tuple[str, DictionaryLocale]] = []
    default_locale = default.locale()
    for raw_name, raw_locale in list(raw_locales.items())[:MAX_LOCALES]:
        name = _locale_name(raw_name)
        if not name:
            continue
        locales.append(
            (
                name,
                _normalize_locale(
                    raw_locale if isinstance(raw_locale, dict) else {},
                    default_locale,
                ),
            )
        )
    if not locales:
        return default
    requested = _locale_name(value.get("active_locale"))
    active = next(
        (name for name, _dictionary in locales if name.casefold() == requested.casefold()),
        locales[0][0],
    )
    return RetrievalDictionary(active_locale=active, locales=tuple(locales))


def _normalize_locale(
    value: dict[str, Any], fallback: DictionaryLocale
) -> DictionaryLocale:
    expansions = value.get("token_expansions")
    if not isinstance(expansions, dict):
        expansions = dict(fallback.token_expansions)
    normalized_expansions: list[tuple[str, tuple[str, ...]]] = []
    for raw_key, raw_values in list(expansions.items())[:MAX_TERMS]:
        key = _term(raw_key)
        if not key or not isinstance(raw_values, (list, tuple)):
            continue
        values = _terms(raw_values)
        if values:
            normalized_expansions.append((key, values))
    return DictionaryLocale(
        historical=_terms(value.get("historical"), fallback.historical),
        first_meeting=_terms(value.get("first_meeting"), fallback.first_meeting),
        tool=_terms(value.get("tool"), fallback.tool),
        stop_words=_terms(value.get("stop_words"), fallback.stop_words),
        token_expansions=tuple(normalized_expansions),
    )


def _terms(value: Any, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    values = value if isinstance(value, (list, tuple)) else fallback
    output: list[str] = []
    seen: set[str] = set()
    for raw_value in list(values)[:MAX_TERMS]:
        item = _term(raw_value)
        folded = item.casefold()
        if item and folded not in seen:
            output.append(item)
            seen.add(folded)
    return tuple(output)


def _term(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_TERM_CHARS].strip()


def _locale_name(value: Any) -> str:
    return " ".join(str(value or "").split())[:32].strip() or "en"
