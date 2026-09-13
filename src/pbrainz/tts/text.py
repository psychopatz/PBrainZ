"""Deterministic text preparation shared by every generated TTS path."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_TTS_TEXT = 12_000


@dataclass(frozen=True, slots=True)
class NonverbalCueRule:
    """A recognized written cue and the compact speech it should produce."""

    name: str
    replacement: str
    aliases: tuple[str, ...]


# Keep this catalog data-driven. Adding a cue should only require adding an
# alias/replacement row, not another branch to the normalization algorithm.
NONVERBAL_CUE_RULES: tuple[NonverbalCueRule, ...] = (
    NonverbalCueRule("chuckle", "Ha ha!", ("chuckle", "chuckles", "chuckling")),
    NonverbalCueRule("laugh", "Ha ha!", ("laugh", "laughs", "laughing", "laughter")),
    NonverbalCueRule("gasp", "Huh!", ("gasp", "gasps", "gasping")),
    NonverbalCueRule("sigh", "Sigh.", ("sigh", "sighs", "sighing")),
    NonverbalCueRule("cough", "Ahem.", ("cough", "coughs", "coughing")),
    NonverbalCueRule("clear_throat", "Ahem.", ("clear throat", "clears throat")),
    NonverbalCueRule("groan", "Ugh.", ("groan", "groans", "groaning")),
    NonverbalCueRule("moan", "Mmm.", ("moan", "moans", "moaning")),
    NonverbalCueRule("snort", "Heh.", ("snort", "snorts", "snorting")),
    NonverbalCueRule("whimper", "Oh.", ("whimper", "whimpers", "whimpering")),
)


def _cue_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


_CUE_REPLACEMENTS = {
    _cue_key(alias): rule.replacement
    for rule in NONVERBAL_CUE_RULES
    for alias in rule.aliases
}
_CUE_ALTERNATIVES = "|".join(
    re.escape(alias)
    for alias in sorted(_CUE_REPLACEMENTS, key=len, reverse=True)
)
_CUE_PATTERN = re.compile(
    rf"(?:"
    rf"\(\s*\*+\s*(?P<paren_star>{_CUE_ALTERNATIVES})"
    rf"\s*[,!?.;:]*\s*\*+\s*\)|"
    rf"\[\s*\*+\s*(?P<bracket_star>{_CUE_ALTERNATIVES})"
    rf"\s*[,!?.;:]*\s*\*+\s*\]|"
    rf"\*+\s*(?P<star>{_CUE_ALTERNATIVES})"
    rf"\s*[,!?.;:]*\s*\*+|"
    rf"\(\s*(?P<paren>{_CUE_ALTERNATIVES})\s*[,!?.;:]*\s*\)|"
    rf"\[\s*(?P<bracket>{_CUE_ALTERNATIVES})\s*[,!?.;:]*\s*\]"
    rf")",
    re.IGNORECASE,
)

_FENCED_CODE_PATTERN = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]{0,256}>")
_IMAGE_PATTERN = re.compile(r"!\[([^\]]{0,512})\]\([^\n)]{0,2048}\)")
_LINK_PATTERN = re.compile(r"\[([^\]]{0,512})\]\([^\n)]{0,2048}\)")
_REFERENCE_LINK_PATTERN = re.compile(r"\[([^\]]{0,512})\]\[[^\]]{0,256}\]")
_BOLD_PATTERN = re.compile(r"(\*\*|__)(?=\S)(.*?\S)\1")
_ITALIC_PATTERN = re.compile(r"(\*|_)(?=\S)(.*?\S)\1")
_STRIKE_PATTERN = re.compile(r"~~(?=\S)(.*?\S)~~")
_STAGE_MARKER_PATTERN = re.compile(r"\*\s+(?=\S)(.*?)\s*\*")
_STAGE_DIRECTION_LEADS = frozenset(
    {
        "blink",
        "blinks",
        "cough",
        "coughs",
        "cross",
        "crosses",
        "fold",
        "folds",
        "frown",
        "frowns",
        "gasp",
        "gasps",
        "gesture",
        "gestures",
        "glance",
        "glances",
        "grin",
        "grins",
        "groan",
        "groans",
        "laugh",
        "laughs",
        "lean",
        "leans",
        "look",
        "looks",
        "lower",
        "lowers",
        "mumble",
        "mumbles",
        "mutters",
        "mutter",
        "nod",
        "nods",
        "pick",
        "picks",
        "point",
        "points",
        "raise",
        "raises",
        "rub",
        "rubs",
        "rubbing",
        "scratch",
        "scratches",
        "shake",
        "shakes",
        "shrug",
        "shrugs",
        "sigh",
        "sighs",
        "smile",
        "smiles",
        "snort",
        "snorts",
        "scoff",
        "scoffs",
        "stare",
        "stares",
        "step",
        "steps",
        "turn",
        "turns",
        "walk",
        "walks",
        "wave",
        "waves",
        "whisper",
        "whispers",
        "wince",
        "winces",
        "wink",
        "winks",
        "yawn",
        "yawns",
    }
)
_PAREN_STAGE_DIRECTION_PATTERN = re.compile(
    r"\(\s*\*(?P<cue>[^*\n]{1,180})\*\s*\)"
)
_STAGE_DIRECTION_PATTERN = re.compile(
    r"(?<!\w)\*(?P<cue>[^*\n]{1,180})\*(?!\w)"
)
_HEADING_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_QUOTE_PATTERN = re.compile(r"(?m)^\s{0,3}>\s?")
_DASH_LIST_PATTERN = re.compile(r"(?m)^\s{0,3}[-+]\s+")
_STAR_LIST_PATTERN = re.compile(r"(?m)^\s{0,3}\*\s+(?![^\n]*\*)")
_ORDERED_LIST_PATTERN = re.compile(r"(?m)^\s{0,3}\d+[.)]\s+")
_HORIZONTAL_RULE_PATTERN = re.compile(r"(?m)^\s{0,3}(?:[-*_]\s*){3,}$")
_RESIDUAL_MARKDOWN_PATTERN = re.compile(r"[*_~`|#\[\]]")
_COLON_PATTERN = re.compile(r"\s*:\s*")
_INLINE_DASH_PATTERN = re.compile(r"\s+[-–—]\s+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _replace_nonverbal_cues(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        groups = match.groupdict()
        for cue in groups.values():
            if cue:
                return _CUE_REPLACEMENTS[_cue_key(cue)]
        return match.group(0)

    return _CUE_PATTERN.sub(replacement, value)


def _is_stage_direction(cue: str) -> bool:
    normalized = " ".join(cue.casefold().split())
    return (
        bool(normalized)
        and ":" not in normalized
        and normalized.split(" ", 1)[0] in _STAGE_DIRECTION_LEADS
    )


def _remove_stage_directions(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        return " " if _is_stage_direction(match.group("cue")) else match.group(0)

    value = _PAREN_STAGE_DIRECTION_PATTERN.sub(replacement, value)
    return _STAGE_DIRECTION_PATTERN.sub(replacement, value)


def normalize_tts_text(text: object, *, max_chars: int = MAX_TTS_TEXT) -> str:
    """Turn generated presentation text into natural, bounded speech text.

    Known nonverbal cues are replaced with short pronounceable equivalents.
    Recognized action directions are removed, while ordinary italicized words
    are retained as dialogue.
    """

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = _replace_nonverbal_cues(value)
    value = _remove_stage_directions(value)
    value = _FENCED_CODE_PATTERN.sub(r"\1", value)
    value = _HTML_TAG_PATTERN.sub("", value)
    value = _IMAGE_PATTERN.sub(r"\1", value)
    value = _LINK_PATTERN.sub(r"\1", value)
    value = _REFERENCE_LINK_PATTERN.sub(r"\1", value)
    value = _STAGE_MARKER_PATTERN.sub(r"\1", value)
    value = _HORIZONTAL_RULE_PATTERN.sub(" ", value)
    value = _HEADING_PATTERN.sub("", value)
    value = _QUOTE_PATTERN.sub("", value)
    value = _DASH_LIST_PATTERN.sub("", value)
    value = _STAR_LIST_PATTERN.sub("", value)
    value = _ORDERED_LIST_PATTERN.sub("", value)
    value = _BOLD_PATTERN.sub(r"\2", value)
    value = _STRIKE_PATTERN.sub(r"\1", value)
    value = _ITALIC_PATTERN.sub(r"\2", value)
    value = re.sub(r"\\([\\*_`~\[\]()])", r"\1", value)
    value = _RESIDUAL_MARKDOWN_PATTERN.sub("", value)
    value = _COLON_PATTERN.sub(". ", value)
    value = _INLINE_DASH_PATTERN.sub(" ", value)
    value = re.sub(r"([.!?])\s*\.\s*", r"\1 ", value)
    value = _WHITESPACE_PATTERN.sub(" ", value).strip()
    return value[: max(0, int(max_chars))].strip()


__all__ = [
    "MAX_TTS_TEXT",
    "NONVERBAL_CUE_RULES",
    "NonverbalCueRule",
    "normalize_tts_text",
]
