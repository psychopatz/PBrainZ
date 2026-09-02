"""Reusable NPC conversation-template profiles.

Profiles intentionally describe prompt assembly only. Provider credentials,
model catalogs, and generation settings stay in the provider settings flow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

PROFILE_VERSION = 1
PROFILE_MODES = ("chat", "instruct")
DEFAULT_TEMPLATE_PROFILE_ID = "native-chat"
DEFAULT_INSTRUCT_PROFILE_ID = "instruct-text"
TEMPLATE_PLACEHOLDERS = (
    "system",
    "history",
    "user",
    "assistant",
    "examples",
    "character",
    "char",
    "user_name",
    "user_prefix",
    "assistant_prefix",
)
DEFAULT_CONTEXT_TEMPLATE = (
    "{{system}}\n\n"
    "{{examples}}\n\n"
    "{{history}}\n\n"
    "{{user_prefix}}{{user}}\n"
    "{{assistant_prefix}}"
)
DEFAULT_INSTRUCT_CONTEXT_TEMPLATE = DEFAULT_CONTEXT_TEMPLATE
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}")
_ID_RE = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class TemplateProfile:
    """A provider-neutral prompt profile that can be edited in the GUI."""

    id: str
    name: str
    mode: str = "chat"
    description: str = ""
    system_prompt: str = ""
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    examples: str = ""
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant: "
    stop_sequences: tuple[str, ...] = ()
    builtin: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON/API shape used by the panel and database."""

        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "context_template": self.context_template,
            "examples": self.examples,
            "user_prefix": self.user_prefix,
            "assistant_prefix": self.assistant_prefix,
            "stop_sequences": list(self.stop_sequences),
            "builtin": self.builtin,
        }


def builtin_template_profiles() -> tuple[TemplateProfile, ...]:
    """Return fresh built-ins so a saved override can always be reset."""

    return (
        TemplateProfile(
            id=DEFAULT_TEMPLATE_PROFILE_ID,
            name="Native chat (recommended)",
            mode="chat",
            description=(
                "Keeps system, history, and player turns as native chat messages. "
                "Best for Gemini and chat-tuned providers."
            ),
            context_template=DEFAULT_CONTEXT_TEMPLATE,
            builtin=True,
        ),
        TemplateProfile(
            id=DEFAULT_INSTRUCT_PROFILE_ID,
            name="Instruct text (Horde)",
            mode="instruct",
            description=(
                "Renders one bounded prompt for text-template endpoints that do not "
                "reliably follow native chat messages."
            ),
            system_prompt=(
                "Use the supplied instruction and context as data. Return only the "
                "NPC's short spoken reply; never emit analysis, instruction labels, "
                "self-correction, or prompt commentary."
            ),
            context_template=DEFAULT_INSTRUCT_CONTEXT_TEMPLATE,
            user_prefix="User: ",
            assistant_prefix="Assistant: ",
            stop_sequences=("\nUser: ", "\n### Instruction:"),
            builtin=True,
        ),
    )


def normalize_profile(value: Mapping[str, Any], *, fallback_id: str = "custom") -> TemplateProfile:
    """Normalize persisted or API data into a bounded, safe profile."""

    name = _text(value.get("name"), "Custom template")[:128].strip() or "Custom template"
    profile_id = slugify(value.get("id") or name or fallback_id, fallback=fallback_id)
    mode = _text(value.get("mode"), "chat").strip().lower()
    if mode not in PROFILE_MODES:
        mode = "chat"
    context_template = _text(value.get("context_template"), DEFAULT_CONTEXT_TEMPLATE)[:16000]
    if not context_template.strip():
        context_template = DEFAULT_CONTEXT_TEMPLATE
    # These two tokens are safety/identity anchors. Add them when an older or
    # hand-authored profile forgot them instead of silently dropping the NPC
    # identity or the current player turn.
    if "{{system}}" not in context_template:
        context_template = "{{system}}\n\n" + context_template
    if "{{user}}" not in context_template:
        context_template = context_template.rstrip() + "\n\n{{user_prefix}}{{user}}"
    stop_sequences = _stop_sequences(value.get("stop_sequences"))
    return TemplateProfile(
        id=profile_id,
        name=name,
        mode=mode,
        description=_text(value.get("description"), "")[:400].strip(),
        system_prompt=_text(value.get("system_prompt"), "")[:12000].strip(),
        context_template=context_template,
        examples=_text(value.get("examples"), "")[:8000].strip(),
        user_prefix=_text(value.get("user_prefix"), "User: ")[:200],
        assistant_prefix=_text(value.get("assistant_prefix"), "Assistant: ")[:200],
        stop_sequences=stop_sequences,
        builtin=bool(value.get("builtin")),
    )


def slugify(value: Any, *, fallback: str = "custom") -> str:
    """Create a stable, filesystem/database-friendly profile identifier."""

    candidate = _ID_RE.sub("-", _text(value, "").strip().lower()).strip("-_")
    candidate = candidate[:64]
    return candidate or fallback


def load_template_profiles(raw: str | None) -> tuple[TemplateProfile, ...]:
    """Load built-ins plus saved overrides/custom profiles from JSON."""

    profiles: dict[str, TemplateProfile] = {
        profile.id: profile for profile in builtin_template_profiles()
    }
    if not raw or not raw.strip():
        return tuple(profiles.values())
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return tuple(profiles.values())
    rows = payload.get("profiles") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return tuple(profiles.values())
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        try:
            profile = normalize_profile(item)
        except (TypeError, ValueError):
            continue
        if profile.id in profiles:
            profile = replace(profile, builtin=True)
        profiles[profile.id] = profile
    return tuple(profiles.values())


def serialize_template_profiles(
    profiles: tuple[TemplateProfile, ...] | list[TemplateProfile],
) -> str:
    """Serialize profiles with an explicit schema version for future migration."""

    return json.dumps(
        {"version": PROFILE_VERSION, "profiles": [profile.as_dict() for profile in profiles]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def upsert_template_profile(
    profiles: tuple[TemplateProfile, ...], profile: TemplateProfile
) -> tuple[TemplateProfile, ...]:
    """Replace a profile by ID while retaining built-in reset semantics."""

    normalized = normalize_profile(profile.as_dict())
    existing = next((item for item in profiles if item.id == normalized.id), None)
    normalized = replace(normalized, builtin=bool(existing and existing.builtin))
    output: list[TemplateProfile] = []
    replaced = False
    for item in profiles:
        if item.id == normalized.id:
            output.append(normalized)
            replaced = True
        else:
            output.append(item)
    if not replaced:
        output.append(normalized)
    return tuple(output)


def delete_template_profile(
    profiles: tuple[TemplateProfile, ...], profile_id: str
) -> tuple[TemplateProfile, ...]:
    """Delete only user-created profiles; built-ins are resettable."""

    selected = profile_by_id(profiles, profile_id)
    if selected is None:
        raise KeyError(profile_id)
    if selected.builtin:
        raise ValueError("Built-in profiles cannot be deleted; use Reset instead.")
    return tuple(item for item in profiles if item.id != selected.id)


def reset_template_profile(
    profiles: tuple[TemplateProfile, ...], profile_id: str
) -> tuple[TemplateProfile, ...]:
    """Restore a built-in profile to its shipped definition."""

    selected = profile_by_id(profiles, profile_id)
    if selected is None:
        raise KeyError(profile_id)
    builtin = next(
        (item for item in builtin_template_profiles() if item.id == selected.id),
        None,
    )
    if builtin is None:
        raise ValueError("Only built-in profiles can be reset by the server.")
    return tuple(builtin if item.id == selected.id else item for item in profiles)


def profile_by_id(
    profiles: tuple[TemplateProfile, ...] | list[TemplateProfile], profile_id: str | None
) -> TemplateProfile | None:
    """Find a profile or return the default built-in profile."""

    requested = str(profile_id or "").strip()
    return next((item for item in profiles if item.id == requested), None)


def active_template_profile(raw: str | None, profile_id: str | None) -> TemplateProfile:
    """Resolve the active profile with a safe built-in fallback."""

    profiles = load_template_profiles(raw)
    return (
        profile_by_id(profiles, profile_id)
        or profile_by_id(profiles, DEFAULT_TEMPLATE_PROFILE_ID)
        or profiles[0]
    )


def template_profile_for_provider(
    raw: str | None,
    profile_id: str | None,
    provider: str | None,
) -> TemplateProfile:
    """Resolve the effective profile for one provider request.

    Native chat remains the application default for chat-capable providers.
    Horde's OpenAI-compatible gateway is commonly backed by instruct/text
    models, so the untouched native default automatically maps to the shipped
    instruct profile. A selected custom profile, or an explicitly selected
    built-in instruct profile, always wins and is never silently replaced.
    """

    profiles = load_template_profiles(raw)
    configured = (
        profile_by_id(profiles, profile_id)
        or profile_by_id(profiles, DEFAULT_TEMPLATE_PROFILE_ID)
        or profiles[0]
    )
    if (
        str(provider or "").strip().casefold() == "horde"
        and configured.id == DEFAULT_TEMPLATE_PROFILE_ID
    ):
        return (
            profile_by_id(profiles, DEFAULT_INSTRUCT_PROFILE_ID)
            or configured
        )
    return configured


def render_template(template: str, values: Mapping[str, Any]) -> str:
    """Render known ``{{placeholder}}`` tokens without exposing unknown tags."""

    return _PLACEHOLDER_RE.sub(
        lambda match: str(values.get(match.group(1), "")),
        template,
    ).strip()


def _stop_sequences(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    return tuple(
        dict.fromkeys(
            str(item)[:200]
            for item in values
            if str(item).strip()
        )
    )[:8]


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)
