"""Provider-neutral normalization for game semantic actions.

Providers may return native function calls or plain text.  Both forms are
converted into the same bounded action records before Project Hoomans sees
them; this module does not execute gameplay or decide whether an action is
allowed.
"""

from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from typing import Any

LOGGER = logging.getLogger(__name__)

_ACTION_TAG_RE = re.compile(
    r"<projecthoomans-action>\s*(?P<body>\{.*?\})\s*</projecthoomans-action>",
    re.IGNORECASE | re.DOTALL,
)
_SOCIAL_INSULT_RE = re.compile(
    r"\b(?:"
    r"fuck(?:ing|er|ed)?|shit(?:ty|head)?|bitch|bastard|asshole|"
    r"idiot|moron|stupid|dumb(?:ass)?|jerk|loser|"
    r"hate\s+you|screw\s+you|shut\s+up|piece\s+of\s+shit"
    r")\b",
    re.IGNORECASE,
)


def is_social_insult(value: object) -> bool:
    return _SOCIAL_INSULT_RE.search(str(value or "")) is not None


def exposed_tool_names(request: dict[str, Any]) -> set[str]:
    context = request.get("conversation_context") or request.get("context") or request
    if not isinstance(context, dict):
        return set()
    names: set[str] = set()
    for tool in context.get("available_tools", []):
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]).strip())
    for tool_id in context.get("available_tool_ids") or []:
        if isinstance(tool_id, str) and tool_id.startswith("projecthoomans.llm:"):
            names.add(tool_id[len("projecthoomans.llm:") :].strip())
    return {name for name in names if name}


def extract_text_tool_calls(
    response: str,
    request: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract bounded action envelopes from a provider's plain-text reply."""
    exposed = exposed_tool_names(request)
    calls: list[dict[str, Any]] = []
    matches = list(_ACTION_TAG_RE.finditer(str(response or "")))[:8]
    for index, match in enumerate(matches, start=1):
        try:
            value = json.loads(match.group("body"))
        except (JSONDecodeError, TypeError):
            LOGGER.warning(
                "Provider text action was not valid JSON npc=%s request=%s index=%s",
                str(request.get("npc_id") or "unknown"),
                str(request.get("request_id") or "unknown"),
                index,
            )
            continue
        if not isinstance(value, dict):
            continue
        function = value.get("function") if isinstance(value.get("function"), dict) else value
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments") or {}
        if not name or name not in exposed:
            LOGGER.warning(
                "Provider text action rejected npc=%s request=%s name=%s",
                str(request.get("npc_id") or "unknown"),
                str(request.get("request_id") or "unknown"),
                name or "<missing>",
            )
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            {
                "id": f"text-action:{request.get('request_id') or 'unknown'}:{index}",
                "name": name,
                "arguments": {
                    str(key): item for key, item in list(arguments.items())[:16]
                },
                "origin": "provider_text_action",
            }
        )
    cleaned = _ACTION_TAG_RE.sub("", str(response or ""))
    return cleaned.strip(), calls


def ensure_social_intent(
    calls: list[dict[str, Any]],
    request: dict[str, Any],
    *,
    reason: str = "provider_text_action_fallback",
) -> list[dict[str, Any]]:
    """Add the same social action when a provider returns only plain text."""
    if any(
        tool_call_name(call) == "social_react"
        for call in calls
    ):
        return calls
    if "social_react" not in exposed_tool_names(request):
        return calls
    context = request.get("conversation_context") or request.get("context") or request
    message = ""
    if isinstance(context, dict):
        message = str(
            context.get("message")
            or context.get("current_player_message")
            or context.get("currentPlayerMessage")
            or ""
        )
    if not message:
        for item in reversed(request.get("messages") or []):
            if isinstance(item, dict) and item.get("role") == "user":
                message = str(item.get("content") or "")
                if message:
                    break
    if not is_social_insult(message):
        return calls
    request_id = str(request.get("request_id") or "unknown")
    LOGGER.info(
        "Provider-neutral semantic fallback npc=%s request=%s reaction=insult reason=%s",
        str(request.get("npc_id") or "unknown"),
        request_id,
        reason,
    )
    return [
        *calls,
        {
            "id": f"fallback-insult:{request_id}",
            "name": "social_react",
            "arguments": {"kind": "insult", "intensity": "normal"},
            "origin": "provider_neutral_social_fallback",
        },
    ]


def tool_call_name(call: object) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()
