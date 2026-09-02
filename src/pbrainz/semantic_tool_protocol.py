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
_ORPHAN_ACTION_TAG_RE = re.compile(
    r"<projecthoomans-action\b[^>]*>.*$",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_CLOSE_TAG_RE = re.compile(
    r"</projecthoomans-action\s*>",
    re.IGNORECASE,
)
_PROVIDER_SCAFFOLD_LINE_RE = re.compile(
    r"(?im)^\s*(?:instruction|response|answer|analysis|"
    r"self[- ]correction(?:\s+check)?|final\s+check|new\s+attempt|"
    r"player(?:['’]s)?\s+last\s+message|"
    r"[A-Za-z][\w -]*['’]s\s+required\s+action)\s*:?"
)
_GENERIC_SOCIAL_REPLY_RE = re.compile(
    r"^(?:"
    r"(?:i['’]?ll|i\s+will)\s+(?:check|handle|take\s+care\s+of|look\s+into)\b.*|"
    r"let\s+me\s+(?:check|handle|look\s+into)\b.*|"
    r"(?:give\s+me|one)\s+moment\b.*|"
    r"i\s+hear\s+you|noted|understood|okay|sure"
    r")[.!?\s]*$",
    re.IGNORECASE,
)
_SOCIAL_INSULT_RE = re.compile(
    r"\b(?:"
    r"fuck(?:ing|er|ed)?|shit(?:ty|head)?|bitch|bastard|asshole|"
    r"idiot|moron|stupid|dumb(?:ass)?|jerk|loser|scumbag|prick|"
    r"suck(?:s|ed)?|crap|damn(?:it|ed)?|"
    r"hate\s+you|screw\s+you|shut\s+(?:the\s+)?(?:hell\s+)?up|"
    r"go\s+to\s+hell|piece\s+of\s+shit"
    r")\b",
    re.IGNORECASE,
)
_SOCIAL_SEXUAL_ADVANCE_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:want|wanna|would\s+like)\s+(?:to\s+)?"
    r"(?:have\s+sex|fuck|sleep\s+with)"
    r"(?:\s+(?:with\s+)?(?:you|me|u|babe|baby))?|"
    r"have\s+sex\s+with\s+(?:me|you|u)|"
    r"sleep\s+with\s+(?:me|you|u)|"
    r"fuck\s+me|"
    r"come\s+to\s+bed\s+with\s+me"
    r")\b",
    re.IGNORECASE,
)
_IDENTITY_NAME_RE = re.compile(
    r"\b(?:what(?:'s| is)\s+(?:your|ur)\s+name|"
    r"who\s+are\s+you|tell\s+(?:me\s+)?your\s+name|"
    r"may\s+i\s+know\s+your\s+name|introduce\s+yourself)\b",
    re.IGNORECASE,
)
_SOCIAL_REACTION_RE = (
    ("admire", re.compile(
        r"\b(?:i\s+)?(?:admire|respect)\s+(?:you|your)\b|"
        r"\b(?:you(?:'re| are)|that's)\s+(?:amazing|incredible|impressive)\b",
        re.IGNORECASE,
    )),
    ("flirt", re.compile(
        r"\b(?:i\s+(?:like|love)\s+you|you(?:'re| are)\s+"
        r"(?:(?:really|very|incredibly|so)\s+)?"
        r"(?:beautiful|handsome|attractive|cute|hot)|"
        r"(?:go|want)\s+on\s+a\s+date|want\s+to\s+date\s+you|"
        r"take\s+you\s+out|kiss(?:\s+(?:me|you))?)\b",
        re.IGNORECASE,
    )),
    ("comfort", re.compile(
        r"\b(?:i'?m\s+here\s+for\s+you|you'?re\s+not\s+alone|"
        r"are\s+you\s+okay|it'?ll\s+be\s+okay|don'?t\s+worry)\b",
        re.IGNORECASE,
    )),
    ("apologize", re.compile(
        r"\b(?:i'?m\s+sorry|please\s+forgive\s+me|my\s+apologies)\b",
        re.IGNORECASE,
    )),
    ("praise", re.compile(
        r"\b(?:good\s+job|well\s+done|nice\s+work|"
        r"you(?:'re| are)\s+(?:great|awesome|brave|smart))\b",
        re.IGNORECASE,
    )),
)


def is_social_insult(value: object) -> bool:
    message = str(value or "")
    # "fuck" is intentionally part of the hostile vocabulary for phrases
    # such as "fuck you", but a proposition such as "wanna fuck babe" is a
    # different social action and must not be written to the insult channel.
    if _SOCIAL_SEXUAL_ADVANCE_RE.search(message) is not None:
        return False
    return _SOCIAL_INSULT_RE.search(message) is not None


def is_name_question(value: object) -> bool:
    return _IDENTITY_NAME_RE.search(str(value or "")) is not None


def infer_social_intent(value: object) -> dict[str, Any] | None:
    """Infer a conservative gameplay kind plus a presentation subtype.

    Subtypes are descriptive metadata.  The server still authorizes the
    gameplay ``reaction`` and never trusts this classifier for relationship
    deltas or consent.
    """
    message = str(value or "")
    if _SOCIAL_SEXUAL_ADVANCE_RE.search(message) is not None:
        return {
            "reaction": "flirt",
            "subtype": "sexual_advance",
            "explicit": True,
        }
    if is_social_insult(message):
        return {
            "reaction": "insult",
            "subtype": "hostile_abuse",
            "explicit": False,
        }
    for kind, pattern in _SOCIAL_REACTION_RE:
        if pattern.search(message):
            subtype = None
            if kind in {"admire", "praise"}:
                subtype = "compliment"
            elif kind == "flirt":
                subtype = "romantic_interest"
            result: dict[str, Any] = {"reaction": kind}
            if subtype:
                result["subtype"] = subtype
            return result
    return None


def infer_social_reaction(value: object) -> str | None:
    """Backward-compatible gameplay kind accessor."""
    intent = infer_social_intent(value)
    return str(intent["reaction"]) if intent else None


def social_reply_repair_needed(message: object, response: object) -> bool:
    """Return whether one bounded text-only repair is worth the extra call.

    Only high-confidence explicit social subtypes use this path.  A natural
    NPC refusal such as "No." is preserved; generic provider acknowledgements
    are repaired into direct in-world dialogue.
    """
    intent = infer_social_intent(message)
    if not intent or intent.get("subtype") not in {
        "sexual_advance",
        "hostile_abuse",
    }:
        return False
    text = str(response or "")
    text = _ACTION_TAG_RE.sub("", text)
    text = _ORPHAN_ACTION_TAG_RE.sub("", text)
    text = _ACTION_CLOSE_TAG_RE.sub("", text).strip()
    return not text or _GENERIC_SOCIAL_REPLY_RE.fullmatch(text) is not None


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
    # Horde and other text-only gateways can stop in the middle of the
    # provider-emitted action envelope.  Never show that protocol fragment as
    # NPC dialogue; the provider-neutral intent fallback will still recover
    # explicit social/name requests from the player's message.
    cleaned = _ORPHAN_ACTION_TAG_RE.sub("", cleaned)
    cleaned = _ACTION_CLOSE_TAG_RE.sub("", cleaned)
    return cleaned.strip(), calls


def is_provider_scaffold(response: object) -> bool:
    """Return whether provider text contains a leaked prompt-template label."""
    text = str(response or "").strip()
    return bool(text and _PROVIDER_SCAFFOLD_LINE_RE.search(text))


def strip_provider_scaffold(response: object) -> str:
    """Remove chat-template instructions that leaked into provider text.

    Horde-style text endpoints can emit their imagined ``Instruction`` /
    ``Response`` turn instead of only the NPC reply.  Treat that whole text as
    untrusted presentation data; semantic calls are extracted independently.
    Keep any dialogue before the scaffold. Returning an empty string means the
    provider emitted no usable dialogue before its template block.
    """
    text = str(response or "").strip()
    match = _PROVIDER_SCAFFOLD_LINE_RE.search(text)
    if match is None:
        return text
    prefix = text[: match.start()].strip()
    # Horde sometimes puts a Markdown-like divider between the reply and its
    # copied template. It is not part of NPC dialogue.
    prefix = re.sub(r"(?:\n\s*[-_=]{3,}\s*)+$", "", prefix).strip()
    return prefix


def _request_message(request: dict[str, Any]) -> str:
    context = request.get("conversation_context") or request.get("context") or request
    if isinstance(context, dict):
        message = str(
            context.get("message")
            or context.get("current_player_message")
            or context.get("currentPlayerMessage")
            or ""
        )
        if message:
            return message
    for item in reversed(request.get("messages") or []):
        if isinstance(item, dict) and item.get("role") == "user":
            message = str(item.get("content") or "")
            if message:
                return message
    return ""


def ensure_social_intent(
    calls: list[dict[str, Any]],
    request: dict[str, Any],
    *,
    reason: str = "provider_text_action_fallback",
) -> list[dict[str, Any]]:
    """Recover and normalize social intent across native and text providers.

    High-confidence sexual propositions and hostile abuse are allowed to
    correct a provider's contradictory ``kind``.  This prevents ``fuck you``
    and ``wanna fuck babe`` from sharing one relationship path while leaving
    ambiguous language under provider/game authority.
    """
    message = _request_message(request)
    intent = infer_social_intent(message)
    for index, call in enumerate(calls):
        if tool_call_name(call) != "social_react":
            continue
        if not intent:
            return calls
        arguments = call.get("arguments") if isinstance(call, dict) else None
        arguments = dict(arguments) if isinstance(arguments, dict) else {}
        current_reaction = str(
            arguments.get("kind") or arguments.get("reaction") or ""
        ).strip().lower()
        subtype = intent.get("subtype")
        high_confidence = subtype in {"sexual_advance", "hostile_abuse"}
        same_kind = current_reaction == str(intent["reaction"])
        if not high_confidence and (arguments.get("subtype") or not same_kind):
            return calls
        updated = dict(call)
        if high_confidence:
            arguments["kind"] = intent["reaction"]
        if subtype:
            arguments["subtype"] = subtype
        if "explicit" in intent:
            arguments["explicit"] = intent["explicit"]
        updated["arguments"] = arguments
        normalized_calls = list(calls)
        normalized_calls[index] = updated
        return normalized_calls
    if "social_react" not in exposed_tool_names(request) or not intent:
        return calls
    reaction = str(intent["reaction"])
    request_id = str(request.get("request_id") or "unknown")
    LOGGER.info(
        "Provider-neutral semantic fallback npc=%s request=%s reaction=%s subtype=%s reason=%s",
        str(request.get("npc_id") or "unknown"),
        request_id,
        reaction,
        str(intent.get("subtype") or ""),
        reason,
    )
    arguments: dict[str, Any] = {"kind": reaction, "intensity": "normal"}
    if intent.get("subtype"):
        arguments["subtype"] = intent["subtype"]
    if "explicit" in intent:
        arguments["explicit"] = intent["explicit"]
    return [
        *calls,
        {
            "id": f"fallback-{reaction}:{request_id}",
            "name": "social_react",
            "arguments": arguments,
            "origin": "provider_neutral_social_fallback",
        },
    ]


def ensure_identity_intent(
    calls: list[dict[str, Any]],
    request: dict[str, Any],
    *,
    reason: str = "provider_text_action_fallback",
) -> list[dict[str, Any]]:
    """Add the authoritative name-disclosure action for a name question."""
    if any(tool_call_name(call) == "ask_name" for call in calls):
        return calls
    if "ask_name" not in exposed_tool_names(request):
        return calls
    if not is_name_question(_request_message(request)):
        return calls
    request_id = str(request.get("request_id") or "unknown")
    LOGGER.info(
        "Provider-neutral semantic fallback npc=%s request=%s tool=ask_name reason=%s",
        str(request.get("npc_id") or "unknown"),
        request_id,
        reason,
    )
    return [
        *calls,
        {
            "id": f"fallback-ask-name:{request_id}",
            "name": "ask_name",
            "arguments": {},
            "origin": "provider_neutral_identity_fallback",
        },
    ]


def tool_call_name(call: object) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()
