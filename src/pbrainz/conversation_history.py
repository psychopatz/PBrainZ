"""Provider-safe projection of recent game conversation history."""

from __future__ import annotations

from pbrainz.api.models import ChatMessage
from pbrainz.memory.types import ConversationTurn

_CURRENT_MESSAGE_MARKER = "\n\n[Current player message]\n"
HISTORY_START_ANCHOR = (
    "[Conversation history begins with an NPC reply; the preceding player "
    "message is outside this retained window.]"
)


def normalize_recent_turns(
    turns: tuple[ConversationTurn, ...],
) -> tuple[tuple[ConversationTurn, ...], int]:
    """Return provider-safe history while preserving every retained reply.

    Bridge history can begin in the middle of a conversation window. A
    leading NPC response is not a valid first conversational turn for several
    chat APIs, so add a neutral synthetic user anchor before it. Keep all
    actual turns, including consecutive turns from the same role, because
    game events can legitimately produce them.
    """

    normalized: list[ConversationTurn] = []
    anchored_leading_assistant = 0
    for turn in turns:
        content = str(turn.content or "").strip()
        if not content:
            continue
        role = "user" if str(turn.role).casefold() == "user" else "assistant"
        if not normalized and role == "assistant":
            normalized.append(
                ConversationTurn(
                    role="user",
                    content=HISTORY_START_ANCHOR,
                    metadata={"synthetic": "history_start_anchor"},
                )
            )
            anchored_leading_assistant += 1
        if role == turn.role and content == turn.content:
            normalized.append(turn)
        else:
            normalized.append(
                ConversationTurn(
                    role=role,
                    content=content,
                    created_at=turn.created_at,
                    turn_index=turn.turn_index,
                    metadata=turn.metadata,
                    message_id=turn.message_id,
                    game_day=turn.game_day,
                    world_age_hours=turn.world_age_hours,
                    speaker_uuid=turn.speaker_uuid,
                    speaker_name=turn.speaker_name,
                    speaker_kind=turn.speaker_kind,
                )
            )
    return tuple(normalized), anchored_leading_assistant


def coalesce_adjacent_text_messages(
    messages: list[ChatMessage],
    *,
    terminal_is_current: bool = False,
) -> int:
    """Merge adjacent text turns with the same chat role.

    The game context is represented as a user message so it remains data, not
    an instruction channel. The current player message is also a user message,
    which can produce consecutive user turns when there is no assistant reply
    in the retained history. Coalescing preserves all text while keeping the
    common request contract provider-neutral.

    Tool and function messages are deliberately excluded: their structured
    lifecycle must remain intact for providers that support tool calling.
    """

    if not messages:
        return 0
    coalesced = 0
    normalized = [messages[0]]
    mergeable_roles = {"system", "user", "assistant"}
    last_index = len(messages) - 1
    for index, message in enumerate(messages[1:], start=1):
        previous = normalized[-1]
        if (
            previous.role == message.role
            and message.role in mergeable_roles
            and isinstance(previous.content, str)
            and isinstance(message.content, str)
        ):
            separator = (
                _CURRENT_MESSAGE_MARKER
                if terminal_is_current and index == last_index
                else "\n\n"
            )
            previous.content = (
                f"{previous.content.rstrip()}{separator}{message.content.lstrip()}"
            )
            if previous.name is None and message.name is not None:
                previous.name = message.name
            coalesced += 1
            continue
        normalized.append(message)
    messages[:] = normalized
    return coalesced


def select_recent_turns(
    turns: tuple[ConversationTurn, ...], limit: int
) -> list[ConversationTurn]:
    """Keep the recent window while retaining a synthetic history anchor."""

    bounded_limit = max(1, int(limit))
    recent = list(turns[-bounded_limit:])
    anchor = next(
        (
            turn
            for turn in turns
            if (turn.metadata or {}).get("synthetic") == "history_start_anchor"
        ),
        None,
    )
    if anchor is not None and anchor not in recent:
        recent.insert(0, anchor)
    return recent


def trim_current_message(content: str, limit: int) -> str:
    """Trim a coalesced user turn without discarding its live input."""

    if _CURRENT_MESSAGE_MARKER not in content:
        return content[:limit]
    context, current = content.split(_CURRENT_MESSAGE_MARKER, 1)
    if len(current) >= limit:
        return current[:limit]
    context_limit = max(0, limit - len(_CURRENT_MESSAGE_MARKER) - len(current))
    return f"{context[:context_limit]}{_CURRENT_MESSAGE_MARKER}{current}"
