"""Typed, idempotent memory primitives received from the game client."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from .locator import MemoryIdentity
from .sqlite import SQLiteMemoryStore
from .types import MemoryRecord, MemoryScope, MemoryType, MemoryVisibility

LOGGER = logging.getLogger(__name__)
MAX_EVENT_ID = 256
MAX_PRIMITIVE_TYPE = 64


def _text(value: Any, limit: int = 256) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    if integer:
        return int(number)
    return number


def normalize_event_time(value: Any) -> dict[str, Any]:
    """Validate the machine fields while preserving a derived display label."""

    source = _mapping(value)
    kind = _text(source.get("kind"), 64).casefold()
    if kind in {"in_world", "in-world", "calendar"}:
        kind = "in_world_calendar"
    if kind == "in_world_calendar":
        game_day = _number(source.get("game_day", source.get("gameDay")), integer=True)
        world_age = _number(
            source.get("world_age_hours", source.get("worldAgeHours"))
        )
        year = _number(source.get("year"), integer=True)
        month = _number(source.get("month"), integer=True)
        day = _number(source.get("day"), integer=True)
        hour = _number(source.get("hour"), integer=True)
        minute = _number(source.get("minute", source.get("minutes")), integer=True)
        if game_day is None or world_age is None:
            raise ValueError("in-world event time requires game_day and world_age_hours")
        calendar_values = (year, month, day)
        if any(value is not None for value in calendar_values) and not (
            year is not None
            and 1 <= year <= 9999
            and month is not None
            and 1 <= month <= 12
            and day is not None
            and 1 <= day <= 31
        ):
            raise ValueError("in-world event time has invalid calendar fields")
        if hour is not None and not 0 <= hour <= 23:
            raise ValueError("in-world event time has invalid hour")
        if minute is not None and not 0 <= minute <= 59:
            raise ValueError("in-world event time has invalid minute")
        hour = hour if hour is not None else 0
        minute = minute if minute is not None else 0
        iso = _text(source.get("iso"), 32) if year is not None else ""
        if not iso and year is not None:
            iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
        label = _text(source.get("label"), 160)
        if not label:
            label = (
                f"{iso[:10]} (Day {game_day}, {hour:02d}:{minute:02d})"
                if iso
                else f"In-world day {game_day} ({hour:02d}:{minute:02d})"
            )
        return {
            "kind": kind,
            "game_day": game_day,
            "world_age_hours": world_age,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "iso": iso,
            "label": label,
        }
    if kind in {"pre_outbreak", "before_outbreak", "pre-outbreak"}:
        return {
            "kind": "pre_outbreak",
            "phase": "before_outbreak",
            "game_day": None,
            "world_age_hours": None,
            "calendar_date": None,
            "label": "Before the outbreak",
        }
    raise ValueError("event time kind must be in_world_calendar or pre_outbreak")


def _identity(event: Mapping[str, Any]) -> MemoryIdentity:
    world = _mapping(
        event.get("world")
        or event.get("memory_context")
        or event.get("memoryContext")
    )
    context = dict(world)
    context.update(event)
    raw_uuid = (
        event.get("world_uuid")
        or event.get("worldUUID")
        or event.get("save_uuid")
        or event.get("saveUUID")
        or context.get("world_uuid")
    )
    if raw_uuid:
        return MemoryIdentity.from_mapping(raw_uuid, context)
    return MemoryIdentity.from_context(context)


def _stable_id(primitive_type: str, player_uuid: str, npc_uuid: str, relation: str) -> str:
    parts = ["pnc", "memory", "v1", primitive_type, player_uuid, npc_uuid]
    if relation:
        parts.append(relation)
    return ":".join(_text(part, MAX_EVENT_ID) for part in parts)


def _first_meeting(event: Mapping[str, Any], name: str, event_time: Mapping[str, Any]) -> str:
    return f"The player first met {name} on {event_time['label']}."


def _pre_outbreak(event: Mapping[str, Any], name: str, event_time: Mapping[str, Any]) -> str:
    relation = _text(event.get("relationship_kind", event.get("relationshipKind")), 64)
    relation = relation.replace("_", " ").casefold()
    if relation in {"father", "mother", "brother", "sister"}:
        return f"Before the outbreak, {name} was the player's {relation}."
    if relation in {"lover", "partner", "spouse"}:
        return f"Before the outbreak, {name} and the player were romantic partners."
    if relation == "friend":
        return f"Before the outbreak, {name} and the player were friends."
    relationship = relation or "a lifelong"
    return f"Before the outbreak, {name} and the player shared {relationship} relationship."


_RENDERERS: dict[str, Callable[[Mapping[str, Any], str, Mapping[str, Any]], str]] = {
    "first_meeting": _first_meeting,
    "pre_outbreak_relationship": _pre_outbreak,
}


def register_primitive(
    primitive_type: str,
    renderer: Callable[[Mapping[str, Any], str, Mapping[str, Any]], str],
) -> None:
    """Register a deterministic renderer for a future primitive type."""

    key = _text(primitive_type, MAX_PRIMITIVE_TYPE)
    if not key or not callable(renderer):
        raise ValueError("primitive type and renderer are required")
    _RENDERERS[key] = renderer


class MemoryPrimitiveService:
    """Validate and store typed event memories in the existing memory layer."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._stores: dict[str, SQLiteMemoryStore] = {}

    def record_batch(self, batch: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(batch, Mapping):
            raise ValueError("memory primitive batch must be an object")
        events = batch.get("memory_primitives") or batch.get("primitives") or batch.get("events")
        if not isinstance(events, list):
            return ()
        acknowledged: list[str] = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            source_id = _text(event.get("event_id") or event.get("eventID"), MAX_EVENT_ID)
            try:
                self.record(event)
            except (TypeError, ValueError) as error:
                LOGGER.warning(
                    "Skipping invalid memory primitive event=%s: %s",
                    source_id or "unknown",
                    error,
                )
                continue
            if source_id:
                acknowledged.append(source_id)
        return tuple(dict.fromkeys(acknowledged))

    def record(self, event: Mapping[str, Any]) -> MemoryRecord:
        primitive_type = _text(
            event.get("primitive_type") or event.get("primitiveType"),
            MAX_PRIMITIVE_TYPE,
        )
        renderer = _RENDERERS.get(primitive_type)
        if renderer is None:
            raise ValueError(f"unknown memory primitive type: {primitive_type}")
        player_uuid = _text(event.get("player_uuid") or event.get("playerUUID"))
        npc_uuid = _text(event.get("npc_uuid") or event.get("npcUUID"))
        if not player_uuid or not npc_uuid:
            raise ValueError("memory primitive requires player_uuid and npc_uuid")
        identity = _identity(event)
        event_time = normalize_event_time(event.get("event_time") or event.get("eventTime"))
        name = _text(event.get("npc_name") or event.get("npcName"), 256) or "the NPC"
        relation = _text(event.get("relationship_kind") or event.get("relationshipKind"), 64)
        memory_id = _stable_id(primitive_type, player_uuid, npc_uuid, relation)
        scope = MemoryScope(identity.world_uuid, player_uuid, npc_uuid)
        store = self._store(identity)
        store.upsert_entity(npc_uuid, name, entity_kind="npc", source="memory_primitive")
        store.upsert_entity(
            player_uuid,
            _text(event.get("player_name") or event.get("playerName"), 256),
            entity_kind="player",
            source="memory_primitive",
        )
        provenance = {
            "source_system": _text(event.get("source_system") or event.get("sourceSystem"), 64)
            or "project_hoomans",
            "source": _text(event.get("source"), 128) or "game_event",
            "source_event_id": _text(event.get("event_id") or event.get("eventID"), MAX_EVENT_ID),
            "primitive_type": primitive_type,
            "authoritative": event.get("authoritative") is True,
            "event_time": event_time,
        }
        if relation:
            provenance["relationship_kind"] = relation
        variant_key = _text(event.get("variant_key") or event.get("variantKey"), 128)
        if variant_key:
            provenance["variant_key"] = variant_key
        content = _text(renderer(event, name, event_time), 1200)
        return store.remember(
            MemoryRecord(
                memory_id=memory_id,
                scope=scope,
                memory_type=MemoryType.PERSONAL_EVENT,
                content=content,
                tags=("primitive", primitive_type),
                importance=0.9 if primitive_type == "pre_outbreak_relationship" else 0.8,
                provenance=provenance,
                visibility=MemoryVisibility.SYSTEM
                if primitive_type == "pre_outbreak_relationship"
                else MemoryVisibility.PUBLIC,
                game_day=event_time.get("game_day"),
                participants=(player_uuid, npc_uuid),
                entity_refs=(npc_uuid,),
            )
        )

    def _store(self, identity: MemoryIdentity) -> SQLiteMemoryStore:
        root = identity.root_for(self.settings)
        key = identity.cache_key + "|" + str(root)
        if key not in self._stores:
            self._stores[key] = SQLiteMemoryStore(root, identity.world_uuid)
        return self._stores[key]
