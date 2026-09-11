"""Resolve the durable storage location for a memory world.

The game must identify the active world; PBrainZ must not guess by looking for
the newest save directory.  Single-player requests may therefore provide a
relative path below ``<Zomboid>/Saves``.  Client-side multiplayer requests use
an external, namespaced identity because the server's save directory is not
owned by this process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pbrainz.paths import normalize_zomboid_path

_SINGLE_PLAYER = {"singleplayer"}
_MULTIPLAYER = {"multiplayer"}
_MAX_ID_PART = 256
_MAX_SAVE_DEPTH = 4


class MemoryLocationError(ValueError):
    """Raised when a game-provided save locator is unsafe or incomplete."""


def _text(value: Any, limit: int = _MAX_ID_PART) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _mode(value: Any) -> str | None:
    candidate = _text(value, 64).casefold().replace("_", "-")
    if candidate in _SINGLE_PLAYER:
        return "singleplayer"
    if candidate in _MULTIPLAYER:
        return "multiplayer"
    return None


def normalize_save_relative_path(value: Any) -> str:
    """Normalize a game save path while retaining only relative components.

    An empty result is intentionally not treated as a valid save path.  The
    resolver later rejects it instead of silently selecting another storage
    location.
    """

    raw_text = str(value or "").strip()
    if "\x00" in raw_text or len(raw_text) > _MAX_ID_PART:
        return ""
    text = raw_text.replace("\\", "/")
    if not text or text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
        return ""
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or ".." in parts or len(parts) > _MAX_SAVE_DEPTH:
        return ""
    return "/".join(parts)


def _safe_component(value: str) -> str:
    # The canonical ID is metadata and a cache key, not a filesystem path. A
    # delimiter is escaped so the ID remains unambiguous for diagnostics.
    return value.replace("|", "%7C")


@dataclass(frozen=True, slots=True)
class MemoryIdentity:
    """Stable logical identity plus the information needed to locate it."""

    world_uuid: str
    storage_mode: str
    save_relative_path: str = ""
    server_instance_id: str = ""
    server_world_generation: str = ""

    def __post_init__(self) -> None:
        if not self.world_uuid.strip():
            raise MemoryLocationError("canonical memory world identity is required")
        if self.storage_mode == "singleplayer":
            if not self.world_uuid.startswith("sp-v1|") or not self.save_relative_path:
                raise MemoryLocationError("invalid single-player memory identity")
        elif self.storage_mode == "multiplayer":
            if (
                not self.world_uuid.startswith("mp-v1|")
                or not self.server_instance_id
                or not self.server_world_generation
            ):
                raise MemoryLocationError("invalid multiplayer memory identity")
        else:
            raise MemoryLocationError("invalid memory storage mode")

    @classmethod
    def from_mapping(
        cls,
        raw_world_uuid: Any,
        context: Mapping[str, Any] | None = None,
    ) -> MemoryIdentity:
        """Build an identity from the explicit bridge lifecycle fields."""

        raw = _text(raw_world_uuid, _MAX_ID_PART)
        if not raw:
            raise MemoryLocationError("world_uuid is required")
        values = context or {}
        mode = _mode(
            _first(values, "world_mode", "worldMode", "game_mode", "gameMode", "mode")
        )
        if mode is None:
            raise MemoryLocationError(
                "world_mode must be exactly 'singleplayer' or 'multiplayer'"
            )
        raw_save_path = _first(
            values,
            "save_relative_path",
            "saveRelativePath",
            "save_path",
            "savePath",
        )
        save_path = normalize_save_relative_path(raw_save_path)

        server_id = _text(
            _first(
                values,
                "server_instance_id",
                "serverInstanceId",
                "server_id",
                "serverId",
            )
        )
        generation = _text(
            _first(
                values,
                "server_world_generation",
                "serverWorldGeneration",
                "server_world_id",
                "serverWorldId",
                "world_generation",
                "worldGeneration",
            )
        )
        if mode == "singleplayer":
            if raw_save_path is None or not save_path:
                raise MemoryLocationError(
                    "single-player memory requires a valid save_relative_path"
                )
            canonical = f"sp-v1|{_safe_component(save_path)}"
            return cls(canonical, mode, save_path)
        if not server_id or not generation:
            raise MemoryLocationError(
                "multiplayer memory requires server_instance_id and "
                "server_world_generation"
            )
        canonical = f"mp-v1|{_safe_component(server_id)}|{_safe_component(generation)}"
        return cls(canonical, mode, "", server_id, generation)

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> MemoryIdentity:
        """Build the canonical identity when the game has no UUID field.

        The game-side bridge reports the stable save/server components rather
        than duplicating PBrainZ's canonical-ID format. Keeping construction
        here prevents the Lua and Python sides from drifting apart.
        """

        values = context if isinstance(context, Mapping) else {}
        mode = _mode(
            _first(values, "world_mode", "worldMode", "game_mode", "gameMode", "mode")
        )
        if mode == "singleplayer":
            save_path = normalize_save_relative_path(
                _first(
                    values,
                    "save_relative_path",
                    "saveRelativePath",
                    "save_path",
                    "savePath",
                )
            )
            if not save_path:
                raise MemoryLocationError(
                    "single-player memory requires a valid save_relative_path"
                )
            return cls.from_mapping(f"sp-v1|{_safe_component(save_path)}", values)
        if mode == "multiplayer":
            server_id = _text(
                _first(
                    values,
                    "server_instance_id",
                    "serverInstanceId",
                    "server_id",
                    "serverId",
                )
            )
            generation = _text(
                _first(
                    values,
                    "server_world_generation",
                    "serverWorldGeneration",
                    "server_world_id",
                    "serverWorldId",
                    "world_generation",
                    "worldGeneration",
                )
            )
            if not server_id or not generation:
                raise MemoryLocationError(
                    "multiplayer memory requires server_instance_id and "
                    "server_world_generation"
                )
            canonical = (
                f"mp-v1|{_safe_component(server_id)}|{_safe_component(generation)}"
            )
            return cls(canonical, mode, "", server_id, generation)
        raise MemoryLocationError(
            "world_mode must be exactly 'singleplayer' or 'multiplayer'"
        )

    @property
    def cache_key(self) -> str:
        return f"{self.storage_mode}|{self.world_uuid}|{self.save_relative_path}"

    def root_for(self, settings: Any) -> Path:
        """Return the approved memory directory for this identity."""

        from .sqlite import memory_root_for_settings

        # An explicit root remains authoritative for portable/test/custom
        # setups. Multiplayer data always uses this external root.
        if self.storage_mode != "singleplayer" or getattr(settings, "memory_root", None):
            return memory_root_for_settings(settings)
        if not self.save_relative_path:
            raise MemoryLocationError("single-player memory requires save_relative_path")

        saves_root = (
            normalize_zomboid_path(getattr(settings, "zomboid_path", None)) / "Saves"
        ).resolve()
        relative = Path(*self.save_relative_path.split("/"))
        save_root = (saves_root / relative).resolve()
        try:
            save_root.relative_to(saves_root)
        except ValueError as error:
            raise MemoryLocationError(
                "save_relative_path escapes the Zomboid Saves directory"
            ) from error
        if not save_root.is_dir():
            raise MemoryLocationError(
                f"active save directory does not exist: {self.save_relative_path}"
            )
        memory_root = save_root / "PBrainZ" / "memory"
        try:
            memory_root.resolve().relative_to(save_root)
        except ValueError as error:
            raise MemoryLocationError(
                "save-local PBrainZ memory directory escapes the save directory"
            ) from error
        return memory_root


def _save_memory_roots(settings: Any, limit: int = 128) -> list[tuple[Path, str]]:
    """Find only the supported save-local memory locations for the browser."""

    saves_root = (
        normalize_zomboid_path(getattr(settings, "zomboid_path", None)) / "Saves"
    ).resolve()
    if not saves_root.is_dir():
        return []
    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    # Standard PZ layout is Saves/<mode>/<save>/PBrainZ/memory. The one-level
    # pattern also keeps this compatible with custom save layouts.
    for pattern in ("*/PBrainZ/memory", "*/*/PBrainZ/memory"):
        try:
            candidates = saves_root.glob(pattern)
        except OSError:
            continue
        for candidate in candidates:
            try:
                root = candidate.resolve()
                relative_root = root.relative_to(saves_root)
                save_relative = str(relative_root.parent.parent).replace(os.sep, "/")
                if root in seen or not root.is_dir() or not save_relative:
                    continue
            except (OSError, ValueError):
                continue
            seen.add(root)
            roots.append((root, save_relative))
            if len(roots) >= limit:
                return roots
    return roots


def list_memory_worlds(settings: Any, limit: int = 64) -> list[dict[str, Any]]:
    """List external and approved save-local memory databases."""

    from .sqlite import SQLiteMemoryStore, memory_root_for_settings

    bounded = max(1, min(int(limit), 128))
    locations: list[tuple[Path, str, str | None]] = [
        (memory_root_for_settings(settings), "external", None)
    ]
    locations.extend(
        (root, "save_local", save_relative)
        for root, save_relative in _save_memory_roots(settings)
    )
    worlds: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root, storage_kind, save_relative in locations:
        for item in SQLiteMemoryStore.list_worlds(root, limit=bounded):
            key = (str(item["world_uuid"]), str(item["path"]))
            if key in seen:
                continue
            seen.add(key)
            item = dict(item)
            item["storage_kind"] = storage_kind
            if save_relative:
                item["save_relative_path"] = save_relative
            item["_root"] = root
            worlds.append(item)
            if len(worlds) >= bounded:
                return worlds
    return worlds


def public_memory_world(item: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the internal filesystem locator before returning API JSON."""

    return {key: value for key, value in item.items() if not key.startswith("_")}
