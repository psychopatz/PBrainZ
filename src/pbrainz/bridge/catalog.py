"""Bounded cache and hydration helpers for Core-owned tool catalogs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .protocol import CORE_NAMESPACE, TOOL_CATALOG_COMMAND, BridgeClientError
from .state import BridgeState

MAX_CATALOGS = 8
MAX_TOOLS = 256


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    catalog_id: str
    catalog_version: int
    definitions: dict[str, dict[str, Any]]


class ToolCatalogCache:
    """Keep static tool schemas in memory; live availability stays in requests."""

    def __init__(self, max_catalogs: int = MAX_CATALOGS) -> None:
        self._catalogs: OrderedDict[str, ToolCatalog] = OrderedDict()
        self.max_catalogs = max(1, min(int(max_catalogs), MAX_CATALOGS))

    def get(self, catalog_id: str | None) -> ToolCatalog | None:
        if not catalog_id:
            return None
        catalog = self._catalogs.get(catalog_id)
        if catalog is not None:
            self._catalogs.move_to_end(catalog_id)
        return catalog

    def add(self, payload: object) -> ToolCatalog:
        if not isinstance(payload, dict):
            raise BridgeClientError("tool catalog is not an object")
        catalog_id = payload.get("catalog_id")
        catalog_version = payload.get("catalog_version")
        rows = payload.get("tools")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise BridgeClientError("tool catalog has no catalog ID")
        if isinstance(catalog_version, bool) or not isinstance(catalog_version, int):
            raise BridgeClientError("tool catalog has an invalid version")
        if not isinstance(rows, list) or len(rows) > MAX_TOOLS:
            raise BridgeClientError("tool catalog has an invalid tool list")

        definitions: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise BridgeClientError("tool catalog contains an invalid row")
            tool_id = row.get("id")
            definition = row.get("definition")
            if not isinstance(tool_id, str) or not tool_id:
                raise BridgeClientError("tool catalog contains an invalid tool ID")
            if not isinstance(definition, dict):
                raise BridgeClientError(f"tool catalog entry '{tool_id}' has no definition")
            if tool_id in definitions:
                raise BridgeClientError(f"tool catalog contains duplicate tool '{tool_id}'")
            definitions[tool_id] = definition

        catalog = ToolCatalog(catalog_id, catalog_version, definitions)
        self._catalogs[catalog_id] = catalog
        self._catalogs.move_to_end(catalog_id)
        while len(self._catalogs) > self.max_catalogs:
            self._catalogs.popitem(last=False)
        return catalog

    async def ensure(self, client: Any, state: BridgeState) -> ToolCatalog | None:
        """Fetch a catalog once when the Core runtime advertises an unknown ID."""
        catalog_id = state.tool_catalog_id
        if not catalog_id or not state.runtime_id:
            return None
        cached = self.get(catalog_id)
        if cached is not None:
            return cached
        payload = await client.call(
            CORE_NAMESPACE,
            TOOL_CATALOG_COMMAND,
            {},
            state.runtime_id,
        )
        if not isinstance(payload, dict) or payload.get("catalog_id") != catalog_id:
            raise BridgeClientError("tool catalog changed during handshake")
        return self.add(payload)


def hydrate_request(request: dict[str, Any], cache: ToolCatalogCache) -> dict[str, Any]:
    """Expand only the currently allowed tool IDs before provider invocation."""
    context_key = "conversation_context" if isinstance(
        request.get("conversation_context"), dict
    ) else "context"
    context = request.get(context_key)
    if not isinstance(context, dict):
        return request
    catalog_id = context.get("tool_catalog_id")
    tool_ids = context.get("available_tool_ids")
    if not isinstance(catalog_id, str) or not isinstance(tool_ids, list):
        return request
    catalog = cache.get(catalog_id)
    if catalog is None:
        return request

    selected: list[dict[str, Any]] = []
    for value in tool_ids[:MAX_TOOLS]:
        if not isinstance(value, str):
            continue
        definition = catalog.definitions.get(value)
        if definition is not None:
            selected.append(definition)

    hydrated_context = dict(context)
    hydrated_context["available_tools"] = selected
    hydrated_context["resolved_tool_catalog_id"] = catalog.catalog_id
    hydrated = dict(request)
    hydrated[context_key] = hydrated_context
    return hydrated
