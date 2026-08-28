"""P BrainZ API router hub.

Endpoint implementations live in focused modules while this file owns the
application's route assembly.
"""

import logging

from fastapi import APIRouter

from .chat_routes import create_chat_completion
from .chat_routes import router as chat_router
from .control_routes import (
    clear_ui_debug_traces,
    control_panel_chat,
    mock_chat,
    refresh_ui_models,
    seed_mock_memories,
    ui_debug_traces,
    ui_delete_memory,
    ui_logs,
    ui_memory,
    ui_memory_worlds,
    ui_status,
    update_ui_bridge,
    update_ui_debug_trace_settings,
    update_ui_settings,
)
from .control_routes import router as control_router
from .system_routes import health, list_models
from .system_routes import router as system_router

router = APIRouter()
# Copy the child routes into one flat application router so route registration
# stays explicit and independent of FastAPI's lazy include behavior.
for child_router in (system_router, control_router, chat_router):
    router.routes.extend(child_router.routes)
LOGGER = logging.getLogger(__name__)

__all__ = [
    "router",
    "health",
    "ui_status",
    "update_ui_settings",
    "update_ui_bridge",
    "refresh_ui_models",
    "ui_logs",
    "ui_debug_traces",
    "update_ui_debug_trace_settings",
    "clear_ui_debug_traces",
    "list_models",
    "control_panel_chat",
    "ui_memory_worlds",
    "ui_memory",
    "ui_delete_memory",
    "seed_mock_memories",
    "mock_chat",
    "create_chat_completion",
]
