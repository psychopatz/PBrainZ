"""P BrainZ API router hub.

Endpoint implementations live in focused modules while this file owns the
application's route assembly.
"""

import logging

from fastapi import APIRouter

from .chat_routes import create_chat_completion
from .chat_routes import router as chat_router
from .control_routes import (
    control_panel_chat,
    refresh_ui_models,
    ui_logs,
    ui_status,
    update_ui_bridge,
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
    "list_models",
    "control_panel_chat",
    "create_chat_completion",
]
