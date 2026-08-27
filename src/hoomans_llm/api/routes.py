"""Stable API router hub for HoomansLLM.

Endpoint implementations live in focused modules. This file intentionally
keeps the historical ``hoomans_llm.api.routes.router`` import path intact and
re-exports the route callables used by integrations and tests.
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
# FastAPI versions that use lazy included-router objects do not recursively
# flatten nested routers when the compatibility hub is included by the app.
# The child routers already contain fully configured APIRoute objects, so copy
# those route objects into the stable hub and keep the public assembly flat.
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
