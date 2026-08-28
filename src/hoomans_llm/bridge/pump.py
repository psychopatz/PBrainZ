"""Long-running poll loop for Project Hoomans bridge requests."""

from __future__ import annotations

import asyncio
import logging

from hoomans_llm.config import Settings
from hoomans_llm.conversation_service import ConversationService
from hoomans_llm.exceptions import ProviderError
from hoomans_llm.providers.registry import ProviderRegistry
from hoomans_llm.tts import TTSService

from .client import BridgeClient
from .handler import complete_and_deliver, preview, request_message
from .protocol import NAMESPACE, BridgeClientError
from .state import BridgeRuntimeMonitor
from .transport import FileBridgeTransport

LOGGER = logging.getLogger(__name__)


async def run_bridge_pump(
    settings: Settings,
    providers: ProviderRegistry,
    monitor: BridgeRuntimeMonitor,
    tts_service: TTSService | None = None,
) -> None:
    """Poll Project Hoomans chat requests and deliver provider responses."""
    transport = FileBridgeTransport(settings.bridge_root)
    client = BridgeClient(
        transport,
        timeout=max(2.0, min(settings.request_timeout, 30.0)),
    )
    conversation_service = ConversationService(settings, providers)
    observed_runtime_id: str | None = None
    observed_state: tuple[bool, bool, str | None, str | None] | None = None
    while True:
        state = monitor.read()
        state_signature = (state.available, state.ready, state.runtime_id, state.lifecycle)
        if state_signature != observed_state:
            LOGGER.info(
                "Project Hoomans bridge state available=%s ready=%s "
                "lifecycle=%s runtime=%s message=%s",
                state.available,
                state.ready,
                state.lifecycle or "unknown",
                state.runtime_id or "unknown",
                state.message,
            )
            observed_state = state_signature
        if state.ready and state.runtime_id and state.runtime_id != observed_runtime_id:
            recovered = transport.recover_stale_slots(state.runtime_id)
            LOGGER.info(
                "Project Hoomans bridge runtime selected runtime=%s "
                "stale_slots_recovered=%s root=%s",
                state.runtime_id,
                recovered,
                transport.root,
            )
            observed_runtime_id = state.runtime_id
        if not state.ready or not state.runtime_id:
            await asyncio.sleep(settings.bridge_poll_interval)
            continue
        try:
            request = await client.call(NAMESPACE, "pollChat", {}, state.runtime_id)
            if request.get("status") != "pending":
                await asyncio.sleep(settings.bridge_poll_interval)
                continue
            LOGGER.info(
                "NPC task received from Project Hoomans npc=%s request=%s message=%s",
                str(request.get("npc_id") or "unknown"),
                str(request.get("request_id") or "unknown"),
                preview(request_message(request)),
            )
            await complete_and_deliver(
                providers,
                client,
                state,
                request,
                conversation_service=conversation_service,
                tts_service=tts_service,
            )
        except asyncio.CancelledError:
            raise
        except (BridgeClientError, ProviderError, ValueError, TypeError) as error:
            LOGGER.warning(
                "Project Hoomans bridge cycle failed root=%s runtime=%s: %s",
                transport.root,
                state.runtime_id or "unknown",
                error,
            )
            await asyncio.sleep(settings.bridge_poll_interval)
