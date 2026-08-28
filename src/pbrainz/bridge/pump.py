"""Long-running poll loop for Project Hoomans bridge requests."""

from __future__ import annotations

import asyncio
import logging
import time

from pbrainz.config import Settings
from pbrainz.conversation_service import ConversationService, TraceWriter
from pbrainz.exceptions import ProviderError
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.tts import TTSService

from .catalog import ToolCatalogCache, hydrate_request
from .client import BridgeClient
from .handler import complete_and_deliver, preview, request_message
from .protocol import NAMESPACE, BridgeClientError, BridgeCommandError
from .state import BridgeRuntimeMonitor
from .transport import FileBridgeTransport
from .voice import VoicePacketConsumer, voice_channel_available

LOGGER = logging.getLogger(__name__)


async def run_bridge_pump(
    settings: Settings,
    providers: ProviderRegistry,
    monitor: BridgeRuntimeMonitor,
    tts_service: TTSService | None = None,
    *,
    trace_writer: TraceWriter | None = None,
) -> None:
    """Poll Project Hoomans chat requests and deliver provider responses."""
    # The monitor and transport must use the same resolved path. This also
    # lets the controller re-point the worker after a settings change.
    transport = FileBridgeTransport(monitor.root)
    client = BridgeClient(
        transport,
        timeout=max(2.0, min(settings.request_timeout, 30.0)),
    )
    conversation_service = ConversationService(
        settings, providers, trace_writer=trace_writer
    )
    observed_runtime_id: str | None = None
    observed_state: tuple[bool, bool, str | None, str | None, str | None] | None = None
    observed_catalog: tuple[str | None, str | None] | None = None
    catalog_retry_at = 0.0
    catalog_cache = ToolCatalogCache()
    voice_consumer = VoicePacketConsumer(tts_service) if tts_service else None
    sync_supported: bool | None = None
    while True:
        state = monitor.read()
        state_signature = (
            state.available, state.ready, state.runtime_id, state.lifecycle,
            state.tool_catalog_id,
        )
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
            sync_supported = None
            if voice_consumer:
                voice_consumer.reset()
        catalog_signature = (state.runtime_id, state.tool_catalog_id)
        if (
            state.ready
            and state.runtime_id
            and catalog_signature != observed_catalog
            and time.monotonic() >= catalog_retry_at
        ):
            if state.tool_catalog_id:
                try:
                    await catalog_cache.ensure(client, state)
                    LOGGER.info(
                        "Project Hoomans tool catalog synchronized id=%s version=%s",
                        state.tool_catalog_id,
                        state.tool_catalog_version,
                    )
                    observed_catalog = catalog_signature
                except BridgeClientError as error:
                    LOGGER.warning("Could not synchronize Core tool catalog: %s", error)
                    catalog_retry_at = time.monotonic() + 5.0
            else:
                observed_catalog = catalog_signature
        if not state.ready or not state.runtime_id:
            await asyncio.sleep(settings.bridge_poll_interval)
            continue
        try:
            if voice_consumer and voice_channel_available(state):
                try:
                    await voice_consumer.poll(client, state)
                except (BridgeClientError, ValueError, TypeError) as error:
                    LOGGER.warning("Core voice packet cycle failed: %s", error)
            if sync_supported is not False:
                try:
                    sync_batch = await client.call(
                        NAMESPACE,
                        "pollConversationSync",
                        {},
                        state.runtime_id,
                    )
                    sync_supported = True
                    message_ids = conversation_service.record_message_batch(sync_batch)
                    if message_ids:
                        await client.call(
                            NAMESPACE,
                            "ackConversationSync",
                            {"message_ids": list(message_ids)},
                            state.runtime_id,
                        )
                        LOGGER.info(
                            "Conversation memory sync acknowledged messages=%s",
                            len(message_ids),
                        )
                except BridgeCommandError as error:
                    if error.code in {"UNKNOWN_COMMAND", "UNKNOWN_NAMESPACE"}:
                        sync_supported = False
                        LOGGER.warning(
                            "Project Hoomans does not expose conversation memory sync; "
                            "continuing without outbox ingestion"
                        )
                    else:
                        raise
            request = await client.call(NAMESPACE, "pollChat", {}, state.runtime_id)
            if request.get("status") != "pending":
                await asyncio.sleep(settings.bridge_poll_interval)
                continue
            request = hydrate_request(request, catalog_cache)
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
