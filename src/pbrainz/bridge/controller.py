"""Runtime lifecycle control for the Project Hoomans bridge worker."""

from __future__ import annotations

import asyncio

from pbrainz.config import Settings
from pbrainz.conversation_service import TraceWriter
from pbrainz.providers.registry import ProviderRegistry
from pbrainz.tts import TTSService

from .memory_context import ActiveMemoryContextCache
from .pump import run_bridge_pump
from .state import BridgeRuntimeMonitor


class BridgeController:
    """Start and stop the bounded bridge pump without restarting the server."""

    def __init__(
        self,
        settings: Settings,
        providers: ProviderRegistry,
        monitor: BridgeRuntimeMonitor,
        tts_service: TTSService | None = None,
        *,
        trace_writer: TraceWriter | None = None,
        active_memory_context: ActiveMemoryContextCache | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.monitor = monitor
        self.tts_service = tts_service
        self.trace_writer = trace_writer
        self.active_memory_context = active_memory_context
        self._enabled = settings.bridge_required
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    async def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(
                run_bridge_pump(
                    self.settings,
                    self.providers,
                    self.monitor,
                    self.tts_service,
                    trace_writer=self.trace_writer,
                    active_memory_context=self.active_memory_context,
                ),
                name="p-brainz-bridge",
            )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.settings.bridge_required = enabled
        if enabled:
            await self.start()
        else:
            await self.stop()

    async def set_zomboid_path(self, path: str) -> None:
        """Re-point the monitor and worker transport after a path change."""

        self.settings.zomboid_path = path
        self.monitor.set_root(
            self.settings.bridge_root,
            zomboid_path=path,
        )
        was_running = self.running
        if was_running:
            await self.stop()
            await self.start()

    def as_dict(self) -> dict[str, object]:
        return {
            "bridge": self.monitor.read().as_dict(),
            "worker_enabled": self.enabled,
            "worker_running": self.running,
        }
