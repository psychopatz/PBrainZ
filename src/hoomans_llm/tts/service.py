"""Application-owned TTS service facade and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from hoomans_llm.config import Settings

from ..conversation_runtime import Utterance, VoiceBinding, VoiceBindingCache
from .audio import AudioOutput, PiperProvider
from .catalog import VoiceCatalog
from .models import (
    MAX_TEST_TEXT,
    FailureCallback,
    InstallProgressCallback,
    SpeechCallback,
    SynthesizedAudio,
    TTSException,
    VoiceModel,
    VoicePresetRepository,
)
from .scheduler import SpeechScheduler

LOGGER = logging.getLogger(__name__)


class TTSService:
    """Application-owned optional TTS service and diagnostics facade."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = VoiceCatalog(settings)
        self.presets = VoicePresetRepository(settings)
        self.provider = PiperProvider(self.catalog, self.presets, settings)
        self.output = AudioOutput(settings)
        self.scheduler = SpeechScheduler(self.provider, self.output, settings)
        self.voice_bindings = VoiceBindingCache()
        self.last_error: str | None = None
        self._catalog_refresh_lock = asyncio.Lock()
        self._preview_tasks: set[asyncio.Task[None]] = set()
        self._install_tasks: dict[str, asyncio.Task[None]] = {}
        self._install_jobs: dict[str, dict[str, Any]] = {}
        self._active_install_job_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.tts_enabled)

    async def start(self) -> None:
        if self.enabled and self.provider.available and self.output.available:
            await self.scheduler.start()
        elif self.enabled:
            self.last_error = self._availability_error()
            LOGGER.warning("TTS is enabled but degraded to text-only: %s", self.last_error)

    async def stop(self) -> None:
        for task in tuple(self._preview_tasks):
            task.cancel()
        if self._preview_tasks:
            await asyncio.gather(*self._preview_tasks, return_exceptions=True)
        self._preview_tasks.clear()
        for task in tuple(self._install_tasks.values()):
            task.cancel()
        if self._install_tasks:
            await asyncio.gather(*self._install_tasks.values(), return_exceptions=True)
        self._install_tasks.clear()
        self._active_install_job_id = None
        await self.scheduler.stop()
        self.voice_bindings.clear()

    async def reconfigure(self) -> None:
        await self.scheduler.stop()
        self.voice_bindings.clear()
        self.provider.cache.resize(self.settings.tts_model_cache_size)
        self.scheduler = SpeechScheduler(self.provider, self.output, self.settings)
        self.last_error = None
        await self.start()

    async def refresh_catalog(self, *, force: bool = False) -> bool:
        """Refresh the remote Piper index without blocking the event loop."""

        async with self._catalog_refresh_lock:
            return await asyncio.to_thread(self.catalog.refresh_remote, force=force)

    def install_voice(
        self, model_id: str, progress: InstallProgressCallback | None = None
    ) -> VoiceModel:
        """Install a voice selected from the already loaded remote catalog."""

        try:
            return self.catalog.install(model_id, progress=progress)
        except TTSException:
            raise
        except (OSError, ValueError, TypeError) as error:
            raise TTSException(str(error)) from error

    def start_voice_install(self, model_id: str) -> dict[str, Any]:
        """Start one background voice installation and return its job snapshot."""

        normalized_id = str(model_id).strip()
        model = self.catalog.get(normalized_id)
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {normalized_id}")
        if model.installed:
            return self._completed_install_snapshot(model)
        if self._active_install_job_id:
            active = self._install_jobs.get(self._active_install_job_id)
            if active and active.get("voice_model_id") == model.id:
                snapshot = self._install_snapshot(active)
                snapshot["already_running"] = True
                return snapshot
            raise TTSException("another Piper voice installation is already in progress")

        job_id = uuid4().hex
        total_bytes = max(0, model.model_size_bytes) + max(0, model.config_size_bytes)
        self._install_jobs[job_id] = {
            "job_id": job_id,
            "voice_model_id": model.id,
            "state": "queued",
            "stage": "Queued",
            "completed_bytes": 0,
            "total_bytes": total_bytes,
            "error": "",
        }
        self._active_install_job_id = job_id
        self._install_tasks[job_id] = asyncio.create_task(
            self._run_install_job(job_id, model.id),
            name=f"tts-install-{model.id}",
        )
        return self._install_snapshot(self._install_jobs[job_id])

    def install_status(self, job_id: str) -> dict[str, Any]:
        """Return a background voice installation snapshot."""

        job = self._install_jobs.get(str(job_id))
        if job is None:
            raise TTSException(f"Piper voice installation job not found: {job_id}")
        return self._install_snapshot(job)

    async def _run_install_job(self, job_id: str, model_id: str) -> None:
        job = self._install_jobs[job_id]
        job["state"] = "installing"
        job["stage"] = "Starting download"

        def progress(stage: str, completed: int, total: int) -> None:
            current = self._install_jobs.get(job_id)
            if current is None:
                return
            current["state"] = "installing"
            current["stage"] = stage
            current["completed_bytes"] = max(0, int(completed))
            current["total_bytes"] = max(0, int(total))

        try:
            model = await asyncio.to_thread(self.install_voice, model_id, progress)
            job["state"] = "complete"
            job["stage"] = "Installed"
            job["completed_bytes"] = job["total_bytes"]
            job["voice"] = model.as_dict(include_download=True)
        except asyncio.CancelledError:
            job["state"] = "cancelled"
            job["stage"] = "Cancelled"
            raise
        except Exception as error:  # Installation errors are reported to the GUI job.
            job["state"] = "failed"
            job["stage"] = "Installation failed"
            job["error"] = str(error)[:500]
            self.last_error = job["error"]
            LOGGER.warning("Piper voice installation failed: %s", error)
        finally:
            if self._active_install_job_id == job_id:
                self._active_install_job_id = None
            self._install_tasks.pop(job_id, None)

    @staticmethod
    def _completed_install_snapshot(model: VoiceModel) -> dict[str, Any]:
        return {
            "job_id": "",
            "voice_model_id": model.id,
            "state": "complete",
            "stage": "Already installed",
            "completed_bytes": 1,
            "total_bytes": 1,
            "progress": 100.0,
            "voice": model.as_dict(include_download=True),
            "already_running": False,
        }

    @staticmethod
    def _install_snapshot(job: dict[str, Any]) -> dict[str, Any]:
        completed = max(0, int(job.get("completed_bytes") or 0))
        total = max(0, int(job.get("total_bytes") or 0))
        progress = round(min(100.0, completed * 100.0 / total), 1) if total else 0.0
        snapshot: dict[str, Any] = {
            "job_id": str(job.get("job_id") or ""),
            "voice_model_id": str(job.get("voice_model_id") or ""),
            "state": str(job.get("state") or "unknown"),
            "stage": str(job.get("stage") or ""),
            "completed_bytes": completed,
            "total_bytes": total,
            "progress": progress,
        }
        if job.get("error"):
            snapshot["error"] = str(job["error"])
        if job.get("voice"):
            snapshot["voice"] = job["voice"]
        return snapshot

    async def preview_voice(self, model_id: str) -> VoiceModel:
        """Play a catalog sample without downloading the voice model."""

        if not self.output.available:
            raise TTSException("no supported audio output command is available")
        model = self.catalog.get(model_id)
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {model_id}")
        sample_path = await asyncio.to_thread(self.catalog.download_sample, model.id)
        task = asyncio.create_task(self._play_preview(sample_path), name=f"tts-preview-{model.id}")
        self._preview_tasks.add(task)
        task.add_done_callback(self._preview_tasks.discard)
        return model

    async def _play_preview(self, path: Path) -> None:
        process: asyncio.subprocess.Process | None = None
        try:
            process = await self.output.start(path)
            await process.wait()
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
            if process is not None:
                try:
                    await process.wait()
                except (OSError, asyncio.CancelledError):
                    pass
            raise
        except Exception as error:
            self.last_error = f"Piper voice preview failed: {error}"
            LOGGER.warning("Piper voice preview failed: %s", error)
        finally:
            path.unlink(missing_ok=True)

    def can_synthesize(self, binding: VoiceBinding | None) -> bool:
        return self.enabled and self.provider.can_synthesize(binding) and self.output.available

    def resolve_voice_binding(
        self,
        conversation_id: str,
        npc_uuid: str,
        binding: VoiceBinding | None,
    ) -> VoiceBinding | None:
        """Remember Lua's compact identity and reuse it for later utterances."""

        if binding is not None:
            if binding.npc_uuid != str(npc_uuid)[:256]:
                return None
            return self.voice_bindings.remember(conversation_id, binding)
        return self.voice_bindings.get(conversation_id, npc_uuid)

    async def enqueue(
        self,
        utterance: Utterance,
        *,
        on_started: SpeechCallback | None = None,
        on_finished: SpeechCallback | None = None,
        on_failed: FailureCallback | None = None,
    ) -> bool:
        accepted = await self.scheduler.enqueue(
            utterance,
            on_started=on_started,
            on_finished=on_finished,
            on_failed=self._record_failure(on_failed),
        )
        if not accepted:
            self.last_error = (
                self._availability_error()
                if not self.can_synthesize(utterance.voice_binding)
                else "bounded TTS queue is full"
            )
        return accepted

    async def test_voice(self, slot: str, text: str) -> bool:
        binding = VoiceBinding("tts-test", slot)
        if not self.output.available:
            self.last_error = "audio output backend is unavailable"
            return False
        if not self.provider.can_synthesize(binding):
            self.last_error = self._voice_test_error(binding)
            return False
        task = asyncio.create_task(
            self._play_voice_test(binding, text[:MAX_TEST_TEXT]),
            name=f"tts-test-{slot}",
        )
        self._preview_tasks.add(task)
        task.add_done_callback(self._preview_tasks.discard)
        return True

    async def _play_voice_test(self, binding: VoiceBinding, text: str) -> None:
        """Synthesize and play one installed voice without enabling NPC TTS."""

        audio: SynthesizedAudio | None = None
        process: asyncio.subprocess.Process | None = None
        try:
            audio = await asyncio.to_thread(self.provider.synthesize, text, binding)
            process = await self.output.start(audio.path)
            await process.wait()
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await process.wait()
                except (OSError, asyncio.CancelledError):
                    pass
            raise
        except Exception as error:
            self.last_error = f"Piper voice test failed: {error}"
            LOGGER.warning("Piper voice test failed: %s", error)
        finally:
            if audio is not None:
                audio.path.unlink(missing_ok=True)

    def _voice_test_error(self, binding: VoiceBinding) -> str:
        if not self.provider.available:
            return "Piper executable/package is unavailable"
        preset = self.presets.all().get(binding.slot)
        if not preset:
            return f"no Piper preset configured for {binding.slot}"
        model = self.catalog.get(preset.voice_model_id)
        if not model or not model.installed:
            return f"Piper model is not installed: {preset.voice_model_id}"
        return "voice model or preset is unavailable"

    def status(self) -> dict[str, Any]:
        self.catalog.refresh()
        active = [
            utterance.speaker_npc_uuid
            for runtime in self.scheduler.runtimes.values()
            for utterance in runtime.active_utterances.values()
        ]
        return {
            "enabled": self.enabled,
            "backend": "Piper",
            "piper_available": self.provider.available,
            "audio_output_available": self.output.available,
            "audio_backend": self.output.command_name,
            "audio_devices": self.output.devices(),
            "output_device": self.settings.tts_output_device or "system/default",
            "master_volume": self.settings.tts_master_volume,
            "catalog_language": self.settings.tts_voice_catalog_language,
            "model_cache_size": self.settings.tts_model_cache_size,
            "max_simultaneous_playback": self.settings.tts_max_simultaneous_playback,
            "natural_gap_ms": self.settings.tts_natural_gap_ms,
            "synthesis_timeout": self.settings.tts_synthesis_timeout,
            "audio_buffer_ms": self.settings.tts_audio_buffer_ms,
            "model_root": str(self.catalog.roots[0]),
            "catalog": [
                model.as_dict(include_download=True) for model in self.catalog.models.values()
            ],
            "catalog_source": "official Piper catalog"
            if self.catalog.remote_models
            else "local files",
            "catalog_last_updated": self.catalog.last_remote_fetch_at,
            "catalog_error": self.catalog.last_remote_error,
            "catalog_total": len(self.catalog.models),
            "catalog_installed": sum(
                1 for model in self.catalog.models.values() if model.installed
            ),
            "catalog_available": sum(
                1
                for model in self.catalog.models.values()
                if not model.installed and model.model_url and model.config_url
            ),
            "presets": [preset.as_dict() for preset in self.presets.all().values()],
            "loaded_models": list(self.provider.cache.loaded_model_ids),
            "model_load_count": self.provider.cache.load_count,
            "model_eviction_count": self.provider.cache.eviction_count,
            "average_synthesis_latency_ms": round(
                sum(self.provider.synthesis_latencies_ms)
                / len(self.provider.synthesis_latencies_ms),
                2,
            )
            if self.provider.synthesis_latencies_ms
            else 0,
            "cached_voice_bindings": len(self.voice_bindings),
            "synthesis_queue_depth": self.scheduler.synthesis.queue.qsize(),
            "active_synthesis_workers": self.scheduler.synthesis.active_workers,
            "synthesis_workers": self.scheduler.synthesis.worker_count,
            "playback_queue_depth": sum(
                len(runtime.tts_ready_queue) for runtime in self.scheduler.runtimes.values()
            ),
            "currently_speaking_npcs": list(dict.fromkeys(active)),
            "active_utterance_ids": [
                utterance.utterance_id
                for runtime in self.scheduler.runtimes.values()
                for utterance in runtime.active_utterances.values()
            ],
            "conversations": [
                {
                    "conversation_id": runtime.conversation_id,
                    "kind": runtime.kind.value,
                    "state": runtime.state,
                    "generated_ahead": runtime.generated_ahead_count,
                    "tts_ready_ahead": runtime.tts_ready_ahead_count,
                }
                for runtime in self.scheduler.runtimes.values()
            ],
            "max_generated_ahead": self.settings.tts_max_generated_ahead,
            "max_tts_ready_ahead": self.settings.tts_max_tts_ready_ahead,
            "last_error": self.last_error or self.catalog.last_error,
        }

    def _availability_error(self) -> str:
        if not self.provider.available:
            return "Piper executable/package is unavailable"
        if not self.output.available:
            return "audio output backend is unavailable"
        return "voice model or preset is unavailable"

    def _record_failure(self, callback: FailureCallback | None) -> FailureCallback | None:
        if not callback:
            return None

        async def wrapped(utterance: Utterance, error: Exception) -> None:
            self.last_error = str(error)[:500]
            await callback(utterance, error)

        return wrapped


