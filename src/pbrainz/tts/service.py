"""Application-owned TTS service facade and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from pbrainz.config import Settings
from pbrainz.tts.text import normalize_tts_text

from ..conversation_runtime import Utterance, VoiceBinding, VoiceBindingCache
from .audio import AudioOutput, PiperProvider
from .catalog import VoiceCatalog
from .models import (
    DEFAULT_TTS_PRESETS,
    MAX_TEST_TEXT,
    FailureCallback,
    InstallProgressCallback,
    SpeechCallback,
    SynthesizedAudio,
    TTSException,
    TTSVoicePreset,
    VoiceModel,
    VoicePresetRepository,
)
from .scheduler import SpeechScheduler

LOGGER = logging.getLogger(__name__)
DEFAULT_INSTALL_RETRY_DELAYS = (0.0, 5.0, 15.0)


class TTSService:
    """Application-owned optional TTS service and diagnostics facade."""

    def __init__(
        self,
        settings: Settings,
        save_settings: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.settings = settings
        self._save_settings = save_settings
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
        self._default_install_task: asyncio.Task[None] | None = None
        self._default_install_status: dict[str, Any] | None = None
        self._default_setup_complete = False

    @property
    def enabled(self) -> bool:
        return bool(self.settings.tts_enabled)

    async def start(self) -> None:
        if not self.enabled:
            await self._cancel_default_install()
            return
        self._ensure_default_presets()
        if self.enabled and self.provider.available and self.output.available:
            await self.scheduler.start()
        else:
            self.last_error = self._availability_error()
            LOGGER.warning("TTS is enabled but degraded to text-only: %s", self.last_error)

    async def stop(self) -> None:
        await self._cancel_default_install()
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

    def _ensure_default_presets(self) -> None:
        missing_slots = self.presets.ensure_defaults()
        if not missing_slots or self._save_settings is None:
            return
        try:
            self._save_settings({"tts_voice_presets_json": self.settings.tts_voice_presets_json})
        except Exception as error:  # Persistence failure must not disable local TTS.
            self.last_error = f"Could not save default Piper voice presets: {error}"[:500]
            LOGGER.warning("Could not save default Piper voice presets: %s", error)

    async def _cancel_default_install(self) -> None:
        task = self._default_install_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._default_install_task = None
        self._default_install_status = None

    def _default_install_finished(self, task: asyncio.Task[None]) -> None:
        if self._default_install_task is task:
            self._default_install_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.last_error = f"Default Piper voice setup failed: {error}"[:500]
            self._default_install_status = self._default_failure_status(self.last_error)
            LOGGER.warning("Default Piper voice setup failed: %s", error)

    def _default_voice_ids(self) -> tuple[str, ...]:
        presets = self.presets.all()
        return tuple(
            dict.fromkeys(
                model_id
                for slot, model_id in DEFAULT_TTS_PRESETS
                if presets.get(slot) and presets[slot].voice_model_id == model_id
            )
        )

    def default_voice_setup_needed(self) -> bool:
        """Return whether any shipped default voice is not installed yet."""

        default_ids = self._default_voice_ids()
        return bool(
            default_ids
            and any(
                not (model := self.catalog.models.get(model_id)) or not model.installed
                for model_id in default_ids
            )
        )

    def start_default_install(self) -> dict[str, Any]:
        """Request the default voice setup after explicit user consent."""

        default_ids = self._default_voice_ids()
        if not default_ids:
            raise TTSException("No default Piper voices are configured")
        if self._default_install_task and not self._default_install_task.done():
            return self.default_install_status() or self._default_status(default_ids[0])
        if not self.default_voice_setup_needed():
            self._default_setup_complete = True
            self._default_install_status = {
                **self._default_status(default_ids[0]),
                "state": "complete",
                "stage": "Default voices already installed",
                "completed_bytes": 1,
                "total_bytes": 1,
                "progress": 100.0,
            }
            return dict(self._default_install_status)
        self.last_error = None
        self._default_install_status = self._default_status(default_ids[0])
        self._default_install_task = asyncio.create_task(
            self._install_default_voices(), name="pbrainz-tts-default-voices"
        )
        self._default_install_task.add_done_callback(self._default_install_finished)
        return dict(self._default_install_status)

    async def _install_default_voices(self) -> None:
        """Install the selected built-ins serially after TTS is enabled."""

        default_ids = self._default_voice_ids()
        try:
            for attempt, delay in enumerate(DEFAULT_INSTALL_RETRY_DELAYS, start=1):
                if delay:
                    self._default_install_status = {
                        **(self._default_install_status or self._default_status(default_ids[0])),
                        "state": "preparing",
                        "stage": (
                            f"Retrying default voice download ({attempt}/"
                            f"{len(DEFAULT_INSTALL_RETRY_DELAYS)})"
                        ),
                    }
                    await asyncio.sleep(delay)
                failures = await self._install_default_voice_pass(default_ids)
                if not failures:
                    self._default_setup_complete = True
                    self.last_error = None
                    self._default_install_status = {
                        **(self._default_install_status or self._default_status(default_ids[0])),
                        "state": "complete",
                        "stage": "Default voices installed",
                        "completed_bytes": 1,
                        "total_bytes": 1,
                        "progress": 100.0,
                    }
                    return
                message = "; ".join(failures)
                if attempt < len(DEFAULT_INSTALL_RETRY_DELAYS):
                    LOGGER.warning(
                        "Default Piper voice setup attempt %d/%d failed: %s",
                        attempt,
                        len(DEFAULT_INSTALL_RETRY_DELAYS),
                        message,
                    )
                else:
                    self._default_setup_complete = False
                    self.last_error = f"Default Piper voice setup incomplete: {message}"[:500]
                    self._default_install_status = self._default_failure_status(
                        self.last_error
                    )
                    LOGGER.warning(self.last_error)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._default_setup_complete = False
            self.last_error = f"Default Piper voice setup failed: {error}"[:500]
            self._default_install_status = self._default_failure_status(self.last_error)
            LOGGER.exception("Default Piper voice setup failed unexpectedly")

    async def _install_default_voice_pass(self, default_ids: tuple[str, ...]) -> list[str]:
        await self.refresh_catalog()
        failures: list[str] = []
        for model_id in default_ids:
            self._default_install_status = self._default_status(model_id)
            model = self.catalog.get(model_id)
            if model is None:
                failures.append(f"{model_id}: not present in the Piper catalog")
                continue
            if model.installed:
                continue
            while self._active_install_job_id:
                active_job_id = self._active_install_job_id
                await self._wait_for_install_job(active_job_id)
                if self._active_install_job_id == active_job_id:
                    # The install job publishes its terminal state before
                    # its finally block clears the active-job marker.
                    await asyncio.sleep(0.05)
            try:
                install = self.start_voice_install(model_id)
            except TTSException as error:
                failures.append(f"{model_id}: {error}")
                continue
            job_id = str(install.get("job_id") or "")
            if job_id:
                await self._wait_for_install_job(job_id)
                completed = self._install_jobs.get(job_id, {})
                if str(completed.get("state") or "") != "complete":
                    failures.append(
                        f"{model_id}: {completed.get('error') or 'installation failed'}"
                    )
        return failures

    @staticmethod
    def _default_failure_status(error: str) -> dict[str, Any]:
        return {
            "job_id": "",
            "voice_model_id": "",
            "state": "failed",
            "stage": "Default voice download failed",
            "completed_bytes": 0,
            "total_bytes": 0,
            "progress": 0.0,
            "error": error[:500],
        }

    @staticmethod
    def _default_status(model_id: str) -> dict[str, Any]:
        return {
            "job_id": "",
            "voice_model_id": model_id,
            "state": "preparing",
            "stage": "Preparing default voices",
            "completed_bytes": 0,
            "total_bytes": 0,
            "progress": 0.0,
        }

    async def _wait_for_install_job(self, job_id: str) -> None:
        while True:
            job = self._install_jobs.get(job_id)
            if job is None or str(job.get("state") or "") in {
                "complete",
                "failed",
                "cancelled",
            }:
                return
            await asyncio.sleep(0.1)

    def default_install_status(self) -> dict[str, Any] | None:
        """Return the automatic default-install progress shown by the GUI."""

        if self._default_install_status is None:
            return None
        active_job_id = self._active_install_job_id
        if active_job_id and active_job_id in self._install_jobs:
            return self._install_snapshot(self._install_jobs[active_job_id])
        return dict(self._default_install_status)

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

    async def uninstall_voice(self, model_id: str) -> dict[str, Any]:
        """Stop local playback, remove a voice, and clear dependent state."""

        normalized_id = str(model_id).strip()
        model = self.catalog.get(normalized_id)
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {normalized_id}")
        if not model.installed:
            raise TTSException(f"Piper voice is not installed: {normalized_id}")
        if self._active_install_job_id:
            raise TTSException("wait for the active Piper voice installation to finish")

        await self.scheduler.stop()
        try:
            self.provider.cache.evict(model.id)
            cleared_slots = self.presets.remove_model(model.id)
            self.voice_bindings.clear()
            removed = await asyncio.to_thread(self.catalog.uninstall, model.id)
        finally:
            if self.enabled and self.provider.available and self.output.available:
                await self.scheduler.start()
        if model.id in {model_id for _slot, model_id in DEFAULT_TTS_PRESETS}:
            self._default_setup_complete = False
            self._default_install_status = None
        return {
            "voice": removed.as_dict(include_download=True),
            "cleared_preset_slots": list(cleared_slots),
        }

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
        speaker_kind: str = "npc",
    ) -> VoiceBinding | None:
        """Remember Lua's compact identity and reuse it for later utterances."""

        speaker_kind = str(speaker_kind or "npc").strip().lower() or "npc"
        speaker_id = str(npc_uuid)[:256]
        if binding is not None:
            if (
                binding.speaker_id != speaker_id
                or binding.speaker_kind != speaker_kind
            ):
                return None
            return self.voice_bindings.remember(conversation_id, binding)
        return self.voice_bindings.get(conversation_id, speaker_id, speaker_kind)

    async def enqueue(
        self,
        utterance: Utterance,
        *,
        on_started: SpeechCallback | None = None,
        on_finished: SpeechCallback | None = None,
        on_failed: FailureCallback | None = None,
        wait_for_capacity: bool = False,
    ) -> bool:
        speech_text = normalize_tts_text(utterance.text)
        if not speech_text:
            self.last_error = "TTS skipped: utterance has no speakable text"
            return False
        if speech_text != utterance.text:
            utterance = replace(utterance, text=speech_text)
        accepted = await self.scheduler.enqueue(
            utterance,
            on_started=on_started,
            on_finished=on_finished,
            on_failed=self._record_failure(on_failed),
            wait_for_capacity=wait_for_capacity,
        )
        if not accepted:
            self.last_error = (
                self._availability_error()
                if not self.can_synthesize(utterance.voice_binding)
                else "bounded TTS queue is full"
            )
        return accepted

    async def cancel_conversation(self, conversation_id: str) -> int:
        """Stop local speech that belongs to a no-longer-presented turn."""

        return await self.scheduler.cancel_conversation(str(conversation_id))

    async def test_voice(self, slot: str, text: str) -> bool:
        normalized_text = normalize_tts_text(text)
        if not normalized_text:
            self.last_error = "TTS skipped: test text has no speakable text"
            return False
        binding = VoiceBinding("tts-test", slot)
        if not self.output.available:
            self.last_error = "audio output backend is unavailable"
            return False
        if not self.provider.can_synthesize(binding):
            self.last_error = self._voice_test_error(binding)
            return False
        task = asyncio.create_task(
            self._play_voice_test(binding, normalized_text[:MAX_TEST_TEXT]),
            name=f"tts-test-{slot}",
        )
        self.last_error = None
        self._preview_tasks.add(task)
        task.add_done_callback(self._preview_tasks.discard)
        return True

    async def speak_text(self, text: str, *, source: str = "inference") -> bool:
        """Play generated application text through the first usable preset.

        Direct control-panel inference does not have an NPC voice binding, so
        it uses the first configured preset whose model is installed. The
        generated response is queued as a local preview task; this keeps
        playback independent from the bridge dialogue scheduler and lets the
        HTTP response return as soon as inference completes.
        """

        if not self.enabled:
            return False
        normalized_text = normalize_tts_text(text)
        if not normalized_text:
            return False
        if not self.output.available:
            self.last_error = "audio output backend is unavailable"
            return False
        preset = self._first_installed_preset()
        if preset is None:
            preset = self._fallback_installed_preset()
            if preset is None:
                self.last_error = "no installed Piper voice is available"
                LOGGER.info("TTS inference skipped: no installed voice is available")
                return False
            LOGGER.info(
                "TTS inference using first installed voice because no preset is configured "
                "model=%s",
                preset.voice_model_id,
            )
        binding = VoiceBinding(f"pbrainz-{source}", preset.slot)
        if not self.provider.can_synthesize(binding):
            self.last_error = self._voice_test_error(binding)
            return False
        task = asyncio.create_task(
            self._play_voice_test(binding, normalized_text[:MAX_TEST_TEXT]),
            name=f"tts-{source}-{uuid4().hex[:8]}",
        )
        self.last_error = None
        self._preview_tasks.add(task)
        task.add_done_callback(self._preview_tasks.discard)
        LOGGER.info(
            "TTS inference queued source=%s slot=%s chars=%s",
            source,
            preset.slot,
            len(normalized_text),
        )
        return True

    def _first_installed_preset(self):
        for preset in self.presets.all().values():
            model = self.catalog.get(preset.voice_model_id)
            if model is not None and model.installed:
                return preset
        return None

    def _fallback_installed_preset(self):
        """Return a transient preset so direct app TTS works before setup is complete."""

        if not self.catalog.models:
            self.catalog.refresh()
        installed = sorted(
            (model for model in self.catalog.models.values() if model.installed),
            key=lambda model: (model.id.casefold(), model.id),
        )
        if not installed:
            return None
        return TTSVoicePreset("VoiceFemale:0", installed[0].id)

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
        default_setup_needed = self.default_voice_setup_needed()
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
            "streaming_available": bool(
                getattr(self.provider, "python_available", False)
                and getattr(self.output, "streaming_available", False)
            ),
            "streaming_backend": (
                f"{self.output.command_name} raw PCM"
                if getattr(self.output, "streaming_available", False)
                else None
            ),
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
            "default_voice_setup_needed": default_setup_needed,
            "presets": [preset.as_dict() for preset in self.presets.all().values()],
            "default_voice_install": self.default_install_status(),
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
