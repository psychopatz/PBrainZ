"""Piper synthesis and operating-system audio output adapters."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pbrainz.config import Settings

from ..conversation_runtime import VoiceBinding
from .catalog import VoiceCatalog
from .models import (
    SynthesizedAudio,
    TTSException,
    TTSVoicePreset,
    VoiceModel,
    VoicePresetRepository,
)

LOGGER = logging.getLogger(__name__)


class PiperModelCache:
    """Bounded LRU cache for lazily loaded Piper model handles."""

    def __init__(self, max_size: int = 2) -> None:
        self.max_size = max(1, min(int(max_size), 16))
        self._items: OrderedDict[str, Any] = OrderedDict()
        self.load_count = 0
        self.eviction_count = 0

    def get_or_load(self, model_id: str, loader: Callable[[], Any]) -> Any:
        if model_id in self._items:
            value = self._items.pop(model_id)
            self._items[model_id] = value
            return value
        value = loader()
        self._items[model_id] = value
        self.load_count += 1
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
            self.eviction_count += 1
        return value

    def resize(self, max_size: int) -> None:
        self.max_size = max(1, min(int(max_size), 16))
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
            self.eviction_count += 1

    @property
    def loaded_model_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def evict(self, model_id: str) -> bool:
        """Remove one model handle before its files are deleted."""

        return self._items.pop(str(model_id), None) is not None

    def clear(self) -> None:
        self._items.clear()


class PiperProvider:
    """Piper adapter with Python-package and CLI implementations."""

    def __init__(
        self, catalog: VoiceCatalog, presets: VoicePresetRepository, settings: Settings
    ) -> None:
        self.catalog = catalog
        self.presets = presets
        self.settings = settings
        self.cache = PiperModelCache(settings.tts_model_cache_size)
        self._piper_voice_type: Any = None
        self._python_checked = False
        self.synthesis_latencies_ms: deque[float] = deque(maxlen=128)

    @property
    def executable(self) -> str | None:
        configured = self.settings.tts_piper_executable.strip() or "piper"
        return shutil.which(configured)

    @property
    def python_available(self) -> bool:
        if not self._python_checked:
            self._python_checked = True
            try:
                from piper import PiperVoice  # type: ignore[import-not-found]

                self._piper_voice_type = PiperVoice
            except ImportError:
                self._piper_voice_type = None
        return self._piper_voice_type is not None

    @property
    def available(self) -> bool:
        return self.python_available or self.executable is not None

    def can_synthesize(self, binding: VoiceBinding | None) -> bool:
        if not binding or not self.available:
            return False
        preset = self._preset_for_binding(binding)
        model = self.catalog.get(preset.voice_model_id) if preset else None
        return bool(model and model.installed)

    def synthesize(self, text: str, voice_binding: VoiceBinding) -> SynthesizedAudio:
        if not self.available:
            raise TTSException("Piper is unavailable")
        preset = self._preset_for_binding(voice_binding)
        if not preset:
            raise TTSException(f"no installed Piper voice is available for {voice_binding.slot}")
        model = self.catalog.get(preset.voice_model_id)
        if not model or not model.installed:
            raise TTSException(f"Piper model is not installed: {preset.voice_model_id}")
        fd, output_name = tempfile.mkstemp(prefix="pbrainz-tts-", suffix=".wav")
        os.close(fd)
        output_path = Path(output_name)
        started = time.perf_counter()
        try:
            if self.python_available:
                try:
                    self._synthesize_python(text, voice_binding, preset, model, output_path)
                except Exception as error:
                    LOGGER.warning("Piper Python synthesis failed; trying CLI: %s", error)
                    self._synthesize_cli(text, preset, model, output_path)
            else:
                self._synthesize_cli(text, preset, model, output_path)
            duration_ms = _wav_duration_ms(output_path, text)
            LOGGER.debug(
                "Piper synthesized model=%s chars=%s latency_ms=%s duration_ms=%s pitch=%s",
                model.id,
                len(text),
                round((time.perf_counter() - started) * 1000),
                duration_ms,
                voice_binding.pitch,
            )
            return SynthesizedAudio(output_path, duration_ms, model.id)
        except Exception as error:
            output_path.unlink(missing_ok=True)
            if isinstance(error, TTSException):
                raise
            raise TTSException(str(error)) from error
        finally:
            self.synthesis_latencies_ms.append((time.perf_counter() - started) * 1000)

    def _synthesize_python(
        self,
        text: str,
        binding: VoiceBinding,
        preset: TTSVoicePreset,
        model: VoiceModel,
        output_path: Path,
    ) -> None:
        voice_type = self._piper_voice_type
        voice = self.cache.get_or_load(
            model.id,
            lambda: voice_type.load(model.model_path),
        )
        with wave.open(str(output_path), "wb") as output:
            kwargs: dict[str, Any] = {}
            if preset.optional_speaker_id is not None:
                kwargs["speaker_id"] = preset.optional_speaker_id
            try:
                voice.synthesize_wav(text, output, **kwargs)
            except TypeError:
                voice.synthesize_wav(text, output)

    def _synthesize_cli(
        self,
        text: str,
        preset: TTSVoicePreset,
        model: VoiceModel,
        output_path: Path,
    ) -> None:
        executable = self.executable
        if not executable:
            raise TTSException("Piper executable is unavailable")
        # The CLI cannot retain a live PiperVoice handle, but recording the
        # model in the same bounded cache still makes catalog usage visible and
        # prevents unbounded model bookkeeping in long-running sessions.
        self.cache.get_or_load(model.id, lambda: model.model_path)
        command = [executable, "--model", model.model_path, "--output_file", str(output_path)]
        if preset.optional_speaker_id is not None:
            command.extend(["--speaker", str(preset.optional_speaker_id)])
        try:
            completed = subprocess.run(
                command,
                input=text[:12000],
                capture_output=True,
                text=True,
                timeout=max(2.0, float(self.settings.tts_synthesis_timeout)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TTSException(f"Piper process failed: {error}") from error
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "no output").strip()[:500]
            raise TTSException(f"Piper synthesis failed: {detail}")

    def _preset_for_binding(self, binding: VoiceBinding) -> TTSVoicePreset | None:
        """Resolve a saved preset, or choose an installed voice for first-run playback."""

        configured = self.presets.all().get(binding.slot)
        if configured is not None:
            return configured
        if not self.catalog.models:
            self.catalog.refresh()
        slot_name = binding.slot.casefold()
        expected_gender = (
            "female"
            if slot_name.startswith("voicefemale:")
            else "male"
            if slot_name.startswith("voicemale:")
            else None
        )
        installed = sorted(
            (model for model in self.catalog.models.values() if model.installed),
            key=lambda model: (model.id.casefold(), model.id),
        )
        if expected_gender:
            matching = [model for model in installed if model.gender.casefold() == expected_gender]
            installed = matching or installed
        if not installed:
            return None
        LOGGER.info(
            "Piper using first installed voice for unconfigured slot=%s model=%s",
            binding.slot,
            installed[0].id,
        )
        return TTSVoicePreset(binding.slot, installed[0].id)


class AudioOutput:
    """Start audio through an OS output command; never touches the PZ bridge."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def command_name(self) -> str | None:
        for name in ("ffplay", "pw-play", "paplay", "aplay", "afplay"):
            if shutil.which(name):
                return name
        return None

    @property
    def available(self) -> bool:
        return self.command_name is not None

    def devices(self) -> list[str]:
        """Return best-effort local sink names for the native settings tab."""

        discovered: list[str] = []
        commands = (("pactl", ("list", "short", "sinks")),)
        for executable, arguments in commands:
            if not shutil.which(executable):
                continue
            try:
                completed = subprocess.run(
                    [executable, *arguments],
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if completed.returncode != 0:
                continue
            if executable == "pactl":
                discovered.extend(
                    line.split()[1]
                    for line in completed.stdout.splitlines()
                    if len(line.split()) >= 2
                )
            break
        return list(dict.fromkeys(item for item in discovered if item))[:32]

    def command(self, path: Path) -> list[str]:
        name = self.command_name
        if not name:
            raise TTSException("no supported audio output command is available")
        volume = max(0.0, min(1.0, float(self.settings.tts_master_volume)))
        device = self.settings.tts_output_device.strip()
        if device.casefold() == "system/default":
            device = ""
        if name == "ffplay":
            return [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-volume",
                str(round(volume * 100)),
                str(path),
            ]
        if name == "pw-play":
            return [
                "pw-play",
                *(["--target", device] if device else []),
                "--volume",
                str(volume),
                str(path),
            ]
        if name == "paplay":
            return [
                "paplay",
                *(["--device", device] if device else []),
                "--volume",
                str(round(volume * 65536)),
                str(path),
            ]
        if name == "aplay":
            return ["aplay", *(["-D", device] if device else []), "-q", str(path)]
        return ["afplay", *(["-v", str(volume)] if volume else []), str(path)]

    async def start(self, path: Path) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                *self.command(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError) as error:
            raise TTSException(f"audio output failed: {error}") from error


def _wav_duration_ms(path: Path, text: str) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            if rate:
                return max(1, round(frames / rate * 1000))
    except (OSError, wave.Error):
        pass
    return max(320, round(len(text) / 13.0 * 1000))
