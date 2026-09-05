"""Shared Piper TTS domain objects, limits, and persistence contracts."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pbrainz.config import Settings

from ..conversation_runtime import Utterance

InstallProgressCallback = Callable[[str, int, int], None]
DownloadProgressCallback = Callable[[int, int], None]

VOICE_PRESET_SLOTS = tuple(
    [f"VoiceFemale:{index}" for index in range(4)] + [f"VoiceMale:{index}" for index in range(4)]
)
# These are the voices used to make a fresh PBrainZ installation immediately
# useful.  The service installs only the models that are still missing; a
# user-selected value in a slot is never overwritten.
DEFAULT_TTS_PRESETS: tuple[tuple[str, str], ...] = (
    ("VoiceFemale:0", "en_US-kristin-medium"),
    ("VoiceFemale:1", "en_GB-alba-medium"),
    ("VoiceFemale:2", "en_US-lessac-high"),
    ("VoiceFemale:3", "en_US-ljspeech-high"),
    ("VoiceMale:0", "en_GB-alan-medium"),
    ("VoiceMale:1", "en_US-bryce-medium"),
    ("VoiceMale:2", "en_US-hfc_male-medium"),
    ("VoiceMale:3", "en_US-ryan-high"),
)
MAX_CATALOG_MODELS = 512
MAX_TEST_TEXT = 1200
MAX_REMOTE_CATALOG_BYTES = 4 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_PREVIEW_BYTES = 16 * 1024 * 1024
OFFICIAL_PIPER_CATALOG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json?download=true"
)
OFFICIAL_PIPER_REPOSITORY_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
OFFICIAL_PIPER_SAMPLES_URL = "https://rhasspy.github.io/piper-samples/samples/"

# Piper's public catalog does not publish speaker gender. These hints make the
# UI filters useful for common named voices while keeping unknown voices
# visible as "unknown". A local voice_metadata.json gender always wins.
_FEMALE_VOICE_NAMES = frozenset(
    {
        "aegis_female",
        "alba",
        "alma",
        "amy",
        "anna",
        "berta",
        "cori",
        "daniela",
        "eva_k",
        "gosia",
        "hfc_female",
        "irina",
        "jenny_dioco",
        "joy",
        "kasandra",
        "kathleen",
        "kerstin",
        "kristin",
        "lada",
        "lessac",
        "lili",
        "lisa",
        "ljspeech",
        "maider",
        "marylux",
        "maya",
        "nathalie",
        "natia",
        "padmavathi",
        "paola",
        "priyamvada",
        "ramona",
        "rapunzelina",
        "serena",
        "southern_english_female",
        "tetiana",
        "ugla",
    }
)
_MALE_VOICE_NAMES = frozenset(
    {
        "aivars",
        "alan",
        "alex",
        "amir",
        "antton",
        "arjun",
        "artur",
        "bryce",
        "cadu",
        "claude",
        "danny",
        "darkman",
        "davefx",
        "denis",
        "dimitar",
        "dmitri",
        "edon",
        "edresson",
        "faber",
        "fasih",
        "gilles",
        "gor",
        "harri",
        "hfc_male",
        "imre",
        "jeff",
        "jirka",
        "joe",
        "john",
        "kareem",
        "karlsson",
        "mihai",
        "mike",
        "mykyta",
        "norman",
        "northern_english_male",
        "oleksa",
        "pavoque",
        "pim",
        "pratham",
        "riccardo",
        "rohan",
        "ronnie",
        "ruslan",
        "ryan",
        "steinn",
        "thorsten",
        "tom",
        "venkatesh",
    }
)


class TTSException(RuntimeError):
    """A recoverable local synthesis or playback failure."""


def _infer_voice_gender(name: str, num_speakers: int = 1) -> str:
    """Return a best-effort gender label for Piper catalog entries."""

    normalized = name.casefold().replace("-", "_").strip()
    if normalized in _FEMALE_VOICE_NAMES or "_female" in normalized:
        return "female"
    if normalized in _MALE_VOICE_NAMES or "_male" in normalized:
        return "male"
    if num_speakers > 1:
        return "mixed"
    return "unknown"

SpeechCallback = Callable[[Utterance], Awaitable[None]]
FailureCallback = Callable[[Utterance, Exception], Awaitable[None]]

@dataclass(frozen=True, slots=True)
class VoiceModel:
    id: str
    display_name: str
    language: str = "unknown"
    locale: str = "unknown"
    gender: str = "unknown"
    quality: str = "unknown"
    num_speakers: int = 1
    speaker_map: dict[str, int] = field(default_factory=dict)
    installed: bool = True
    model_path: str = ""
    config_path: str = ""
    source: str = "local"
    model_url: str = ""
    config_url: str = ""
    model_size_bytes: int = 0
    model_md5: str = ""
    config_size_bytes: int = 0
    config_md5: str = ""
    sample_url: str = ""

    def as_dict(
        self, *, include_paths: bool = True, include_download: bool = False
    ) -> dict[str, Any]:
        value = {
            "id": self.id,
            "display_name": self.display_name,
            "language": self.language,
            "locale": self.locale,
            "gender": self.gender,
            "quality": self.quality,
            "num_speakers": self.num_speakers,
            "speaker_map": self.speaker_map,
            "installed": self.installed,
        }
        if include_paths:
            value.update({"model_path": self.model_path, "config_path": self.config_path})
        if include_download:
            value.update(
                {
                    "source": self.source,
                    "available_online": bool(self.model_url and self.config_url),
                    "model_url": self.model_url,
                    "config_url": self.config_url,
                    "model_size_bytes": self.model_size_bytes,
                    "config_size_bytes": self.config_size_bytes,
                    "sample_url": self.sample_url,
                    "preview_available": bool(self.sample_url),
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class TTSVoicePreset:
    slot: str
    voice_model_id: str
    optional_speaker_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "voice_model_id": self.voice_model_id,
            "optional_speaker_id": self.optional_speaker_id,
        }


@dataclass(frozen=True, slots=True)
class SynthesizedAudioChunk:
    """One bounded PCM chunk produced by a streaming TTS backend."""

    sample_rate: int
    sample_width: int
    sample_channels: int
    pcm: bytes
    duration_ms: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.sample_width <= 0 or self.sample_channels <= 0:
            raise ValueError("audio chunk format is invalid")
        if not self.pcm:
            raise ValueError("audio chunk is empty")
        if self.duration_ms <= 0:
            raise ValueError("audio chunk duration is invalid")


class VoicePresetRepository:
    """Local-only slot mapping; no Piper IDs enter bridge payloads."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def all(self) -> dict[str, TTSVoicePreset]:
        try:
            value = json.loads(self.settings.tts_voice_presets_json or "{}")
        except json.JSONDecodeError:
            value = {}
        if not isinstance(value, dict):
            value = {}
        output: dict[str, TTSVoicePreset] = {}
        for slot in VOICE_PRESET_SLOTS:
            raw = value.get(slot)
            if isinstance(raw, str):
                raw = {"voice_model_id": raw}
            if not isinstance(raw, dict):
                continue
            model_id = str(raw.get("voice_model_id") or "").strip()
            if not model_id:
                continue
            speaker_id = raw.get("optional_speaker_id")
            try:
                speaker_id = int(speaker_id) if speaker_id is not None else None
            except (TypeError, ValueError):
                speaker_id = None
            output[slot] = TTSVoicePreset(slot, model_id[:256], speaker_id)
        return output

    def replace(
        self, values: Iterable[TTSVoicePreset | dict[str, Any]]
    ) -> dict[str, TTSVoicePreset]:
        normalized: dict[str, dict[str, Any]] = {}
        for value in values:
            if isinstance(value, TTSVoicePreset):
                preset = value
            elif isinstance(value, dict):
                slot = str(value.get("slot") or "")
                model_id = str(value.get("voice_model_id") or "").strip()
                if slot not in VOICE_PRESET_SLOTS or not model_id:
                    continue
                speaker = value.get("optional_speaker_id")
                try:
                    speaker = int(speaker) if speaker is not None else None
                except (TypeError, ValueError):
                    speaker = None
                preset = TTSVoicePreset(slot, model_id[:256], speaker)
            else:
                continue
            if preset.slot in VOICE_PRESET_SLOTS and preset.voice_model_id:
                normalized[preset.slot] = preset.as_dict()
        self.settings.tts_voice_presets_json = json.dumps(normalized, separators=(",", ":"))
        return self.all()

    def ensure_defaults(self) -> tuple[str, ...]:
        """Fill empty slots with the built-in PBrainZ voice selections."""

        current = self.all()
        missing = [
            TTSVoicePreset(slot, model_id)
            for slot, model_id in DEFAULT_TTS_PRESETS
            if slot not in current
        ]
        if missing:
            self.replace([*current.values(), *missing])
        return tuple(preset.slot for preset in missing)

    def remove_model(self, model_id: str) -> tuple[str, ...]:
        """Clear every preset slot that points at a removed voice model."""

        normalized_id = str(model_id).strip()
        current = self.all()
        removed_slots = tuple(
            slot for slot, preset in current.items() if preset.voice_model_id == normalized_id
        )
        if removed_slots:
            self.replace(
                preset for slot, preset in current.items() if slot not in removed_slots
            )
        return removed_slots


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    path: Path
    duration_ms: int
    model_id: str


def _is_number(value: object) -> bool:
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _official_voice_file_url(relative_path: str) -> str:
    return f"{OFFICIAL_PIPER_REPOSITORY_URL}{quote(relative_path, safe='/')}?download=true"


def _official_voice_sample_url(language_code: str, voice_name: str, quality: str) -> str:
    language_family = language_code.split("_", 1)[0]
    parts = (language_family, language_code, voice_name, quality, "speaker_0.mp3")
    return OFFICIAL_PIPER_SAMPLES_URL + "/".join(quote(part, safe="") for part in parts)


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
