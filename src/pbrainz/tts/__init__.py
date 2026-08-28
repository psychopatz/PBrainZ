"""Public API for the modular local Piper TTS subsystem."""

from .audio import AudioOutput, PiperModelCache, PiperProvider
from .catalog import VoiceCatalog
from .models import (
    DEFAULT_TTS_PRESETS,
    MAX_CATALOG_MODELS,
    MAX_DOWNLOAD_BYTES,
    MAX_PREVIEW_BYTES,
    MAX_REMOTE_CATALOG_BYTES,
    MAX_TEST_TEXT,
    OFFICIAL_PIPER_CATALOG_URL,
    OFFICIAL_PIPER_REPOSITORY_URL,
    OFFICIAL_PIPER_SAMPLES_URL,
    VOICE_PRESET_SLOTS,
    DownloadProgressCallback,
    FailureCallback,
    InstallProgressCallback,
    SpeechCallback,
    SynthesizedAudio,
    TTSException,
    TTSVoicePreset,
    VoiceModel,
    VoicePresetRepository,
)
from .scheduler import SpeechScheduler, SynthesisQueue
from .service import TTSService

__all__ = [
    "AudioOutput",
    "DownloadProgressCallback",
    "DEFAULT_TTS_PRESETS",
    "FailureCallback",
    "InstallProgressCallback",
    "MAX_CATALOG_MODELS",
    "MAX_DOWNLOAD_BYTES",
    "MAX_PREVIEW_BYTES",
    "MAX_REMOTE_CATALOG_BYTES",
    "MAX_TEST_TEXT",
    "OFFICIAL_PIPER_CATALOG_URL",
    "OFFICIAL_PIPER_REPOSITORY_URL",
    "OFFICIAL_PIPER_SAMPLES_URL",
    "PiperModelCache",
    "PiperProvider",
    "SynthesisQueue",
    "SpeechCallback",
    "SpeechScheduler",
    "SynthesizedAudio",
    "TTSException",
    "TTSService",
    "TTSVoicePreset",
    "VOICE_PRESET_SLOTS",
    "VoiceCatalog",
    "VoiceModel",
    "VoicePresetRepository",
]
