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
    SynthesizedAudioChunk,
    TTSException,
    TTSVoicePreset,
    VoiceModel,
    VoicePresetRepository,
)
from .scheduler import SpeechScheduler, SynthesisQueue
from .service import TTSService
from .text import MAX_TTS_TEXT, NONVERBAL_CUE_RULES, NonverbalCueRule, normalize_tts_text

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
    "MAX_TTS_TEXT",
    "MAX_TEST_TEXT",
    "OFFICIAL_PIPER_CATALOG_URL",
    "OFFICIAL_PIPER_REPOSITORY_URL",
    "OFFICIAL_PIPER_SAMPLES_URL",
    "PiperModelCache",
    "PiperProvider",
    "NONVERBAL_CUE_RULES",
    "NonverbalCueRule",
    "SynthesisQueue",
    "SpeechCallback",
    "SpeechScheduler",
    "SynthesizedAudio",
    "SynthesizedAudioChunk",
    "TTSException",
    "TTSService",
    "TTSVoicePreset",
    "VOICE_PRESET_SLOTS",
    "VoiceCatalog",
    "VoiceModel",
    "VoicePresetRepository",
    "normalize_tts_text",
]
