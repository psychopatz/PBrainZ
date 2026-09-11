"""Lightweight local DSP effects for presentation-only TTS audio."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from ..conversation_runtime import AudioPresentation
from .models import SynthesizedAudio, SynthesizedAudioChunk, TTSException

STREAM_FRAMES = 4096


def _alpha(cutoff_hz: float, sample_rate: int) -> float:
    """Return a stable one-pole filter coefficient for one audio stream."""

    cutoff = max(10.0, min(float(cutoff_hz), sample_rate * 0.45))
    return 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)


def _compress(sample: float, threshold: float = 0.42, ratio: float = 0.35) -> float:
    magnitude = abs(sample)
    if magnitude <= threshold:
        return sample
    compressed = threshold + (magnitude - threshold) * ratio
    return math.copysign(compressed, sample)


@dataclass(slots=True)
class _ChannelState:
    radio_highpass_low: float = 0.0
    radio_bandpass_low: float = 0.0
    telephone_highpass_low: float = 0.0
    telephone_bandpass_low: float = 0.0
    muffled_low: float = 0.0
    underwater_low_one: float = 0.0
    underwater_low_two: float = 0.0


class AudioEffectStream:
    """Process sequential PCM chunks while retaining filter state."""

    def __init__(self, presentation: AudioPresentation) -> None:
        self.presentation = presentation
        self._sample_rate: int | None = None
        self._sample_width: int | None = None
        self._sample_channels: int | None = None
        self._states: list[_ChannelState] = []
        self._radio_highpass_alpha = 0.0
        self._radio_bandpass_alpha = 0.0
        self._telephone_highpass_alpha = 0.0
        self._telephone_bandpass_alpha = 0.0
        self._muffled_alpha = 0.0
        self._underwater_alpha = 0.0

    def process(self, chunk: SynthesizedAudioChunk) -> SynthesizedAudioChunk:
        if not self.presentation.requires_processing:
            return chunk
        self._ensure_format(chunk)
        if chunk.sample_width != 2:
            return chunk
        if chunk.sample_channels <= 0 or len(chunk.pcm) % 2:
            raise TTSException("audio effect input is not valid 16-bit PCM")

        samples = array("h")
        samples.frombytes(chunk.pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if len(samples) % chunk.sample_channels:
            raise TTSException("audio effect input has an incomplete sample frame")

        presentation = self.presentation
        intensity = presentation.intensity
        profile = presentation.effect_profile
        environment = presentation.environment
        for index, raw in enumerate(samples):
            channel = index % chunk.sample_channels
            state = self._states[channel]
            dry = max(-1.0, min(1.0, raw / 32768.0))
            wet = dry

            if profile == "radio":
                state.radio_highpass_low += self._radio_highpass_alpha * (
                    wet - state.radio_highpass_low
                )
                wet -= state.radio_highpass_low
                state.radio_bandpass_low += self._radio_bandpass_alpha * (
                    wet - state.radio_bandpass_low
                )
                wet = state.radio_bandpass_low
                wet = _compress(wet)
                drive = 1.35
                wet = math.tanh(wet * drive) / math.tanh(drive)
            elif profile == "telephone":
                state.telephone_highpass_low += self._telephone_highpass_alpha * (
                    wet - state.telephone_highpass_low
                )
                wet -= state.telephone_highpass_low
                state.telephone_bandpass_low += self._telephone_bandpass_alpha * (
                    wet - state.telephone_bandpass_low
                )
                wet = _compress(state.telephone_bandpass_low, threshold=0.5, ratio=0.5)
            elif profile == "muffled":
                state.muffled_low += self._muffled_alpha * (wet - state.muffled_low)
                wet = state.muffled_low
            elif profile == "underwater":
                wet = self._underwater(wet, state)

            if environment == "underwater":
                wet = self._underwater(wet, state)
            elif environment == "muffled":
                state.muffled_low += self._muffled_alpha * (wet - state.muffled_low)
                wet = state.muffled_low

            value = max(-1.0, min(1.0, dry + (wet - dry) * intensity))
            samples[index] = max(-32768, min(32767, round(value * 32767.0)))

        if sys.byteorder != "little":
            samples.byteswap()
        return SynthesizedAudioChunk(
            sample_rate=chunk.sample_rate,
            sample_width=chunk.sample_width,
            sample_channels=chunk.sample_channels,
            pcm=samples.tobytes(),
            duration_ms=chunk.duration_ms,
        )

    def _ensure_format(self, chunk: SynthesizedAudioChunk) -> None:
        if self._sample_rate is None:
            self._sample_rate = chunk.sample_rate
            self._sample_width = chunk.sample_width
            self._sample_channels = chunk.sample_channels
            self._states = [_ChannelState() for _ in range(chunk.sample_channels)]
            self._radio_highpass_alpha = _alpha(350.0, chunk.sample_rate)
            self._radio_bandpass_alpha = _alpha(3000.0, chunk.sample_rate)
            self._telephone_highpass_alpha = _alpha(300.0, chunk.sample_rate)
            self._telephone_bandpass_alpha = _alpha(3400.0, chunk.sample_rate)
            self._muffled_alpha = _alpha(1800.0, chunk.sample_rate)
            self._underwater_alpha = _alpha(950.0, chunk.sample_rate)
            return
        if (
            chunk.sample_rate != self._sample_rate
            or chunk.sample_width != self._sample_width
            or chunk.sample_channels != self._sample_channels
        ):
            raise TTSException("audio effect input changed format mid-utterance")

    def _underwater(self, sample: float, state: _ChannelState) -> float:
        state.underwater_low_one += self._underwater_alpha * (
            sample - state.underwater_low_one
        )
        state.underwater_low_two += self._underwater_alpha * (
            state.underwater_low_one - state.underwater_low_two
        )
        return state.underwater_low_two


class AudioEffectProcessor:
    """Create stateful effect streams and process completed WAV output."""

    def stream(self, presentation: AudioPresentation) -> AudioEffectStream:
        return AudioEffectStream(presentation)

    def process_wav(
        self, audio: SynthesizedAudio, presentation: AudioPresentation
    ) -> SynthesizedAudio:
        if not presentation.requires_processing:
            return audio

        fd, output_name = tempfile.mkstemp(prefix="pbrainz-tts-fx-", suffix=".wav")
        os.close(fd)
        output_path = Path(output_name)
        stream = self.stream(presentation)
        try:
            with wave.open(str(audio.path), "rb") as source, wave.open(
                str(output_path), "wb"
            ) as target:
                target.setparams(source.getparams())
                while True:
                    pcm = source.readframes(STREAM_FRAMES)
                    if not pcm:
                        break
                    duration_ms = max(
                        1,
                        round(
                            len(pcm)
                            / (source.getframerate() * source.getsampwidth())
                            / source.getnchannels()
                            * 1000
                        ),
                    )
                    processed = stream.process(
                        SynthesizedAudioChunk(
                            sample_rate=source.getframerate(),
                            sample_width=source.getsampwidth(),
                            sample_channels=source.getnchannels(),
                            pcm=pcm,
                            duration_ms=duration_ms,
                        )
                    )
                    target.writeframes(processed.pcm)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return SynthesizedAudio(output_path, audio.duration_ms, audio.model_id)


__all__ = ["AudioEffectProcessor", "AudioEffectStream"]
