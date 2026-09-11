"""Lightweight local DSP effects for presentation-only TTS audio."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from ..conversation_runtime import AudioPresentation
from .models import SynthesizedAudio, SynthesizedAudioChunk, TTSException

STREAM_FRAMES = 4096
_TWO_PI = 2.0 * math.pi
_RADIO_DRIVE = 2.4
_RADIO_SATURATION = math.tanh(_RADIO_DRIVE)
_TELEPHONE_DRIVE = 1.5
_TELEPHONE_SATURATION = math.tanh(_TELEPHONE_DRIVE)


def _cutoff(cutoff_hz: float, sample_rate: int) -> float:
    return max(10.0, min(float(cutoff_hz), sample_rate * 0.45))


@dataclass(frozen=True, slots=True)
class _BiquadCoefficients:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


@dataclass(slots=True)
class _BiquadState:
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


def _lowpass(cutoff_hz: float, sample_rate: int, q: float = 0.7071) -> _BiquadCoefficients:
    """Return a normalized RBJ second-order low-pass filter."""

    cutoff = _cutoff(cutoff_hz, sample_rate)
    omega = _TWO_PI * cutoff / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * q)
    a0 = 1.0 + alpha
    return _BiquadCoefficients(
        (1.0 - cosine) / (2.0 * a0),
        (1.0 - cosine) / a0,
        (1.0 - cosine) / (2.0 * a0),
        -2.0 * cosine / a0,
        (1.0 - alpha) / a0,
    )


def _highpass(cutoff_hz: float, sample_rate: int, q: float = 0.7071) -> _BiquadCoefficients:
    """Return a normalized RBJ second-order high-pass filter."""

    cutoff = _cutoff(cutoff_hz, sample_rate)
    omega = _TWO_PI * cutoff / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * q)
    a0 = 1.0 + alpha
    return _BiquadCoefficients(
        (1.0 + cosine) / (2.0 * a0),
        -(1.0 + cosine) / a0,
        (1.0 + cosine) / (2.0 * a0),
        -2.0 * cosine / a0,
        (1.0 - alpha) / a0,
    )


def _biquad(
    sample: float, state: _BiquadState, coefficients: _BiquadCoefficients
) -> float:
    output = (
        coefficients.b0 * sample
        + coefficients.b1 * state.x1
        + coefficients.b2 * state.x2
        - coefficients.a1 * state.y1
        - coefficients.a2 * state.y2
    )
    state.x2 = state.x1
    state.x1 = sample
    state.y2 = state.y1
    state.y1 = output
    return output


def _time_alpha(seconds: float, sample_rate: int) -> float:
    return 1.0 - math.exp(-1.0 / max(1.0, seconds * sample_rate))


def _compress(
    sample: float,
    state: _ChannelState,
    attack_alpha: float,
    release_alpha: float,
    threshold: float,
    ratio: float,
) -> float:
    """Apply a small envelope compressor suitable for speech."""

    level = abs(sample)
    alpha = attack_alpha if level > state.compressor_envelope else release_alpha
    state.compressor_envelope += alpha * (level - state.compressor_envelope)
    envelope = state.compressor_envelope
    if envelope <= threshold:
        return sample
    target = threshold + (envelope - threshold) * ratio
    return sample * min(1.0, target / max(envelope, 1e-6))


@dataclass(slots=True)
class _ChannelState:
    radio_highpass: _BiquadState = field(default_factory=_BiquadState)
    radio_lowpass: _BiquadState = field(default_factory=_BiquadState)
    telephone_highpass: _BiquadState = field(default_factory=_BiquadState)
    telephone_lowpass: _BiquadState = field(default_factory=_BiquadState)
    muffled_lowpass_one: _BiquadState = field(default_factory=_BiquadState)
    muffled_lowpass_two: _BiquadState = field(default_factory=_BiquadState)
    underwater_lowpass_one: _BiquadState = field(default_factory=_BiquadState)
    underwater_lowpass_two: _BiquadState = field(default_factory=_BiquadState)
    underwater_bass: _BiquadState = field(default_factory=_BiquadState)
    radio_static_highpass: _BiquadState = field(default_factory=_BiquadState)
    radio_static_lowpass: _BiquadState = field(default_factory=_BiquadState)
    compressor_envelope: float = 0.0
    radio_crackle: float = 0.0
    noise_seed: int = 0x13579BDF


class AudioEffectStream:
    """Process sequential PCM chunks while retaining filter state."""

    def __init__(self, presentation: AudioPresentation, ambient_volume: float = 1.0) -> None:
        self.presentation = presentation
        self.ambient_volume = max(0.0, min(2.0, float(ambient_volume)))
        self._sample_rate: int | None = None
        self._sample_width: int | None = None
        self._sample_channels: int | None = None
        self._states: list[_ChannelState] = []
        self._radio_highpass: _BiquadCoefficients | None = None
        self._radio_lowpass: _BiquadCoefficients | None = None
        self._telephone_highpass: _BiquadCoefficients | None = None
        self._telephone_lowpass: _BiquadCoefficients | None = None
        self._muffled_lowpass: _BiquadCoefficients | None = None
        self._underwater_lowpass_one: _BiquadCoefficients | None = None
        self._underwater_lowpass_two: _BiquadCoefficients | None = None
        self._underwater_bass: _BiquadCoefficients | None = None
        self._compressor_attack = 0.0
        self._compressor_release = 0.0

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
            wet = self._apply_profile(dry, state, profile)
            if environment == "underwater" and profile != "underwater":
                wet = self._apply_underwater(wet, state)
            elif environment == "muffled" and profile != "muffled":
                wet = self._apply_muffled(wet, state)

            value = dry + (wet - dry) * intensity
            value += self._apply_ambient(state, profile)
            value = max(-1.0, min(1.0, value))
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

    def _apply_profile(self, sample: float, state: _ChannelState, profile: str) -> float:
        if profile == "radio":
            assert self._radio_highpass is not None
            assert self._radio_lowpass is not None
            wet = _biquad(sample, state.radio_highpass, self._radio_highpass)
            wet = _biquad(wet, state.radio_lowpass, self._radio_lowpass)
            wet = _compress(
                wet,
                state,
                self._compressor_attack,
                self._compressor_release,
                threshold=0.16,
                ratio=0.22,
            )
            wet = math.tanh(wet * _RADIO_DRIVE) / _RADIO_SATURATION
            wet = round(wet * 96.0) / 96.0
            return wet * 1.15
        if profile == "telephone":
            assert self._telephone_highpass is not None
            assert self._telephone_lowpass is not None
            wet = _biquad(sample, state.telephone_highpass, self._telephone_highpass)
            wet = _biquad(wet, state.telephone_lowpass, self._telephone_lowpass)
            wet = _compress(
                wet,
                state,
                self._compressor_attack,
                self._compressor_release,
                threshold=0.22,
                ratio=0.35,
            )
            return math.tanh(wet * _TELEPHONE_DRIVE) / _TELEPHONE_SATURATION
        if profile == "muffled":
            return self._apply_muffled(sample, state)
        if profile == "underwater":
            return self._apply_underwater(sample, state)
        return sample

    def _apply_muffled(self, sample: float, state: _ChannelState) -> float:
        assert self._muffled_lowpass is not None
        wet = _biquad(sample, state.muffled_lowpass_one, self._muffled_lowpass)
        wet = _biquad(wet, state.muffled_lowpass_two, self._muffled_lowpass)
        return wet * 0.82

    def _apply_underwater(self, sample: float, state: _ChannelState) -> float:
        assert self._underwater_lowpass_one is not None
        assert self._underwater_lowpass_two is not None
        assert self._underwater_bass is not None
        wet = _biquad(sample, state.underwater_lowpass_one, self._underwater_lowpass_one)
        wet = _biquad(wet, state.underwater_lowpass_two, self._underwater_lowpass_two)
        bass = _biquad(wet, state.underwater_bass, self._underwater_bass)
        return (wet * 0.78) + (bass * 0.22)

    def _apply_ambient(self, state: _ChannelState, profile: str) -> float:
        """Return background ambience independently of the processed voice."""

        if profile != "radio" or self.ambient_volume <= 0.0:
            return 0.0
        assert self._radio_highpass is not None
        assert self._radio_lowpass is not None
        state.noise_seed = (1664525 * state.noise_seed + 1013904223) & 0xFFFFFFFF
        noise = (state.noise_seed / 2147483648.0) - 1.0
        static = _biquad(noise, state.radio_static_highpass, self._radio_highpass)
        static = _biquad(static, state.radio_static_lowpass, self._radio_lowpass)
        if (state.noise_seed & 0x3FFF) == 0:
            state.radio_crackle = 1.0
        state.radio_crackle *= 0.96
        static *= 1.0 + (state.radio_crackle * 3.0)
        return static * 0.085 * self.ambient_volume

    def _ensure_format(self, chunk: SynthesizedAudioChunk) -> None:
        if self._sample_rate is None:
            self._sample_rate = chunk.sample_rate
            self._sample_width = chunk.sample_width
            self._sample_channels = chunk.sample_channels
            self._states = [
                _ChannelState(noise_seed=0x13579BDF + channel)
                for channel in range(chunk.sample_channels)
            ]
            self._radio_highpass = _highpass(280.0, chunk.sample_rate)
            self._radio_lowpass = _lowpass(2800.0, chunk.sample_rate)
            self._telephone_highpass = _highpass(300.0, chunk.sample_rate)
            self._telephone_lowpass = _lowpass(3400.0, chunk.sample_rate)
            self._muffled_lowpass = _lowpass(950.0, chunk.sample_rate)
            self._underwater_lowpass_one = _lowpass(950.0, chunk.sample_rate, q=0.8)
            self._underwater_lowpass_two = _lowpass(520.0, chunk.sample_rate, q=0.8)
            self._underwater_bass = _lowpass(180.0, chunk.sample_rate, q=0.75)
            self._compressor_attack = _time_alpha(0.008, chunk.sample_rate)
            self._compressor_release = _time_alpha(0.09, chunk.sample_rate)
            return
        if (
            chunk.sample_rate != self._sample_rate
            or chunk.sample_width != self._sample_width
            or chunk.sample_channels != self._sample_channels
        ):
            raise TTSException("audio effect input changed format mid-utterance")


class AudioEffectProcessor:
    """Create stateful effect streams and process completed WAV output."""

    def __init__(self, ambient_volume: float = 1.0) -> None:
        self.ambient_volume = max(0.0, min(2.0, float(ambient_volume)))

    def stream(
        self, presentation: AudioPresentation, ambient_volume: float | None = None
    ) -> AudioEffectStream:
        volume = self.ambient_volume if ambient_volume is None else ambient_volume
        return AudioEffectStream(presentation, volume)

    def process_wav(
        self,
        audio: SynthesizedAudio,
        presentation: AudioPresentation,
        ambient_volume: float | None = None,
    ) -> SynthesizedAudio:
        if not presentation.requires_processing:
            return audio

        fd, output_name = tempfile.mkstemp(prefix="pbrainz-tts-fx-", suffix=".wav")
        os.close(fd)
        output_path = Path(output_name)
        stream = self.stream(presentation, ambient_volume)
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
