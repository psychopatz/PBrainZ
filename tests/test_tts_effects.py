from __future__ import annotations

import math
import wave
from array import array

from pbrainz.conversation_runtime import AudioPresentation
from pbrainz.tts.effects import AudioEffectProcessor
from pbrainz.tts.models import SynthesizedAudio, SynthesizedAudioChunk

SAMPLE_RATE = 16_000


def _tone(frequency: float, duration_ms: int = 1000) -> SynthesizedAudioChunk:
    count = round(SAMPLE_RATE * duration_ms / 1000)
    samples = array(
        "h",
        [
            round(math.sin(2 * math.pi * frequency * index / SAMPLE_RATE) * 12_000)
            for index in range(count)
        ],
    )
    return SynthesizedAudioChunk(
        sample_rate=SAMPLE_RATE,
        sample_width=2,
        sample_channels=1,
        pcm=samples.tobytes(),
        duration_ms=duration_ms,
    )


def _rms(chunk: SynthesizedAudioChunk, *, skip_samples: int = 1600) -> float:
    samples = array("h")
    samples.frombytes(chunk.pcm)
    values = samples[skip_samples:]
    return math.sqrt(sum(value * value for value in values) / max(1, len(values)))


def test_audio_presentation_normalizes_aliases_and_bounds_intensity() -> None:
    presentation = AudioPresentation.from_mapping(
        {
            "audio": {
                "effectProfile": "walkie-talkie",
                "environment": "water",
                "intensity": 5,
            }
        }
    )

    assert presentation.as_dict() == {
        "effect_profile": "radio",
        "environment": "underwater",
        "intensity": 1.0,
    }


def test_no_effect_is_byte_exact() -> None:
    source = _tone(440)
    processed = AudioEffectProcessor().stream(AudioPresentation()).process(source)

    assert processed.pcm == source.pcm


def test_zero_intensity_is_byte_exact() -> None:
    source = _tone(440)
    presentation = AudioPresentation.from_mapping(
        {"effect_profile": "radio", "intensity": 0}
    )

    processed = AudioEffectProcessor().stream(presentation).process(source)

    assert processed.pcm == source.pcm


def test_radio_effect_changes_pcm_without_clipping() -> None:
    source = _tone(440)
    processed = AudioEffectProcessor().stream(
        AudioPresentation(effect_profile="radio")
    ).process(source)
    samples = array("h")
    samples.frombytes(processed.pcm)

    assert processed.pcm != source.pcm
    assert max(samples) <= 32767
    assert min(samples) >= -32768


def test_underwater_effect_reduces_high_frequency_energy() -> None:
    processor = AudioEffectProcessor()
    stream = processor.stream(AudioPresentation(effect_profile="underwater"))
    low = stream.process(_tone(250))
    # Use a fresh stream so the high-tone measurement has no state from low.
    high = processor.stream(AudioPresentation(effect_profile="underwater")).process(
        _tone(4000)
    )

    assert _rms(high) < _rms(low) * 0.5


def test_streaming_chunks_keep_filter_state() -> None:
    source = _tone(440, duration_ms=250)
    split_at = len(source.pcm) // 2
    first = SynthesizedAudioChunk(
        SAMPLE_RATE,
        2,
        1,
        source.pcm[:split_at],
        source.duration_ms // 2,
    )
    second = SynthesizedAudioChunk(
        SAMPLE_RATE,
        2,
        1,
        source.pcm[split_at:],
        source.duration_ms - source.duration_ms // 2,
    )
    presentation = AudioPresentation(effect_profile="radio")
    processor = AudioEffectProcessor()
    whole = processor.stream(presentation).process(source)
    stream = processor.stream(presentation)
    split = stream.process(first).pcm + stream.process(second).pcm

    assert split == whole.pcm


def test_completed_wav_path_applies_effect_and_preserves_source(tmp_path) -> None:
    source_chunk = _tone(440)
    source_path = tmp_path / "source.wav"
    with wave.open(str(source_path), "wb") as output:
        output.setnchannels(source_chunk.sample_channels)
        output.setsampwidth(source_chunk.sample_width)
        output.setframerate(source_chunk.sample_rate)
        output.writeframes(source_chunk.pcm)

    source = SynthesizedAudio(source_path, source_chunk.duration_ms, "test-model")
    processed = AudioEffectProcessor().process_wav(
        source, AudioPresentation(effect_profile="radio")
    )
    try:
        assert processed.path != source.path
        assert source.path.exists()
        with wave.open(str(processed.path), "rb") as output:
            assert output.getparams() == (1, 2, SAMPLE_RATE, 16000, "NONE", "not compressed")
            assert output.readframes(16000) != source_chunk.pcm
    finally:
        processed.path.unlink(missing_ok=True)
