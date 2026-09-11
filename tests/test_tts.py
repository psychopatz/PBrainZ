import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pbrainz.config import Settings
from pbrainz.conversation_runtime import Utterance, VoiceBinding
from pbrainz.tts import (
    DEFAULT_TTS_PRESETS,
    PiperModelCache,
    SpeechScheduler,
    SynthesizedAudio,
    SynthesizedAudioChunk,
    TTSException,
    TTSService,
    TTSVoicePreset,
    VoiceCatalog,
    VoiceModel,
    VoicePresetRepository,
)


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        database_path=str(tmp_path / "settings.db"),
        tts_model_root=str(tmp_path / "piper"),
        **overrides,
    )


def test_catalog_reads_piper_metadata_and_explicit_gender_overlay(tmp_path) -> None:
    root = tmp_path / "piper"
    root.mkdir()
    model_path = root / "en_US-example-female.onnx"
    model_path.write_bytes(b"placeholder")
    (root / f"{model_path.name}.json").write_text(
        json.dumps(
            {
                "language": {"code": "en_US", "family": "English"},
                "quality": "high",
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )
    (root / "voice_metadata.json").write_text(
        json.dumps({"voices": {model_path.stem: {"gender": "female", "display_name": "Example"}}}),
        encoding="utf-8",
    )
    catalog = VoiceCatalog(_settings(tmp_path))

    models = catalog.refresh()

    assert len(models) == 1
    assert models[0].display_name == "Example"
    assert models[0].language == "English"
    assert models[0].locale == "en_US"
    assert models[0].gender == "female"
    assert catalog.filter(gender="female", quality="high")[0].id == model_path.stem


class _CatalogResponse:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self._read = False

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self.value


def _remote_catalog(model_id: str, model_bytes: bytes, config_bytes: bytes) -> dict:
    return {
        model_id: {
            "name": "amy",
            "language": {
                "code": "en_US",
                "family": "en",
                "name_english": "English",
            },
            "quality": "medium",
            "num_speakers": 1,
            "speaker_id_map": {},
            "files": {
                f"en/en_US/amy/medium/{model_id}.onnx": {
                    "size_bytes": len(model_bytes),
                    "md5_digest": hashlib.md5(model_bytes, usedforsecurity=False).hexdigest(),
                },
                f"en/en_US/amy/medium/{model_id}.onnx.json": {
                    "size_bytes": len(config_bytes),
                    "md5_digest": hashlib.md5(config_bytes, usedforsecurity=False).hexdigest(),
                },
            },
        }
    }


def test_remote_catalog_distinguishes_missing_and_installed_voice_files(
    monkeypatch, tmp_path
) -> None:
    model_id = "en_US-amy-medium"
    catalog_json = _remote_catalog(model_id, b"model", b"config")
    settings = _settings(tmp_path)
    catalog = VoiceCatalog(settings)
    monkeypatch.setattr(
        "pbrainz.tts.catalog.urlopen",
        lambda *_args, **_kwargs: _CatalogResponse(json.dumps(catalog_json).encode()),
    )

    assert catalog.refresh_remote(force=True)
    assert catalog.get(model_id) is not None
    assert catalog.get(model_id).installed is False
    assert catalog.get(model_id).gender == "female"
    assert catalog.get(model_id).model_url
    assert catalog.get(model_id).config_url
    assert catalog.get(model_id).sample_url.endswith("/samples/en/en_US/amy/medium/speaker_0.mp3")

    root = Path(settings.tts_model_root)
    root.mkdir(exist_ok=True)
    (root / f"{model_id}.onnx").write_bytes(b"model")
    catalog.refresh()
    assert catalog.get(model_id).installed is False
    (root / f"{model_id}.onnx.json").write_bytes(b"config")
    catalog.refresh()
    assert catalog.get(model_id).installed is True


def test_installed_catalog_filter_uses_piper_display_language(tmp_path) -> None:
    model_id = "en_US-amy-medium"
    settings = _settings(tmp_path)
    catalog = VoiceCatalog(settings)
    catalog.remote_models = catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"model", b"config")
    )
    root = Path(settings.tts_model_root)
    root.mkdir(parents=True)
    (root / f"{model_id}.onnx").write_bytes(b"model")
    (root / f"{model_id}.onnx.json").write_text(
        json.dumps(
            {
                "language": {
                    "code": "en_US",
                    "family": "en",
                    "name_english": "English",
                }
            }
        ),
        encoding="utf-8",
    )

    catalog.refresh()

    assert catalog.get(model_id).language == "English"
    assert [model.id for model in catalog.filter(language="English", installed_only=True)] == [
        model_id
    ]


def test_catalog_uninstall_removes_model_and_descriptor(tmp_path) -> None:
    model_id = "en_US-amy-medium"
    settings = _settings(tmp_path)
    catalog = VoiceCatalog(settings)
    catalog.remote_models = catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"model", b"config")
    )
    root = Path(settings.tts_model_root)
    root.mkdir(parents=True)
    model_path = root / f"{model_id}.onnx"
    config_path = root / f"{model_id}.onnx.json"
    model_path.write_bytes(b"model")
    config_path.write_text(json.dumps({"language": {"code": "en_US"}}), encoding="utf-8")
    catalog.refresh()

    removed = catalog.uninstall(model_id)

    assert removed.installed is False
    assert not model_path.exists()
    assert not config_path.exists()
    assert catalog.filter(installed_only=True) == []


def test_remote_catalog_marks_multi_speaker_voice_as_mixed(tmp_path) -> None:
    model_id = "en_US-google-medium"
    catalog_json = _remote_catalog(model_id, b"model", b"config")
    entry = catalog_json[model_id]
    entry["name"] = "google"
    entry["num_speakers"] = 16

    catalog = VoiceCatalog(_settings(tmp_path))
    catalog.remote_models = catalog._normalize_remote_catalog(catalog_json)
    catalog.refresh()

    assert catalog.get(model_id).gender == "mixed"


def test_catalog_downloads_a_sample_without_installing_the_voice(monkeypatch, tmp_path) -> None:
    model_id = "en_US-amy-medium"
    catalog = VoiceCatalog(_settings(tmp_path))
    catalog.remote_models = catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"model", b"config")
    )
    catalog.refresh()
    monkeypatch.setattr(
        "pbrainz.tts.catalog.urlopen",
        lambda *_args, **_kwargs: _CatalogResponse(b"sample audio"),
    )

    sample_path = catalog.download_sample(model_id)
    try:
        assert sample_path.read_bytes() == b"sample audio"
        assert catalog.get(model_id).installed is False
    finally:
        sample_path.unlink(missing_ok=True)


def test_catalog_install_downloads_and_verifies_both_piper_files(monkeypatch, tmp_path) -> None:
    model_id = "en_US-amy-medium"
    model_bytes = b"verified model"
    config_bytes = b"verified config"
    catalog = VoiceCatalog(_settings(tmp_path))
    catalog.remote_models = catalog._normalize_remote_catalog(
        _remote_catalog(model_id, model_bytes, config_bytes)
    )
    catalog.refresh()

    def fake_urlopen(request, **_kwargs):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        return _CatalogResponse(
            model_bytes if url.endswith(".onnx?download=true") else config_bytes
        )

    monkeypatch.setattr("pbrainz.tts.catalog.urlopen", fake_urlopen)
    progress: list[tuple[str, int, int]] = []
    installed = catalog.install(model_id, progress=lambda *update: progress.append(update))

    assert installed.installed is True
    assert Path(installed.model_path).read_bytes() == model_bytes
    assert Path(installed.config_path).read_bytes() == config_bytes
    assert progress[0] == ("Preparing download", 0, len(model_bytes) + len(config_bytes))
    assert progress[-1] == (
        "Downloading config",
        len(model_bytes) + len(config_bytes),
        len(model_bytes) + len(config_bytes),
    )


@pytest.mark.asyncio
async def test_voice_install_jobs_deduplicate_and_report_completion(monkeypatch, tmp_path) -> None:
    model_id = "en_US-amy-medium"
    model_bytes = b"verified model"
    config_bytes = b"verified config"
    service = TTSService(_settings(tmp_path))
    service.catalog.remote_models = service.catalog._normalize_remote_catalog(
        _remote_catalog(model_id, model_bytes, config_bytes)
    )
    service.catalog.refresh()

    def fake_urlopen(request, **_kwargs):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        return _CatalogResponse(
            model_bytes if url.endswith(".onnx?download=true") else config_bytes
        )

    monkeypatch.setattr("pbrainz.tts.catalog.urlopen", fake_urlopen)
    first = service.start_voice_install(model_id)
    duplicate = service.start_voice_install(model_id)

    assert first["job_id"] == duplicate["job_id"]
    assert duplicate["already_running"] is True
    await _wait_until(lambda: service.install_status(first["job_id"])["state"] == "complete")
    completed = service.install_status(first["job_id"])
    assert completed["progress"] == 100.0
    assert completed["voice"]["installed"] is True
    await service.stop()


def test_catalog_install_rejects_checksum_mismatch(monkeypatch, tmp_path) -> None:
    model_id = "en_US-amy-medium"
    settings = _settings(tmp_path)
    catalog = VoiceCatalog(settings)
    catalog.remote_models = catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"verified model", b"verified config")
    )
    catalog.refresh()
    monkeypatch.setattr(
        "pbrainz.tts.catalog.urlopen",
        lambda *_args, **_kwargs: _CatalogResponse(b"tampered model"),
    )

    with pytest.raises(TTSException, match="checksum"):
        catalog.install(model_id)
    assert not (Path(settings.tts_model_root) / f"{model_id}.onnx").exists()


def test_presets_are_restricted_to_compact_voice_slots(tmp_path) -> None:
    settings = _settings(tmp_path)
    repository = VoicePresetRepository(settings)

    repository.replace(
        [
            TTSVoicePreset("VoiceFemale:0", "voice-a", 0),
            {"slot": "not-a-slot", "voice_model_id": "voice-secret"},
            {"slot": "VoiceMale:1", "voice_model_id": "voice-b", "optional_speaker_id": "2"},
        ]
    )

    assert set(repository.all()) == {"VoiceFemale:0", "VoiceMale:1"}
    assert repository.all()["VoiceMale:1"].optional_speaker_id == 2
    assert "voice-secret" not in settings.tts_voice_presets_json


def test_preset_repository_fills_empty_slots_without_overwriting_custom_choices(tmp_path) -> None:
    repository = VoicePresetRepository(_settings(tmp_path))
    repository.replace([{"slot": "VoiceMale:0", "voice_model_id": "my-custom-voice"}])

    missing_slots = repository.ensure_defaults()

    assert missing_slots == tuple(
        slot for slot, _model_id in DEFAULT_TTS_PRESETS if slot != "VoiceMale:0"
    )
    presets = repository.all()
    assert presets["VoiceMale:0"].voice_model_id == "my-custom-voice"
    assert all(
        presets[slot].voice_model_id == model_id
        for slot, model_id in DEFAULT_TTS_PRESETS
        if slot != "VoiceMale:0"
    )


@pytest.mark.asyncio
async def test_default_voice_setup_installs_selected_defaults_in_order(
    monkeypatch, tmp_path
) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    service.presets.ensure_defaults()
    service.catalog.models = {
        model_id: VoiceModel(
            model_id,
            model_id,
            language="English",
            gender="female" if slot.startswith("VoiceFemale") else "male",
            quality="medium",
            installed=False,
            model_url="https://example.invalid/model.onnx",
            config_url="https://example.invalid/model.onnx.json",
        )
        for slot, model_id in DEFAULT_TTS_PRESETS
    }
    monkeypatch.setattr(service, "refresh_catalog", _async_noop)
    installed: list[str] = []

    def fake_start_voice_install(model_id: str) -> dict[str, object]:
        installed.append(model_id)
        model = service.catalog.models[model_id]
        service.catalog.models[model_id] = replace(model, installed=True)
        return {"job_id": "", "state": "complete"}

    monkeypatch.setattr(service, "start_voice_install", fake_start_voice_install)

    await service._install_default_voices()

    assert installed == [model_id for _slot, model_id in DEFAULT_TTS_PRESETS]
    assert service._default_setup_complete is True


@pytest.mark.asyncio
async def test_tts_start_waits_for_explicit_default_download_request(monkeypatch, tmp_path) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    monkeypatch.setattr(service.scheduler, "start", _async_noop)

    await service.start()

    try:
        assert service._default_install_task is None
        assert service.default_voice_setup_needed() is True
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_default_voice_setup_retries_and_keeps_terminal_failure(
    monkeypatch, tmp_path
) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    service.presets.ensure_defaults()
    service.catalog.models = {
        model_id: VoiceModel(
            model_id,
            model_id,
            installed=False,
            model_url="https://example.invalid/model.onnx",
            config_url="https://example.invalid/model.onnx.json",
        )
        for _slot, model_id in DEFAULT_TTS_PRESETS
    }
    monkeypatch.setattr(service, "refresh_catalog", _async_noop)
    monkeypatch.setattr("pbrainz.tts.service.DEFAULT_INSTALL_RETRY_DELAYS", (0.0, 0.0))
    attempts: dict[str, int] = {}

    def flaky_start_voice_install(model_id: str) -> dict[str, object]:
        attempts[model_id] = attempts.get(model_id, 0) + 1
        if attempts[model_id] == 1:
            raise TTSException("Windows Defender blocked the download")
        model = service.catalog.models[model_id]
        service.catalog.models[model_id] = replace(model, installed=True)
        return {"job_id": "", "state": "complete"}

    monkeypatch.setattr(service, "start_voice_install", flaky_start_voice_install)

    await service._install_default_voices()

    assert set(attempts.values()) == {2}
    assert service._default_setup_complete is True
    assert service.default_install_status()["state"] == "complete"


@pytest.mark.asyncio
async def test_default_voice_setup_exposes_failed_state_for_manual_retry(
    monkeypatch, tmp_path
) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    service.presets.ensure_defaults()
    service.catalog.models = {
        model_id: VoiceModel(
            model_id,
            model_id,
            installed=False,
            model_url="https://example.invalid/model.onnx",
            config_url="https://example.invalid/model.onnx.json",
        )
        for _slot, model_id in DEFAULT_TTS_PRESETS
    }
    monkeypatch.setattr(service, "refresh_catalog", _async_noop)
    monkeypatch.setattr("pbrainz.tts.service.DEFAULT_INSTALL_RETRY_DELAYS", (0.0, 0.0))

    def failed_start_voice_install(_model_id: str) -> dict[str, object]:
        raise TTSException("network unavailable")

    monkeypatch.setattr(
        service,
        "start_voice_install",
        failed_start_voice_install,
    )

    await service._install_default_voices()

    failure = service.default_install_status()
    assert failure is not None
    assert failure["state"] == "failed"
    assert "network unavailable" in failure["error"]
    assert service._default_setup_complete is False


async def _async_noop(*_args, **_kwargs) -> bool:
    return True


def test_preset_repository_removes_all_slots_for_uninstalled_model(tmp_path) -> None:
    repository = VoicePresetRepository(_settings(tmp_path))
    repository.replace(
        [
            {"slot": "VoiceFemale:0", "voice_model_id": "voice-a"},
            {"slot": "VoiceMale:0", "voice_model_id": "voice-a"},
            {"slot": "VoiceMale:1", "voice_model_id": "voice-b"},
        ]
    )

    removed = repository.remove_model("voice-a")

    assert removed == ("VoiceFemale:0", "VoiceMale:0")
    assert set(repository.all()) == {"VoiceMale:1"}


def test_model_cache_is_bounded_lru() -> None:
    cache = PiperModelCache(2)
    cache.get_or_load("a", lambda: "A")
    cache.get_or_load("b", lambda: "B")
    cache.get_or_load("a", lambda: "new-A")
    cache.get_or_load("c", lambda: "C")

    assert cache.loaded_model_ids == ("a", "c")
    assert cache.eviction_count == 1
    assert cache.load_count == 3
    assert cache.evict("a") is True
    assert cache.loaded_model_ids == ("c",)
    assert cache.evict("missing") is False


@pytest.mark.asyncio
async def test_uninstall_voice_clears_cache_and_preset_references(tmp_path) -> None:
    model_id = "en_US-amy-medium"
    service = TTSService(_settings(tmp_path, tts_enabled=False))
    service.catalog.remote_models = service.catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"model", b"config")
    )
    root = Path(service.settings.tts_model_root)
    root.mkdir(parents=True)
    (root / f"{model_id}.onnx").write_bytes(b"model")
    (root / f"{model_id}.onnx.json").write_text(
        json.dumps({"language": {"code": "en_US"}}), encoding="utf-8"
    )
    service.catalog.refresh()
    service.presets.replace(
        [
            {"slot": "VoiceFemale:0", "voice_model_id": model_id},
            {"slot": "VoiceMale:0", "voice_model_id": "other-model"},
        ]
    )
    service.provider.cache.get_or_load(model_id, lambda: object())

    result = await service.uninstall_voice(model_id)

    assert result["cleared_preset_slots"] == ["VoiceFemale:0"]
    assert service.provider.cache.loaded_model_ids == ()
    assert "VoiceFemale:0" not in service.presets.all()
    assert service.presets.all()["VoiceMale:0"].voice_model_id == "other-model"
    assert not (root / f"{model_id}.onnx").exists()
    await service.stop()


@pytest.mark.asyncio
async def test_tts_is_disabled_without_starting_workers(tmp_path) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=False))

    await service.start()
    try:
        assert service.scheduler.synthesis._tasks == []
        assert not service.can_synthesize(None)
        assert not await service.test_voice("VoiceFemale:0", "test")
        assert "Enable TTS" not in (service.last_error or "")
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_installed_voice_test_plays_when_tts_is_disabled(tmp_path) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=False))
    provider = _FakeProvider(tmp_path)
    output = _FakeOutput()
    service.provider = provider
    service.output = output
    service.presets.replace(
        [{"slot": "VoiceMale:0", "voice_model_id": "installed-model"}]
    )

    await service.start()
    try:
        assert service.scheduler.synthesis._tasks == []
        assert await service.test_voice("VoiceMale:0", "manual test")
        await asyncio.wait_for(_wait_until(lambda: len(output.processes) == 1), timeout=2)
        assert provider.calls == ["manual test"]
        output.processes[0].release()
        await asyncio.wait_for(
            _wait_until(lambda: not (tmp_path / "test-audio-1.wav").exists()), timeout=2
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_enabled_tts_autoplays_generated_text_with_first_installed_preset(tmp_path) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    model_id = "en_US-amy-medium"
    root = Path(service.settings.tts_model_root)
    root.mkdir(parents=True)
    (root / f"{model_id}.onnx").write_bytes(b"model")
    (root / f"{model_id}.onnx.json").write_text(
        json.dumps({"language": {"code": "en_US", "name_english": "English"}}),
        encoding="utf-8",
    )
    service.catalog.refresh()
    service.presets.replace([{"slot": "VoiceFemale:0", "voice_model_id": model_id}])
    provider = _FakeProvider(tmp_path)
    output = _FakeOutput()
    service.provider = provider
    service.output = output

    try:
        assert await service.speak_text("generated response", source="control-panel")
        await asyncio.wait_for(_wait_until(lambda: len(output.processes) == 1), timeout=2)
        assert provider.calls == ["generated response"]
        output.processes[0].release()
        await asyncio.wait_for(
            _wait_until(lambda: not (tmp_path / "test-audio-1.wav").exists()), timeout=2
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_enabled_tts_uses_an_installed_voice_before_presets_are_configured(tmp_path) -> None:
    service = TTSService(_settings(tmp_path, tts_enabled=True))
    model_id = "en_US-amy-medium"
    root = Path(service.settings.tts_model_root)
    root.mkdir(parents=True)
    (root / f"{model_id}.onnx").write_bytes(b"model")
    (root / f"{model_id}.onnx.json").write_text(
        json.dumps({"language": {"code": "en_US", "name_english": "English"}}),
        encoding="utf-8",
    )
    service.catalog.refresh()
    provider = _FakeProvider(tmp_path)
    output = _FakeOutput()
    service.provider = provider
    service.output = output

    try:
        assert await service.speak_text("fallback response", source="control-panel")
        await asyncio.wait_for(_wait_until(lambda: len(output.processes) == 1), timeout=2)
        assert provider.calls == ["fallback response"]
        output.processes[0].release()
        await asyncio.wait_for(
            _wait_until(lambda: not (tmp_path / "test-audio-1.wav").exists()), timeout=2
        )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_tts_preview_plays_an_uninstalled_sample_without_enabling_tts(
    monkeypatch, tmp_path
) -> None:
    model_id = "en_US-amy-medium"
    service = TTSService(_settings(tmp_path, tts_enabled=False))
    service.catalog.remote_models = service.catalog._normalize_remote_catalog(
        _remote_catalog(model_id, b"model", b"config")
    )
    service.catalog.refresh()
    sample_path = tmp_path / "sample.mp3"
    sample_path.write_bytes(b"sample audio")
    monkeypatch.setattr(service.catalog, "download_sample", lambda _model_id: sample_path)
    output = _FakeOutput()
    service.output = output

    model = await service.preview_voice(model_id)
    await asyncio.wait_for(_wait_until(lambda: len(output.processes) == 1), timeout=2)
    assert model.id == model_id
    assert model.installed is False
    output.processes[0].release()
    await asyncio.wait_for(_wait_until(lambda: not sample_path.exists()), timeout=2)
    await service.stop()


def test_voice_model_serializes_catalog_fields() -> None:
    model = VoiceModel("id", "Name", language="English", locale="en_US")

    assert model.as_dict(include_paths=False) == {
        "id": "id",
        "display_name": "Name",
        "language": "English",
        "locale": "en_US",
        "gender": "unknown",
        "quality": "unknown",
        "num_speakers": 1,
        "speaker_map": {},
        "installed": True,
    }


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self._finished = asyncio.Event()

    async def wait(self) -> int:
        await self._finished.wait()
        self.returncode = 0
        return 0

    def release(self) -> None:
        self._finished.set()

    def terminate(self) -> None:
        self.release()


class _FakeProvider:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []

    def can_synthesize(self, _binding) -> bool:
        return True

    def synthesize(self, text: str, _binding) -> SynthesizedAudio:
        self.calls.append(text)
        path = self.root / f"test-audio-{len(self.calls)}.wav"
        path.write_bytes(b"test")
        return SynthesizedAudio(path, 300, "fake-model")


class _FakeOutput:
    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    @property
    def available(self) -> bool:
        return True

    async def start(self, _path: Path) -> _FakeProcess:
        process = _FakeProcess()
        self.processes.append(process)
        return process


class _FakePipe:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, value: bytes) -> None:
        self.writes.append(value)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeStreamProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.stdin = _FakePipe()


class _FakeStreamingProvider(_FakeProvider):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.stream_calls: list[str] = []

    def can_stream(self, _binding) -> bool:
        return True

    def stream_synthesize(self, text: str, _binding):
        self.stream_calls.append(text)
        yield SynthesizedAudioChunk(22050, 2, 1, b"first", 40)
        yield SynthesizedAudioChunk(22050, 2, 1, b"second", 50)


class _FakeStreamingOutput(_FakeOutput):
    streaming_available = True

    def __init__(self) -> None:
        super().__init__()
        self.streams: list[_FakeStreamProcess] = []

    async def start_stream(self, _rate: int, _width: int, _channels: int) -> _FakeStreamProcess:
        process = _FakeStreamProcess()
        self.streams.append(process)
        return process


@pytest.mark.asyncio
async def test_scheduler_starts_streaming_audio_before_process_finishes(tmp_path) -> None:
    settings = _settings(tmp_path, tts_natural_gap_ms=0, tts_audio_buffer_ms=1)
    provider = _FakeStreamingProvider(tmp_path)
    output = _FakeStreamingOutput()
    scheduler = SpeechScheduler(provider, output, settings)
    started: list[str] = []

    async def on_started(utterance: Utterance) -> None:
        started.append(utterance.utterance_id)

    utterance = Utterance(
        "stream-line",
        "stream-conversation",
        0,
        "npc-stream",
        "first sentence. second sentence.",
        voice_binding=VoiceBinding("npc-stream", "VoiceFemale:0"),
    )
    await scheduler.start()
    try:
        assert await scheduler.enqueue(utterance, on_started=on_started)
        await asyncio.wait_for(_wait_until(lambda: started == ["stream-line"]), timeout=2)
        assert provider.stream_calls == [utterance.text]
        assert len(output.streams) == 1
        assert output.streams[0].stdin.writes[0] == b"first"
        output.streams[0].release()
        await asyncio.wait_for(_wait_until(lambda: output.streams[0].stdin.closed), timeout=2)
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_keeps_queued_streaming_lines_in_streaming_order(tmp_path) -> None:
    settings = _settings(tmp_path, tts_natural_gap_ms=0, tts_audio_buffer_ms=1)
    provider = _FakeStreamingProvider(tmp_path)
    output = _FakeStreamingOutput()
    scheduler = SpeechScheduler(provider, output, settings)
    started: list[str] = []

    async def on_started(utterance: Utterance) -> None:
        started.append(utterance.utterance_id)

    first = Utterance(
        "stream-line-one",
        "stream-conversation",
        0,
        "npc-stream",
        "first line.",
        voice_binding=VoiceBinding("npc-stream", "VoiceFemale:0"),
    )
    second = Utterance(
        "stream-line-two",
        "stream-conversation",
        1,
        "npc-stream",
        "second line.",
        voice_binding=VoiceBinding("npc-stream", "VoiceFemale:0"),
    )
    await scheduler.start()
    try:
        assert await scheduler.enqueue(first, on_started=on_started)
        assert await scheduler.enqueue(second, on_started=on_started)
        await asyncio.wait_for(_wait_until(lambda: started == [first.utterance_id]), timeout=2)
        assert len(output.streams) == 1

        output.streams[0].release()
        await asyncio.wait_for(
            _wait_until(lambda: started == [first.utterance_id, second.utterance_id]),
            timeout=2,
        )
        assert len(output.streams) == 2
        output.streams[1].release()
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_synthesizes_next_line_while_current_line_plays(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        tts_max_generated_ahead=3,
        tts_max_tts_ready_ahead=1,
        tts_natural_gap_ms=0,
    )
    provider = _FakeProvider(tmp_path)
    output = _FakeOutput()
    scheduler = SpeechScheduler(provider, output, settings)
    started: list[str] = []

    async def on_started(utterance: Utterance) -> None:
        started.append(utterance.utterance_id)

    first = Utterance(
        "line-one",
        "conversation-one",
        0,
        "npc-a",
        "first",
        voice_binding=VoiceBinding("npc-a", "VoiceFemale:0"),
    )
    second = Utterance(
        "line-two",
        "conversation-one",
        1,
        "npc-a",
        "second",
        voice_binding=VoiceBinding("npc-a", "VoiceFemale:0"),
    )
    await scheduler.start()
    try:
        assert await scheduler.enqueue(first, on_started=on_started)
        assert await scheduler.enqueue(second, on_started=on_started)
        await asyncio.wait_for(_wait_until(lambda: started == ["line-one"]), timeout=2)
        await asyncio.wait_for(_wait_until(lambda: len(provider.calls) == 2), timeout=2)
        assert started == ["line-one"]

        output.processes[0].release()
        await asyncio.wait_for(_wait_until(lambda: started == ["line-one", "line-two"]), timeout=2)
        output.processes[1].release()
    finally:
        await scheduler.stop()


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.005)
