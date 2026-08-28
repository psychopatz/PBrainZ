"""Piper voice discovery, catalog normalization, downloads, and previews."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pbrainz.branding import PRODUCT_BINARY_NAME
from pbrainz.config import Settings

from .models import (
    MAX_CATALOG_MODELS,
    MAX_DOWNLOAD_BYTES,
    MAX_PREVIEW_BYTES,
    MAX_REMOTE_CATALOG_BYTES,
    OFFICIAL_PIPER_CATALOG_URL,
    DownloadProgressCallback,
    InstallProgressCallback,
    TTSException,
    VoiceModel,
    _infer_voice_gender,
    _is_number,
    _official_voice_file_url,
    _official_voice_sample_url,
    _safe_int,
)

LOGGER = logging.getLogger(__name__)

class VoiceCatalog:
    """Merge local Piper files with Piper's downloadable voice catalog."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models: dict[str, VoiceModel] = {}
        self.remote_models: dict[str, dict[str, Any]] = {}
        self.last_scan_at: float | None = None
        self.last_error: str | None = None
        self.last_remote_fetch_at: float | None = None
        self.last_remote_error: str | None = None

    @property
    def roots(self) -> tuple[Path, ...]:
        configured = self.settings.tts_model_root or os.getenv("PIPER_MODEL_ROOT")
        if configured:
            return (Path(configured).expanduser(),)
        database_path = self.settings.database_path
        if database_path:
            return (Path(database_path).expanduser().parent / "piper",)
        return (Path.home() / ".local" / "share" / "piper",)

    @property
    def metadata_path(self) -> Path:
        configured = self.settings.tts_metadata_path
        if configured:
            return Path(configured).expanduser()
        return self.roots[0] / "voice_metadata.json"

    @property
    def remote_catalog_url(self) -> str:
        return self.settings.tts_voice_catalog_url.strip() or OFFICIAL_PIPER_CATALOG_URL

    @property
    def remote_cache_path(self) -> Path:
        return self.roots[0] / "voice_catalog.json"

    def refresh(self) -> list[VoiceModel]:
        discovered: dict[str, VoiceModel] = {}
        self.last_error = None
        overlay = self._read_overlay()
        try:
            paths = sorted(
                path for root in self.roots if root.is_dir() for path in root.rglob("*.onnx")
            )[:MAX_CATALOG_MODELS]
            for model_path in paths:
                model = self._read_model(
                    model_path,
                    overlay.get(model_path.stem, {}),
                    self.remote_models.get(model_path.stem),
                )
                discovered[model.id] = model
        except (OSError, ValueError, TypeError) as error:
            self.last_error = str(error)
            LOGGER.warning("Piper voice catalog scan failed: %s", error)
        for model_id, remote in self.remote_models.items():
            if model_id not in discovered:
                discovered[model_id] = self._remote_model(
                    model_id, remote, overlay.get(model_id, {})
                )
        self.models = discovered
        self.last_scan_at = time.time()
        return list(self.models.values())

    def refresh_remote(self, *, force: bool = False) -> bool:
        """Refresh the official catalog, falling back to its local cache."""

        if not self.remote_models:
            self._load_remote_cache()
        ttl = max(60, int(self.settings.tts_voice_catalog_ttl_seconds))
        if (
            not force
            and self.remote_models
            and self.last_remote_fetch_at is not None
            and time.time() - self.last_remote_fetch_at < ttl
        ):
            self.refresh()
            return True
        try:
            request = Request(
                self.remote_catalog_url,
                headers={"User-Agent": f"{PRODUCT_BINARY_NAME}/0.1 Piper voice catalog"},
            )
            with urlopen(request, timeout=12) as response:  # nosec B310 - configured catalog URL
                raw = response.read(MAX_REMOTE_CATALOG_BYTES + 1)
            if len(raw) > MAX_REMOTE_CATALOG_BYTES:
                raise TTSException("Piper voice catalog is larger than the safety limit")
            value = json.loads(raw.decode("utf-8"))
            normalized = self._normalize_remote_catalog(value)
            if not normalized:
                raise TTSException("Piper voice catalog did not contain usable voices")
            self.remote_models = normalized
            self.last_remote_fetch_at = time.time()
            self.last_remote_error = None
            self._write_remote_cache(raw)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            TTSException,
        ) as error:
            self.last_remote_error = str(error)
            LOGGER.warning("Piper remote voice catalog unavailable: %s", error)
        self.refresh()
        return bool(self.remote_models)

    def install(
        self, model_id: str, progress: InstallProgressCallback | None = None
    ) -> VoiceModel:
        """Download one catalog voice and verify both Piper files before activation."""

        self.refresh()
        model = self.models.get(str(model_id))
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {model_id}")
        if not model.model_url or not model.config_url:
            if model.installed:
                return model
            raise TTSException(f"Piper voice has no downloadable files: {model_id}")
        root = self.roots[0]
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / Path(model.model_path).name
        config_path = root / Path(model.config_path).name
        total_bytes = max(0, model.model_size_bytes) + max(0, model.config_size_bytes)
        if progress:
            progress("Preparing download", 0, total_bytes)
        model_bytes = self._download_verified(
            model.model_url,
            model_path,
            expected_size=model.model_size_bytes,
            expected_md5=model.model_md5,
            progress=(
                lambda completed, total: progress("Downloading model", completed, total)
                if progress
                else None
            ),
            progress_total=total_bytes,
        )
        try:
            self._download_verified(
                model.config_url,
                config_path,
                expected_size=model.config_size_bytes,
                expected_md5=model.config_md5,
                progress=(
                    lambda completed, total: progress("Downloading config", completed, total)
                    if progress
                    else None
                ),
                progress_offset=model_bytes,
                progress_total=total_bytes,
            )
        except Exception:
            model_path.unlink(missing_ok=True)
            raise
        self.refresh()
        installed = self.models.get(model.id)
        if installed is None or not installed.installed:
            raise TTSException(
                f"Piper voice installation did not produce a usable model: {model.id}"
            )
        return installed

    def uninstall(self, model_id: str) -> VoiceModel:
        """Delete the selected Piper model and its matching JSON descriptor."""

        self.refresh()
        model = self.models.get(str(model_id).strip())
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {model_id}")
        if not model.installed:
            raise TTSException(f"Piper voice is not installed: {model.id}")

        paths = (Path(model.model_path), Path(model.config_path))
        roots = tuple(root.expanduser().resolve() for root in self.roots)
        for path in paths:
            resolved = path.expanduser().resolve()
            if not any(resolved.parent == root or root in resolved.parents for root in roots):
                raise TTSException(
                    f"Piper voice path is outside the managed model directory: {path}"
                )
        try:
            for path in paths:
                path.unlink(missing_ok=True)
        except OSError as error:
            raise TTSException(f"Could not uninstall Piper voice {model.id}: {error}") from error
        self.refresh()
        return self.models.get(model.id) or replace(model, installed=False)

    def download_sample(self, model_id: str) -> Path:
        """Download a bounded pre-generated Piper sample for one catalog voice."""

        self.refresh()
        model = self.models.get(str(model_id))
        if model is None:
            raise TTSException(f"Piper voice is not in the catalog: {model_id}")
        if not model.sample_url:
            raise TTSException(f"Piper voice has no preview sample: {model_id}")
        descriptor, output_name = tempfile.mkstemp(prefix="pbrainz-tts-preview-", suffix=".mp3")
        os.close(descriptor)
        output_path = Path(output_name)
        total = 0
        try:
            request = Request(
                model.sample_url,
                headers={"User-Agent": f"{PRODUCT_BINARY_NAME}/0.1 Piper voice preview"},
            )
            with urlopen(request, timeout=30) as response, output_path.open("wb") as output:  # nosec B310 - generated samples URL
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_PREVIEW_BYTES:
                        raise TTSException("Piper voice preview exceeded the safety limit")
                    output.write(chunk)
            if total <= 0:
                raise TTSException("Piper voice preview was empty")
            return output_path
        except (OSError, TTSException) as error:
            output_path.unlink(missing_ok=True)
            if isinstance(error, TTSException):
                raise
            raise TTSException(f"Piper voice preview download failed: {error}") from error
        except (ValueError, TypeError) as error:
            output_path.unlink(missing_ok=True)
            raise TTSException(f"Piper voice preview download failed: {error}") from error

    def get(self, model_id: str) -> VoiceModel | None:
        if not self.models:
            self.refresh()
        return self.models.get(str(model_id))

    def filter(
        self,
        *,
        language: str = "",
        gender: str = "Any",
        quality: str = "Any",
        installed_only: bool = False,
    ) -> list[VoiceModel]:
        if not self.models:
            self.refresh()
        language = language.casefold().strip()
        gender = gender.casefold().strip()
        quality = quality.casefold().strip()
        return [
            model
            for model in self.models.values()
            if (not language or language in {model.language.casefold(), model.locale.casefold()})
            and (not gender or gender == "any" or model.gender.casefold() == gender)
            and (not quality or quality == "any" or model.quality.casefold() == quality)
            and (not installed_only or model.installed)
        ]

    def _read_overlay(self) -> dict[str, dict[str, Any]]:
        path = self.metadata_path
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.last_error = f"metadata overlay: {error}"
            LOGGER.warning("Piper voice metadata overlay is unreadable: %s", error)
            return {}
        if isinstance(value, dict) and isinstance(value.get("voices"), dict):
            value = value["voices"]
        if not isinstance(value, dict):
            return {}
        return {
            str(model_id): metadata
            for model_id, metadata in value.items()
            if isinstance(metadata, dict)
        }

    def _write_remote_cache(self, raw: bytes) -> None:
        path = self.remote_cache_path
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        except OSError as error:
            LOGGER.warning("Could not cache Piper voice catalog: %s", error)
            temporary.unlink(missing_ok=True)

    def _load_remote_cache(self) -> bool:
        path = self.remote_cache_path
        try:
            if not path.is_file():
                return False
            raw = path.read_bytes()
            if len(raw) > MAX_REMOTE_CATALOG_BYTES:
                return False
            normalized = self._normalize_remote_catalog(json.loads(raw.decode("utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
        if not normalized:
            return False
        self.remote_models = normalized
        self.last_remote_fetch_at = path.stat().st_mtime
        return True

    @staticmethod
    def _normalize_remote_catalog(value: object) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for raw_id, raw_entry in value.items():
            if not isinstance(raw_entry, dict):
                continue
            files = raw_entry.get("files")
            if not isinstance(files, dict):
                continue
            model_file = next(
                (
                    str(path)
                    for path in files
                    if str(path).endswith(".onnx") and not str(path).endswith(".onnx.json")
                ),
                "",
            )
            config_file = next(
                (str(path) for path in files if str(path).endswith(".onnx.json")), ""
            )
            if not model_file or not config_file:
                continue
            if any(
                part in {"", ".", ".."}
                for part in (*model_file.split("/"), *config_file.split("/"))
            ):
                continue
            model_name = Path(model_file).name
            config_name = Path(config_file).name
            if (
                Path(model_file).name != model_file.split("/")[-1]
                or Path(config_file).name != config_file.split("/")[-1]
                or Path(model_name).stem != str(raw_id)
            ):
                continue
            model_info = files.get(model_file)
            config_info = files.get(config_file)
            if not isinstance(model_info, dict) or not isinstance(config_info, dict):
                continue
            language = raw_entry.get("language")
            if not isinstance(language, dict):
                language = {}
            language_code = str(language.get("code") or "unknown")
            voice_name = str(raw_entry.get("name") or raw_id)
            quality = str(raw_entry.get("quality") or "unknown").lower()
            num_speakers = max(1, int(raw_entry.get("num_speakers") or 1))
            normalized[str(raw_id)[:256]] = {
                "model_file": model_name,
                "config_file": config_name,
                "language": str(
                    language.get("name_english")
                    or language.get("family")
                    or language.get("code")
                    or "unknown"
                ),
                "locale": language_code,
                "display_name": voice_name,
                "gender": str(
                    raw_entry.get("gender") or _infer_voice_gender(voice_name, num_speakers)
                ).lower(),
                "quality": quality,
                "num_speakers": num_speakers,
                "speaker_map": raw_entry.get("speaker_id_map")
                if isinstance(raw_entry.get("speaker_id_map"), dict)
                else {},
                "model_url": _official_voice_file_url(model_file),
                "config_url": _official_voice_file_url(config_file),
                "model_size_bytes": _safe_int(model_info.get("size_bytes")),
                "model_md5": str(model_info.get("md5_digest") or "").lower(),
                "config_size_bytes": _safe_int(config_info.get("size_bytes")),
                "config_md5": str(config_info.get("md5_digest") or "").lower(),
                "sample_url": _official_voice_sample_url(language_code, voice_name, quality),
            }
        return dict(list(normalized.items())[:MAX_CATALOG_MODELS])

    @staticmethod
    def _read_model(
        path: Path, overlay: dict[str, Any], remote: dict[str, Any] | None = None
    ) -> VoiceModel:
        config_path = Path(f"{path}.json")
        metadata: dict[str, Any] = {}
        if config_path.is_file():
            try:
                value = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    metadata = value
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        language = metadata.get("language")
        if isinstance(language, dict):
            locale = str(language.get("code") or language.get("locale") or "unknown")
            language = str(
                language.get("name_english")
                or language.get("name_native")
                or language.get("family")
                or locale
            )
        else:
            language = str(language or metadata.get("language_code") or "unknown")
            locale = str(metadata.get("locale") or metadata.get("language_code") or language)
        speaker_map = metadata.get("speaker_id_map") or metadata.get("speaker_map") or {}
        normalized_speaker_map = (
            {str(key): int(value) for key, value in speaker_map.items() if _is_number(value)}
            if isinstance(speaker_map, dict)
            else {}
        )
        model_id = path.stem
        return VoiceModel(
            id=model_id,
            display_name=str(
                overlay.get("display_name")
                or metadata.get("display_name")
                or (remote or {}).get("display_name")
                or model_id
            ),
            language=str(overlay.get("language") or language or (remote or {}).get("language")),
            locale=str(overlay.get("locale") or locale or (remote or {}).get("locale")),
            gender=str(
                overlay.get("gender")
                or metadata.get("gender")
                or (remote or {}).get("gender")
                or _infer_voice_gender(
                    str(
                        overlay.get("display_name")
                        or metadata.get("display_name")
                        or (remote or {}).get("display_name")
                        or model_id
                    ),
                    int((remote or {}).get("num_speakers") or metadata.get("num_speakers") or 1),
                )
            ).lower(),
            quality=str(
                overlay.get("quality")
                or metadata.get("quality")
                or (remote or {}).get("quality")
                or "unknown"
            ).lower(),
            num_speakers=max(
                1, int(metadata.get("num_speakers") or (remote or {}).get("num_speakers") or 1)
            ),
            speaker_map=normalized_speaker_map,
            installed=config_path.is_file(),
            model_path=str(path),
            config_path=str(config_path),
            source="official" if remote else "local",
            model_url=str((remote or {}).get("model_url") or ""),
            config_url=str((remote or {}).get("config_url") or ""),
            model_size_bytes=_safe_int((remote or {}).get("model_size_bytes")),
            model_md5=str((remote or {}).get("model_md5") or ""),
            config_size_bytes=_safe_int((remote or {}).get("config_size_bytes")),
            config_md5=str((remote or {}).get("config_md5") or ""),
            sample_url=str((remote or {}).get("sample_url") or ""),
        )

    def _remote_model(
        self, model_id: str, remote: dict[str, Any], overlay: dict[str, Any]
    ) -> VoiceModel:
        root = self.roots[0]
        model_path = root / str(remote.get("model_file") or f"{model_id}.onnx")
        config_path = root / str(remote.get("config_file") or f"{model_id}.onnx.json")
        return VoiceModel(
            id=model_id,
            display_name=str(overlay.get("display_name") or remote.get("display_name") or model_id),
            language=str(overlay.get("language") or remote.get("language") or "unknown"),
            locale=str(overlay.get("locale") or remote.get("locale") or "unknown"),
            gender=str(
                overlay.get("gender")
                or remote.get("gender")
                or _infer_voice_gender(
                    str(remote.get("display_name") or model_id),
                    int(remote.get("num_speakers") or 1),
                )
            ).lower(),
            quality=str(overlay.get("quality") or remote.get("quality") or "unknown").lower(),
            num_speakers=max(1, int(remote.get("num_speakers") or 1)),
            speaker_map=remote.get("speaker_map")
            if isinstance(remote.get("speaker_map"), dict)
            else {},
            installed=model_path.is_file() and config_path.is_file(),
            model_path=str(model_path),
            config_path=str(config_path) if config_path.is_file() else str(config_path),
            source="official",
            model_url=str(remote.get("model_url") or ""),
            config_url=str(remote.get("config_url") or ""),
            model_size_bytes=_safe_int(remote.get("model_size_bytes")),
            model_md5=str(remote.get("model_md5") or ""),
            config_size_bytes=_safe_int(remote.get("config_size_bytes")),
            config_md5=str(remote.get("config_md5") or ""),
            sample_url=str(remote.get("sample_url") or ""),
        )

    @staticmethod
    def _download_verified(
        url: str,
        target: Path,
        *,
        expected_size: int,
        expected_md5: str,
        progress: DownloadProgressCallback | None = None,
        progress_offset: int = 0,
        progress_total: int = 0,
    ) -> int:
        if target.is_file() and _file_matches(target, expected_size, expected_md5):
            completed = expected_size or target.stat().st_size
            if progress:
                progress(progress_offset + completed, progress_total)
            return completed
        temporary = target.with_name(f".{target.name}.part")
        digest = hashlib.md5(usedforsecurity=False)
        total = 0
        try:
            request = Request(
                url, headers={"User-Agent": f"{PRODUCT_BINARY_NAME}/0.1 Piper voice installer"}
            )
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:  # nosec B310 - catalog URL
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise TTSException("Piper voice download exceeded the safety limit")
                    digest.update(chunk)
                    output.write(chunk)
                    if progress:
                        progress(progress_offset + total, progress_total)
            if expected_size and total != expected_size:
                raise TTSException(
                    f"downloaded {target.name} has size {total}, expected {expected_size}"
                )
            if expected_md5 and digest.hexdigest().lower() != expected_md5.lower():
                raise TTSException(f"downloaded {target.name} failed checksum verification")
            os.replace(temporary, target)
            return total
        finally:
            temporary.unlink(missing_ok=True)

def _file_matches(path: Path, expected_size: int, expected_md5: str) -> bool:
    try:
        if not path.is_file() or (expected_size and path.stat().st_size != expected_size):
            return False
        if not expected_md5:
            return path.stat().st_size > 0
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().lower() == expected_md5.lower()
    except OSError:
        return False
