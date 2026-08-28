"""Bounded fixed-slot file transport for the PsychopatzCore bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pbrainz.paths import bridge_root_for

from .protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SLOT_COUNT,
    BridgeClientError,
    BridgeRequest,
    BridgeResponse,
)


class FileBridgeTransport:
    """Bounded fixed-slot transport matching PsychopatzBridgeFileTransport."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = bridge_root_for(explicit_bridge_root=root)
        self.requests = self.root / "requests"
        self.responses = self.root / "responses"
        self.state = self.root / "state"

    def ensure_directories(self) -> None:
        for directory in (self.requests, self.responses, self.state):
            directory.mkdir(parents=True, exist_ok=True)

    def write_request(self, request: BridgeRequest) -> int:
        self.ensure_directories()
        for slot in range(SLOT_COUNT):
            lock_path = self._path(self.requests, slot, ".lock")
            try:
                lock_path.open("x", encoding="ascii").close()
            except FileExistsError:
                continue
            if self._occupied(slot):
                lock_path.unlink(missing_ok=True)
                continue
            request_path = self._path(self.requests, slot, ".json")
            temporary_path = self.requests / f"{self._name(slot)}.{request.request_id}.tmp"
            encoded = json.dumps(
                request.as_dict(), ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
            if len(encoded) > MAX_REQUEST_BYTES:
                lock_path.unlink(missing_ok=True)
                raise BridgeClientError("bridge request exceeds the protocol size limit")
            try:
                temporary_path.write_bytes(encoded)
                os.replace(temporary_path, request_path)
            except OSError as error:
                temporary_path.unlink(missing_ok=True)
                lock_path.unlink(missing_ok=True)
                raise BridgeClientError(f"could not publish bridge request: {error}") from error
            return slot
        raise BridgeClientError("all PsychopatzCore bridge request slots are busy")

    def read_response(self, slot: int, request_id: str) -> BridgeResponse | None:
        marker_path = self._path(self.responses, slot, ".ready.txt")
        try:
            marker = marker_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as error:
            raise BridgeClientError(f"could not read bridge response marker: {error}") from error
        if marker != request_id:
            return None
        response_path = self._path(self.responses, slot, ".json")
        try:
            if response_path.stat().st_size > MAX_RESPONSE_BYTES:
                raise BridgeClientError("bridge response exceeds the protocol size limit")
            value = json.loads(response_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BridgeClientError(f"could not decode bridge response: {error}") from error
        return BridgeResponse.from_dict(value, request_id)

    def recover_stale_slots(self, runtime_id: str) -> int:
        """Release slot files that target a previous game runtime."""
        self.ensure_directories()
        recovered = 0
        for slot in range(SLOT_COUNT):
            request = self._read_object(self._path(self.requests, slot, ".json"))
            response = self._read_object(self._path(self.responses, slot, ".json"))
            request_runtime = request.get("target_runtime_id") if request else None
            response_runtime = response.get("runtime_id") if response else None
            stale_request = isinstance(request_runtime, str) and bool(
                request_runtime
            ) and request_runtime != runtime_id
            stale_response = isinstance(response_runtime, str) and bool(
                response_runtime
            ) and response_runtime != runtime_id
            if stale_request or stale_response:
                self.release(slot)
                recovered += 1
        return recovered

    def release(self, slot: int) -> None:
        for path in (
            self._path(self.requests, slot, ".json"),
            self._path(self.responses, slot, ".json"),
            self._path(self.responses, slot, ".ready.txt"),
            self._path(self.requests, slot, ".lock"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _occupied(self, slot: int) -> bool:
        return any(
            path.exists()
            for path in (
                self._path(self.requests, slot, ".json"),
                self._path(self.responses, slot, ".json"),
                self._path(self.responses, slot, ".ready.txt"),
            )
        )

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _name(slot: int) -> str:
        return f"slot-{slot:02d}"

    @classmethod
    def _path(cls, directory: Path, slot: int, suffix: str) -> Path:
        return directory / f"{cls._name(slot)}{suffix}"
