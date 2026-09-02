"""Process-level lock preventing multiple PBrainZ runtimes per install."""

from __future__ import annotations

import os
from pathlib import Path

from pbrainz.branding import PRODUCT_NAME, PROJECT_NAME
from pbrainz.database import application_root

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class SingleInstanceLock:
    """Hold an operating-system lock for the lifetime of the application."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._handle = None

    def acquire(self) -> bool:
        """Try to acquire the lock without removing stale lock files."""

        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            _lock(handle)
        except (BlockingIOError, OSError):
            handle.close()
            return False
        self._handle = handle
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.flush()
        return True

    def release(self) -> None:
        """Release the OS lock while retaining the harmless lock marker."""

        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> SingleInstanceLock:
        if not self.acquire():
            raise RuntimeError(f"{PRODUCT_NAME} is already running")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def default_lock_path() -> Path:
    """Return the lock location inside the portable application data root."""

    return application_root() / "data" / f".{PROJECT_NAME}.lock"


def _lock(handle) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
