"""Threaded requests from native GUI components to the local API."""

from __future__ import annotations

import json
import threading
import tkinter as tk
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

RequestSuccess = Callable[[dict[str, Any]], None]
RequestFailure = Callable[[Exception], None]
RequestFn = Callable[..., None]


class ApiRequestRunner:
    """Keep blocking HTTP work off Tk's event loop."""

    def __init__(self, root: tk.Misc, host: str, port: int) -> None:
        self.root = root
        self.base_url = f"http://{self._local_host(host)}:{port}"

    @staticmethod
    def _local_host(host: str) -> str:
        return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host

    def run(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        success: RequestSuccess,
        *,
        failure: RequestFailure | None = None,
        timeout: float = 8,
    ) -> None:
        def worker() -> None:
            try:
                data = self.request(method, path, payload, timeout=timeout)
                self.root.after(0, success, data)
            except Exception as request_error:  # Network errors belong in the GUI status.
                callback = failure or self._raise_to_gui
                self.root.after(0, callback, request_error)

        threading.Thread(target=worker, name="p-brainz-gui-request", daemon=True).start()

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: float = 8,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                value = json.loads(detail)
                detail = value.get("detail") or value.get("error", {}).get("message") or detail
            except (TypeError, ValueError):
                pass
            raise RuntimeError(detail or f"HTTP {error.code}") from error

    def _raise_to_gui(self, error: Exception) -> None:
        raise error
