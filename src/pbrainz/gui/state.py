"""Shared Tk variables and mutable panel state."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field

from pbrainz.branding import PRODUCT_NAME


@dataclass
class PanelState:
    """State shared by the window shell and individual tabs."""

    status: tk.StringVar
    bridge_status: tk.StringVar
    bridge_details: tk.StringVar
    bridge_worker: tk.BooleanVar
    provider: tk.StringVar
    model: tk.StringVar
    timeout: tk.StringVar
    poll_interval: tk.StringVar
    theme: tk.StringVar
    openai_base_url: tk.StringVar
    openai_key: tk.StringVar
    ollama_base_url: tk.StringVar
    ollama_key: tk.StringVar
    lmstudio_base_url: tk.StringVar
    lmstudio_key: tk.StringVar
    custom_base_url: tk.StringVar
    custom_key: tk.StringVar
    gemini_key: tk.StringVar
    provider_info: tk.StringVar
    provider_models: dict[str, list[str]] = field(default_factory=dict)
    provider_configured: dict[str, bool] = field(default_factory=dict)
    settings_dirty: bool = False
    applying_status: bool = False
    refresh_in_flight: bool = False
    logs_in_flight: bool = False
    closed: bool = False

    @classmethod
    def create(cls, root: tk.Misc) -> PanelState:
        """Create all shared Tk variables on the panel's interpreter."""

        return cls(
            status=tk.StringVar(root, value=f"Starting {PRODUCT_NAME}…"),
            bridge_status=tk.StringVar(root, value="Checking Project Hoomans bridge…"),
            bridge_details=tk.StringVar(root, value=""),
            bridge_worker=tk.BooleanVar(root, value=False),
            provider=tk.StringVar(root),
            model=tk.StringVar(root),
            timeout=tk.StringVar(root, value="120"),
            poll_interval=tk.StringVar(root, value="0.5"),
            theme=tk.StringVar(root, value="light"),
            openai_base_url=tk.StringVar(root, value="https://api.openai.com/v1"),
            openai_key=tk.StringVar(root),
            ollama_base_url=tk.StringVar(root, value="http://127.0.0.1:11434/v1"),
            ollama_key=tk.StringVar(root),
            lmstudio_base_url=tk.StringVar(root, value="http://127.0.0.1:1234/v1"),
            lmstudio_key=tk.StringVar(root),
            custom_base_url=tk.StringVar(root),
            custom_key=tk.StringVar(root),
            gemini_key=tk.StringVar(root),
            provider_info=tk.StringVar(root, value=""),
        )

    def bind_dirty_tracking(self) -> None:
        """Mark editable values dirty while allowing status refreshes to hydrate them."""

        for variable in (
            self.provider,
            self.model,
            self.timeout,
            self.poll_interval,
            self.theme,
            self.openai_base_url,
            self.openai_key,
            self.ollama_base_url,
            self.ollama_key,
            self.lmstudio_base_url,
            self.lmstudio_key,
            self.custom_base_url,
            self.custom_key,
            self.gemini_key,
        ):
            variable.trace_add("write", self._mark_settings_dirty)

    def _mark_settings_dirty(self, *_args: object) -> None:
        if not self.applying_status:
            self.settings_dirty = True
