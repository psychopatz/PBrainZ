"""Native Piper TTS tab assembled from focused view mixins."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from pbrainz.branding import PRODUCT_NAME

from ..request import RequestFn
from .catalog_view import CatalogViewMixin
from .output_view import OutputViewMixin
from .presets_view import PresetsViewMixin


class TTSTab(OutputViewMixin, CatalogViewMixin, PresetsViewMixin):
    """Coordinate TTS controls while each view mixin owns one concern."""

    _SLOTS = tuple(
        [f"VoiceFemale:{index}" for index in range(4)]
        + [f"VoiceMale:{index}" for index in range(4)]
    )

    def __init__(self, parent: ttk.Frame, request: RequestFn) -> None:
        self.parent = parent
        self._request = request
        self._applying = False
        self._dirty = False
        self._models: list[dict[str, Any]] = []
        self._model_by_id: dict[str, dict[str, Any]] = {}
        self._preset_vars: dict[str, tk.StringVar] = {}
        self._preset_boxes: dict[str, ttk.Combobox] = {}
        self._preset_selected_ids: dict[str, str] = {}
        self._preset_option_ids: dict[str, dict[str, str]] = {}
        self._preset_gender_filters: dict[str, tk.BooleanVar] = {}
        self._preset_dirty = False
        self._non_preset_dirty = False
        self._preset_save_in_flight = False
        self._sort_column = "voice"
        self._sort_descending = False
        self._sort_heading_labels: dict[str, str] = {}
        self._install_job_id: str | None = None
        self._install_model_id: str | None = None
        self._install_request_in_flight = False
        self._uninstall_request_in_flight = False
        self._install_poll_after: str | None = None
        self._install_poll_failures = 0
        self._automatic_install_in_flight = False
        self._default_install_prompted = False
        self._default_install_request_in_flight = False
        self._default_install_refresh_after: str | None = None
        self._volume_scale_var = tk.DoubleVar(parent, value=1.0)
        self.enabled = tk.BooleanVar(parent, value=False)
        self.device = tk.StringVar(parent, value="system/default")
        self.volume = tk.StringVar(parent, value="1.0")
        self.language = tk.StringVar(parent, value="English")
        self.gender = tk.StringVar(parent, value="Any")
        self.quality = tk.StringVar(parent, value="Any")
        self.availability = tk.StringVar(parent, value="All voices")
        self.status = tk.StringVar(parent, value="TTS status unavailable")
        self.catalog_status = tk.StringVar(parent, value="Voice catalog unavailable")
        self._model_tree: ttk.Treeview
        self._install_button: ttk.Button
        self._uninstall_button: ttk.Button
        self._preview_button: ttk.Button
        self._build(parent)
        for variable in (
            self.enabled,
            self.device,
            self.volume,
        ):
            variable.trace_add("write", self._mark_dirty)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        outer = ttk.Frame(parent, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        self._build_output(outer)
        self._build_catalog(outer)
        self._build_presets(outer, parent)

        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="e")
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Save TTS settings", command=self.save).pack(side="left")

    def apply_status(self, data: dict[str, Any]) -> None:
        self._applying = True
        try:
            server_preset_ids: dict[str, str] = {}
            if not self._dirty:
                self.enabled.set(bool(data.get("enabled")))
                self.device.set(str(data.get("output_device") or "system/default"))
                volume = float(data.get("master_volume", 1.0))
                self.volume.set(f"{volume:.2f}")
                self._volume_scale_var.set(volume)
                self.language.set(
                    str(data.get("catalog_language") or self.language.get() or "English")
                )
                for preset in data.get("presets", []):
                    slot = str(preset.get("slot") or "")
                    if slot in self._preset_vars:
                        server_preset_ids[slot] = str(preset.get("voice_model_id") or "")
                self._preset_selected_ids = server_preset_ids
            self._models = list(data.get("catalog", []))
            self._model_by_id = {str(model.get("id")): model for model in self._models}
            languages = sorted(
                {
                    str(model.get("language") or "")
                    for model in self._models
                    if model.get("language")
                }
            )
            self._language_box["values"] = ["All languages", *languages]
            if self.language.get() not in self._language_box["values"]:
                self.language.set("All languages")
            devices = [str(device) for device in data.get("audio_devices", []) if device]
            current_device = str(data.get("output_device") or "system/default")
            self._device_box["values"] = list(
                dict.fromkeys(["system/default", *devices, current_device])
            )
            self._update_preset_options()
            speaking = (
                ", ".join(str(item) for item in data.get("currently_speaking_npcs", [])) or "none"
            )
            status = (
                f"TTS {'enabled' if data.get('enabled') else 'disabled'} · "
                f"Piper {'available' if data.get('piper_available') else 'unavailable'} · "
                f"audio {data.get('audio_backend') or 'unavailable'} · speaking: {speaking}"
            )
            last_error = str(data.get("last_error") or "").strip()
            self.status.set(f"{status} · {last_error}" if last_error else status)
            installed = int(
                data.get(
                    "catalog_installed", sum(1 for model in self._models if model.get("installed"))
                )
            )
            available = int(
                data.get(
                    "catalog_available",
                    sum(1 for model in self._models if not model.get("installed")),
                )
            )
            source = str(data.get("catalog_source") or "voice catalog")
            error = str(data.get("catalog_error") or "")
            self.catalog_status.set(
                f"{installed} installed · {available} available to download · {source}"
                + (f" · {error}" if error else "")
            )
            self._sync_default_install(data)
            self._render_models()
        finally:
            self._applying = False

    def save(self) -> None:
        try:
            payload: dict[str, Any] = {
                "enabled": self.enabled.get(),
                "output_device": self.device.get().strip(),
                "master_volume": float(self.volume.get()),
                "catalog_language": self.language.get(),
                "voice_presets": self._preset_payload(),
            }
        except ValueError:
            messagebox.showerror(
                PRODUCT_NAME,
                "TTS volume must be a valid number between 0 and 1.",
                parent=self.parent.winfo_toplevel(),
            )
            return
        self._request(
            "POST",
            "/api/tts/settings",
            payload,
            self._saved,
            failure=self._save_failed,
        )

    def _saved(self, data: dict[str, Any]) -> None:
        self._dirty = False
        self._non_preset_dirty = False
        self._preset_dirty = False
        self._preset_save_in_flight = False
        self.apply_status(data)
        self.status.set("TTS settings saved. " + self.status.get())

    def _save_failed(self, error: Exception) -> None:
        self.status.set(f"TTS settings failed: {error}")
