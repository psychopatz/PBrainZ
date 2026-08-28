"""Runtime settings tab for the native control panel."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from pbrainz.branding import PRODUCT_NAME
from pbrainz.paths import default_zomboid_path

from .state import PanelState

SaveFn = Callable[[dict[str, Any]], None]
RefreshFn = Callable[[], None]
ThemeFn = Callable[[str], None]


class SettingsTab:
    """Own runtime and appearance controls."""

    def __init__(
        self,
        parent: ttk.Frame,
        state: PanelState,
        save: SaveFn,
        refresh: RefreshFn,
        theme_changed: ThemeFn,
    ) -> None:
        self.parent = parent
        self.state = state
        self._save_callback = save
        self._theme_changed_callback = theme_changed
        self._tts_applying = False
        self._tts_dirty = False
        self.tts_workers = tk.StringVar(parent, value="1")
        self.tts_cache_size = tk.StringVar(parent, value="2")
        self.tts_playback_voices = tk.StringVar(parent, value="4")
        self.tts_generated_ahead = tk.StringVar(parent, value="3")
        self.tts_ready_ahead = tk.StringVar(parent, value="1")
        self.tts_gap_ms = tk.StringVar(parent, value="180")
        self.tts_synthesis_timeout = tk.StringVar(parent, value="45")
        self.tts_buffer_ms = tk.StringVar(parent, value="50")
        self._build(parent, refresh)
        for variable in self._tts_variables():
            variable.trace_add("write", self._mark_tts_dirty)

    def _build(self, parent: ttk.Frame, refresh: RefreshFn) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        outer = ttk.Frame(parent, padding=18)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        runtime = ttk.LabelFrame(outer, text="Runtime settings", padding=12)
        runtime.grid(row=0, column=0, sticky="new")
        runtime.columnconfigure(1, weight=1)
        ttk.Label(runtime, text="Request timeout (seconds)").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(runtime, textvariable=self.state.timeout, width=16).grid(
            row=0, column=1, sticky="ew", pady=4
        )
        ttk.Label(runtime, text="Bridge poll interval (seconds)").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(runtime, textvariable=self.state.poll_interval, width=16).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Label(runtime, text="Project Zomboid data directory").grid(
            row=2, column=0, sticky="w", pady=(16, 4)
        )
        path_frame = ttk.Frame(runtime)
        path_frame.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(16, 4))
        path_frame.columnconfigure(0, weight=1)
        ttk.Entry(path_frame, textvariable=self.state.zomboid_path).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(path_frame, text="Browse…", command=self._browse_zomboid_path).grid(
            row=0, column=1, padx=(8, 0)
        )
        ttk.Button(path_frame, text="Auto-detect", command=self._auto_detect_zomboid_path).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Label(
            runtime,
            text="Folder containing console.txt",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(runtime, text="Control-panel theme").grid(
            row=4, column=0, sticky="w", pady=(16, 4)
        )
        self._theme_box = ttk.Combobox(
            runtime,
            textvariable=self.state.theme,
            values=("light", "dark"),
            state="readonly",
            width=16,
        )
        self._theme_box.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(16, 4))
        self._theme_box.bind("<<ComboboxSelected>>", self._on_theme_changed)
        actions = ttk.Frame(runtime)
        actions.grid(row=5, column=0, columnspan=3, sticky="e", pady=(16, 0))
        ttk.Button(actions, text="Refresh", command=refresh).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Save settings", command=self._save).pack(side="left")

        performance = ttk.LabelFrame(outer, text="TTS performance", padding=12)
        performance.grid(row=1, column=0, sticky="new", pady=(12, 0))
        performance.columnconfigure(1, weight=1)
        performance.columnconfigure(3, weight=1)
        performance.columnconfigure(5, weight=1)
        entries = (
            ("Synthesis workers", self.tts_workers),
            ("Loaded-model cache", self.tts_cache_size),
            ("Playback voices", self.tts_playback_voices),
            ("Generated ahead", self.tts_generated_ahead),
            ("TTS-ready ahead", self.tts_ready_ahead),
            ("Natural gap (ms)", self.tts_gap_ms),
            ("Synthesis timeout (s)", self.tts_synthesis_timeout),
            ("Audio buffer (ms)", self.tts_buffer_ms),
        )
        for index, (label, variable) in enumerate(entries):
            column = (index % 3) * 2
            row = index // 3
            ttk.Label(performance, text=label).grid(
                row=row, column=column, sticky="w", padx=(0, 6), pady=3
            )
            ttk.Entry(performance, textvariable=variable, width=8).grid(
                row=row, column=column + 1, sticky="w", padx=(0, 18), pady=3
            )

    def _on_theme_changed(self, _event: object | None = None) -> None:
        self._theme_changed_callback(self.state.theme.get())

    def _browse_zomboid_path(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.parent.winfo_toplevel(),
            initialdir=self.state.zomboid_path.get() or str(Path.home()),
            title="Choose Project Zomboid data directory",
        )
        if selected:
            self.state.zomboid_path.set(selected)

    def _auto_detect_zomboid_path(self) -> None:
        self.state.zomboid_path.set(str(default_zomboid_path()))

    def _tts_variables(self) -> tuple[tk.StringVar, ...]:
        return (
            self.tts_workers,
            self.tts_cache_size,
            self.tts_playback_voices,
            self.tts_generated_ahead,
            self.tts_ready_ahead,
            self.tts_gap_ms,
            self.tts_synthesis_timeout,
            self.tts_buffer_ms,
        )

    def _mark_tts_dirty(self, *_args: object) -> None:
        if not self._tts_applying:
            self._tts_dirty = True
            self.state.settings_dirty = True

    def apply_status(self, data: dict[str, Any], preserve_unsaved: bool) -> None:
        if preserve_unsaved and self._tts_dirty:
            return
        self._tts_applying = True
        try:
            self.tts_workers.set(str(data.get("tts_synthesis_workers", 1)))
            self.tts_cache_size.set(str(data.get("tts_model_cache_size", 2)))
            self.tts_playback_voices.set(str(data.get("tts_max_simultaneous_playback", 4)))
            self.tts_generated_ahead.set(str(data.get("tts_max_generated_ahead", 3)))
            self.tts_ready_ahead.set(str(data.get("tts_max_tts_ready_ahead", 1)))
            self.tts_gap_ms.set(str(data.get("tts_natural_gap_ms", 180)))
            self.tts_synthesis_timeout.set(str(data.get("tts_synthesis_timeout", 45)))
            self.tts_buffer_ms.set(str(data.get("tts_audio_buffer_ms", 50)))
        finally:
            self._tts_applying = False

    def mark_saved(self) -> None:
        self._tts_dirty = False

    def _save(self) -> None:
        try:
            timeout = float(self.state.timeout.get())
            poll_interval = float(self.state.poll_interval.get())
            tts_values = {
                "tts_synthesis_workers": int(self.tts_workers.get()),
                "tts_model_cache_size": int(self.tts_cache_size.get()),
                "tts_max_simultaneous_playback": int(self.tts_playback_voices.get()),
                "tts_max_generated_ahead": int(self.tts_generated_ahead.get()),
                "tts_max_tts_ready_ahead": int(self.tts_ready_ahead.get()),
                "tts_natural_gap_ms": int(self.tts_gap_ms.get()),
                "tts_synthesis_timeout": float(self.tts_synthesis_timeout.get()),
                "tts_audio_buffer_ms": int(self.tts_buffer_ms.get()),
            }
        except ValueError:
            messagebox.showerror(
                PRODUCT_NAME,
                "Runtime and TTS performance settings must be valid numbers.",
                parent=self.parent.winfo_toplevel(),
            )
            return
        self._save_callback(
            {
                "request_timeout": timeout,
                "bridge_poll_interval": poll_interval,
                "zomboid_path": self.state.zomboid_path.get().strip() or None,
                "ui_theme": self.state.theme.get(),
                **tts_values,
            }
        )

    def save(self) -> None:
        """Save runtime settings for the panel shell."""

        self._save()
