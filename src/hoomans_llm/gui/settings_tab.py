"""Runtime settings tab for the native control panel."""

from __future__ import annotations

from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

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
        self._build(parent, refresh)

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
        ttk.Label(runtime, text="Control-panel theme").grid(
            row=2, column=0, sticky="w", pady=(16, 4)
        )
        self._theme_box = ttk.Combobox(
            runtime,
            textvariable=self.state.theme,
            values=("light", "dark"),
            state="readonly",
            width=16,
        )
        self._theme_box.grid(row=2, column=1, sticky="ew", pady=(16, 4))
        self._theme_box.bind("<<ComboboxSelected>>", self._on_theme_changed)
        actions = ttk.Frame(runtime)
        actions.grid(row=3, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(actions, text="Refresh", command=refresh).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Save settings", command=self._save).pack(side="left")

    def _on_theme_changed(self, _event: object | None = None) -> None:
        self._theme_changed_callback(self.state.theme.get())

    def _save(self) -> None:
        try:
            timeout = float(self.state.timeout.get())
            poll_interval = float(self.state.poll_interval.get())
        except ValueError:
            messagebox.showerror(
                "HoomansLLM",
                "Timeout and poll interval must be numbers.",
                parent=self.parent.winfo_toplevel(),
            )
            return
        self._save_callback(
            {
                "request_timeout": timeout,
                "bridge_poll_interval": poll_interval,
                "ui_theme": self.state.theme.get(),
            }
        )

    def save(self) -> None:
        """Save runtime settings; kept public for the panel compatibility API."""

        self._save()
