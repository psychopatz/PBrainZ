"""Local speech-output controls for the native TTS tab."""

from __future__ import annotations

from tkinter import ttk


class OutputViewMixin:
    def _build_output(self, outer: ttk.Frame) -> None:
        controls = ttk.LabelFrame(outer, text="Local speech output", padding=9)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        ttk.Checkbutton(controls, text="Enable TTS", variable=self.enabled).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(controls, text="Backend: Piper").grid(
            row=0, column=2, columnspan=2, sticky="e", padx=(12, 0)
        )
        ttk.Label(controls, text="Output device").grid(row=1, column=0, sticky="w", pady=2)
        self._device_box = ttk.Combobox(controls, textvariable=self.device, width=25)
        self._device_box.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(controls, text="Master volume").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self._volume_scale_var,
            command=lambda value: self.volume.set(f"{float(value):.2f}"),
        ).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(controls, textvariable=self.status, wraplength=700).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )

    def _mark_dirty(self, *_args: object) -> None:
        if not self._applying:
            self._dirty = True
            self._non_preset_dirty = True
