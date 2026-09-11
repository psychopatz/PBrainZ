"""Local speech-output controls for the native TTS tab."""

from __future__ import annotations

from tkinter import ttk

TEST_EFFECT_OPTIONS = (
    ("None", "none", 1.0),
    ("Walkie-talkie / radio", "radio", 0.85),
    ("Telephone", "telephone", 1.0),
    ("Muffled", "muffled", 1.0),
    ("Underwater", "underwater", 1.0),
)


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
        ttk.Label(controls, text="Test effect").grid(row=3, column=0, sticky="w", pady=2)
        self._test_effect_box = ttk.Combobox(
            controls,
            textvariable=self.test_effect,
            values=tuple(label for label, _profile, _intensity in TEST_EFFECT_OPTIONS),
            state="readonly",
            width=25,
        )
        self._test_effect_box.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(
            controls,
            text="Used by voice test buttons only.",
        ).grid(row=3, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=2)
        ttk.Label(controls, text="Radio static volume").grid(
            row=4, column=0, sticky="w", pady=2
        )
        ttk.Scale(
            controls,
            from_=0,
            to=2,
            variable=self._ambient_scale_var,
            command=lambda value: self.ambient_volume.set(f"{float(value):.2f}"),
        ).grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Label(controls, textvariable=self.ambient_volume).grid(
            row=4, column=2, sticky="w", padx=(12, 0), pady=2
        )
        ttk.Label(controls, textvariable=self.status, wraplength=700).grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )

    def _test_audio_presentation_payload(self) -> dict[str, object]:
        selected = self.test_effect.get()
        for label, profile, intensity in TEST_EFFECT_OPTIONS:
            if label == selected:
                return {
                    "effect_profile": profile,
                    "environment": "normal",
                    "effect_intensity": intensity,
                    "ambient_volume": float(self.ambient_volume.get()),
                }
        return {
            "effect_profile": "none",
            "environment": "normal",
            "effect_intensity": 1.0,
            "ambient_volume": float(self.ambient_volume.get()),
        }

    def _mark_dirty(self, *_args: object) -> None:
        if not self._applying:
            self._dirty = True
            self._non_preset_dirty = True
