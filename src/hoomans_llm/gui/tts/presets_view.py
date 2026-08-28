"""Project Hoomans voice-preset controls for the native TTS tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


class PresetsViewMixin:
    def _build_presets(self, outer: ttk.Frame, parent: tk.Misc) -> None:
        presets = ttk.LabelFrame(outer, text="Project Hoomans voice presets", padding=9)
        presets.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        presets.columnconfigure(1, weight=1)
        presets.columnconfigure(4, weight=1)
        for column, prefix, title, expected_gender in (
            (0, "VoiceFemale", "Female voices", "female"),
            (3, "VoiceMale", "Male voices", "male"),
        ):
            header = ttk.Frame(presets)
            header.grid(
                row=0,
                column=column,
                columnspan=3,
                sticky="w",
                padx=(18, 0) if column else 0,
            )
            ttk.Label(header, text=title).pack(side="left")
            gender_filter = tk.BooleanVar(parent, value=True)
            self._preset_gender_filters[prefix] = gender_filter
            ttk.Checkbutton(
                header,
                text=f"{expected_gender} only",
                variable=gender_filter,
                command=self._preset_gender_filter_changed,
            ).pack(side="left", padx=(10, 0))
        for row in range(4):
            for column, prefix in ((0, "VoiceFemale"), (3, "VoiceMale")):
                slot = f"{prefix}:{row}"
                variable = tk.StringVar(parent)
                self._preset_vars[slot] = variable
                ttk.Label(presets, text=slot, width=14).grid(
                    row=row + 1,
                    column=column,
                    sticky="w",
                    pady=2,
                    padx=(0, 4 if column == 0 else 0),
                )
                box = ttk.Combobox(presets, textvariable=variable, state="readonly", width=24)
                box.grid(row=row + 1, column=column + 1, sticky="ew", padx=(0, 5), pady=2)
                box.bind(
                    "<<ComboboxSelected>>",
                    lambda _event, selected=slot: self._preset_changed(selected),
                )
                self._preset_boxes[slot] = box
                ttk.Button(
                    presets,
                    text="Test",
                    width=7,
                    command=lambda selected=slot: self._test(selected),
                ).grid(row=row + 1, column=column + 2, sticky="e", pady=2)

    def _preset_gender_filter_changed(self) -> None:
        self._update_preset_options()

    def _update_preset_options(self) -> None:
        """Show installed models and annotate every option with its gender."""

        installed_models = [model for model in self._models if model.get("installed")]
        for slot, box in self._preset_boxes.items():
            prefix = slot.split(":", 1)[0]
            expected_gender = "female" if prefix == "VoiceFemale" else "male"
            gender_only = self._preset_gender_filters[prefix].get()
            options: list[str] = []
            option_ids: dict[str, str] = {}
            for model in sorted(installed_models, key=self._model_sort_key):
                model_id = str(model.get("id") or "")
                model_gender = str(model.get("gender") or "unknown").casefold()
                if not model_id or (gender_only and model_gender != expected_gender):
                    continue
                option = self._preset_option_label(model)
                options.append(option)
                option_ids[option] = model_id
            box["values"] = options
            self._preset_option_ids[slot] = option_ids
            if not self._dirty:
                model_id = self._preset_selected_ids.get(slot, "")
                model = self._model_by_id.get(model_id)
                box.set(
                    self._preset_option_label(model) if model and model.get("installed") else ""
                )

    @staticmethod
    def _preset_option_label(model: dict[str, Any] | None) -> str:
        if not model:
            return ""
        model_id = str(model.get("id") or "")
        gender = str(model.get("gender") or "unknown")
        return f"{model_id} ({gender})"

    def _selected_preset_model_id(self, slot: str) -> str:
        selected = self._preset_vars[slot].get()
        return self._preset_option_ids.get(slot, {}).get(
            selected, self._preset_selected_ids.get(slot, selected)
        )

    def _preset_changed(self, slot: str) -> None:
        if self._applying:
            return
        model_id = self._preset_option_ids.get(slot, {}).get(self._preset_vars[slot].get(), "")
        if not model_id:
            return
        self._preset_selected_ids[slot] = model_id
        self._preset_dirty = True
        self._dirty = True
        self._save_preset_changes()

    def _save_preset_changes(self) -> None:
        if not self._preset_dirty or self._preset_save_in_flight:
            return
        self._preset_save_in_flight = True
        self._request(
            "POST",
            "/api/tts/settings",
            {"voice_presets": self._preset_payload()},
            self._presets_saved,
            failure=self._presets_save_failed,
        )

    def _preset_payload(self) -> list[dict[str, str]]:
        return [
            {"slot": slot, "voice_model_id": self._selected_preset_model_id(slot)}
            for slot in self._SLOTS
            if self._selected_preset_model_id(slot)
        ]

    def _presets_saved(self, _data: dict[str, Any]) -> None:
        self._preset_save_in_flight = False
        self._preset_dirty = False
        if not self._non_preset_dirty:
            self._dirty = False
        self.status.set("Voice presets saved automatically.")

    def _presets_save_failed(self, error: Exception) -> None:
        self._preset_save_in_flight = False
        self.status.set(f"Voice presets failed to save: {error}")

    def _test(self, slot: str) -> None:
        model_id = self._selected_preset_model_id(slot)
        if not model_id:
            self.status.set(f"Configure an installed Piper model for {slot} first.")
            return
        model = self._model_by_id.get(model_id)
        if model is not None and not model.get("installed"):
            self.status.set(
                f"Install {model.get('display_name') or model.get('id')} before testing it."
            )
            return
        self._request(
            "POST",
            "/api/tts/test",
            {"slot": slot, "text": f"This is the HoomansLLM {slot} voice test."},
            lambda _data: self.status.set(f"Playing {slot} voice test."),
            failure=lambda error: self.status.set(f"Voice test failed: {error}"),
        )

