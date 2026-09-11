"""Native editor for the retrieval planner's player-editable word dictionaries."""

from __future__ import annotations

import copy
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Any

from pbrainz.retrieval_dictionary import default_retrieval_dictionary, load_retrieval_dictionary


class DictionaryTab:
    """Edit retrieval cues, routing stop-words, and token expansions by locale."""

    _CATEGORIES = (
        (
            "historical",
            "Historical cues",
            "Questions or phrases that should search prior memories and dialogue.",
        ),
        (
            "first_meeting",
            "First-meeting cues",
            "Phrases that retrieve the saved first-meeting primitive for this NPC.",
        ),
        (
            "tool",
            "Action/tool cues",
            "Words that make relevant native gameplay tools available to the model.",
        ),
        (
            "stop_words",
            "Routing stop-words",
            "Common words ignored when matching a request to a tool description.",
        ),
        (
            "token_expansions",
            "Token expansions",
            "One rule per line: source = related word, another related word. "
            "For localization, map a local cue to a game term, such as "
            "sígueme = follow.",
        ),
    )

    def __init__(self, parent: tk.Misc, request: Callable[..., None]) -> None:
        self._request = request
        self._loading = False
        self._dirty = False
        self._dictionary = default_retrieval_dictionary().as_dict()
        self._locales: dict[str, dict[str, Any]] = {}
        self._loaded_locale = ""
        self._locale = tk.StringVar(parent)
        self._status = tk.StringVar(parent, value="Loading shipped English defaults…")
        self._areas: dict[str, tk.Text] = {}
        self._build(parent)
        self.apply_status({"retrieval_dictionary": self._dictionary})

    def _build(self, parent: tk.Misc) -> None:
        outer = ttk.Frame(parent, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        ttk.Label(
            outer,
            text=(
                "Customize the words PBrainZ uses to decide when to retrieve memories "
                "or expose tools. Add a locale for your language, select it as active, "
                "then save. Matching is case-insensitive and supports Unicode phrases."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Active locale").grid(row=0, column=0, sticky="w")
        self._locale_box = ttk.Combobox(
            toolbar,
            textvariable=self._locale,
            state="readonly",
            width=24,
        )
        self._locale_box.grid(row=0, column=1, sticky="w", padx=(8, 8))
        self._locale_box.bind("<<ComboboxSelected>>", self._locale_changed)
        ttk.Button(toolbar, text="New locale", command=self._new_locale).grid(
            row=0, column=2, padx=(0, 5)
        )
        ttk.Button(toolbar, text="Delete locale", command=self._delete_locale).grid(
            row=0, column=3, padx=(0, 5)
        )
        ttk.Button(toolbar, text="Copy English defaults", command=self._reset_locale).grid(
            row=0, column=4
        )

        pages = ttk.Notebook(outer)
        pages.grid(row=2, column=0, sticky="nsew")
        for key, title, description in self._CATEGORIES:
            page = ttk.Frame(pages, padding=8)
            page.columnconfigure(0, weight=1)
            page.rowconfigure(1, weight=1)
            pages.add(page, text=title)
            ttk.Label(page, text=description, wraplength=700, justify="left").grid(
                row=0, column=0, sticky="ew", pady=(0, 6)
            )
            area = scrolledtext.ScrolledText(
                page,
                height=12,
                wrap="word",
                undo=True,
                font=("TkFixedFont", 9),
            )
            area.grid(row=1, column=0, sticky="nsew")
            area.bind("<KeyRelease>", self._mark_dirty)
            self._areas[key] = area

        footer = ttk.Frame(outer)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Save dictionary", command=self._save).pack(side="left")
        ttk.Label(footer, textvariable=self._status).pack(side="left", padx=(12, 0))

    @property
    def text_widgets(self) -> tuple[tk.Text, ...]:
        """Expose dictionary editors to the shared light/dark theme."""

        return tuple(self._areas.values())

    def _mark_dirty(self, _event: object | None = None) -> None:
        if not self._loading:
            self._dirty = True
            self._status.set("Unsaved dictionary changes")

    def apply_status(self, data: dict[str, Any], preserve_unsaved: bool = False) -> None:
        """Hydrate the editor from API status without overwriting local edits."""

        del preserve_unsaved  # Dictionary edits have their own independent dirty flag.
        if self._dirty:
            return
        raw = data.get("retrieval_dictionary")
        if not isinstance(raw, dict):
            return
        normalized = load_retrieval_dictionary(raw).as_dict()
        locales = normalized.get("locales")
        if not isinstance(locales, dict) or not locales:
            return
        self._dictionary = normalized
        self._locales = {
            str(name): copy.deepcopy(value)
            for name, value in locales.items()
            if isinstance(value, dict)
        }
        if not self._locales:
            return
        active = str(normalized.get("active_locale") or "")
        if active not in self._locales:
            active = next(iter(self._locales))
        self._loading = True
        try:
            self._locale_box.configure(values=tuple(self._locales))
            self._locale.set(active)
            self._load_locale(active)
            self._dirty = False
            self._status.set(f"Active locale: {active}")
        finally:
            self._loading = False

    def _locale_changed(self, _event: object | None = None) -> None:
        if self._loading:
            return
        self._remember_loaded_locale()
        selected = self._locale.get().strip()
        if selected not in self._locales:
            return
        self._load_locale(selected)
        self._dirty = True
        self._status.set(f"Unsaved changes — active locale: {selected}")

    def _load_locale(self, name: str) -> None:
        locale = self._locales.get(name, {})
        self._loading = True
        try:
            for key, area in self._areas.items():
                area.delete("1.0", "end")
                if key == "token_expansions":
                    expansions = locale.get(key, {})
                    lines = []
                    if isinstance(expansions, dict):
                        lines = [
                            f"{source} = {', '.join(str(value) for value in values)}"
                            for source, values in expansions.items()
                            if isinstance(values, (list, tuple))
                        ]
                else:
                    values = locale.get(key, [])
                    lines = [str(value) for value in values] if isinstance(values, list) else []
                area.insert("1.0", "\n".join(lines))
            self._loaded_locale = name
        finally:
            self._loading = False

    def _remember_loaded_locale(self) -> None:
        if self._loaded_locale and self._loaded_locale in self._locales:
            self._locales[self._loaded_locale] = self._collect_locale()

    def _collect_locale(self) -> dict[str, Any]:
        locale: dict[str, Any] = {}
        for key, area in self._areas.items():
            lines = [
                line.strip()
                for line in area.get("1.0", "end").splitlines()
                if line.strip()
            ]
            if key != "token_expansions":
                locale[key] = lines
                continue
            expansions: dict[str, list[str]] = {}
            for line in lines:
                if "=" not in line:
                    raise ValueError(
                        "Token expansions must use 'source = related, word' format."
                    )
                source, raw_values = line.split("=", 1)
                source = source.strip()
                values = [value.strip() for value in raw_values.split(",") if value.strip()]
                if not source or not values:
                    raise ValueError(
                        "Every token expansion needs a source and at least one related word."
                    )
                expansions[source] = values
            locale[key] = expansions
        return locale

    def _collect_dictionary(self) -> dict[str, Any]:
        self._remember_loaded_locale()
        return {
            "version": 1,
            "active_locale": self._locale.get().strip() or "en",
            "locales": copy.deepcopy(self._locales),
        }

    def _new_locale(self) -> None:
        name = simpledialog.askstring(
            "New locale",
            "Locale name (for example: es, de, or ja-JP):",
            parent=self._locale_box.winfo_toplevel(),
        )
        name = " ".join(str(name or "").split())[:32]
        if not name:
            return
        if any(existing.casefold() == name.casefold() for existing in self._locales):
            messagebox.showerror("Locale exists", f"The locale '{name}' already exists.")
            return
        self._remember_loaded_locale()
        source = self._locales.get(self._loaded_locale, {})
        self._locales[name] = copy.deepcopy(source)
        self._dictionary["locales"] = self._locales
        self._loading = True
        try:
            self._locale_box.configure(values=tuple(self._locales))
            self._locale.set(name)
            self._load_locale(name)
        finally:
            self._loading = False
        self._dirty = True
        self._status.set(f"Unsaved changes — new locale: {name}")

    def _delete_locale(self) -> None:
        selected = self._locale.get().strip()
        if len(self._locales) <= 1:
            messagebox.showinfo("Locale required", "Keep at least one locale configured.")
            return
        if not messagebox.askyesno("Delete locale", f"Delete the '{selected}' locale?"):
            return
        self._locales.pop(selected, None)
        selected = next(iter(self._locales))
        self._loading = True
        try:
            self._locale_box.configure(values=tuple(self._locales))
            self._locale.set(selected)
            self._load_locale(selected)
        finally:
            self._loading = False
        self._dirty = True
        self._status.set(f"Unsaved changes — active locale: {selected}")

    def _reset_locale(self) -> None:
        selected = self._locale.get().strip()
        if not messagebox.askyesno(
            "Copy English defaults",
            f"Replace all words in '{selected}' with the shipped English defaults?",
        ):
            return
        default_locale = default_retrieval_dictionary().as_dict()["locales"]["en"]
        self._locales[selected] = copy.deepcopy(default_locale)
        self._load_locale(selected)
        self._dirty = True
        self._status.set(f"Unsaved changes — copied English defaults to {selected}")

    def _save(self) -> None:
        try:
            payload = self._collect_dictionary()
            normalized = load_retrieval_dictionary(payload).as_dict()
        except ValueError as error:
            self._status.set(f"Dictionary error: {error}")
            return
        self._status.set("Saving dictionary…")
        self._request(
            "POST",
            "/api/retrieval-dictionary",
            {"dictionary": normalized},
            self._save_succeeded,
            failure=self._save_failed,
        )

    def _save_succeeded(self, data: dict[str, Any]) -> None:
        self._dirty = False
        self.apply_status(data)
        self._status.set("Dictionary saved and active for new requests.")

    def _save_failed(self, error: Exception) -> None:
        self._status.set(f"Dictionary not saved: {error}")
