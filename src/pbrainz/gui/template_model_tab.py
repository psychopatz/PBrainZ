"""Native editor for reusable NPC conversation-template profiles."""

from __future__ import annotations

import copy
import tkinter as tk
import uuid
from collections.abc import Callable
from tkinter import scrolledtext, ttk
from typing import Any

from pbrainz.template_profiles import (
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_TEMPLATE_PROFILE_ID,
    TEMPLATE_PLACEHOLDERS,
    normalize_profile,
    render_template,
)

SaveProfileFn = Callable[[dict[str, Any]], None]
ProfileActionFn = Callable[[str], None]

# Compatibility constants for integrations that imported the original tab.
# Provider selection itself now belongs exclusively to Control Panel.
TEMPLATE_MODEL_PROVIDERS = ("gemini", "horde")
DEFAULT_TEMPLATE_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class TemplateModelTab:
    """Edit prompt templates without duplicating provider configuration."""

    def __init__(
        self,
        parent: ttk.Frame,
        save_profile: SaveProfileFn,
        activate_profile: ProfileActionFn,
        delete_profile: ProfileActionFn,
        reset_profile: ProfileActionFn,
    ) -> None:
        self._save_profile = save_profile
        self._activate_profile = activate_profile
        self._delete_profile = delete_profile
        self._reset_profile = reset_profile
        self._profiles: dict[str, dict[str, Any]] = {}
        self._selected_id = ""
        self._active_id = DEFAULT_TEMPLATE_PROFILE_ID
        self._follow_active = True
        self._loading = False
        self._dirty = False
        self._name = tk.StringVar(parent)
        self._description = tk.StringVar(parent)
        self._mode = tk.StringVar(parent, value="chat")
        self._status = tk.StringVar(parent, value="Choose a template profile.")
        self._mode_note = tk.StringVar(parent, value="")
        self._editor_status = tk.StringVar(parent, value="")
        self._user_prefix = tk.StringVar(parent, value="User: ")
        self._assistant_prefix = tk.StringVar(parent, value="Assistant: ")
        self._system_prompt: tk.Text
        self._context_template: tk.Text
        self._examples: tk.Text
        self._stop_sequences: tk.Text
        self._preview: tk.Text
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        outer = ttk.Frame(parent, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(
            outer,
            text=(
                "Manage the prompt template sent to NPC models. Provider credentials, "
                "model catalogs, and generation settings remain in Control panel. "
                "The default Native chat profile automatically uses Instruct text "
                "for Horde; activate a custom profile to override it. Changes apply "
                "to new NPC turns after activation."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        split = ttk.Panedwindow(outer, orient="horizontal")
        split.grid(row=1, column=0, sticky="nsew")
        library = ttk.Frame(split, padding=(0, 0, 10, 0))
        editor = ttk.Frame(split)
        split.add(library, weight=0)
        split.add(editor, weight=1)
        library.rowconfigure(1, weight=1)
        library.columnconfigure(0, weight=1)

        ttk.Label(library, text="Template profiles").grid(row=0, column=0, sticky="w")
        self._profile_list = tk.Listbox(
            library,
            width=28,
            height=18,
            exportselection=False,
            activestyle="none",
        )
        self._profile_list.grid(row=1, column=0, sticky="nsew", pady=(5, 8))
        self._profile_list.bind("<<ListboxSelect>>", self._profile_selected)
        library_actions = ttk.Frame(library)
        library_actions.grid(row=2, column=0, sticky="ew")
        for index in range(2):
            library_actions.columnconfigure(index, weight=1)
        ttk.Button(library_actions, text="New", command=self._new_profile).grid(
            row=0, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Button(library_actions, text="Duplicate", command=self._duplicate_profile).grid(
            row=0, column=1, sticky="ew", padx=(3, 0), pady=2
        )
        ttk.Button(library_actions, text="Delete", command=self._delete_selected).grid(
            row=1, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Button(library_actions, text="Reset", command=self._reset_selected).grid(
            row=1, column=1, sticky="ew", padx=(3, 0), pady=2
        )

        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)
        heading = ttk.Frame(editor)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        heading.columnconfigure(1, weight=1)
        ttk.Label(heading, text="Profile name").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(heading, textvariable=self._name).grid(row=0, column=1, sticky="ew")
        ttk.Label(heading, text="Description").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        ttk.Entry(heading, textvariable=self._description).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(heading, text="Format").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        mode_box = ttk.Combobox(
            heading,
            textvariable=self._mode,
            values=("chat", "instruct"),
            state="readonly",
            width=18,
        )
        mode_box.grid(row=2, column=1, sticky="w", pady=(6, 0))
        mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(heading, textvariable=self._mode_note, wraplength=520).grid(
            row=3, column=1, sticky="w", pady=(4, 0)
        )

        pages = ttk.Notebook(editor)
        pages.grid(row=1, column=0, sticky="nsew")
        prompt_page = ttk.Frame(pages, padding=8)
        preview_page = ttk.Frame(pages, padding=8)
        pages.add(prompt_page, text="Prompt fields")
        pages.add(preview_page, text="Preview")
        prompt_page.columnconfigure(0, weight=1)
        prompt_page.rowconfigure(3, weight=1)

        ttk.Label(
            prompt_page,
            text="Template instructions (appended to the code-owned NPC safety rules)",
        ).grid(row=0, column=0, sticky="w")
        self._system_prompt = self._text_area(prompt_page, row=1, height=4)
        ttk.Label(
            prompt_page,
            text=(
                "Context template (used in instruct mode). Tokens: "
                + ", ".join(f"{{{{{token}}}}}" for token in TEMPLATE_PLACEHOLDERS)
            ),
            wraplength=650,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._context_template = self._text_area(prompt_page, row=3, height=6)
        ttk.Label(prompt_page, text="Example dialogue (optional)").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._examples = self._text_area(prompt_page, row=5, height=3)

        advanced = ttk.LabelFrame(prompt_page, text="Turn formatting", padding=6)
        advanced.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        advanced.columnconfigure(1, weight=1)
        ttk.Label(advanced, text="User prefix").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(advanced, textvariable=self._user_prefix).grid(row=0, column=1, sticky="ew")
        ttk.Label(advanced, text="Assistant prefix").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(5, 0)
        )
        ttk.Entry(advanced, textvariable=self._assistant_prefix).grid(
            row=1, column=1, sticky="ew", pady=(5, 0)
        )
        ttk.Label(advanced, text="Stop sequences (one per line)").grid(
            row=2, column=0, sticky="nw", padx=(0, 8), pady=(5, 0)
        )
        self._stop_sequences = self._text_area(advanced, row=2, height=2, column=1)

        preview_page.columnconfigure(0, weight=1)
        preview_page.rowconfigure(0, weight=1)
        self._preview = scrolledtext.ScrolledText(
            preview_page,
            state="disabled",
            wrap="word",
            height=20,
            font=("TkFixedFont", 9),
        )
        self._preview.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(editor)
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Save profile", command=self._save).pack(side="left")
        ttk.Button(footer, text="Use this profile", command=self._activate).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(footer, textvariable=self._editor_status).pack(side="left", padx=(12, 0))
        ttk.Label(outer, textvariable=self._status).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )

        for variable in (
            self._name,
            self._description,
            self._mode,
            self._user_prefix,
            self._assistant_prefix,
        ):
            variable.trace_add("write", self._value_changed)
        for widget in (
            self._system_prompt,
            self._context_template,
            self._examples,
            self._stop_sequences,
        ):
            widget.bind("<KeyRelease>", self._mark_dirty)

    @staticmethod
    def _text_area(
        parent: tk.Misc, *, row: int, height: int, column: int = 0
    ) -> tk.Text:
        area = scrolledtext.ScrolledText(parent, height=height, wrap="word", undo=True)
        area.grid(row=row, column=column, sticky="nsew")
        return area

    @property
    def text_widgets(self) -> tuple[tk.Text, ...]:
        """Expose editor widgets to the shared light/dark theme."""

        return (
            self._system_prompt,
            self._context_template,
            self._examples,
            self._stop_sequences,
            self._preview,
        )

    def apply_status(self, data: dict[str, Any], preserve_unsaved: bool = False) -> None:
        """Hydrate profiles from the status payload without losing active edits."""

        rows = data.get("template_profiles") or []
        incoming = {
            str(row.get("id")): copy.deepcopy(row)
            for row in rows
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        if not incoming:
            return
        local_unsaved = (
            copy.deepcopy(self._profiles.get(self._selected_id))
            if self._dirty and self._selected_id
            else None
        )
        self._profiles = incoming
        if local_unsaved and self._selected_id not in self._profiles:
            self._profiles[self._selected_id] = local_unsaved
        self._active_id = str(data.get("active_template_profile_id") or "")
        self._refresh_profile_list()
        if self._follow_active or self._selected_id not in self._profiles:
            selected = (
                self._active_id
                if self._active_id in self._profiles
                else next(iter(self._profiles))
            )
            self._select_profile(selected, load=True)
        elif self._dirty or preserve_unsaved:
            self._update_preview()
        else:
            self._load_profile(self._profiles[self._selected_id])

    def set_action_status(self, value: str) -> None:
        self._editor_status.set(value)

    def mark_server_synced(self) -> None:
        """Allow a successful save/reset response to replace local edits."""

        self._dirty = False

    def sync_active_on_next_status(self) -> None:
        """Make the next server response select the newly active profile."""

        self._follow_active = True

    def _refresh_profile_list(self) -> None:
        selected_index = (
            list(self._profiles).index(self._selected_id)
            if self._selected_id in self._profiles
            else -1
        )
        self._profile_list.delete(0, "end")
        for profile in self._profiles.values():
            marker = "● " if profile.get("id") == self._active_id else "  "
            mode = "instruct" if profile.get("mode") == "instruct" else "chat"
            builtin = " · built-in" if profile.get("builtin") else ""
            self._profile_list.insert(
                "end",
                f"{marker}{profile.get('name', 'Unnamed')} [{mode}]{builtin}",
            )
        if 0 <= selected_index < self._profile_list.size():
            self._profile_list.selection_set(selected_index)
            self._profile_list.see(selected_index)

    def _profile_selected(self, _event: object | None = None) -> None:
        selection = self._profile_list.curselection()
        if not selection:
            return
        profile_id = list(self._profiles)[selection[0]]
        self._follow_active = False
        self._select_profile(profile_id, load=True)

    def _select_profile(self, profile_id: str, *, load: bool) -> None:
        if profile_id not in self._profiles:
            return
        self._selected_id = profile_id
        self._refresh_profile_list()
        if load:
            self._load_profile(self._profiles[profile_id])

    def _load_profile(self, profile: dict[str, Any]) -> None:
        self._loading = True
        try:
            self._name.set(str(profile.get("name") or ""))
            self._description.set(str(profile.get("description") or ""))
            self._mode.set(str(profile.get("mode") or "chat"))
            self._set_text(self._system_prompt, str(profile.get("system_prompt") or ""))
            self._set_text(
                self._context_template,
                str(profile.get("context_template") or DEFAULT_CONTEXT_TEMPLATE),
            )
            self._set_text(self._examples, str(profile.get("examples") or ""))
            self._user_prefix.set(str(profile.get("user_prefix") or "User: "))
            self._assistant_prefix.set(str(profile.get("assistant_prefix") or "Assistant: "))
            self._set_text(
                self._stop_sequences,
                "\n".join(str(item) for item in profile.get("stop_sequences") or []),
            )
            self._editor_status.set(
                "Built-in profile" if profile.get("builtin") else "Custom profile"
            )
            self._dirty = False
        finally:
            self._loading = False
        self._update_mode_note()
        self._update_preview()

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.edit_modified(False)

    def _value_changed(self, *_args: object) -> None:
        self._mark_dirty()

    def _mark_dirty(self, _event: object | None = None) -> None:
        if self._loading:
            return
        self._dirty = True
        self._editor_status.set("Unsaved changes")
        self._update_preview()

    def _mode_changed(self, _event: object | None = None) -> None:
        self._mark_dirty()
        self._update_mode_note()

    def _update_mode_note(self) -> None:
        if self._mode.get() == "instruct":
            self._mode_note.set("One rendered prompt for text-template endpoints such as Horde.")
        else:
            self._mode_note.set("Native chat messages for Gemini and chat-tuned providers.")

    def _profile_payload(self) -> dict[str, Any]:
        return {
            "id": self._selected_id,
            "name": self._name.get().strip(),
            "description": self._description.get().strip(),
            "mode": self._mode.get(),
            "system_prompt": self._system_prompt.get("1.0", "end-1c"),
            "context_template": self._context_template.get("1.0", "end-1c"),
            "examples": self._examples.get("1.0", "end-1c"),
            "user_prefix": self._user_prefix.get(),
            "assistant_prefix": self._assistant_prefix.get(),
            "stop_sequences": [
                line
                for line in self._stop_sequences.get("1.0", "end-1c").splitlines()
                if line.strip()
            ],
            "builtin": bool(self._profiles.get(self._selected_id, {}).get("builtin")),
        }

    def _save(self) -> None:
        payload = self._profile_payload()
        if not payload["name"]:
            self._editor_status.set("Enter a profile name first.")
            return
        self._editor_status.set("Saving profile…")
        self._save_profile(payload)

    def _activate(self) -> None:
        if not self._selected_id:
            return
        if self._dirty:
            self._editor_status.set("Save the profile before activating it.")
            return
        self._editor_status.set("Activating profile…")
        self._activate_profile(self._selected_id)

    def _new_profile(self) -> None:
        profile_id = f"custom-{uuid.uuid4().hex[:8]}"
        profile = normalize_profile(
            {
                "id": profile_id,
                "name": "New template",
                "mode": "instruct",
                "context_template": DEFAULT_CONTEXT_TEMPLATE,
            }
        ).as_dict()
        self._profiles[profile_id] = profile
        self._follow_active = False
        self._select_profile(profile_id, load=True)
        self._mark_dirty()

    def _duplicate_profile(self) -> None:
        if not self._selected_id:
            return
        source = copy.deepcopy(
            self._profile_payload() if self._dirty else self._profiles[self._selected_id]
        )
        profile_id = f"custom-{uuid.uuid4().hex[:8]}"
        source.update(
            {
                "id": profile_id,
                "name": f"{source.get('name', 'Template')} copy",
                "builtin": False,
            }
        )
        self._profiles[profile_id] = normalize_profile(source).as_dict()
        self._follow_active = False
        self._select_profile(profile_id, load=True)
        self._mark_dirty()

    def _delete_selected(self) -> None:
        profile = self._profiles.get(self._selected_id)
        if not profile:
            return
        if profile.get("builtin"):
            self._editor_status.set(
                "Built-in profiles cannot be deleted; use Duplicate first."
            )
            return
        self._editor_status.set("Deleting profile…")
        self._delete_profile(self._selected_id)

    def _reset_selected(self) -> None:
        profile = self._profiles.get(self._selected_id)
        if not profile:
            return
        if not profile.get("builtin"):
            self._editor_status.set(
                "Reset is available for built-in profiles; Duplicate for a custom starting point."
            )
            return
        self._editor_status.set("Resetting profile…")
        self._reset_profile(self._selected_id)

    def _update_preview(self) -> None:
        if not hasattr(self, "_preview") or not self._selected_id:
            return
        try:
            profile = normalize_profile(self._profile_payload())
        except (TypeError, ValueError):
            return
        values = {
            "system": "[Core NPC rules are always supplied by PBrainZ]\n"
            + profile.system_prompt,
            "history": "User: We need to move before dark.\nAssistant: Then we should leave now.",
            "user": "Where is the shelter?",
            "assistant": "",
            "examples": profile.examples,
            "character": "name: Emilio\npersonality: guarded, practical",
            "char": "Emilio",
            "user_name": "Player",
            "user_prefix": profile.user_prefix,
            "assistant_prefix": profile.assistant_prefix,
        }
        if profile.mode == "instruct":
            body = render_template(profile.context_template, values)
        else:
            body = (
                "Native chat messages\n\n"
                f"SYSTEM\n{values['system']}\n\n"
                f"EXAMPLES\n{profile.examples or '[none]'}\n\n"
                "USER\nWhere is the shelter?"
            )
        preview = (
            f"Profile: {profile.name}\n"
            f"Mode: {profile.mode}\n"
            f"Stop sequences: {', '.join(profile.stop_sequences) or '[none]'}\n\n"
            "Resolved sample\n"
            "────────────────\n"
            f"{body}"
        )
        self._preview.configure(state="normal")
        self._preview.delete("1.0", "end")
        self._preview.insert("1.0", preview)
        self._preview.configure(state="disabled")
