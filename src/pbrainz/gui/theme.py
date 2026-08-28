"""Theme styling shared by all native GUI tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def apply_theme(
    root: tk.Misc,
    theme: str,
    *,
    log_view: tk.Text | None = None,
    chat_view: tk.Text | None = None,
    chat_input: tk.Text | None = None,
    memory_detail: tk.Text | None = None,
    debug_detail: tk.Text | None = None,
) -> str:
    """Apply the light/dark palette and return the normalized theme name."""

    theme = theme if theme in {"light", "dark"} else "light"
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    if theme == "dark":
        background = "#1f2937"
        foreground = "#f9fafb"
        field_background = "#111827"
        button_background = "#374151"
        active_background = "#2563eb"
        border = "#4b5563"
    else:
        background = "#ffffff"
        foreground = "#111827"
        field_background = "#ffffff"
        button_background = "#f3f4f6"
        active_background = "#2563eb"
        border = "#cbd5e1"

    root.configure(background=background)
    style.configure(".", background=background, foreground=foreground)
    style.configure("TFrame", background=background)
    style.configure("TLabel", background=background, foreground=foreground)
    style.configure("TLabelframe", background=background, foreground=foreground)
    style.configure("TLabelframe.Label", background=background, foreground=foreground)
    style.configure(
        "TNotebook",
        background=background,
        bordercolor=border,
        tabmargins=(2, 2, 2, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=button_background,
        foreground=foreground,
        bordercolor=border,
        padding=(12, 6),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", active_background), ("active", active_background)],
        foreground=[("selected", "#ffffff"), ("active", "#ffffff")],
    )
    style.configure("TCheckbutton", background=background, foreground=foreground)
    style.map("TCheckbutton", background=[("active", background)])
    style.configure(
        "TButton",
        background=button_background,
        foreground=foreground,
        bordercolor=border,
        lightcolor=border,
        darkcolor=border,
    )
    style.map(
        "TButton",
        background=[("pressed", active_background), ("active", active_background)],
        foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
    )
    style.configure(
        "TEntry",
        fieldbackground=field_background,
        foreground=foreground,
        insertcolor=foreground,
    )
    style.configure(
        "TCombobox",
        fieldbackground=field_background,
        foreground=foreground,
        background=button_background,
        arrowcolor=foreground,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", field_background)],
        foreground=[("readonly", foreground)],
        selectbackground=[("readonly", active_background)],
        selectforeground=[("readonly", "#ffffff")],
    )
    style.configure(
        "Treeview",
        background=field_background,
        fieldbackground=field_background,
        foreground=foreground,
        bordercolor=border,
        lightcolor=border,
        darkcolor=border,
        rowheight=24,
    )
    style.map(
        "Treeview",
        background=[("selected", active_background)],
        foreground=[("selected", "#ffffff")],
    )
    style.configure(
        "Treeview.Heading",
        background=button_background,
        foreground=foreground,
        bordercolor=border,
    )
    style.map(
        "Treeview.Heading",
        background=[("active", active_background)],
        foreground=[("active", "#ffffff")],
    )
    root.option_add("*TCombobox*Listbox.background", field_background)
    root.option_add("*TCombobox*Listbox.foreground", foreground)
    root.option_add("*TCombobox*Listbox.selectBackground", active_background)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    for widget in (log_view, chat_view, chat_input, memory_detail, debug_detail):
        if widget is not None:
            widget.configure(
                background=field_background,
                foreground=foreground,
                insertbackground=foreground,
                selectbackground=active_background,
                selectforeground="#ffffff",
            )
    return theme
