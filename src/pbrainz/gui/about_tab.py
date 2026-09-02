"""About information for the native PBrainZ control panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pbrainz.branding import PRODUCT_NAME, PRODUCT_VERSION

from .icons import scale_icon


class AboutTab:
    """Display product identity and the local-first runtime purpose."""

    def __init__(self, parent: ttk.Frame, brand_icon: tk.PhotoImage | None = None) -> None:
        self._brand_icon = brand_icon
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        outer = ttk.Frame(parent, padding=28)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        row = 0
        if self._brand_icon is not None:
            self._about_icon = scale_icon(self._brand_icon, 256)
            ttk.Label(outer, image=self._about_icon, anchor="center").grid(
                row=row, column=0, sticky="ew", pady=(0, 14)
            )
            row += 1
        ttk.Label(
            outer, text=PRODUCT_NAME, font=("TkDefaultFont", 22, "bold"), anchor="center"
        ).grid(row=row, column=0, sticky="ew", pady=(0, 4))
        row += 1
        ttk.Label(outer, text=f"Version {PRODUCT_VERSION}", anchor="center").grid(
            row=row, column=0, sticky="ew", pady=(0, 22)
        )
        row += 1

        details = ttk.LabelFrame(outer, text="About PBrainZ", padding=14)
        details.grid(row=row, column=0, sticky="ew")
        details.columnconfigure(0, weight=1)
        ttk.Label(
            details,
            text=(
                "PBrainZ is a portable, local-first control panel for provider "
                "connections, game bridges, conversation memory, and voice playback."
            ),
            justify=tk.LEFT,
            wraplength=620,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            details,
            text=(
                "Your settings, downloaded voice models, and activity history stay "
                "in the portable data folder beside the application."
            ),
            justify=tk.LEFT,
            wraplength=620,
        ).grid(row=1, column=0, sticky="w", pady=(14, 0))
