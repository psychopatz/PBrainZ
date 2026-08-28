"""Shared icon loading and high-resolution Tk scaling helpers."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def load_icon(master: tk.Misc, filename: str) -> tk.PhotoImage | None:
    """Load one bundled PNG without allowing a bad asset to break the GUI."""

    path = ASSET_ROOT / filename
    if not path.is_file():
        return None
    try:
        return tk.PhotoImage(master=master, file=str(path))
    except (OSError, tk.TclError):
        return None


def scale_icon(image: tk.PhotoImage, target_size: int) -> tk.PhotoImage:
    """Downsample a high-resolution icon for a specific UI role."""

    reduction = max(1, math.ceil(image.width() / max(1, target_size)))
    return image.subsample(reduction, reduction)
