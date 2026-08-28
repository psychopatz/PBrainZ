"""Manual native-panel tab arrangement.

Edit ``TAB_ORDER`` when a new panel tab is added or the product flow changes.
The order is intentionally explicit and About is validated as the final tab.
"""

TAB_LABELS = {
    "control": "Control panel",
    "chat": "Chat test",
    "memory": "Memories",
    "debug": "Debug trace",
    "tts": "TTS",
    "settings": "Settings",
    "about": "About",
}

# Product-owned ordering. Keep About last so operational controls and settings
# remain together before product information.
TAB_ORDER = ("control", "chat", "memory", "debug", "tts", "settings", "about")


def ordered_tab_keys() -> tuple[str, ...]:
    """Return the manually configured tab order after validating its contract."""

    if len(TAB_ORDER) != len(TAB_LABELS) or set(TAB_ORDER) != set(TAB_LABELS):
        raise RuntimeError("TAB_ORDER must contain each registered tab exactly once")
    if TAB_ORDER[-1] != "about":
        raise RuntimeError("About must remain the final control-panel tab")
    return TAB_ORDER
