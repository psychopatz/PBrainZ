from pbrainz.gui.tab_layout import TAB_LABELS, TAB_ORDER, ordered_tab_keys
from pbrainz.gui.template_model_tab import (
    DEFAULT_TEMPLATE_PROVIDER,
    TEMPLATE_MODEL_PROVIDERS,
)


def test_panel_tabs_follow_manual_order_with_about_last() -> None:
    assert ordered_tab_keys() == TAB_ORDER
    assert "about" in ordered_tab_keys()
    assert "template" in ordered_tab_keys()
    assert ordered_tab_keys().index("chat") < ordered_tab_keys().index("template")
    assert TAB_LABELS["template"] == "Templates"
    assert TEMPLATE_MODEL_PROVIDERS == ("gemini", "horde")
    assert DEFAULT_TEMPLATE_PROVIDER == "gemini"
    assert ordered_tab_keys().index("settings") < ordered_tab_keys().index("about")
    assert ordered_tab_keys()[-1] == "about"
