from pbrainz.gui.tab_layout import TAB_ORDER, ordered_tab_keys


def test_panel_tabs_follow_manual_order_with_about_last() -> None:
    assert ordered_tab_keys() == TAB_ORDER
    assert "about" in ordered_tab_keys()
    assert ordered_tab_keys().index("settings") < ordered_tab_keys().index("about")
    assert ordered_tab_keys()[-1] == "about"
