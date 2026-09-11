from __future__ import annotations

from types import SimpleNamespace

from pbrainz.gui.panel import PBrainZControlPanel


class _Variable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _panel_without_tk() -> PBrainZControlPanel:
    panel = PBrainZControlPanel.__new__(PBrainZControlPanel)
    panel.state = SimpleNamespace(
        openai_base_url=_Variable("https://api.openai.com/v1"),
        openai_key=_Variable("openai-secret"),
        ollama_base_url=_Variable("http://127.0.0.1:11434/v1"),
        ollama_key=_Variable(),
        lmstudio_base_url=_Variable("http://127.0.0.1:1234/v1"),
        lmstudio_key=_Variable(),
        custom_base_url=_Variable(),
        custom_key=_Variable(),
        horde_base_url=_Variable("https://oai.aihorde.net/v1"),
        horde_key=_Variable(),
        gemini_key=_Variable("gemini-secret"),
    )
    return panel


def test_provider_settings_payload_includes_visible_credentials() -> None:
    panel = _panel_without_tk()

    payload = panel._provider_settings_payload()

    assert payload["gemini_api_key"] == "gemini-secret"
    assert payload["openai_api_key"] == "openai-secret"


def test_api_test_saves_settings_before_chat_request() -> None:
    panel = _panel_without_tk()
    requests: list[tuple[str, str, dict[str, object], object, dict[str, object]]] = []
    applied: list[dict[str, object]] = []
    panel._run_request = lambda method, path, payload, success, **kwargs: requests.append(
        (method, path, payload, success, kwargs)
    )
    panel._apply_status = lambda data: applied.append(data)

    panel.test_provider_api(
        "gemini", "gemini-2.5-flash", lambda _data: None, lambda _error: None, 30
    )

    assert len(requests) == 1
    assert requests[0][1] == "/api/settings"
    assert requests[0][2]["default_provider"] == "gemini"
    assert requests[0][2]["gemini_api_key"] == "gemini-secret"

    save_success = requests[0][3]
    assert callable(save_success)
    save_success({"default_provider": "gemini"})

    assert applied == [{"default_provider": "gemini"}]
    assert len(requests) == 2
    assert requests[1][1] == "/api/chat"
    assert requests[1][2]["provider"] == "gemini"
    assert requests[1][2]["model"] == "gemini-2.5-flash"
