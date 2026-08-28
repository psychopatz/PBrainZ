"""Window shell and cross-tab coordination for HoomansLLM's native GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .chat_tab import ChatTab
from .control_tab import ControlTab
from .request import ApiRequestRunner, RequestFailure, RequestSuccess
from .settings_tab import SettingsTab
from .state import PanelState
from .theme import apply_theme
from .tts_tab import TTSTab


class HoomansLLMControlPanel:
    """Coordinate independent native tabs around one local API client."""

    def __init__(self, host: str, port: int) -> None:
        self.root = tk.Tk()
        self.root.title("HoomansLLM Control Panel")
        self.root.geometry("780x700")
        self.root.minsize(680, 580)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.state = PanelState.create(self.root)
        self.state.bind_dirty_tracking()
        self._requests = ApiRequestRunner(self.root, host, port)
        self.base_url = self._requests.base_url
        self.control_tab: ControlTab
        self.chat_tab: ChatTab
        self.tts_tab: TTSTab
        self.settings_tab: SettingsTab
        self._build_ui()
        self._apply_theme(self.state.theme.get())
        self.root.after(100, self.refresh)
        self.root.after(150, self.refresh_logs)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        control_frame = ttk.Frame(notebook)
        settings_frame = ttk.Frame(notebook)
        chat_frame = ttk.Frame(notebook)
        tts_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Control panel")
        notebook.add(settings_frame, text="Settings")
        notebook.add(chat_frame, text="Chat test")
        notebook.add(tts_frame, text="TTS")

        self.control_tab = ControlTab(
            control_frame,
            self.state,
            self._run_request,
            self._apply_status,
            self._request_failed,
        )
        self.settings_tab = SettingsTab(
            settings_frame,
            self.state,
            self.save_settings,
            self.refresh,
            self._theme_changed,
        )
        self.chat_tab = ChatTab(chat_frame, self._run_request, self._provider_request_timeout)
        self.tts_tab = TTSTab(tts_frame, self._run_request)

    def refresh(self) -> None:
        if self.state.closed or self.state.refresh_in_flight:
            return
        self.state.refresh_in_flight = True
        self._run_request("GET", "/api/settings", None, self._apply_status)

    def refresh_logs(self) -> None:
        if self.state.closed or self.state.logs_in_flight:
            return
        self.state.logs_in_flight = True

        def success(data: dict[str, Any]) -> None:
            self.state.logs_in_flight = False
            entries = data.get("entries", [])
            self.control_tab.log_view.configure(state="normal")
            self.control_tab.log_view.delete("1.0", "end")
            for entry in entries:
                self.control_tab.log_view.insert(
                    "end",
                    f"{entry.get('created_at', '')} [{entry.get('level', '')}] "
                    f"{entry.get('source', '')}: {entry.get('message', '')}\n",
                )
            self.control_tab.log_view.configure(state="disabled")
            self.control_tab.log_view.see("end")
            self.root.after(2500, self.refresh_logs)

        def failure(_error: Exception) -> None:
            self.state.logs_in_flight = False
            self.root.after(3000, self.refresh_logs)

        self._run_request("GET", "/api/logs", None, success, failure=failure)

    def save_settings(self, runtime_values: dict[str, Any] | None = None) -> None:
        if runtime_values is None:
            self.settings_tab.save()
            return
        self._run_request(
            "POST",
            "/api/settings",
            {
                "default_provider": self.state.provider.get() or None,
                "default_model": self.state.model.get() or None,
                **runtime_values,
                "openai_base_url": self.state.openai_base_url.get().strip() or None,
                "openai_api_key": self.state.openai_key.get() or None,
                "ollama_base_url": self.state.ollama_base_url.get().strip() or None,
                "ollama_api_key": self.state.ollama_key.get() or None,
                "lmstudio_base_url": self.state.lmstudio_base_url.get().strip() or None,
                "lmstudio_api_key": self.state.lmstudio_key.get() or None,
                "custom_base_url": self.state.custom_base_url.get().strip() or None,
                "custom_api_key": self.state.custom_key.get() or None,
                "gemini_api_key": self.state.gemini_key.get() or None,
            },
            lambda data: self._apply_status(data, "Settings saved."),
        )

    # These forwarding methods preserve the small public surface exposed by
    # the former monolithic desktop_gui module while each tab owns its logic.
    def toggle_bridge(self) -> None:
        self.control_tab.toggle_bridge()

    def refresh_models(self) -> None:
        self.control_tab.refresh_models()

    def test_api(self) -> None:
        self.control_tab.test_api()

    def send_chat_message(self) -> None:
        self.chat_tab.send()

    def _provider_request_timeout(self) -> float:
        try:
            return max(8.0, float(self.state.timeout.get()))
        except ValueError:
            return 120.0

    def _theme_changed(self, theme: str) -> None:
        self._apply_theme(theme)

    def _apply_theme(self, theme: str) -> None:
        log_view = getattr(self.control_tab, "log_view", None)
        text_widgets = getattr(self.chat_tab, "text_widgets", (None, None))
        apply_theme(
            self.root,
            theme,
            log_view=log_view,
            chat_view=text_widgets[0],
            chat_input=text_widgets[1],
        )

    def _apply_status(self, data: dict[str, Any], success_message: str | None = None) -> None:
        self.state.refresh_in_flight = False
        self.state.status.set(success_message or "SERVER ONLINE")
        preserve_unsaved = self.state.settings_dirty and success_message != "Settings saved."
        if success_message == "Settings saved.":
            self.state.settings_dirty = False
        self.state.applying_status = True
        try:
            if success_message == "Settings saved.":
                self.settings_tab.mark_saved()
            self.control_tab.apply_status_values(data, preserve_unsaved)
            self.settings_tab.apply_status(data, preserve_unsaved)
            if not preserve_unsaved:
                self._set_if_changed(self.state.timeout, str(data.get("request_timeout", 120)))
                self._set_if_changed(
                    self.state.poll_interval, str(data.get("bridge_poll_interval", 0.5))
                )
                self._set_if_changed(self.state.theme, data.get("ui_theme") or "light")
                self._apply_theme(self.state.theme.get())
        finally:
            self.state.applying_status = False

        providers = self.state.provider_models
        preferred = data.get("default_provider") or next(iter(providers), "")
        self.chat_tab.set_providers(providers, preferred)
        self.tts_tab.refresh()
        if success_message == "Settings saved.":
            self.state.settings_dirty = False
        self.root.after(2000, self.refresh)

    @staticmethod
    def _set_if_changed(variable: tk.StringVar, value: str) -> None:
        if variable.get() != value:
            variable.set(value)

    def _run_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        success: RequestSuccess,
        *,
        failure: RequestFailure | None = None,
        timeout: float = 8,
    ) -> None:
        self._requests.run(
            method,
            path,
            payload,
            success,
            failure=failure or self._request_failed,
            timeout=timeout,
        )

    def _request_failed(self, error: Exception) -> None:
        self.state.refresh_in_flight = False
        if isinstance(error, RuntimeError):
            self.state.status.set("SERVER ERROR")
            self.state.bridge_status.set("HoomansLLM rejected the request")
        else:
            self.state.status.set("SERVER OFFLINE")
            self.state.bridge_status.set("Waiting for HoomansLLM server…")
        self.state.bridge_details.set(str(error))
        self.root.after(2500, self.refresh)

    def close(self) -> None:
        self.state.closed = True
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
