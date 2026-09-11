"""Window shell and cross-tab coordination for PBrainZ's native GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from pbrainz.branding import PRODUCT_BINARY_NAME, PRODUCT_NAME, PRODUCT_VERSION

from .about_tab import AboutTab
from .chat_tab import ChatTab
from .control_tab import ControlTab
from .debug_tab import DebugTab
from .dictionary_tab import DictionaryTab
from .icons import load_icon
from .memory_tab import MemoryTab
from .request import ApiRequestRunner, RequestFailure, RequestSuccess
from .settings_tab import SettingsTab
from .state import PanelState
from .tab_layout import TAB_LABELS, ordered_tab_keys
from .template_model_tab import TemplateModelTab
from .theme import apply_theme
from .tts.tab import TTSTab


class PBrainZControlPanel:
    """Coordinate independent native tabs around one local API client."""

    def __init__(self, host: str, port: int, initial_theme: str = "light") -> None:
        self.root = tk.Tk(className=PRODUCT_BINARY_NAME)
        self.root.title(f"{PRODUCT_BINARY_NAME} v{PRODUCT_VERSION}")
        self._window_icon: tk.PhotoImage | None = None
        self._brand_icon: tk.PhotoImage | None = None
        self._set_window_icon()
        self.root.geometry("780x700")
        self.root.minsize(680, 580)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.state = PanelState.create(self.root, theme=initial_theme)
        self.state.bind_dirty_tracking()
        self._requests = ApiRequestRunner(self.root, host, port)
        self.base_url = self._requests.base_url
        self._chat_selection_after: str | None = None
        self._chat_selection_pending: tuple[str, str] | None = None
        self._chat_selection_generation = 0
        self.control_tab: ControlTab
        self.chat_tab: ChatTab
        self.template_model_tab: TemplateModelTab
        self.retrieval_dictionary_tab: DictionaryTab
        self.memory_tab: MemoryTab
        self.debug_tab: DebugTab
        self.tts_tab: TTSTab
        self.about_tab: AboutTab
        self.settings_tab: SettingsTab
        self._build_ui()
        self._apply_theme(self.state.theme.get())
        self.root.after(100, self.refresh)
        self.root.after(150, self.refresh_logs)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        frames: dict[str, ttk.Frame] = {}
        for tab_key in ordered_tab_keys():
            frame = ttk.Frame(notebook)
            frames[tab_key] = frame
            notebook.add(frame, text=TAB_LABELS[tab_key])

        control_frame = frames["control"]
        settings_frame = frames["settings"]
        chat_frame = frames["chat"]
        template_frame = frames["template"]
        memory_frame = frames["memory"]
        self._memory_frame = memory_frame
        debug_frame = frames["debug"]
        self._debug_frame = debug_frame
        tts_frame = frames["tts"]
        about_frame = frames["about"]

        self.control_tab = ControlTab(
            control_frame,
            self.state,
            self._run_request,
            self._apply_status,
            self._request_failed,
            save_selection=self.save_selection,
            save_settings=self.save_settings,
            test_api=self.test_provider_api,
            brand_icon=self._brand_icon,
        )
        self.settings_tab = SettingsTab(
            settings_frame,
            self.state,
            self.save_settings,
            self.refresh,
            self._theme_changed,
        )
        self.chat_tab = ChatTab(
            chat_frame,
            self._run_request,
            self._provider_request_timeout,
            self.save_chat_selection,
        )
        template_pages = ttk.Notebook(template_frame)
        template_pages.pack(fill="both", expand=True)
        template_profile_page = ttk.Frame(template_pages)
        dictionary_page = ttk.Frame(template_pages)
        template_pages.add(template_profile_page, text="Template profiles")
        template_pages.add(dictionary_page, text="Dictionaries")
        self.template_model_tab = TemplateModelTab(
            template_profile_page,
            self.save_template_profile,
            self.activate_template_profile,
            self.delete_template_profile,
            self.reset_template_profile,
        )
        self.retrieval_dictionary_tab = DictionaryTab(dictionary_page, self._run_request)
        self.memory_tab = MemoryTab(
            memory_frame,
            self._run_request,
            self._provider_request_timeout,
        )
        self.debug_tab = DebugTab(
            debug_frame,
            self._run_request,
            self._provider_request_timeout,
        )
        self.tts_tab = TTSTab(tts_frame, self._run_request)
        self.about_tab = AboutTab(about_frame, brand_icon=self._brand_icon)
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _tab_changed(self, _event: object | None = None) -> None:
        if self.notebook.select() == str(self._memory_frame):
            self.memory_tab.refresh()
        elif self.notebook.select() == str(self._debug_frame):
            self.debug_tab.refresh()

    def _set_window_icon(self) -> None:
        self._window_icon = load_icon(self.root, "pbrainz.png")
        self._brand_icon = load_icon(self.root, "pbrainz-mark.png")
        if self._window_icon is not None:
            self.root.iconphoto(True, self._window_icon)

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
                **self._provider_settings_payload(),
            },
            lambda data: self._apply_status(data, "Settings saved."),
        )

    def test_provider_api(
        self,
        provider: str,
        model: str,
        success: RequestSuccess,
        failure: RequestFailure,
        timeout: float,
    ) -> None:
        """Save the visible provider fields before sending the API test."""

        payload = {
            "default_provider": provider,
            "default_model": model,
            **self._provider_settings_payload(),
        }

        def settings_saved(data: dict[str, Any]) -> None:
            # Keep other in-progress edits in the UI while refreshing the
            # server's configured-provider flags from the save response.
            self._apply_status(data)
            self._run_request(
                "POST",
                "/api/chat",
                {
                    "provider": provider,
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Reply with exactly: {PRODUCT_NAME} API OK",
                        }
                    ],
                },
                success,
                failure=failure,
                timeout=timeout,
            )

        self._run_request(
            "POST",
            "/api/settings",
            payload,
            settings_saved,
            failure=failure,
            timeout=8,
        )

    def _provider_settings_payload(self) -> dict[str, Any]:
        return {
            "openai_base_url": self.state.openai_base_url.get().strip() or None,
            "openai_api_key": self.state.openai_key.get() or None,
            "ollama_base_url": self.state.ollama_base_url.get().strip() or None,
            "ollama_api_key": self.state.ollama_key.get() or None,
            "lmstudio_base_url": self.state.lmstudio_base_url.get().strip() or None,
            "lmstudio_api_key": self.state.lmstudio_key.get() or None,
            "custom_base_url": self.state.custom_base_url.get().strip() or None,
            "custom_api_key": self.state.custom_key.get() or None,
            "horde_base_url": self.state.horde_base_url.get().strip() or None,
            "horde_api_key": self.state.horde_key.get() or None,
            "gemini_api_key": self.state.gemini_key.get() or None,
        }

    def save_selection(self, provider: str, model: str) -> None:
        """Persist the active provider/model so new requests use it immediately."""

        self._chat_selection_pending = (provider, model)
        self._chat_selection_generation += 1
        if self._chat_selection_after is not None:
            self.root.after_cancel(self._chat_selection_after)
        self._chat_selection_after = self.root.after(180, self._persist_chat_selection)

    def save_chat_selection(self, provider: str, model: str) -> None:
        """Persist Chat test's provider/model choice after the user settles it."""

        self.save_selection(provider, model)

    def save_template_profile(self, profile: dict[str, Any]) -> None:
        """Persist one editable NPC prompt-template profile."""
        self._run_request(
            "POST",
            "/api/template-profiles",
            {"profile": profile},
            self._template_profile_saved,
            failure=self._template_profile_failed,
        )

    def _template_profile_saved(self, data: dict[str, Any]) -> None:
        self.template_model_tab.mark_server_synced()
        self.template_model_tab.set_action_status("Profile saved")
        self._apply_status(data)

    def activate_template_profile(self, profile_id: str) -> None:
        """Make one saved profile active for new NPC turns."""

        self._run_request(
            "POST",
            "/api/template-profiles/active",
            {"profile_id": profile_id},
            self._template_profile_activated,
            failure=self._template_profile_failed,
        )

    def _template_profile_activated(self, data: dict[str, Any]) -> None:
        self.template_model_tab.sync_active_on_next_status()
        self.template_model_tab.set_action_status("Profile active")
        self._apply_status(data)

    def delete_template_profile(self, profile_id: str) -> None:
        """Delete a user-created template profile."""

        self._run_request(
            "POST",
            "/api/template-profiles/delete",
            {"profile_id": profile_id},
            self._template_profile_deleted,
            failure=self._template_profile_failed,
        )

    def _template_profile_deleted(self, data: dict[str, Any]) -> None:
        self.template_model_tab.mark_server_synced()
        self.template_model_tab.set_action_status("Profile deleted")
        self._apply_status(data)

    def reset_template_profile(self, profile_id: str) -> None:
        """Restore a shipped template profile."""

        self._run_request(
            "POST",
            "/api/template-profiles/reset",
            {"profile_id": profile_id},
            self._template_profile_reset,
            failure=self._template_profile_failed,
        )

    def _template_profile_reset(self, data: dict[str, Any]) -> None:
        self.template_model_tab.mark_server_synced()
        self.template_model_tab.set_action_status("Profile reset")
        self._apply_status(data)

    def _template_profile_failed(self, error: Exception) -> None:
        self.template_model_tab.set_action_status(f"Template profile error: {error}")
        self._request_failed(error)

    def _persist_chat_selection(self) -> None:
        self._chat_selection_after = None
        if self._chat_selection_pending is None:
            return
        provider, model = self._chat_selection_pending
        generation = self._chat_selection_generation
        self._run_request(
            "POST",
            "/api/settings",
            {
                "default_provider": provider,
                "default_model": model,
                **self._provider_settings_payload(),
            },
            lambda data: self._chat_selection_saved(generation, data),
            failure=lambda error: self._chat_selection_failed(generation, error),
        )

    def _chat_selection_saved(self, generation: int, data: dict[str, Any]) -> None:
        if generation != self._chat_selection_generation:
            return
        self.chat_tab.set_selection_status("Selection saved")
        self._apply_status(data)
        provider = str(data.get("default_provider") or "")
        model = str(data.get("default_model") or "")
        self.chat_tab.set_selection(provider, model)

    def _chat_selection_failed(self, generation: int, error: Exception) -> None:
        if generation != self._chat_selection_generation:
            return
        self.chat_tab.set_selection_status(f"Selection not saved: {error}")
        self._request_failed(error)

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
            memory_detail=getattr(self.memory_tab, "detail_view", None),
            debug_detail=getattr(self.debug_tab, "detail_view", None),
            template_views=(
                *getattr(self.template_model_tab, "text_widgets", ()),
                *getattr(self.retrieval_dictionary_tab, "text_widgets", ()),
            ),
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
                zomboid_path = data.get("zomboid_path")
                if zomboid_path:
                    self._set_if_changed(self.state.zomboid_path, str(zomboid_path))
                self._set_if_changed(self.state.theme, data.get("ui_theme") or "light")
                self._apply_theme(self.state.theme.get())
        finally:
            self.state.applying_status = False

        providers = self.state.provider_models
        preferred = data.get("default_provider") or next(iter(providers), "")
        self.chat_tab.set_providers(providers, preferred)
        self.template_model_tab.apply_status(data, preserve_unsaved)
        self.retrieval_dictionary_tab.apply_status(data, preserve_unsaved)
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
            self.state.bridge_status.set(f"{PRODUCT_NAME} rejected the request")
        else:
            self.state.status.set("SERVER OFFLINE")
            self.state.bridge_status.set(f"Waiting for {PRODUCT_NAME} server…")
        self.state.bridge_details.set(str(error))
        self.root.after(2500, self.refresh)

    def close(self) -> None:
        self.state.closed = True
        if self._chat_selection_after is not None:
            self.root.after_cancel(self._chat_selection_after)
            self._chat_selection_after = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
