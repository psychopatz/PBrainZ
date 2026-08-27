"""Native, dependency-free Tk control panel for HoomansLLM."""

from __future__ import annotations

import json
import threading
import tkinter as tk
import urllib.error
import urllib.request
from collections.abc import Callable
from tkinter import messagebox, scrolledtext, ttk
from typing import Any


class HoomansLLMControlPanel:
    """Small native GUI that talks to the local HoomansLLM HTTP API."""

    def __init__(self, host: str, port: int) -> None:
        self.base_url = f"http://{self._local_host(host)}:{port}"
        self.root = tk.Tk()
        self.root.title("HoomansLLM Control Panel")
        self.root.geometry("780x700")
        self.root.minsize(680, 580)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._refresh_in_flight = False
        self._logs_in_flight = False
        self._closed = False
        self._status = tk.StringVar(value="Starting HoomansLLM…")
        self._bridge_status = tk.StringVar(value="Checking Project Hoomans bridge…")
        self._bridge_details = tk.StringVar(value="")
        self._bridge_worker = tk.BooleanVar(value=False)
        self._provider = tk.StringVar()
        self._model = tk.StringVar()
        self._timeout = tk.StringVar(value="120")
        self._poll_interval = tk.StringVar(value="0.5")
        self._theme = tk.StringVar(value="light")
        self._openai_base_url = tk.StringVar(value="https://api.openai.com/v1")
        self._openai_key = tk.StringVar()
        self._ollama_base_url = tk.StringVar(value="http://127.0.0.1:11434/v1")
        self._ollama_key = tk.StringVar()
        self._lmstudio_base_url = tk.StringVar(value="http://127.0.0.1:1234/v1")
        self._lmstudio_key = tk.StringVar()
        self._custom_base_url = tk.StringVar()
        self._custom_key = tk.StringVar()
        self._gemini_key = tk.StringVar()
        self._provider_info = tk.StringVar(value="")
        self._provider_models: dict[str, list[str]] = {}
        self._provider_configured: dict[str, bool] = {}
        self._provider_statuses: list[dict[str, Any]] = []
        self._settings_dirty = False
        self._applying_status = False
        self._rendered_provider: str | None = None
        self._api_test_in_flight = False
        self._api_test_status = tk.StringVar(value="")
        self._chat_provider = tk.StringVar()
        self._chat_model = tk.StringVar()
        self._chat_messages: list[dict[str, str]] = []
        self._chat_request_in_flight = False
        self._log_view: scrolledtext.ScrolledText
        self._chat_view: scrolledtext.ScrolledText
        self._chat_input: tk.Text

        for variable in (
            self._provider,
            self._model,
            self._timeout,
            self._poll_interval,
            self._theme,
            self._openai_base_url,
            self._openai_key,
            self._ollama_base_url,
            self._ollama_key,
            self._lmstudio_base_url,
            self._lmstudio_key,
            self._custom_base_url,
            self._custom_key,
            self._gemini_key,
        ):
            variable.trace_add("write", self._mark_settings_dirty)

        self._apply_theme(self._theme.get())
        self._build_ui()
        self.root.after(100, self.refresh)
        self.root.after(150, self.refresh_logs)

    @staticmethod
    def _local_host(host: str) -> str:
        return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        control_tab = ttk.Frame(notebook)
        chat_tab = ttk.Frame(notebook)
        notebook.add(control_tab, text="Control panel")
        notebook.add(chat_tab, text="Chat test")

        outer = ttk.Frame(control_tab, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        outer.rowconfigure(3, weight=1)

        heading = ttk.Frame(outer)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="HoomansLLM", font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(heading, text="Project Hoomans local LLM control panel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(heading, textvariable=self._status).grid(row=0, column=1, rowspan=2, sticky="e")

        bridge = ttk.LabelFrame(outer, text="Project Hoomans bridge", padding=12)
        bridge.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        bridge.columnconfigure(0, weight=1)
        ttk.Label(
            bridge,
            textvariable=self._bridge_status,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            bridge,
            text="Enable HoomansLLM bridge worker",
            variable=self._bridge_worker,
            command=self.toggle_bridge,
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(bridge, textvariable=self._bridge_details, wraplength=720).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            bridge,
            text=(
                "The in-game Project Hoomans bridge is authoritative. Enable it in the game too; "
                "this switch controls HoomansLLM polling."
            ),
            wraplength=720,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        content = ttk.Frame(outer)
        content.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        provider = ttk.LabelFrame(content, text="Provider and credentials", padding=12)
        provider.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        provider.columnconfigure(1, weight=1)
        ttk.Label(provider, text="Default provider").grid(row=0, column=0, sticky="w", pady=4)
        self._provider_box = ttk.Combobox(
            provider, textvariable=self._provider, state="readonly", width=24
        )
        self._provider_box.grid(row=0, column=1, sticky="ew", pady=4)
        self._provider_box.bind("<<ComboboxSelected>>", self._provider_changed)
        ttk.Label(provider, text="Default model").grid(row=1, column=0, sticky="w", pady=4)
        model_row = ttk.Frame(provider)
        model_row.grid(row=1, column=1, sticky="ew", pady=4)
        model_row.columnconfigure(0, weight=1)
        self._model_box = ttk.Combobox(
            model_row, textvariable=self._model, state="readonly", width=20
        )
        self._model_box.grid(row=0, column=0, sticky="ew")
        ttk.Button(model_row, text="Refresh models", command=self.refresh_models).grid(
            row=0, column=1, padx=(6, 0)
        )
        self._provider_settings = ttk.LabelFrame(provider, padding=8)
        self._provider_settings.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._provider_settings.columnconfigure(1, weight=1)
        ttk.Label(provider, textvariable=self._provider_info, wraplength=330).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self._render_provider_settings()

        runtime = ttk.LabelFrame(content, text="Runtime settings", padding=12)
        runtime.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        runtime.columnconfigure(1, weight=1)
        ttk.Label(runtime, text="Request timeout (seconds)").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(runtime, textvariable=self._timeout, width=12).grid(
            row=0, column=1, sticky="ew", pady=4
        )
        ttk.Label(runtime, text="Bridge poll interval (seconds)").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(runtime, textvariable=self._poll_interval, width=12).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Label(
            runtime,
            text=(
                "Model lists are cached per provider for fast startup. Use Refresh models "
                "to update only the selected provider."
            ),
            wraplength=300,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(runtime, text="Theme").grid(row=3, column=0, sticky="w", pady=(16, 4))
        self._theme_box = ttk.Combobox(
            runtime,
            textvariable=self._theme,
            values=("light", "dark"),
            state="readonly",
            width=12,
        )
        self._theme_box.grid(row=3, column=1, sticky="ew", pady=(16, 4))
        self._theme_box.bind("<<ComboboxSelected>>", self._theme_changed)
        actions = ttk.Frame(runtime)
        actions.grid(row=4, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Test API", command=self.test_api).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Save settings", command=self.save_settings).pack(side="left")
        ttk.Label(runtime, textvariable=self._api_test_status, wraplength=300).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        activity = ttk.LabelFrame(outer, text="Recent activity", padding=8)
        activity.grid(row=3, column=0, sticky="nsew")
        activity.rowconfigure(0, weight=1)
        activity.columnconfigure(0, weight=1)
        self._log_view = scrolledtext.ScrolledText(
            activity, height=8, state="disabled", wrap="word", font=("TkFixedFont", 9)
        )
        self._log_view.grid(row=0, column=0, sticky="nsew")

        ttk.Label(outer, text="API keys are masked and never written to the activity log.").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        self._build_chat_tab(chat_tab)

    def _build_chat_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        selector = ttk.LabelFrame(tab, text="Provider test", padding=12)
        selector.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        selector.columnconfigure(1, weight=1)
        selector.columnconfigure(3, weight=1)
        ttk.Label(selector, text="Provider").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._chat_provider_box = ttk.Combobox(
            selector, textvariable=self._chat_provider, state="readonly", width=22
        )
        self._chat_provider_box.grid(row=0, column=1, sticky="ew")
        self._chat_provider_box.bind("<<ComboboxSelected>>", self._chat_provider_changed)
        ttk.Label(selector, text="Model").grid(row=0, column=2, sticky="w", padx=(18, 8))
        self._chat_model_box = ttk.Combobox(
            selector, textvariable=self._chat_model, state="readonly", width=28
        )
        self._chat_model_box.grid(row=0, column=3, sticky="ew")
        self._chat_model_box.bind("<<ComboboxSelected>>", self._chat_model_changed)

        self._chat_view = scrolledtext.ScrolledText(
            tab, state="disabled", wrap="word", height=18, font=("TkDefaultFont", 10)
        )
        self._chat_view.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))

        composer = ttk.Frame(tab, padding=(18, 0, 18, 18))
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)
        self._chat_input = tk.Text(composer, height=4, wrap="word", undo=True)
        self._chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._chat_input.bind("<Control-Return>", self._send_chat_from_shortcut)
        self._chat_send_button = ttk.Button(
            composer, text="Send", command=self.send_chat_message, width=12
        )
        self._chat_send_button.grid(row=0, column=1, sticky="ns")
        ttk.Label(composer, text="Ctrl+Enter sends the message; Enter adds a new line.").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

    def refresh(self) -> None:
        if self._closed or self._refresh_in_flight:
            return
        self._refresh_in_flight = True
        self._run_request("GET", "/api/settings", None, self._apply_status)

    def refresh_logs(self) -> None:
        if self._closed or self._logs_in_flight:
            return
        self._logs_in_flight = True

        def success(data: dict[str, Any]) -> None:
            self._logs_in_flight = False
            entries = data.get("entries", [])
            self._log_view.configure(state="normal")
            self._log_view.delete("1.0", "end")
            for entry in entries:
                self._log_view.insert(
                    "end",
                    f"{entry.get('created_at', '')} [{entry.get('level', '')}] "
                    f"{entry.get('source', '')}: {entry.get('message', '')}\n",
                )
            self._log_view.configure(state="disabled")
            self._log_view.see("end")
            self.root.after(2500, self.refresh_logs)

        def failure(_error: Exception) -> None:
            self._logs_in_flight = False
            self.root.after(3000, self.refresh_logs)

        self._run_request("GET", "/api/logs", None, success, failure=failure)

    def toggle_bridge(self) -> None:
        enabled = self._bridge_worker.get()
        self._run_request(
            "POST",
            "/api/bridge",
            {"enabled": enabled},
            self._apply_status,
            failure=lambda error: self._bridge_toggle_failed(not enabled, error),
        )

    def refresh_models(self) -> None:
        provider = self._provider.get()
        if not provider:
            return
        self._run_request(
            "POST",
            "/api/models/refresh",
            {"provider": provider},
            self._apply_status,
            success_message=f"Refreshing {provider} models…",
        )

    def test_api(self) -> None:
        if self._api_test_in_flight:
            return
        provider = self._provider.get()
        model = self._model.get()
        if not provider or not model:
            self._api_test_status.set("Select a provider and refreshed model first.")
            return
        self._api_test_in_flight = True
        self._api_test_status.set(f"Testing {provider}/{model}…")
        self._run_request(
            "POST",
            "/api/chat",
            {
                "provider": provider,
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly: HoomansLLM API OK",
                    }
                ],
            },
            self._api_test_succeeded,
            failure=self._api_test_failed,
            timeout=self._provider_request_timeout(),
        )

    def _api_test_succeeded(self, data: dict[str, Any]) -> None:
        self._api_test_in_flight = False
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        compact = " ".join(str(content).split())
        if len(compact) > 140:
            compact = f"{compact[:137]}…"
        self._api_test_status.set(f"API test passed: {compact or '(empty response)'}")

    def _api_test_failed(self, error: Exception) -> None:
        self._api_test_in_flight = False
        self._api_test_status.set(f"API test failed: {error}")

    def _send_chat_from_shortcut(self, _event: object) -> str:
        self.send_chat_message()
        return "break"

    def send_chat_message(self) -> None:
        if self._chat_request_in_flight:
            return
        provider = self._chat_provider.get()
        model = self._chat_model.get()
        message = self._chat_input.get("1.0", "end-1c").strip()
        if not provider or not model:
            self._append_chat_message("System", "Select a provider and model first.")
            return
        if not message:
            return
        self._chat_messages.append({"role": "user", "content": message})
        self._append_chat_message("You", message)
        self._chat_input.delete("1.0", "end")
        self._chat_request_in_flight = True
        self._chat_send_button.configure(state="disabled")
        self._run_request(
            "POST",
            "/api/chat",
            {"provider": provider, "model": model, "messages": self._chat_messages},
            self._chat_succeeded,
            failure=self._chat_failed,
            timeout=self._provider_request_timeout(),
        )

    def _chat_succeeded(self, data: dict[str, Any]) -> None:
        self._chat_request_in_flight = False
        self._chat_send_button.configure(state="normal")
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        if not content:
            content = "(empty response)"
        self._chat_messages.append({"role": "assistant", "content": str(content)})
        self._append_chat_message("Assistant", str(content))

    def _chat_failed(self, error: Exception) -> None:
        self._chat_request_in_flight = False
        self._chat_send_button.configure(state="normal")
        self._append_chat_message("System", f"Request failed: {error}")

    def _append_chat_message(self, speaker: str, content: str) -> None:
        self._chat_view.configure(state="normal")
        self._chat_view.insert("end", f"{speaker}:\n{content}\n\n")
        self._chat_view.configure(state="disabled")
        self._chat_view.see("end")

    def _chat_provider_changed(self, _event: object | None = None) -> None:
        models = self._provider_models.get(self._chat_provider.get(), [])
        self._chat_model_box["values"] = models
        if not models:
            self._chat_model.set("")
        elif self._chat_model.get() not in models:
            self._chat_model.set(models[0])

    def _chat_model_changed(self, _event: object | None = None) -> None:
        return

    def _provider_request_timeout(self) -> float:
        try:
            return max(8.0, float(self._timeout.get()))
        except ValueError:
            return 120.0

    def save_settings(self) -> None:
        try:
            timeout = float(self._timeout.get())
            poll_interval = float(self._poll_interval.get())
        except ValueError:
            self._show_error("Timeout and poll interval must be numbers.")
            return
        self._run_request(
            "POST",
            "/api/settings",
            {
                "default_provider": self._provider.get() or None,
                "default_model": self._model.get() or None,
                "request_timeout": timeout,
                "bridge_poll_interval": poll_interval,
                "ui_theme": self._theme.get(),
                "openai_base_url": self._openai_base_url.get().strip() or None,
                "openai_api_key": self._openai_key.get() or None,
                "ollama_base_url": self._ollama_base_url.get().strip() or None,
                "ollama_api_key": self._ollama_key.get() or None,
                "lmstudio_base_url": self._lmstudio_base_url.get().strip() or None,
                "lmstudio_api_key": self._lmstudio_key.get() or None,
                "custom_base_url": self._custom_base_url.get().strip() or None,
                "custom_api_key": self._custom_key.get() or None,
                "gemini_api_key": self._gemini_key.get() or None,
            },
            self._apply_status,
            success_message="Settings saved.",
        )

    def _provider_changed(self, _event: object | None = None) -> None:
        models = self._provider_models.get(self._provider.get(), [])
        self._model_box["values"] = models
        if not models:
            self._model.set("")
        elif self._model.get() not in models:
            self._model.set(models[0])
        configured = self._provider_configured.get(self._provider.get(), False)
        hint = next(
            (
                item.get("api_key_hint")
                for item in self._provider_statuses
                if item.get("name") == self._provider.get()
            ),
            None,
        )
        self._provider_info.set(
            f"Configured ({hint})" if configured and hint else "Configured"
            if configured
            else "API key or endpoint is missing"
        )
        self._render_provider_settings()

    def _render_provider_settings(self) -> None:
        provider = self._provider.get()
        if provider == self._rendered_provider:
            return
        self._rendered_provider = provider
        title = {
            "openai": "OpenAI settings",
            "ollama": "Ollama settings (OpenAI-compatible)",
            "lmstudio": "LM Studio settings (OpenAI-compatible)",
            "custom": "Custom OpenAI-compatible settings",
            "gemini": "Gemini settings",
        }.get(provider, "Provider settings")
        self._provider_settings.configure(text=title)
        for child in self._provider_settings.winfo_children():
            child.destroy()
        compatible_base_urls = {
            "openai": self._openai_base_url,
            "ollama": self._ollama_base_url,
            "lmstudio": self._lmstudio_base_url,
            "custom": self._custom_base_url,
        }
        compatible_keys = {
            "openai": self._openai_key,
            "ollama": self._ollama_key,
            "lmstudio": self._lmstudio_key,
            "custom": self._custom_key,
        }
        if provider in compatible_base_urls:
            ttk.Label(self._provider_settings, text="API endpoint").grid(
                row=0, column=0, sticky="w", pady=4
            )
            ttk.Entry(
                self._provider_settings,
                textvariable=compatible_base_urls[provider],
                width=25,
            ).grid(row=0, column=1, sticky="ew", pady=4)
            ttk.Label(self._provider_settings, text="API key").grid(
                row=1, column=0, sticky="w", pady=4
            )
            ttk.Entry(
                self._provider_settings,
                textvariable=compatible_keys[provider],
                show="*",
                width=25,
            ).grid(row=1, column=1, sticky="ew", pady=4)
            note = {
                "openai": "OpenAI Cloud requires an API key.",
                "ollama": "Ollama normally does not need an API key.",
                "lmstudio": "LM Studio normally does not need an API key.",
                "custom": "API key is optional; set the endpoint for your router.",
            }[provider]
        elif provider == "gemini":
            ttk.Label(self._provider_settings, text="API key").grid(
                row=0, column=0, sticky="w", pady=4
            )
            ttk.Entry(
                self._provider_settings,
                textvariable=self._gemini_key,
                show="*",
                width=25,
            ).grid(row=0, column=1, sticky="ew", pady=4)
            note = "Leave the key blank to keep the saved key."
        else:
            note = "Choose a configured provider to edit its credentials."
        ttk.Label(
            self._provider_settings,
            text=f"{note} Values are stored in the local SQLite database.",
            wraplength=330,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _mark_settings_dirty(self, *_args: object) -> None:
        if not self._applying_status:
            self._settings_dirty = True

    def _theme_changed(self, _event: object | None = None) -> None:
        self._apply_theme(self._theme.get())

    def _apply_theme(self, theme: str) -> None:
        theme = theme if theme in {"light", "dark"} else "light"
        style = ttk.Style(self.root)
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

        self.root.configure(background=background)
        style.configure(".", background=background, foreground=foreground)
        style.configure("TFrame", background=background)
        style.configure("TLabel", background=background, foreground=foreground)
        style.configure("TLabelframe", background=background, foreground=foreground)
        style.configure("TLabelframe.Label", background=background, foreground=foreground)
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
        self.root.option_add("*TCombobox*Listbox.background", field_background)
        self.root.option_add("*TCombobox*Listbox.foreground", foreground)
        self.root.option_add("*TCombobox*Listbox.selectBackground", active_background)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        log_view = getattr(self, "_log_view", None)
        if log_view is not None:
            log_view.configure(
                background=field_background,
                foreground=foreground,
                insertbackground=foreground,
                selectbackground=active_background,
                selectforeground="#ffffff",
            )
        chat_view = getattr(self, "_chat_view", None)
        if chat_view is not None:
            chat_view.configure(
                background=field_background,
                foreground=foreground,
                insertbackground=foreground,
                selectbackground=active_background,
                selectforeground="#ffffff",
            )
        chat_input = getattr(self, "_chat_input", None)
        if chat_input is not None:
            chat_input.configure(
                background=field_background,
                foreground=foreground,
                insertbackground=foreground,
                selectbackground=active_background,
                selectforeground="#ffffff",
            )

    def _apply_status(self, data: dict[str, Any], success_message: str | None = None) -> None:
        self._refresh_in_flight = False
        self._status.set(success_message or "SERVER ONLINE")
        preserve_unsaved = self._settings_dirty and success_message != "Settings saved."
        if success_message == "Settings saved.":
            self._settings_dirty = False
        self._applying_status = True
        try:
            self._apply_status_values(data, preserve_unsaved)
        finally:
            self._applying_status = False
        if success_message == "Settings saved.":
            self._applying_status = True
            try:
                self._openai_key.set("")
                self._ollama_key.set("")
                self._lmstudio_key.set("")
                self._custom_key.set("")
                self._gemini_key.set("")
            finally:
                self._applying_status = False
            self._settings_dirty = False
        self.root.after(2000, self.refresh)

    def _apply_status_values(self, data: dict[str, Any], preserve_unsaved: bool) -> None:
        bridge = data.get("bridge") or {}
        ready = bool(bridge.get("ready"))
        available = bool(bridge.get("available"))
        state = "READY" if ready else "NOT READY" if available else "NOT DETECTED"
        self._bridge_status.set(f"{state} — {bridge.get('message', 'No bridge status')}")
        self._bridge_details.set(
            " | ".join(
                [
                    f"Game enabled: {'yes' if bridge.get('enabled') else 'no'}",
                    f"Worker: {'running' if data.get('bridge_worker_running') else 'stopped'}",
                    f"Lifecycle: {bridge.get('lifecycle') or '—'}",
                    f"Runtime: {bridge.get('runtime_id') or '—'}",
                ]
            )
        )
        self._bridge_worker.set(bool(data.get("bridge_worker_enabled")))
        self._provider_models = {
            item["name"]: item.get("models", []) for item in data.get("providers", [])
        }
        self._provider_statuses = data.get("providers", [])
        self._provider_configured = {
            item["name"]: bool(item.get("configured")) for item in data.get("providers", [])
        }
        providers = list(self._provider_models)
        self._provider_box["values"] = providers
        server_provider = data.get("default_provider") or (providers[0] if providers else "")
        if not preserve_unsaved or self._provider.get() not in providers:
            self._provider.set(server_provider)
            self._model.set(data.get("default_model") or "")
        elif self._model.get() not in self._provider_models.get(self._provider.get(), []):
            models = self._provider_models.get(self._provider.get(), [])
            if models:
                self._model.set(data.get("default_model") or models[0])
        if not preserve_unsaved:
            provider_statuses = {
                item.get("name"): item for item in data.get("providers", [])
            }
            for provider_name, variable in {
                "openai": self._openai_base_url,
                "ollama": self._ollama_base_url,
                "lmstudio": self._lmstudio_base_url,
                "custom": self._custom_base_url,
            }.items():
                status = provider_statuses.get(provider_name) or {}
                fallback = data.get("openai_base_url") if provider_name == "openai" else ""
                variable.set(status.get("base_url") or fallback or "")
            self._timeout.set(str(data.get("request_timeout", 120)))
            self._poll_interval.set(str(data.get("bridge_poll_interval", 0.5)))
            self._theme.set(data.get("ui_theme") or "light")
            self._apply_theme(self._theme.get())
        self._provider_changed()
        self._chat_provider_box["values"] = providers
        if not self._chat_provider.get() or self._chat_provider.get() not in providers:
            self._chat_provider.set(server_provider)
        self._chat_provider_changed()

    def _run_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        success: Callable[[dict[str, Any], str | None], None]
        | Callable[[dict[str, Any]], None],
        *,
        failure: Callable[[Exception], None] | None = None,
        success_message: str | None = None,
        timeout: float = 8,
    ) -> None:
        def worker() -> None:
            try:
                data = self._request(method, path, payload, timeout=timeout)
                if success_message is None:
                    self.root.after(0, success, data)
                else:
                    self.root.after(0, success, data, success_message)
            except Exception as request_error:  # Network errors belong in the GUI status.
                callback = failure or self._request_failed
                self.root.after(0, callback, request_error)

        threading.Thread(target=worker, name="hoomans-llm-gui-request", daemon=True).start()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: float = 8,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                value = json.loads(detail)
                detail = value.get("detail") or value.get("error", {}).get("message") or detail
            except (TypeError, ValueError):
                pass
            raise RuntimeError(detail or f"HTTP {error.code}") from error

    def _request_failed(self, error: Exception) -> None:
        self._refresh_in_flight = False
        if isinstance(error, RuntimeError):
            self._status.set("SERVER ERROR")
            self._bridge_status.set("HoomansLLM rejected the request")
        else:
            self._status.set("SERVER OFFLINE")
            self._bridge_status.set("Waiting for HoomansLLM server…")
        self._bridge_details.set(str(error))
        self.root.after(2500, self.refresh)

    def _bridge_toggle_failed(self, previous_state: bool, error: Exception) -> None:
        self._bridge_worker.set(previous_state)
        self._request_failed(error)

    def _show_error(self, message: str) -> None:
        messagebox.showerror("HoomansLLM", message, parent=self.root)

    def close(self) -> None:
        self._closed = True
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
