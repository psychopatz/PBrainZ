"""Bridge, provider, and activity controls for the native panel."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import scrolledtext, ttk
from typing import Any

from pbrainz.branding import PRODUCT_NAME

from .icons import scale_icon
from .request import RequestFn
from .state import PanelState

StatusFn = Callable[[dict[str, Any], str | None], None]
ErrorFn = Callable[[Exception], None]


class ControlTab:
    """Own the control tab's widgets, provider selection, and provider actions."""

    def __init__(
        self,
        parent: ttk.Frame,
        state: PanelState,
        request: RequestFn,
        apply_status: StatusFn,
        request_failed: ErrorFn,
        brand_icon: tk.PhotoImage | None = None,
    ) -> None:
        self.state = state
        self._request = request
        self._apply_status = apply_status
        self._request_failed = request_failed
        self._brand_icon = brand_icon
        self._rendered_provider: str | None = None
        self._api_test_in_flight = False
        self._api_test_status = tk.StringVar(parent, value="")
        self._provider_settings: ttk.LabelFrame
        self.provider_box: ttk.Combobox
        self.model_box: ttk.Combobox
        self.log_view: scrolledtext.ScrolledText
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        outer.rowconfigure(3, weight=1)

        heading = ttk.Frame(outer)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        heading.columnconfigure(1, weight=1)
        heading_icon = (
            scale_icon(self._brand_icon, 40) if self._brand_icon is not None else None
        )
        self._heading_icon = heading_icon
        if heading_icon is not None:
            ttk.Label(heading, image=heading_icon).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(heading, text=PRODUCT_NAME, font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(heading, textvariable=self.state.status).grid(row=0, column=2, sticky="e")

        bridge = ttk.LabelFrame(outer, text="Bridge", padding=12)
        bridge.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        bridge.columnconfigure(0, weight=1)
        ttk.Label(
            bridge,
            textvariable=self.state.bridge_status,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            bridge,
            text="Enable Project Hoomans bridge",
            variable=self.state.bridge_worker,
            command=self.toggle_bridge,
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(bridge, textvariable=self.state.bridge_details, wraplength=720).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        content = ttk.Frame(outer)
        content.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        provider = ttk.LabelFrame(content, text="Provider and credentials", padding=12)
        provider.grid(row=0, column=0, sticky="nsew")
        provider.columnconfigure(1, weight=1)
        ttk.Label(provider, text="Default provider").grid(row=0, column=0, sticky="w", pady=4)
        self.provider_box = ttk.Combobox(
            provider, textvariable=self.state.provider, state="readonly", width=24
        )
        self.provider_box.grid(row=0, column=1, sticky="ew", pady=4)
        self.provider_box.bind("<<ComboboxSelected>>", self._provider_changed)
        ttk.Label(provider, text="Default model").grid(row=1, column=0, sticky="w", pady=4)
        model_row = ttk.Frame(provider)
        model_row.grid(row=1, column=1, sticky="ew", pady=4)
        model_row.columnconfigure(0, weight=1)
        self.model_box = ttk.Combobox(
            model_row, textvariable=self.state.model, state="readonly", width=20
        )
        self.model_box.grid(row=0, column=0, sticky="ew")
        ttk.Button(model_row, text="Refresh models", command=self.refresh_models).grid(
            row=0, column=1, padx=(6, 0)
        )
        self._provider_settings = ttk.LabelFrame(provider, padding=8)
        self._provider_settings.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._provider_settings.columnconfigure(1, weight=1)
        ttk.Label(provider, textvariable=self.state.provider_info, wraplength=680).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        provider_actions = ttk.Frame(provider)
        provider_actions.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(provider_actions, text="Test API", command=self.test_api).pack(
            side="left", padx=(0, 10)
        )
        ttk.Label(provider_actions, textvariable=self._api_test_status).pack(side="left")
        self._render_provider_settings()

        activity = ttk.LabelFrame(outer, text="Recent activity", padding=8)
        activity.grid(row=3, column=0, sticky="nsew")
        activity.rowconfigure(0, weight=1)
        activity.columnconfigure(0, weight=1)
        self.log_view = scrolledtext.ScrolledText(
            activity, height=8, state="disabled", wrap="word", font=("TkFixedFont", 9)
        )
        self.log_view.grid(row=0, column=0, sticky="nsew")

    @property
    def providers(self) -> list[str]:
        return list(self.state.provider_models)

    def apply_status_values(self, data: dict[str, Any], preserve_unsaved: bool) -> None:
        """Hydrate bridge/provider controls from the API status response."""

        bridge = data.get("bridge") or {}
        ready = bool(bridge.get("ready"))
        available = bool(bridge.get("available"))
        state = "READY" if ready else "NOT READY" if available else "NOT DETECTED"
        self.state.bridge_status.set(f"{state} — {bridge.get('message', 'No bridge status')}")
        self.state.bridge_details.set(
            " | ".join(
                [
                    f"Setting: {'on' if data.get('game_bridge_setting_enabled') else 'off'}",
                    f"Worker: {'running' if data.get('bridge_worker_running') else 'stopped'}",
                    f"Lifecycle: {bridge.get('lifecycle') or '—'}",
                    f"Runtime: {bridge.get('runtime_id') or '—'}",
                ]
            )
        )
        self.state.bridge_worker.set(
            bool(data.get("bridge_worker_enabled"))
            and bool(data.get("game_bridge_setting_enabled"))
        )
        self.state.provider_models = {
            item["name"]: item.get("models", []) for item in data.get("providers", [])
        }
        self.state.provider_configured = {
            item["name"]: bool(item.get("configured")) for item in data.get("providers", [])
        }
        providers = self.providers
        self.provider_box["values"] = providers
        server_provider = data.get("default_provider") or (providers[0] if providers else "")
        if not preserve_unsaved or self.state.provider.get() not in providers:
            self.state.provider.set(server_provider)
            self.state.model.set(data.get("default_model") or "")
        elif self.state.model.get() not in self.state.provider_models.get(
            self.state.provider.get(), []
        ):
            models = self.state.provider_models.get(self.state.provider.get(), [])
            if models:
                self.state.model.set(data.get("default_model") or models[0])
        if not preserve_unsaved:
            provider_statuses = {
                item.get("name"): item for item in data.get("providers", [])
            }
            for provider_name, variable in {
                "openai": self.state.openai_base_url,
                "ollama": self.state.ollama_base_url,
                "lmstudio": self.state.lmstudio_base_url,
                "custom": self.state.custom_base_url,
                "horde": self.state.horde_base_url,
            }.items():
                status = provider_statuses.get(provider_name) or {}
                fallback = data.get("openai_base_url") if provider_name == "openai" else ""
                self._set_if_changed(variable, status.get("base_url") or fallback or "")
            for provider_name, variable in {
                "openai": self.state.openai_key,
                "ollama": self.state.ollama_key,
                "lmstudio": self.state.lmstudio_key,
                "custom": self.state.custom_key,
                "horde": self.state.horde_key,
                "gemini": self.state.gemini_key,
            }.items():
                status = provider_statuses.get(provider_name) or {}
                self._set_if_changed(variable, status.get("api_key") or "")
        self._provider_changed()

    def toggle_bridge(self) -> None:
        enabled = self.state.bridge_worker.get()
        self._request(
            "POST",
            "/api/bridge",
            {"enabled": enabled},
            self._apply_status,
            failure=lambda error: self._bridge_toggle_failed(not enabled, error),
        )

    def refresh_models(self) -> None:
        provider = self.state.provider.get()
        if not provider:
            return
        self._request(
            "POST",
            "/api/models/refresh",
            {"provider": provider},
            lambda data: self._apply_status(data, f"Refreshing {provider} models…"),
        )

    def test_api(self) -> None:
        if self._api_test_in_flight:
            return
        provider = self.state.provider.get()
        model = self.state.model.get()
        if not provider or not model:
            self._api_test_status.set("Select a provider and refreshed model first.")
            return
        self._api_test_in_flight = True
        self._api_test_status.set(f"Testing {provider}/{model}…")
        self._request(
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

    def _provider_changed(self, _event: object | None = None) -> None:
        models = self.state.provider_models.get(self.state.provider.get(), [])
        self.model_box["values"] = models
        if not models:
            self.state.model.set("")
        elif self.state.model.get() not in models:
            self.state.model.set(models[0])
        configured = self.state.provider_configured.get(self.state.provider.get(), False)
        self.state.provider_info.set(
            "Configured" if configured else "API key or endpoint is missing"
        )
        self._render_provider_settings()

    @staticmethod
    def _set_if_changed(variable: tk.StringVar, value: str) -> None:
        if variable.get() != value:
            variable.set(value)

    def _render_provider_settings(self) -> None:
        provider = self.state.provider.get()
        if provider == self._rendered_provider:
            return
        self._rendered_provider = provider
        title = {
            "openai": "OpenAI settings",
            "ollama": "Ollama settings (OpenAI-compatible)",
            "lmstudio": "LM Studio settings (OpenAI-compatible)",
            "custom": "Custom OpenAI-compatible settings",
            "horde": "AI Horde settings (free anonymous access)",
            "gemini": "Gemini settings",
        }.get(provider, "Provider settings")
        self._provider_settings.configure(text=title)
        for child in self._provider_settings.winfo_children():
            child.destroy()
        compatible_base_urls = {
            "openai": self.state.openai_base_url,
            "ollama": self.state.ollama_base_url,
            "lmstudio": self.state.lmstudio_base_url,
            "custom": self.state.custom_base_url,
            "horde": self.state.horde_base_url,
        }
        compatible_keys = {
            "openai": self.state.openai_key,
            "ollama": self.state.ollama_key,
            "lmstudio": self.state.lmstudio_key,
            "custom": self.state.custom_key,
            "horde": self.state.horde_key,
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
            key_label = (
                "API key (optional; anonymous by default)"
                if provider == "horde"
                else "API key"
            )
            ttk.Label(self._provider_settings, text=key_label).grid(
                row=1, column=0, sticky="w", pady=4
            )
            ttk.Entry(
                self._provider_settings,
                textvariable=compatible_keys[provider],
                width=25,
            ).grid(row=1, column=1, sticky="ew", pady=4)
        elif provider == "gemini":
            ttk.Label(self._provider_settings, text="API key").grid(
                row=0, column=0, sticky="w", pady=4
            )
            ttk.Entry(
                self._provider_settings,
                textvariable=self.state.gemini_key,
                width=25,
            ).grid(row=0, column=1, sticky="ew", pady=4)

    def _provider_request_timeout(self) -> float:
        try:
            return max(8.0, float(self.state.timeout.get()))
        except ValueError:
            return 120.0

    def _bridge_toggle_failed(self, previous_state: bool, error: Exception) -> None:
        self.state.bridge_worker.set(previous_state)
        self._request_failed(error)
