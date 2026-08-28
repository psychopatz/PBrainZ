"""Direct provider chat tab for testing P BrainZ providers."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import scrolledtext, ttk
from typing import Any

from .request import RequestFn

SelectionSaveFn = Callable[[str, str], None]


class ChatTab:
    """Own chat history, provider selection, and direct test requests."""

    def __init__(
        self,
        parent: ttk.Frame,
        request: RequestFn,
        get_timeout: Callable[[], float],
        save_selection: SelectionSaveFn | None = None,
    ) -> None:
        self._request = request
        self._get_timeout = get_timeout
        self._save_selection = save_selection
        self._provider_models: dict[str, list[str]] = {}
        self._messages: list[dict[str, str]] = []
        self._request_in_flight = False
        self._provider = tk.StringVar(parent)
        self._model = tk.StringVar(parent)
        self._view: scrolledtext.ScrolledText
        self._input: tk.Text
        self._send_button: ttk.Button
        self._selection_status = tk.StringVar(parent, value="Selection is saved automatically")
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        selector = ttk.LabelFrame(parent, text="Provider test", padding=12)
        selector.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        selector.columnconfigure(1, weight=1)
        selector.columnconfigure(3, weight=1)
        ttk.Label(selector, text="Provider").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._provider_box = ttk.Combobox(
            selector, textvariable=self._provider, state="readonly", width=22
        )
        self._provider_box.grid(row=0, column=1, sticky="ew")
        self._provider_box.bind("<<ComboboxSelected>>", self._provider_changed)
        ttk.Label(selector, text="Model").grid(row=0, column=2, sticky="w", padx=(18, 8))
        self._model_box = ttk.Combobox(
            selector, textvariable=self._model, state="readonly", width=28
        )
        self._model_box.grid(row=0, column=3, sticky="ew")
        self._model_box.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(selector, textvariable=self._selection_status).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        self._view = scrolledtext.ScrolledText(
            parent, state="disabled", wrap="word", height=18, font=("TkDefaultFont", 10)
        )
        self._view.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))

        composer = ttk.Frame(parent, padding=(18, 0, 18, 18))
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)
        self._input = tk.Text(composer, height=4, wrap="word", undo=True)
        self._input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._input.bind("<Control-Return>", self._send_from_shortcut)
        self._send_button = ttk.Button(composer, text="Send", command=self.send, width=12)
        self._send_button.grid(row=0, column=1, sticky="ns")

    @property
    def text_widgets(self) -> tuple[tk.Text, tk.Text]:
        return self._view, self._input

    def set_providers(self, models: dict[str, list[str]], preferred: str) -> None:
        self._provider_models = models
        providers = list(models)
        self._provider_box["values"] = providers
        if not self._provider.get() or self._provider.get() not in providers:
            self._provider.set(preferred)
        self._provider_changed()

    def _provider_changed(self, _event: object | None = None) -> None:
        models = self._provider_models.get(self._provider.get(), [])
        self._model_box["values"] = models
        if not models:
            self._model.set("")
        elif self._model.get() not in models:
            self._model.set(models[0])
        if _event is not None:
            self._persist_selection()

    def _model_changed(self, _event: object | None = None) -> None:
        self._persist_selection()

    def _persist_selection(self) -> None:
        provider = self._provider.get()
        model = self._model.get()
        if not self._save_selection or not provider or not model:
            return
        self._selection_status.set("Saving selection…")
        self._save_selection(provider, model)

    def set_selection_status(self, value: str) -> None:
        """Display the result of the panel's automatic selection save."""

        self._selection_status.set(value)

    def _send_from_shortcut(self, _event: object) -> str:
        self.send()
        return "break"

    def send(self) -> None:
        if self._request_in_flight:
            return
        provider = self._provider.get()
        model = self._model.get()
        message = self._input.get("1.0", "end-1c").strip()
        if not provider or not model:
            self._append("System", "Select a provider and model first.")
            return
        if not message:
            return
        self._messages.append({"role": "user", "content": message})
        self._append("You", message)
        self._input.delete("1.0", "end")
        self._request_in_flight = True
        self._send_button.configure(state="disabled")
        self._request(
            "POST",
            "/api/chat",
            {"provider": provider, "model": model, "messages": self._messages},
            self._succeeded,
            failure=self._failed,
            timeout=self._get_timeout(),
        )

    def _succeeded(self, data: dict[str, Any]) -> None:
        self._request_in_flight = False
        self._send_button.configure(state="normal")
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        if not content:
            content = "(empty response)"
        self._messages.append({"role": "assistant", "content": str(content)})
        self._append("Assistant", str(content))

    def _failed(self, error: Exception) -> None:
        self._request_in_flight = False
        self._send_button.configure(state="normal")
        self._append("System", f"Request failed: {error}")

    def _append(self, speaker: str, content: str) -> None:
        self._view.configure(state="normal")
        self._view.insert("end", f"{speaker}:\n{content}\n\n")
        self._view.configure(state="disabled")
        self._view.see("end")
