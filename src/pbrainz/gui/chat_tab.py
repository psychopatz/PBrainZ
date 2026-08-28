"""Direct provider chat tab for testing P BrainZ providers."""

from __future__ import annotations

import tkinter as tk
import uuid
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
        self._mock_mode = tk.BooleanVar(parent, value=False)
        self._mock_world = tk.StringVar(parent, value="pbrainz-mock-world")
        self._mock_player = tk.StringVar(parent, value="mock-player")
        self._mock_npc = tk.StringVar(parent, value="mock-npc")
        self._mock_day = tk.StringVar(parent, value="1")
        self._mock_session_id = f"mock-session:{uuid.uuid4().hex}"
        self._mock_status = tk.StringVar(
            parent, value="Seed a fixture, then ask a memory question."
        )
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
        selector.columnconfigure(5, weight=1)
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
        self._mock_toggle = ttk.Checkbutton(
            selector,
            text="Use structured NPC/RAG mock",
            variable=self._mock_mode,
            command=self._mock_mode_changed,
        )
        self._mock_toggle.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(selector, textvariable=self._selection_status).grid(
            row=1, column=4, columnspan=2, sticky="e", pady=(8, 0)
        )

        self._mock_frame = ttk.LabelFrame(selector, text="Mock save context", padding=8)
        self._mock_frame.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        for column in (1, 3, 5):
            self._mock_frame.columnconfigure(column, weight=1)
        ttk.Label(self._mock_frame, text="World").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Entry(self._mock_frame, textvariable=self._mock_world, width=22).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(self._mock_frame, text="Player").grid(
            row=0, column=2, sticky="w", padx=(12, 6)
        )
        ttk.Entry(self._mock_frame, textvariable=self._mock_player, width=18).grid(
            row=0, column=3, sticky="ew"
        )
        ttk.Label(self._mock_frame, text="NPC").grid(
            row=0, column=4, sticky="w", padx=(12, 6)
        )
        ttk.Entry(self._mock_frame, textvariable=self._mock_npc, width=18).grid(
            row=0, column=5, sticky="ew"
        )
        ttk.Label(self._mock_frame, text="Game day").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(self._mock_frame, textvariable=self._mock_day, width=8).grid(
            row=1, column=1, sticky="w", pady=(8, 0)
        )
        self._seed_button = ttk.Button(
            self._mock_frame, text="Seed sample memories", command=self.seed_mock_memories
        )
        self._seed_button.grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(8, 0)
        )
        ttk.Button(
            self._mock_frame, text="Reset mock session", command=self.reset_mock_session
        ).grid(row=1, column=4, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Label(self._mock_frame, textvariable=self._mock_status).grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )
        self._mock_frame.grid_remove()

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

    def _mock_mode_changed(self) -> None:
        if self._mock_mode.get():
            self._mock_frame.grid()
            self._mock_status.set("Seed a fixture, then ask a memory question.")
        else:
            self._mock_frame.grid_remove()

    def reset_mock_session(self) -> None:
        self._mock_session_id = f"mock-session:{uuid.uuid4().hex}"
        self._mock_status.set("New isolated mock session ready.")

    def seed_mock_memories(self) -> None:
        if self._request_in_flight:
            return
        try:
            game_day = int(self._mock_day.get().strip())
        except ValueError:
            self._mock_status.set("Game day must be a whole number.")
            return
        self._seed_button.configure(state="disabled")
        self._mock_status.set("Seeding sample memories…")
        self._request(
            "POST",
            "/api/mock-chat/seed",
            {
                "world_uuid": self._mock_world.get().strip(),
                "player_uuid": self._mock_player.get().strip(),
                "npc_uuid": self._mock_npc.get().strip(),
                "game_day": game_day,
            },
            self._mock_seed_succeeded,
            failure=self._mock_seed_failed,
            timeout=self._get_timeout(),
        )

    def _mock_seed_succeeded(self, data: dict[str, Any]) -> None:
        self._seed_button.configure(state="normal")
        self.reset_mock_session()
        stats = data.get("stats") or {}
        self._mock_status.set(
            "Seeded sample memories "
            f"({stats.get('memory_count', 0)} memories, "
            f"{stats.get('episode_count', 0)} episodes, "
            f"{stats.get('fact_count', 0)} facts)."
        )

    def _mock_seed_failed(self, error: Exception) -> None:
        self._seed_button.configure(state="normal")
        self._mock_status.set(f"Seed failed: {error}")

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
        self._append("You", message)
        self._input.delete("1.0", "end")
        self._request_in_flight = True
        self._send_button.configure(state="disabled")
        if self._mock_mode.get():
            try:
                game_day = int(self._mock_day.get().strip())
            except ValueError:
                self._request_in_flight = False
                self._send_button.configure(state="normal")
                self._append("System", "Mock game day must be a whole number.")
                return
            self._request(
                "POST",
                "/api/mock-chat",
                {
                    "provider": provider,
                    "model": model,
                    "message": message,
                    "world_uuid": self._mock_world.get().strip(),
                    "player_uuid": self._mock_player.get().strip(),
                    "npc_uuid": self._mock_npc.get().strip(),
                    "session_id": self._mock_session_id,
                    "game_day": game_day,
                },
                self._mock_succeeded,
                failure=self._failed,
                timeout=self._get_timeout(),
            )
            return
        self._messages.append({"role": "user", "content": message})
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

    def _mock_succeeded(self, data: dict[str, Any]) -> None:
        self._request_in_flight = False
        self._send_button.configure(state="normal")
        content = data.get("response_text") or ""
        if not content:
            choices = data.get("choices") or []
            content = (
                ((choices[0].get("message") or {}).get("content") if choices else "")
                or ""
            )
        self._append("Mock NPC", str(content) or "(empty response)")
        retrieved = data.get("retrieved_memories") or []
        diagnostics = data.get("diagnostics") or {}
        ids = ", ".join(str(item.get("memory_id")) for item in retrieved[:6]) or "none"
        self._append(
            "RAG",
            f"retrieval_needed={diagnostics.get('retrieval_needed', False)}; "
            f"records={len(retrieved)}; ids={ids}",
        )
        self._mock_status.set(
            f"Mock response complete · {len(retrieved)} retrieved record(s)"
        )

    def _append(self, speaker: str, content: str) -> None:
        self._view.configure(state="normal")
        self._view.insert("end", f"{speaker}:\n{content}\n\n")
        self._view.configure(state="disabled")
        self._view.see("end")
