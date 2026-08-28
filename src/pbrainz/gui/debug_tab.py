"""Native browser for opt-in LLM request, prompt, tool, and response traces."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from .request import RequestFn


class DebugTab:
    """Show recent diagnostics without polling while the tab is hidden."""

    def __init__(self, parent: ttk.Frame, request: RequestFn, get_timeout) -> None:
        self._request = request
        self._get_timeout = get_timeout
        self._capture = tk.BooleanVar(parent, value=False)
        self._status = tk.StringVar(
            parent,
            value="Capture is off. No prompts or environment payloads are retained.",
        )
        self._request_in_flight = False
        self._settings_in_flight = False
        self._items: dict[str, dict[str, Any]] = {}
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(parent, text="LLM runtime trace", padding=10)
        controls.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        controls.columnconfigure(1, weight=1)
        self._capture_box = ttk.Checkbutton(
            controls,
            text="Capture full prompts, environment, tools, and returned reasoning metadata",
            variable=self._capture,
            command=self._capture_changed,
        )
        self._capture_box.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(controls, text="Refresh", command=self.refresh).grid(
            row=0, column=2, padx=(10, 0)
        )
        ttk.Button(controls, text="Clear", command=self.clear).grid(
            row=0, column=3, padx=(8, 0)
        )
        ttk.Label(controls, textvariable=self._status).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("created", "phase", "source", "npc", "request")
        self._tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse"
        )
        headings = {
            "created": "Created",
            "phase": "Phase",
            "source": "Source",
            "npc": "NPC",
            "request": "Request",
        }
        widths = {"created": 150, "phase": 150, "source": 180, "npc": 150, "request": 260}
        for column in columns:
            self._tree.heading(column, text=headings[column])
            self._tree.column(column, width=widths[column], anchor="w")
        self._tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.bind("<<TreeviewSelect>>", self._trace_selected)

        detail_frame = ttk.LabelFrame(parent, text="Selected trace payload", padding=10)
        detail_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self._detail = scrolledtext.ScrolledText(
            detail_frame,
            state="disabled",
            wrap="none",
            height=12,
            font=("TkFixedFont", 9),
        )
        self._detail.grid(row=0, column=0, sticky="nsew")

    @property
    def detail_view(self) -> tk.Text:
        return self._detail

    def refresh(self) -> None:
        if self._request_in_flight:
            return
        self._request_in_flight = True
        self._request(
            "GET",
            "/api/debug/traces?limit=300",
            None,
            self._loaded,
            failure=self._failed,
            timeout=self._get_timeout(),
        )

    def _loaded(self, data: dict[str, Any]) -> None:
        self._request_in_flight = False
        self._capture.set(bool(data.get("enabled")))
        children = self._tree.get_children()
        if children:
            self._tree.delete(*children)
        self._items = {}
        items = data.get("items") or []
        for index, item in enumerate(items):
            key = str(item.get("id") or index)
            self._items[key] = item
            self._tree.insert(
                "",
                "end",
                iid=key,
                values=(
                    item.get("created_at", ""),
                    item.get("phase", ""),
                    item.get("source", ""),
                    item.get("npc_id", "") or "—",
                    item.get("request_id", "") or "—",
                ),
            )
        state = "ON" if self._capture.get() else "OFF"
        self._status.set(
            f"Capture {state} · {len(items)} recent event(s). "
            "Reasoning is shown only when the provider returned public reasoning metadata."
        )
        if items:
            self._tree.selection_set(str(items[-1].get("id") or len(items) - 1))
            self._trace_selected()
        else:
            self._set_detail("No trace events. Enable capture, then perform an NPC request.")

    def _trace_selected(self, _event: object | None = None) -> None:
        selected = self._tree.selection()
        item = self._items.get(selected[0]) if selected else None
        self._set_detail(
            json.dumps(item, indent=2, ensure_ascii=False, default=str)
            if item
            else "Select a trace event to inspect its payload."
        )

    def _set_detail(self, content: str) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end", content)
        self._detail.configure(state="disabled")
        self._detail.see("1.0")

    def _capture_changed(self) -> None:
        if self._settings_in_flight:
            return
        enabled = self._capture.get()
        if not enabled and not messagebox.askyesno(
            "Disable trace capture",
            "Disable capture and delete all locally retained diagnostic traces?",
            parent=self._tree.winfo_toplevel(),
        ):
            self._capture.set(True)
            return
        self._settings_in_flight = True
        self._capture_box.configure(state="disabled")
        self._request(
            "POST",
            "/api/debug/traces/settings",
            {"enabled": enabled},
            self._settings_saved,
            failure=self._settings_failed,
            timeout=self._get_timeout(),
        )

    def _settings_saved(self, data: dict[str, Any]) -> None:
        self._settings_in_flight = False
        self._capture_box.configure(state="normal")
        self._capture.set(bool(data.get("enabled")))
        self.refresh()

    def _settings_failed(self, error: Exception) -> None:
        self._settings_in_flight = False
        self._capture_box.configure(state="normal")
        self._status.set(f"Could not update capture: {error}")
        self.refresh()

    def clear(self) -> None:
        self._request(
            "POST",
            "/api/debug/traces/clear",
            None,
            lambda _data: self.refresh(),
            failure=self._failed,
            timeout=self._get_timeout(),
        )

    def _failed(self, error: Exception) -> None:
        self._request_in_flight = False
        self._status.set(f"Debug trace request failed: {error}")
