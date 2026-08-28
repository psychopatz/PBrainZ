"""Native browser for the bounded, save-scoped memory layers."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

RequestFn = Callable[..., None]


class MemoryTab:
    """Browse and delete reusable memory records without loading transcripts."""

    def __init__(
        self,
        parent: ttk.Frame,
        request: RequestFn,
        get_timeout: Callable[[], float],
    ) -> None:
        self._request = request
        self._get_timeout = get_timeout
        self._world = tk.StringVar(parent)
        self._search = tk.StringVar(parent)
        self._kind = tk.StringVar(parent, value="all")
        self._status = tk.StringVar(parent, value="No memory world selected")
        self._worlds: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._world_request_in_flight = False
        self._records_request_in_flight = False
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(parent, text="Saved memory browser", padding=10)
        controls.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(5, weight=1)
        ttk.Label(controls, text="Save/world").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._world_box = ttk.Combobox(
            controls, textvariable=self._world, state="readonly", width=28
        )
        self._world_box.grid(row=0, column=1, sticky="ew")
        self._world_box.bind("<<ComboboxSelected>>", self._world_changed)
        ttk.Label(controls, text="Layer").grid(row=0, column=2, sticky="w", padx=(16, 8))
        self._kind_box = ttk.Combobox(
            controls,
            textvariable=self._kind,
            state="readonly",
            width=16,
            values=("all", "memory", "episode", "fact", "day_synopsis"),
        )
        self._kind_box.grid(row=0, column=3, sticky="w")
        self._kind_box.bind("<<ComboboxSelected>>", self._filter_changed)
        ttk.Label(controls, text="Search").grid(row=0, column=4, sticky="e", padx=(16, 8))
        search_box = ttk.Entry(controls, textvariable=self._search)
        search_box.grid(row=0, column=5, sticky="ew")
        search_box.bind("<Return>", self._search_submitted)
        ttk.Button(controls, text="Search", command=self.load_records).grid(
            row=0, column=6, padx=(8, 0)
        )
        ttk.Button(controls, text="Refresh", command=self.refresh).grid(
            row=0, column=7, padx=(8, 0)
        )
        ttk.Label(controls, textvariable=self._status).grid(
            row=1, column=0, columnspan=8, sticky="w", pady=(8, 0)
        )

        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("kind", "type", "day", "npc", "visibility", "importance", "preview")
        self._tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse"
        )
        headings = {
            "kind": "Layer",
            "type": "Type",
            "day": "Day",
            "npc": "NPC",
            "visibility": "Visibility",
            "importance": "Weight",
            "preview": "Saved content",
        }
        widths = {
            "kind": 100,
            "type": 140,
            "day": 55,
            "npc": 140,
            "visibility": 90,
            "importance": 65,
        }
        for column in columns:
            self._tree.heading(column, text=headings[column])
            self._tree.column(column, width=widths.get(column, 360), anchor="w")
        self._tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self._tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self._tree.bind("<<TreeviewSelect>>", self._record_selected)

        detail_frame = ttk.LabelFrame(parent, text="Selected record", padding=10)
        detail_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self._detail = scrolledtext.ScrolledText(
            detail_frame, state="disabled", wrap="word", height=8, font=("TkDefaultFont", 10)
        )
        self._detail.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        actions = ttk.Frame(detail_frame)
        actions.grid(row=0, column=1, sticky="ns")
        self._delete_button = ttk.Button(
            actions,
            text="Delete selected",
            command=self.delete_selected,
            state="disabled",
            width=16,
        )
        self._delete_button.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            actions,
            text=(
                "Raw conversation turns are kept separately from these saved layers.\n"
                "They are the transcript source used when consolidation runs."
            ),
            justify="left",
        ).grid(row=1, column=0, sticky="nw", pady=(14, 0))

    @property
    def detail_view(self) -> tk.Text:
        return self._detail

    def refresh(self) -> None:
        if self._world_request_in_flight:
            return
        self._world_request_in_flight = True
        self._request(
            "GET",
            "/api/memory/worlds",
            None,
            self._worlds_loaded,
            failure=self._request_failed,
            timeout=self._get_timeout(),
        )

    def _worlds_loaded(self, data: dict[str, Any]) -> None:
        self._world_request_in_flight = False
        worlds = [item for item in data.get("worlds", []) if item.get("world_uuid")]
        self._worlds = {str(item["world_uuid"]): item for item in worlds}
        values = list(self._worlds)
        self._world_box["values"] = values
        if self._world.get() not in self._worlds:
            self._world.set(values[0] if values else "")
        self.load_records()

    def _world_changed(self, _event: object | None = None) -> None:
        self.load_records()

    def _filter_changed(self, _event: object | None = None) -> None:
        self.load_records()

    def _search_submitted(self, _event: object) -> str:
        self.load_records()
        return "break"

    def load_records(self) -> None:
        if self._records_request_in_flight:
            return
        world_uuid = self._world.get().strip()
        if not world_uuid:
            self._clear_records("No saved memory worlds found")
            return
        self._records_request_in_flight = True
        self._request(
            "GET",
            "/api/memory"
            f"?world_uuid={_quote(world_uuid)}&limit=100&search={_quote(self._search.get())}"
            f"&record_kind={_quote(self._kind.get())}",
            None,
            self._records_loaded,
            failure=self._request_failed,
            timeout=self._get_timeout(),
        )

    def _records_loaded(self, data: dict[str, Any]) -> None:
        self._records_request_in_flight = False
        children = self._tree.get_children()
        if children:
            self._tree.delete(*children)
        self._records = {}
        for index, record in enumerate(data.get("items", [])):
            key = f"{record.get('record_kind', '')}:{record.get('record_id', '')}:{index}"
            self._records[key] = record
            self._tree.insert(
                "",
                "end",
                iid=key,
                values=(
                    record.get("record_kind", ""),
                    record.get("record_type", ""),
                    record.get("game_day") if record.get("game_day") is not None else "—",
                    record.get("npc_uuid", ""),
                    record.get("visibility", ""),
                    f"{float(record.get('importance', 0)):.2f}",
                    record.get("preview", ""),
                ),
            )
        total = int(data.get("total", len(data.get("items", []))))
        world_info = self._worlds.get(self._world.get(), {})
        raw_turns = int(world_info.get("turn_count") or 0)
        sessions = int(world_info.get("session_count") or 0)
        status = (
            f"{total} saved layer record(s) · showing {len(data.get('items', []))}"
            f" · raw transcript turns: {raw_turns} · sessions: {sessions}"
        )
        if total == 0 and raw_turns:
            status += " · durable layers appear when consolidation runs"
        self._status.set(status)
        self._clear_detail()

    def _record_selected(self, _event: object | None = None) -> None:
        selection = self._tree.selection()
        record = self._records.get(selection[0]) if selection else None
        if record is None:
            self._clear_detail()
            return
        lines = [
            f"Layer: {record.get('record_kind', '')}",
            f"Type: {record.get('record_type', '')}",
            f"World: {record.get('world_uuid', '')}",
            f"Player: {record.get('player_uuid', '')}",
            f"NPC: {record.get('npc_uuid', '')}",
            f"Game day: {record.get('game_day') if record.get('game_day') is not None else '—'}",
            f"Visibility: {record.get('visibility', '')}",
            f"Importance: {float(record.get('importance', 0)):.2f}",
            f"Created: {record.get('created_at', '')}",
            f"Updated: {record.get('updated_at', '')}",
            f"ID: {record.get('record_id', '')}",
            "",
            record.get("content", ""),
        ]
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", "\n".join(str(line) for line in lines))
        self._detail.configure(state="disabled")
        self._delete_button.configure(state="normal")

    def delete_selected(self) -> None:
        selection = self._tree.selection()
        record = self._records.get(selection[0]) if selection else None
        if record is None:
            return
        confirmed = messagebox.askyesno(
            "Delete saved memory",
            "Permanently delete this saved memory-layer record?",
            parent=self._tree.winfo_toplevel(),
        )
        if not confirmed:
            return
        self._delete_button.configure(state="disabled")
        self._request(
            "POST",
            "/api/memory/delete",
            {
                "world_uuid": record.get("world_uuid"),
                "record_kind": record.get("record_kind"),
                "record_id": record.get("record_id"),
                "player_uuid": record.get("player_uuid"),
                "npc_uuid": record.get("npc_uuid"),
                "game_day": record.get("game_day"),
            },
            lambda _data: self.load_records(),
            failure=self._request_failed,
            timeout=self._get_timeout(),
        )

    def _clear_records(self, status: str) -> None:
        self._records_request_in_flight = False
        children = self._tree.get_children()
        if children:
            self._tree.delete(*children)
        self._records = {}
        self._status.set(status)
        self._clear_detail()

    def _clear_detail(self) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.configure(state="disabled")
        self._delete_button.configure(state="disabled")

    def _request_failed(self, error: Exception) -> None:
        self._world_request_in_flight = False
        self._records_request_in_flight = False
        self._status.set(f"Memory request failed: {error}")


def _quote(value: object) -> str:
    from urllib.parse import quote

    return quote(str(value or ""), safe="")
