"""Voice catalog controls, filtering, sorting, installation, and previews."""

from __future__ import annotations

from tkinter import messagebox, ttk
from typing import Any

from pbrainz.branding import PRODUCT_NAME


class CatalogViewMixin:
    def _build_catalog(self, outer: ttk.Frame) -> None:
        catalog = ttk.LabelFrame(outer, text="Piper voice catalog", padding=9)
        catalog.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        catalog.columnconfigure(1, weight=1)
        catalog.rowconfigure(2, weight=1)
        ttk.Label(catalog, text="Language").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._language_box = ttk.Combobox(
            catalog, textvariable=self.language, state="readonly", width=18
        )
        self._language_box.grid(row=0, column=1, sticky="w")
        self._language_box.bind("<<ComboboxSelected>>", self._language_changed)
        ttk.Label(catalog, text="Gender").grid(row=0, column=2, sticky="w", padx=(14, 5))
        self._gender_box = ttk.Combobox(
            catalog,
            textvariable=self.gender,
            values=("Any", "male", "female", "mixed", "unknown"),
            state="readonly",
            width=11,
        )
        self._gender_box.grid(row=0, column=3, sticky="w")
        self._gender_box.bind("<<ComboboxSelected>>", lambda _event: self._render_models())
        ttk.Label(catalog, text="Quality").grid(row=0, column=4, sticky="w", padx=(14, 5))
        self._quality_box = ttk.Combobox(
            catalog,
            textvariable=self.quality,
            values=("Any", "x_low", "low", "medium", "high", "unknown"),
            state="readonly",
            width=11,
        )
        self._quality_box.grid(row=0, column=5, sticky="w")
        self._quality_box.bind("<<ComboboxSelected>>", lambda _event: self._render_models())
        ttk.Label(catalog, text="Availability").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self._availability_box = ttk.Combobox(
            catalog,
            textvariable=self.availability,
            values=("All voices", "Installed", "Not installed"),
            state="readonly",
            width=13,
        )
        self._availability_box.grid(row=1, column=1, sticky="w")
        self._availability_box.bind("<<ComboboxSelected>>", lambda _event: self._render_models())
        ttk.Button(catalog, text="Refresh catalog", command=self.refresh_catalog).grid(
            row=1, column=8, sticky="e", padx=(12, 0)
        )

        self._model_tree = ttk.Treeview(
            catalog,
            columns=("voice", "language", "gender", "quality", "size", "status"),
            show="headings",
            selectmode="browse",
            height=9,
        )
        headings = {
            "voice": ("Voice", 235),
            "language": ("Language", 125),
            "gender": ("Gender", 85),
            "quality": ("Quality", 85),
            "size": ("Download", 90),
            "status": ("Status", 105),
        }
        self._sort_heading_labels = {
            column: heading for column, (heading, _width) in headings.items()
        }
        for column, (heading, width) in headings.items():
            self._model_tree.heading(
                column,
                text=heading,
                command=lambda selected_column=column: self._sort_models(selected_column),
            )
            self._model_tree.column(column, width=width, anchor="w", stretch=column == "voice")
        self._update_sort_headings()
        self._model_tree.grid(row=2, column=0, columnspan=9, sticky="nsew", pady=(8, 5))
        scrollbar = ttk.Scrollbar(catalog, orient="vertical", command=self._model_tree.yview)
        scrollbar.grid(row=2, column=9, sticky="ns", pady=(8, 5))
        self._model_tree.configure(yscrollcommand=scrollbar.set)
        self._model_tree.bind("<<TreeviewSelect>>", self._selected_model_changed)

        catalog_actions = ttk.Frame(catalog)
        catalog_actions.grid(row=3, column=0, columnspan=9, sticky="ew")
        catalog_actions.columnconfigure(0, weight=1)
        ttk.Label(catalog_actions, textvariable=self.catalog_status).grid(
            row=0, column=0, sticky="w"
        )
        self._install_button = ttk.Button(
            catalog_actions,
            text="Install selected voice",
            command=self._install_selected,
            state="disabled",
        )
        self._install_button.grid(row=0, column=2, sticky="e")
        self._uninstall_button = ttk.Button(
            catalog_actions,
            text="Uninstall selected voice",
            command=self._uninstall_selected,
            state="disabled",
        )
        self._uninstall_button.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self._preview_button = ttk.Button(
            catalog_actions,
            text="Test selected voice",
            command=self._preview_selected,
            state="disabled",
        )
        self._preview_button.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self._install_progress = ttk.Progressbar(
            catalog_actions, mode="determinate", maximum=100, length=220
        )
        self._install_progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self._install_progress.grid_remove()
        self._default_install_button = ttk.Button(
            catalog_actions,
            text="Download default voices",
            command=self._download_defaults,
        )
        self._default_install_button.grid(row=0, column=4, sticky="e", padx=(8, 0))

    def _language_changed(self, _event: object | None = None) -> None:
        """Rerender immediately and persist this lightweight filter preference."""

        self._render_models()
        if self._applying:
            return
        self._request(
            "POST",
            "/api/tts/settings",
            {"catalog_language": self.language.get()},
            lambda _data: None,
            failure=lambda error: self.status.set(f"Language preference failed: {error}"),
        )

    def _render_models(self) -> None:
        selected = self._model_tree.selection()
        selected_id = selected[0] if selected else None
        for item in self._model_tree.get_children():
            self._model_tree.delete(item)
        language = self.language.get().casefold().strip()
        gender = self.gender.get().casefold().strip()
        quality = self.quality.get().casefold().strip()
        availability = self.availability.get().casefold().strip()
        models = []
        for model in self._models:
            model_language = str(model.get("language") or "unknown").casefold()
            model_locale = str(model.get("locale") or "unknown").casefold()
            model_gender = str(model.get("gender") or "unknown").casefold()
            model_quality = str(model.get("quality") or "unknown").casefold()
            installed = bool(model.get("installed"))
            if language not in {"", "all languages"} and language not in {
                model_language,
                model_locale,
            }:
                continue
            if gender not in {"", "any", model_gender}:
                continue
            if quality not in {"", "any", model_quality}:
                continue
            if availability == "installed" and not installed:
                continue
            if availability == "not installed" and installed:
                continue
            models.append(model)
        inserted_ids = set()
        for model in sorted(models, key=self._model_sort_key, reverse=self._sort_descending):
            model_id = str(model.get("id") or "")
            if not model_id:
                continue
            display_name = str(model.get("display_name") or model_id)
            if display_name != model_id:
                display_name = f"{display_name} ({model_id})"
            self._model_tree.insert(
                "",
                "end",
                iid=model_id,
                values=(
                    display_name,
                    str(model.get("language") or model.get("locale") or "unknown"),
                    str(model.get("gender") or "unknown"),
                    str(model.get("quality") or "unknown"),
                    _format_size(model.get("model_size_bytes")),
                    "Installed" if model.get("installed") else "Not installed",
                ),
            )
            inserted_ids.add(model_id)
        if selected_id in inserted_ids:
            self._model_tree.selection_set(selected_id)
            self._model_tree.focus(selected_id)
            self._model_tree.see(selected_id)
        self._selected_model_changed()

    def _sort_models(self, column: str) -> None:
        if column not in self._sort_heading_labels:
            return
        if column == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = False
        self._update_sort_headings()
        self._render_models()

    def _update_sort_headings(self) -> None:
        if not hasattr(self, "_model_tree"):
            return
        for column, heading in self._sort_heading_labels.items():
            indicator = " ▼" if self._sort_descending else " ▲"
            self._model_tree.heading(
                column,
                text=f"{heading}{indicator}" if column == self._sort_column else heading,
            )

    def _model_sort_key(self, model: dict[str, Any]) -> tuple[Any, ...]:
        column = self._sort_column
        model_id = str(model.get("id") or "").casefold()
        if column == "language":
            return (
                str(model.get("language") or model.get("locale") or "unknown").casefold(),
                str(model.get("locale") or "unknown").casefold(),
                model_id,
            )
        if column == "gender":
            return (str(model.get("gender") or "unknown").casefold(), model_id)
        if column == "quality":
            quality = str(model.get("quality") or "unknown").casefold()
            quality_rank = {"x_low": 0, "low": 1, "medium": 2, "high": 3}.get(quality, 4)
            return (quality_rank, quality, model_id)
        if column == "size":
            try:
                size = int(model.get("model_size_bytes") or 0)
            except (TypeError, ValueError):
                size = 0
            return (size, model_id)
        if column == "status":
            return (0 if model.get("installed") else 1, model_id)
        return (str(model.get("display_name") or model_id).casefold(), model_id)

    def _selected_model_changed(self, _event: object | None = None) -> None:
        selected = self._model_tree.selection()
        if not selected:
            self._preview_button.configure(state="disabled", text="Test selected voice")
            self._install_button.configure(state="disabled", text="Install selected voice")
            self._uninstall_button.configure(state="disabled", text="Uninstall selected voice")
            return
        model = self._model_by_id.get(selected[0])
        installed = bool(model and model.get("installed"))
        downloadable = bool(model and model.get("available_online"))
        previewable = bool(model and model.get("preview_available"))
        self._preview_button.configure(
            state="normal" if previewable else "disabled",
            text="Test selected voice",
        )
        if self._automatic_install_in_flight:
            self._install_button.configure(state="disabled", text="Installing defaults…")
            self._uninstall_button.configure(state="disabled", text="Installing defaults…")
            return
        if (
            self._install_job_id
            or self._install_request_in_flight
            or self._uninstall_request_in_flight
        ):
            self._install_button.configure(state="disabled", text="Installing…")
            self._uninstall_button.configure(state="disabled", text="Working…")
            return
        if installed:
            self._install_button.configure(state="disabled", text="Already installed")
            self._uninstall_button.configure(state="normal", text="Uninstall selected voice")
        elif downloadable:
            self._install_button.configure(state="normal", text="Install selected voice")
            self._uninstall_button.configure(state="disabled", text="Not installed")
        else:
            self._install_button.configure(state="disabled", text="No download available")
            self._uninstall_button.configure(state="disabled", text="Not installed")

    def _sync_default_install(self, data: dict[str, Any]) -> None:
        """Show, poll, and prompt for the service-owned default setup job."""

        install = data.get("default_voice_install") or {}
        state = str(install.get("state") or "")
        if state in {"preparing", "queued", "installing"}:
            self._automatic_install_in_flight = True
            self._install_model_id = str(install.get("voice_model_id") or "")
            self._show_install_progress(install)
            self._default_install_button.configure(
                state="disabled", text="Installing defaults…"
            )
            self._queue_default_install_poll()
            return
        if state in {"failed", "cancelled"}:
            self._clear_default_install_display()
            error = str(install.get("error") or "Default voice download failed")
            self.catalog_status.set(
                f"Default voice download failed: {error}. Retry is available."
            )
            self._default_install_button.configure(state="normal", text="Retry default voices")
            return
        if state == "complete":
            self._clear_default_install_display()
            self._default_install_button.configure(
                state="disabled", text="Default voices installed"
            )
            return
        if data.get("default_voice_setup_needed"):
            self._default_install_button.configure(state="normal", text="Download default voices")
            if (
                data.get("enabled")
                and not int(data.get("catalog_installed") or 0)
                and not self._default_install_prompted
                and not self._install_job_id
                and not self._install_request_in_flight
                and not self._uninstall_request_in_flight
            ):
                self._default_install_prompted = True
                self.parent.after(50, self._confirm_default_install)
            return
        self._clear_default_install_display()
        self._default_install_button.configure(state="disabled", text="Default voices installed")

    def _confirm_default_install(self) -> None:
        if (
            self._default_install_request_in_flight
            or self._automatic_install_in_flight
            or self._install_job_id
            or self._install_request_in_flight
            or self._uninstall_request_in_flight
        ):
            return
        if not messagebox.askyesno(
            PRODUCT_NAME,
            "There are no models found.\n\nDownload defaults now?",
            parent=self.parent.winfo_toplevel(),
        ):
            self.catalog_status.set(
                "Default voice download skipped. Use Download default voices to retry."
            )
            return
        self._start_default_install()

    def _download_defaults(self) -> None:
        if (
            self._default_install_request_in_flight
            or self._automatic_install_in_flight
            or self._install_job_id
            or self._install_request_in_flight
            or self._uninstall_request_in_flight
        ):
            return
        self._confirm_default_install()

    def _start_default_install(self) -> None:
        self._default_install_request_in_flight = True
        self._default_install_button.configure(state="disabled", text="Starting defaults…")
        self.catalog_status.set("Starting default voice download…")
        self._request(
            "POST",
            "/api/tts/defaults/install",
            None,
            self._default_install_started,
            failure=self._default_install_start_failed,
            timeout=30,
        )

    def _default_install_started(self, data: dict[str, Any]) -> None:
        self._default_install_request_in_flight = False
        self._default_install_prompted = True
        self.apply_status(data)

    def _default_install_start_failed(self, error: Exception) -> None:
        self._default_install_request_in_flight = False
        self._clear_default_install_display()
        self.catalog_status.set(
            f"Default voice download could not start: {error}. Retry is available."
        )
        self._default_install_button.configure(state="normal", text="Retry default voices")

    def _clear_default_install_display(self) -> None:
        self._automatic_install_in_flight = False
        if self._default_install_refresh_after is not None:
            self.parent.after_cancel(self._default_install_refresh_after)
        self._default_install_refresh_after = None
        self._install_model_id = None
        self._install_progress.stop()
        self._install_progress.grid_remove()

    def _queue_default_install_poll(self, delay: int = 500) -> None:
        if self._default_install_refresh_after is not None:
            self.parent.after_cancel(self._default_install_refresh_after)
        self._default_install_refresh_after = self.parent.after(delay, self._poll_default_install)

    def _poll_default_install(self) -> None:
        self._default_install_refresh_after = None
        if not self._automatic_install_in_flight:
            return
        self._request(
            "GET",
            "/api/tts",
            None,
            self.apply_status,
            failure=self._default_install_poll_failed,
            timeout=8,
        )

    def _default_install_poll_failed(self, error: Exception) -> None:
        if not self._automatic_install_in_flight:
            return
        self.catalog_status.set(
            f"Default voice install progress unavailable; retrying… ({error})"
        )
        self._queue_default_install_poll(1000)

    def refresh(self) -> None:
        self._request("GET", "/api/tts", None, self.apply_status)

    def refresh_catalog(self) -> None:
        self.catalog_status.set("Refreshing official Piper voice catalog…")
        self._request(
            "GET",
            "/api/tts?refresh_catalog=true",
            None,
            self.apply_status,
            failure=self._catalog_refresh_failed,
            timeout=30,
        )

    def _catalog_refresh_failed(self, error: Exception) -> None:
        self.catalog_status.set(f"Voice catalog refresh failed: {error}")

    def _install_selected(self) -> None:
        if (
            self._install_job_id
            or self._install_request_in_flight
            or self._uninstall_request_in_flight
            or self._automatic_install_in_flight
        ):
            return
        selected = self._model_tree.selection()
        if not selected:
            return
        model_id = selected[0]
        model = self._model_by_id.get(model_id)
        if not model or model.get("installed"):
            return
        self._install_model_id = model_id
        self._install_request_in_flight = True
        self._install_poll_failures = 0
        self._show_install_progress(
            {
                "stage": "Starting installation",
                "completed_bytes": 0,
                "total_bytes": int(model.get("model_size_bytes") or 0)
                + int(model.get("config_size_bytes") or 0),
                "progress": 0,
            }
        )
        self._install_button.configure(state="disabled", text="Installing…")
        self.catalog_status.set(f"Downloading {model.get('display_name') or model_id}…")
        self._request(
            "POST",
            "/api/tts/voices/install",
            {"voice_model_id": model_id},
            self._install_started,
            failure=self._install_start_failed,
            timeout=600,
        )

    def _uninstall_selected(self) -> None:
        if (
            self._install_job_id
            or self._install_request_in_flight
            or self._uninstall_request_in_flight
            or self._automatic_install_in_flight
        ):
            return
        selected = self._model_tree.selection()
        if not selected:
            return
        model_id = selected[0]
        model = self._model_by_id.get(model_id)
        if not model or not model.get("installed"):
            return
        display_name = str(model.get("display_name") or model_id)
        if not messagebox.askyesno(
            PRODUCT_NAME,
            (
                f"Uninstall {display_name} ({model_id})?\n\n"
                "Any presets using this voice will be cleared."
            ),
            parent=self.parent.winfo_toplevel(),
        ):
            return
        self._uninstall_request_in_flight = True
        self._install_button.configure(state="disabled", text="Working…")
        self._uninstall_button.configure(state="disabled", text="Uninstalling…")
        self.catalog_status.set(f"Uninstalling {display_name}…")
        self._request(
            "POST",
            "/api/tts/voices/uninstall",
            {"voice_model_id": model_id},
            self._uninstall_completed,
            failure=self._uninstall_failed,
            timeout=45,
        )

    def _uninstall_completed(self, data: dict[str, Any]) -> None:
        self._uninstall_request_in_flight = False
        voice = data.get("voice") or {}
        display_name = str(voice.get("display_name") or "selected voice")
        self.apply_status(data)
        self.catalog_status.set(f"Uninstalled {display_name}. {self.catalog_status.get()}")

    def _uninstall_failed(self, error: Exception) -> None:
        self._uninstall_request_in_flight = False
        self._selected_model_changed()
        self.catalog_status.set(f"Voice uninstall failed: {error}")

    def _install_started(self, data: dict[str, Any]) -> None:
        self._install_request_in_flight = False
        install = data.get("install") or {}
        state = str(install.get("state") or "")
        if state == "complete":
            self._install_completed(data)
            return
        job_id = str(install.get("job_id") or "")
        if not job_id:
            self._install_start_failed(RuntimeError("server did not return an install job"))
            return
        self._install_job_id = job_id
        self._show_install_progress(install)
        self._queue_install_poll()

    def _queue_install_poll(self, delay: int = 250) -> None:
        if self._install_poll_after is not None:
            self.parent.after_cancel(self._install_poll_after)
        self._install_poll_after = self.parent.after(delay, self._poll_install)

    def _poll_install(self) -> None:
        self._install_poll_after = None
        if not self._install_job_id:
            return
        self._request(
            "GET",
            f"/api/tts/voices/install/{self._install_job_id}",
            None,
            self._install_status_received,
            failure=self._install_poll_failed,
            timeout=8,
        )

    def _install_status_received(self, data: dict[str, Any]) -> None:
        if not self._install_job_id:
            return
        install = data.get("install") or {}
        if str(install.get("job_id") or "") != self._install_job_id:
            return
        self._install_poll_failures = 0
        self._show_install_progress(install)
        state = str(install.get("state") or "")
        if state in {"queued", "installing"}:
            self._queue_install_poll()
        elif state == "complete":
            self._install_completed(data)
        else:
            self._install_job_failed(str(install.get("error") or "Installation failed"))

    def _install_poll_failed(self, error: Exception) -> None:
        if not self._install_job_id:
            return
        self._install_poll_failures += 1
        if self._install_poll_failures <= 5:
            self.catalog_status.set(f"Install progress unavailable; retrying… ({error})")
            self._queue_install_poll(1000)
            return
        self._install_job_failed(f"Could not read install progress: {error}")

    def _install_completed(self, data: dict[str, Any]) -> None:
        install = data.get("install") or {}
        voice = install.get("voice") or {}
        display_name = str(voice.get("display_name") or self._install_model_id or "voice")
        self._show_install_progress({**install, "progress": 100, "stage": "Installed"})
        self._clear_install_state()
        self.apply_status(data)
        self.catalog_status.set(f"Installed {display_name}. {self.catalog_status.get()}")

    def _install_start_failed(self, error: Exception) -> None:
        self._clear_install_state()
        self._selected_model_changed()
        self.catalog_status.set(f"Voice installation failed: {error}")

    def _install_job_failed(self, error: str) -> None:
        self._clear_install_state()
        self._selected_model_changed()
        self.catalog_status.set(f"Voice installation failed: {error}")

    def _show_install_progress(self, install: dict[str, Any]) -> None:
        completed = max(0, int(install.get("completed_bytes") or 0))
        total = max(0, int(install.get("total_bytes") or 0))
        if total:
            self._install_progress.stop()
            self._install_progress.configure(mode="determinate", maximum=total, value=completed)
            percent = min(100.0, completed * 100.0 / total)
            progress_text = f"{percent:.0f}%"
        else:
            self._install_progress.configure(mode="indeterminate")
            self._install_progress.start(10)
            progress_text = "working"
        self._install_progress.grid()
        stage = str(install.get("stage") or "Installing")
        model = self._model_by_id.get(self._install_model_id or "")
        model_name = str((model or {}).get("display_name") or self._install_model_id or "voice")
        self.catalog_status.set(f"{stage}: {model_name} · {progress_text}")

    def _clear_install_state(self) -> None:
        if self._install_poll_after is not None:
            self.parent.after_cancel(self._install_poll_after)
        self._install_poll_after = None
        self._install_job_id = None
        self._install_model_id = None
        self._install_request_in_flight = False
        self._install_poll_failures = 0
        self._install_progress.stop()
        self._install_progress.grid_remove()

    def _preview_selected(self) -> None:
        selected = self._model_tree.selection()
        if not selected:
            return
        model_id = selected[0]
        model = self._model_by_id.get(model_id)
        if not model or not model.get("preview_available"):
            return
        self._preview_button.configure(state="disabled", text="Loading sample…")
        self._request(
            "POST",
            "/api/tts/voices/preview",
            {
                "voice_model_id": model_id,
                **self._test_audio_presentation_payload(),
            },
            self._preview_started,
            failure=self._preview_failed,
            timeout=45,
        )

    def _preview_started(self, data: dict[str, Any]) -> None:
        # The preview endpoint returns only the queued voice result, not a full
        # catalog snapshot. Keep the current tree and selection intact.
        self._selected_model_changed()
        voice = data.get("voice", {})
        self.status.set(
            f"Playing {self.test_effect.get()} sample for "
            f"{voice.get('display_name') or 'selected voice'}."
        )

    def _preview_failed(self, error: Exception) -> None:
        self._selected_model_changed()
        self.status.set(f"Voice sample failed: {error}")

def _format_size(value: object) -> str:
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        return "unknown"
    if size <= 0:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return "unknown"
