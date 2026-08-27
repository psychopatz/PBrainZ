"""Command-line entry point for the HoomansLLM server."""

import argparse
import json
import threading

import uvicorn

from .config import get_settings
from .database import DEFAULT_ACTIVITY_LIMIT, MAX_ACTIVITY_LIMIT, SettingsDatabase


def main() -> None:
    """Run the API server and optional native control panel."""
    arguments = _parse_args()
    if arguments.activity is not None or arguments.activity_json:
        activity_limit = (
            arguments.activity
            if arguments.activity is not None
            else DEFAULT_ACTIVITY_LIMIT
        )
        _show_activity(
            activity_limit,
            as_json=arguments.activity_json,
        )
        return

    # Keep the activity command lightweight: the FastAPI application and its
    # provider imports are only needed when actually starting the server.
    from .app import app

    settings = get_settings()
    if not settings.open_gui:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
        return

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
    )
    server_thread = threading.Thread(target=server.run, name="hoomans-llm-server", daemon=True)
    server_thread.start()
    try:
        from .desktop_gui import HoomansLLMControlPanel

        HoomansLLMControlPanel(settings.host, settings.port).run()
    except Exception as error:
        print(f"Native GUI is unavailable ({error}); HoomansLLM API remains running.")
        server_thread.join()
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HoomansLLM local server.")
    parser.add_argument(
        "--activity",
        "--recent-activity",
        "--logs",
        nargs="?",
        const=DEFAULT_ACTIVITY_LIMIT,
        type=int,
        metavar="LIMIT",
        help=(
            "Print recent activity and exit. LIMIT defaults to 50 and is capped at "
            f"{MAX_ACTIVITY_LIMIT}."
        ),
    )
    parser.add_argument(
        "--activity-json",
        action="store_true",
        help="Print recent activity as JSON instead of human-readable lines.",
    )
    return parser.parse_args()


def _show_activity(limit: int, *, as_json: bool = False) -> None:
    bounded_limit = max(1, min(limit, MAX_ACTIVITY_LIMIT))
    settings = get_settings()
    database = SettingsDatabase(settings.database_path)
    database.initialize()
    entries = database.recent_logs(bounded_limit)
    if as_json:
        print(json.dumps({"entries": entries}, ensure_ascii=False))
        return
    if not entries:
        print("No recent HoomansLLM activity.")
        return
    for entry in entries:
        print(
            f"{entry['created_at']} [{entry['level']}] "
            f"{entry['source']}: {entry['message']}"
        )


if __name__ == "__main__":
    main()
