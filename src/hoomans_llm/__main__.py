"""Command-line entry point for the HoomansLLM server."""

import threading

import uvicorn

from .app import app
from .config import get_settings


def main() -> None:
    """Run the API server and optional native control panel."""
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


if __name__ == "__main__":
    main()
