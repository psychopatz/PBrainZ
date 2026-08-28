"""SQLite-backed activity logging without adding a logging dependency."""

from __future__ import annotations

import logging
import sys

from pbrainz.branding import LOGGER_NAME
from pbrainz.database import SettingsDatabase


class SQLiteLogHandler(logging.Handler):
    """Write application log records to the bounded local activity table."""

    def __init__(self, database: SettingsDatabase) -> None:
        super().__init__()
        self.database = database

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.database.add_log(record.levelname, record.name, self.format(record))
        except Exception:
            self.handleError(record)


def configure_logging(database: SettingsDatabase, level: str) -> None:
    """Attach one SQLite handler to the P BrainZ logger for this app instance."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        if isinstance(handler, SQLiteLogHandler) or getattr(
            handler, "_pbrainz_console", False
        ):
            logger.removeHandler(handler)
            handler.close()
    handler = SQLiteLogHandler(database)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    # PyInstaller's windowed bootloader can expose ``None`` for both standard
    # streams. Keep the SQLite activity log working in that mode instead of
    # creating a handler that fails every time it receives a record.
    console_stream = sys.stdout or sys.stderr
    if console_stream is not None:
        console = logging.StreamHandler(console_stream)
        console._pbrainz_console = True
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(console)
    logger.propagate = False
