"""SQLite-backed activity logging without adding a logging dependency."""

from __future__ import annotations

import logging

from hoomans_llm.database import SettingsDatabase


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
    """Attach one SQLite handler to the HoomansLLM logger for this app instance."""
    logger = logging.getLogger("hoomans_llm")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        if isinstance(handler, SQLiteLogHandler):
            logger.removeHandler(handler)
            handler.close()
    handler = SQLiteLogHandler(database)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
