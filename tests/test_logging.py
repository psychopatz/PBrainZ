import logging

from pbrainz.database import SettingsDatabase
from pbrainz.logging_setup import configure_logging


def test_module_logs_are_visible_in_recent_activity(tmp_path) -> None:
    database = SettingsDatabase(tmp_path / "pbrainz.db")
    database.initialize()
    logger = logging.getLogger("pbrainz")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate

    try:
        configure_logging(database, "INFO")
        logging.getLogger("pbrainz.bridge.pump").info("bridge inference probe")

        entries = database.recent_logs()
        assert any(
            entry["source"] == "pbrainz.bridge.pump"
            and entry["message"].endswith("bridge inference probe")
            for entry in entries
        )
    finally:
        for handler in list(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(original_level)
        logger.propagate = original_propagate
