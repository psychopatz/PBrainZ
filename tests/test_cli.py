import json

from hoomans_llm.__main__ import _show_activity
from hoomans_llm.config import Settings
from hoomans_llm.database import SettingsDatabase


def test_show_activity_defaults_to_bounded_recent_entries(monkeypatch, tmp_path, capsys) -> None:
    database_path = tmp_path / "hoomansllm.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    for index in range(60):
        database.add_log("INFO", "test", f"event-{index}")
    settings = Settings(database_path=str(database_path), bridge_required=False)
    monkeypatch.setattr("hoomans_llm.__main__.get_settings", lambda: settings)

    _show_activity(50)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 50
    assert lines[0].endswith("event-10")
    assert lines[-1].endswith("event-59")


def test_show_activity_can_emit_json(monkeypatch, tmp_path, capsys) -> None:
    database_path = tmp_path / "hoomansllm.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    database.add_log("INFO", "test", "bridge task")
    settings = Settings(database_path=str(database_path), bridge_required=False)
    monkeypatch.setattr("hoomans_llm.__main__.get_settings", lambda: settings)

    _show_activity(50, as_json=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"][0]["message"] == "bridge task"
