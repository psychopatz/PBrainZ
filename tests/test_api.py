import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from hoomans_llm.api.models import ChatCompletionRequest
from hoomans_llm.app import create_app
from hoomans_llm.config import Settings
from hoomans_llm.database import SettingsDatabase
from hoomans_llm.providers.base import CompletionResult, StreamEvent


class FakeRegistry:
    provider_names = ("openai", "gemini")

    def resolve(self, requested_provider: str | None, model: str) -> tuple[str, str]:
        return (requested_provider or "openai", model)

    def model_ids(self) -> list[tuple[str, str]]:
        return [("openai", "test-model"), ("gemini", "test-gemini")]

    async def list_models(self, provider_name: str) -> list[str]:
        return [f"{provider_name}-discovered-model"]

    async def invalidate(self, _provider_names: set[str]) -> None:
        pass

    async def complete(
        self, provider_name: str, request: ChatCompletionRequest
    ) -> CompletionResult:
        return CompletionResult(model=request.model, text=f"reply from {provider_name}")

    async def stream_events(
        self, provider_name: str, request: ChatCompletionRequest
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(text="streamed ")
        yield StreamEvent(text="reply", finish_reason="stop")

    async def close(self) -> None:
        pass


@contextmanager
def _client(
    bridge_root: str | None = None, *, bridge_required: bool = False
) -> Iterator[TestClient]:
    bridge_path = Path(bridge_root or "/tmp/hoomans-llm-test-bridge-never-created")
    app = create_app(
        Settings(
            enabled_providers="openai,gemini",
            openai_api_key="test",
            gemini_api_key="test",
            bridge_required=bridge_required,
            bridge_root=str(bridge_path),
            database_path=str(bridge_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        providers = FakeRegistry()
        app.state.providers = providers
        app.state.bridge_controller.providers = providers
        app.state.model_catalog.providers = providers
        yield client


def test_health_and_models() -> None:
    with _client() as client:
        health = client.get("/health")
        models = client.get("/v1/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert models.json()["data"][1]["owned_by"] == "gemini"


def test_control_panel_api_reports_bridge_state(tmp_path) -> None:
    with _client(str(tmp_path)) as client:
        status = client.get("/api/settings")

    assert status.status_code == 200
    assert status.json()["bridge_worker_enabled"] is False
    assert status.json()["bridge"]["ready"] is False
    assert status.json()["providers"][1]["name"] == "gemini"


def test_control_panel_uses_a_configured_provider_when_default_is_missing(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai,gemini",
            openai_api_key=None,
            gemini_api_key="test",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(tmp_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        status = client.get("/api/settings")

    assert status.json()["default_provider"] == "gemini"
    assert status.json()["providers"][1]["selected"] is True


def test_control_panel_does_not_select_idle_local_endpoint_over_gemini(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai,ollama,lmstudio,custom,gemini",
            default_provider="openai",
            openai_api_key=None,
            gemini_api_key="test",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(tmp_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        status = client.get("/api/settings")

    assert status.json()["default_provider"] == "gemini"


def test_control_panel_shows_only_provider_cache_models(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai,ollama,lmstudio,custom,gemini",
            openai_api_key="openai-key",
            ollama_models="configured-but-not-cached",
            custom_models="configured-but-not-cached",
            gemini_api_key="gemini-key",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(tmp_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        status = client.get("/api/settings")

    models_by_provider = {
        item["name"]: item["models"] for item in status.json()["providers"]
    }
    assert models_by_provider["ollama"] == []
    assert models_by_provider["custom"] == []
    assert models_by_provider["gemini"] == []


def test_control_panel_loads_models_from_each_provider_cache(tmp_path) -> None:
    database_path = tmp_path / "hoomansllm.db"
    database = SettingsDatabase(database_path)
    database.initialize()
    database.replace_model_catalog("ollama", ["llama3.2"])
    database.replace_model_catalog("custom", ["router-model"])
    app = create_app(
        Settings(
            enabled_providers="openai,ollama,lmstudio,custom,gemini",
            default_provider="ollama",
            openai_api_key="openai-key",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(database_path),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        status = client.get("/api/settings")

    models_by_provider = {
        item["name"]: item["models"] for item in status.json()["providers"]
    }
    assert models_by_provider["ollama"] == ["llama3.2"]
    assert models_by_provider["custom"] == ["router-model"]
    assert models_by_provider["lmstudio"] == []


def test_control_panel_persists_theme(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai",
            openai_api_key="openai-key",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(tmp_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        response = client.post("/api/settings", json={"ui_theme": "dark"})

    assert response.status_code == 200
    assert response.json()["ui_theme"] == "dark"
    database = sqlite3.connect(tmp_path / "hoomansllm.db")
    saved = dict(database.execute("SELECT key, value FROM settings").fetchall())
    database.close()
    assert json.loads(saved["ui_theme"]) == "dark"


def test_control_panel_saves_provider_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with _client(str(tmp_path / "bridge")) as client:
        response = client.post(
            "/api/settings",
            json={
                "default_provider": "gemini",
                "default_model": "gemini-2.5-flash",
                "request_timeout": 45,
                "bridge_poll_interval": 0.75,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["default_provider"] == "gemini"
    assert body["default_model"] == "gemini-2.5-flash"
    database = sqlite3.connect(tmp_path / "bridge" / "hoomansllm.db")
    saved = dict(database.execute("SELECT key, value FROM settings").fetchall())
    database.close()
    assert json.loads(saved["default_provider"]) == "gemini"
    assert json.loads(saved["request_timeout"]) == 45.0


def test_control_panel_can_toggle_bridge_worker(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with _client(str(tmp_path / "bridge")) as client:
        enabled = client.post("/api/bridge", json={"enabled": True})
        disabled = client.post("/api/bridge", json={"enabled": False})

    assert enabled.status_code == 200
    assert enabled.json()["bridge_worker_enabled"] is True
    assert disabled.status_code == 200
    assert disabled.json()["bridge_worker_enabled"] is False
    database = sqlite3.connect(tmp_path / "bridge" / "hoomansllm.db")
    saved = dict(database.execute("SELECT key, value FROM settings").fetchall())
    database.close()
    assert json.loads(saved["bridge_required"]) is False


def test_control_panel_persists_api_key_without_logging_the_secret(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with _client(str(tmp_path / "bridge")) as client:
        response = client.post(
            "/api/settings",
            json={
                "default_provider": "gemini",
                "default_model": "gemini-2.5-flash",
                "gemini_api_key": "new-secret-key",
            },
        )
        logs = client.get("/api/logs")

    assert response.status_code == 200
    assert response.json()["providers"][1]["api_key_hint"] == "••••-key"
    assert "new-secret-key" not in logs.text
    database = sqlite3.connect(tmp_path / "bridge" / "hoomansllm.db")
    saved = dict(database.execute("SELECT key, value FROM settings").fetchall())
    database.close()
    assert json.loads(saved["gemini_api_key"]) == "new-secret-key"


def test_control_panel_refreshes_and_persists_provider_models(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with _client(str(tmp_path / "bridge")) as client:
        response = client.post("/api/models/refresh", json={"provider": "gemini"})

    assert response.status_code == 200
    assert response.json()["providers"][1]["models"] == ["gemini-discovered-model"]
    database = sqlite3.connect(tmp_path / "bridge" / "hoomansllm.db")
    catalog = database.execute(
        "SELECT model_id FROM model_catalog WHERE provider = 'gemini'"
    ).fetchall()
    database.close()
    assert catalog == [("gemini-discovered-model",)]


def test_control_panel_exposes_and_persists_separate_local_profiles(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai,ollama,lmstudio,custom,gemini",
            default_provider="ollama",
            ollama_models="llama3.2",
            openai_api_key="openai-key",
            gemini_api_key="gemini-key",
            bridge_required=False,
            bridge_root=str(tmp_path),
            database_path=str(tmp_path / "hoomansllm.db"),
            auto_refresh_models=False,
        )
    )
    with TestClient(app) as client:
        status = client.get("/api/settings")
        response = client.post(
            "/api/settings",
            json={
                "default_provider": "ollama",
                "default_model": "llama3.2",
                "ollama_base_url": "http://localhost:11434/v1",
                "ollama_api_key": "ollama-local-key",
                "custom_base_url": "http://localhost:9000/v1",
            },
        )

    assert status.status_code == 200
    assert [item["name"] for item in status.json()["providers"]] == [
        "openai",
        "ollama",
        "lmstudio",
        "custom",
        "gemini",
    ]
    assert response.status_code == 200
    ollama_status = next(
        item for item in response.json()["providers"] if item["name"] == "ollama"
    )
    assert ollama_status["base_url"] == "http://localhost:11434/v1"
    assert ollama_status["api_key_hint"] == "••••-key"
    custom_status = next(
        item for item in response.json()["providers"] if item["name"] == "custom"
    )
    assert custom_status["base_url"] == "http://localhost:9000/v1"
    assert custom_status["api_key_hint"] is None
    database = sqlite3.connect(tmp_path / "hoomansllm.db")
    saved = dict(database.execute("SELECT key, value FROM settings").fetchall())
    database.close()
    assert json.loads(saved["ollama_api_key"]) == "ollama-local-key"
    assert json.loads(saved["ollama_base_url"]) == "http://localhost:11434/v1"
    assert json.loads(saved["custom_base_url"]) == "http://localhost:9000/v1"


def test_chat_completion_is_openai_compatible() -> None:
    with _client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "provider": "gemini",
                "model": "test-gemini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "reply from gemini"}


def test_control_panel_chat_can_test_provider_without_bridge() -> None:
    with _client(bridge_required=True) as client:
        response = client.post(
            "/api/chat",
            json={
                "provider": "gemini",
                "model": "test-gemini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "reply from gemini",
    }


def test_streaming_chat_completion_returns_sse() -> None:
    with _client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "streamed " in response.text
    assert "data: [DONE]" in response.text


def test_chat_requires_a_ready_bridge_by_default(tmp_path) -> None:
    app = create_app(
        Settings(
            enabled_providers="openai",
            openai_api_key=None,
            bridge_required=True,
            bridge_root=str(tmp_path),
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "bridge_unavailable"
