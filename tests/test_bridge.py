import json

from pbrainz.bridge import BridgeRuntimeMonitor, ProviderPresenceWriter


def test_provider_presence_writer_publishes_matching_state_and_marker(tmp_path) -> None:
    writer = ProviderPresenceWriter(tmp_path)

    writer.write(
        runtime_id="runtime-123",
        status="ready",
        provider="horde",
        model="aphrodite/test",
        heartbeat_ms=123456,
    )

    state = json.loads(
        (tmp_path / "state" / "provider_state.json").read_text(encoding="utf-8")
    )
    marker = (tmp_path / "state" / "provider_state.ready.txt").read_text(
        encoding="utf-8"
    )
    assert state["schema_version"] == 1
    assert state["runtime_id"] == "runtime-123"
    assert state["ready"] is True
    assert state["heartbeat_ms"] == 123456
    assert marker == "runtime-123"


def test_bridge_runtime_requires_matching_ready_marker(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    runtime = {
        "protocol_version": 1,
        "runtime_id": "runtime-123",
        "enabled": True,
        "lifecycle": "READY",
        "authority": "singleplayer",
        "transport": "file",
        "tool_catalog_id": "catalog-1",
        "tool_catalog_version": 3,
        "namespaces": {
            "psychopatzcore.bridge": {"commands": []},
            "pbrainz.llm": {"commands": []},
        },
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-123", encoding="utf-8")

    result = BridgeRuntimeMonitor(tmp_path).read()

    assert result.ready is True
    assert result.runtime_id == "runtime-123"
    assert result.tool_catalog_id == "catalog-1"
    assert result.tool_catalog_version == 3
    assert result.namespaces == ("psychopatzcore.bridge", "pbrainz.llm")
    assert result.namespaces_known is True


def test_bridge_runtime_can_report_when_pbrainz_namespace_is_missing(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    runtime = {
        "protocol_version": 1,
        "runtime_id": "runtime-core-only",
        "enabled": True,
        "lifecycle": "READY",
        "namespaces": {"psychopatzcore.bridge": {"commands": []}},
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-core-only", encoding="utf-8")

    result = BridgeRuntimeMonitor(tmp_path).read()

    assert result.ready is True
    assert result.namespaces_known is True
    assert "pbrainz.llm" not in result.namespaces


def test_bridge_runtime_rejects_stale_marker(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "runtime.json").write_text(
        json.dumps({"protocol_version": 1, "runtime_id": "runtime-123", "enabled": True}),
        encoding="utf-8",
    )
    (state / "runtime.ready.txt").write_text("old-runtime", encoding="utf-8")

    result = BridgeRuntimeMonitor(tmp_path).read()

    assert result.ready is False
    assert "marker" in result.message
