import json

from pbrainz.bridge import BridgeRuntimeMonitor


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
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-123", encoding="utf-8")

    result = BridgeRuntimeMonitor(tmp_path).read()

    assert result.ready is True
    assert result.runtime_id == "runtime-123"
    assert result.tool_catalog_id == "catalog-1"
    assert result.tool_catalog_version == 3


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
