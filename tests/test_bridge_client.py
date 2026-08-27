import asyncio
import json
from pathlib import Path

import pytest

from hoomans_llm.bridge import BridgeRuntimeMonitor
from hoomans_llm.bridge_client import (
    BridgeRequest,
    FileBridgeTransport,
    run_bridge_pump,
)
from hoomans_llm.config import Settings
from hoomans_llm.providers.base import CompletionResult


def test_file_transport_matches_psychopatzcore_slot_protocol(tmp_path) -> None:
    transport = FileBridgeTransport(tmp_path)
    request = BridgeRequest.create(
        "projecthoomans.llm",
        "pollChat",
        {},
        "runtime-123",
    )

    slot = transport.write_request(request)
    request_path = tmp_path / "requests" / f"slot-{slot:02d}.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload["message_type"] == "request"
    assert payload["protocol_version"] == 1
    assert payload["target_runtime_id"] == "runtime-123"

    response = {
        "message_type": "response",
        "protocol_version": 1,
        "request_id": request.request_id,
        "runtime_id": "runtime-123",
        "status": "ok",
        "request_state": "COMPLETE",
        "result": {"status": "idle"},
        "error": None,
    }
    response_dir = tmp_path / "responses"
    (response_dir / f"slot-{slot:02d}.json").write_text(
        json.dumps(response),
        encoding="utf-8",
    )
    (response_dir / f"slot-{slot:02d}.ready.txt").write_text(
        request.request_id,
        encoding="utf-8",
    )

    received = transport.read_response(slot, request.request_id)
    assert received is not None
    assert received.result == {"status": "idle"}
    transport.release(slot)
    assert not request_path.exists()


def test_file_transport_does_not_overwrite_busy_slots(tmp_path) -> None:
    transport = FileBridgeTransport(tmp_path)
    requests = [
        BridgeRequest.create("projecthoomans.llm", "pollChat", {}, "runtime-123")
        for _ in range(16)
    ]
    slots = [transport.write_request(request) for request in requests]

    try:
        try:
            transport.write_request(
                BridgeRequest.create(
                    "projecthoomans.llm",
                    "pollChat",
                    {},
                    "runtime-123",
                )
            )
        except RuntimeError as error:
            assert "slots" in str(error)
        else:
            raise AssertionError("expected all slots to be busy")
    finally:
        for slot in slots:
            transport.release(slot)


class FakeProviders:
    def resolve(self, _provider: str | None, model: str) -> tuple[str, str]:
        return "gemini", model

    async def complete(self, _provider: str, request) -> CompletionResult:
        return CompletionResult(model=request.model, text="Stay close to the shelter.")


async def _fake_game(root: Path, delivered: asyncio.Event) -> None:
    requests = root / "requests"
    responses = root / "responses"
    handled: set[str] = set()
    while not delivered.is_set():
        for request_path in requests.glob("slot-*.json"):
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request_id = request["request_id"]
            if request_id in handled:
                continue
            handled.add(request_id)
            command = request["command"]
            if command == "pollChat":
                result = {
                    "status": "pending",
                    "request_id": "pnc_llm_1",
                    "npc_id": "npc-1",
                    "messages": [
                        {"role": "system", "content": "You are an NPC."},
                        {"role": "user", "content": "Where should we go?"},
                    ],
                }
            else:
                result = {"accepted": True}
                assert request["arguments"]["response_text"] == (
                    "Stay close to the shelter."
                )
                delivered.set()
            slot = request_path.stem.rsplit("-", 1)[-1]
            response = {
                "message_type": "response",
                "protocol_version": 1,
                "request_id": request_id,
                "runtime_id": "runtime-123",
                "status": "ok",
                "request_state": "COMPLETE",
                "result": result,
                "error": None,
            }
            (responses / f"slot-{slot}.json").write_text(
                json.dumps(response),
                encoding="utf-8",
            )
            (responses / f"slot-{slot}.ready.txt").write_text(
                request_id,
                encoding="utf-8",
            )
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_bridge_pump_completes_and_delivers_an_npc_message(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    runtime = {
        "protocol_version": 1,
        "runtime_id": "runtime-123",
        "enabled": True,
        "lifecycle": "READY",
    }
    (state / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (state / "runtime.ready.txt").write_text("runtime-123", encoding="utf-8")
    delivered = asyncio.Event()
    settings = Settings(
        bridge_required=True,
        bridge_root=str(tmp_path),
        bridge_poll_interval=0.06,
        request_timeout=1,
    )
    pump = asyncio.create_task(
        run_bridge_pump(settings, FakeProviders(), BridgeRuntimeMonitor(tmp_path))
    )
    game = asyncio.create_task(_fake_game(tmp_path, delivered))
    try:
        await asyncio.wait_for(delivered.wait(), timeout=2)
    finally:
        pump.cancel()
        game.cancel()
        await asyncio.gather(pump, game, return_exceptions=True)
