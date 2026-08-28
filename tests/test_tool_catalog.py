import pytest

from pbrainz.bridge import PacketStreamClient, ToolCatalogCache, hydrate_request
from pbrainz.bridge.protocol import BridgeClientError
from pbrainz.bridge.state import BridgeState


def _catalog(catalog_id: str = "catalog-1") -> dict:
    return {
        "catalog_id": catalog_id,
        "catalog_version": 3,
        "tools": [
            {
                "id": "example.debug:inspect",
                "namespace": "example.debug",
                "name": "inspect",
                "kind": "llm_tool",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "description": "Inspect a value.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            }
        ],
    }


def test_tool_catalog_cache_hydrates_only_allowed_ids() -> None:
    cache = ToolCatalogCache()
    cache.add(_catalog())
    request = {
        "conversation_context": {
            "tool_catalog_id": "catalog-1",
            "available_tool_ids": ["example.debug:inspect", "example.debug:missing"],
            "message": "hello",
        }
    }

    hydrated = hydrate_request(request, cache)

    tools = hydrated["conversation_context"]["available_tools"]
    assert [tool["function"]["name"] for tool in tools] == ["inspect"]
    assert request["conversation_context"].get("available_tools") is None


def test_tool_catalog_cache_rejects_malformed_catalog() -> None:
    with pytest.raises(BridgeClientError, match="no catalog ID"):
        ToolCatalogCache().add({"tools": []})


@pytest.mark.asyncio
async def test_tool_catalog_cache_fetches_once_per_catalog_id() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, *_args):
            self.calls += 1
            return _catalog()

    client = Client()
    cache = ToolCatalogCache()
    state = BridgeState(
        available=True,
        enabled=True,
        ready=True,
        runtime_id="runtime-1",
        tool_catalog_id="catalog-1",
        tool_catalog_version=3,
    )

    await cache.ensure(client, state)
    await cache.ensure(client, state)

    assert client.calls == 1


@pytest.mark.asyncio
async def test_packet_stream_client_tracks_events_and_resynchronizes_on_gap() -> None:
    class Client:
        async def call(self, _namespace, _command, _arguments, _runtime_id):
            return {
                "streams": [
                    {
                        "namespace": "example.debug",
                        "channel": "events",
                        "sequence": 9,
                        "events": [],
                        "gap": True,
                        "snapshot": {"active": True},
                    }
                ]
            }

    stream = PacketStreamClient(cursors={"example.debug:events": 2})
    state = BridgeState(available=True, enabled=True, ready=True, runtime_id="runtime-1")

    result = await stream.poll(
        Client(),
        state,
        [{"namespace": "example.debug", "channel": "events"}],
    )

    assert result["streams"][0]["snapshot"] == {"active": True}
    assert stream.cursors["example.debug:events"] == 9
