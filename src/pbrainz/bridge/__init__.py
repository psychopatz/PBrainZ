"""Project Hoomans bridge subsystem."""

from .catalog import ToolCatalog, ToolCatalogCache, hydrate_request
from .client import BridgeClient
from .controller import BridgeController
from .memory_context import ActiveMemoryContext, ActiveMemoryContextCache
from .protocol import (
    CORE_NAMESPACE,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_STRING,
    NAMESPACE,
    POLL_PACKETS_COMMAND,
    PROTOCOL_VERSION,
    SLOT_COUNT,
    TOOL_CATALOG_COMMAND,
    BridgeClientError,
    BridgeCommandError,
    BridgeRequest,
    BridgeResponse,
    BridgeTimeoutError,
)
from .pump import run_bridge_pump
from .state import MAX_RUNTIME_BYTES, BridgeRuntimeMonitor, BridgeState
from .streams import PacketStreamClient
from .transport import FileBridgeTransport
from .voice import (
    VOICE_CHANNEL,
    VOICE_EVENT_TYPE,
    VOICE_NAMESPACE,
    VoicePacketConsumer,
    utterance_from_packet,
    voice_channel_available,
)

__all__ = [
    "BridgeClient",
    "ToolCatalog",
    "ToolCatalogCache",
    "hydrate_request",
    "BridgeClientError",
    "BridgeCommandError",
    "BridgeController",
    "ActiveMemoryContext",
    "ActiveMemoryContextCache",
    "BridgeRequest",
    "BridgeResponse",
    "BridgeRuntimeMonitor",
    "BridgeState",
    "BridgeTimeoutError",
    "FileBridgeTransport",
    "PacketStreamClient",
    "VOICE_CHANNEL",
    "VOICE_EVENT_TYPE",
    "VOICE_NAMESPACE",
    "VoicePacketConsumer",
    "utterance_from_packet",
    "voice_channel_available",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MAX_RUNTIME_BYTES",
    "MAX_STRING",
    "CORE_NAMESPACE",
    "NAMESPACE",
    "POLL_PACKETS_COMMAND",
    "PROTOCOL_VERSION",
    "SLOT_COUNT",
    "TOOL_CATALOG_COMMAND",
    "run_bridge_pump",
]
