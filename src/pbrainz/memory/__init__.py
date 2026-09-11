"""Save-scoped conversation memory primitives.

The memory package deliberately exposes small interfaces instead of leaking
SQLite details into the bridge or provider code.  A future sqlite-vec or
remote retriever can implement the same ``MemoryRetriever`` contract.
"""

from .locator import (
    MemoryIdentity,
    MemoryLocationError,
    list_memory_worlds,
    normalize_save_relative_path,
    public_memory_world,
)
from .policy import is_context_eligible
from .primitives import MemoryPrimitiveService, normalize_event_time, register_primitive
from .sqlite import SQLiteMemoryStore, memory_root_for_settings
from .types import (
    ConversationSession,
    ConversationTurn,
    DaySynopsis,
    MemoryEpisode,
    MemoryQuery,
    MemoryRecord,
    MemoryRetriever,
    MemoryScope,
    MemoryStore,
    MemoryType,
    MemoryVisibility,
    RetrievalMatch,
    StructuredFact,
    TurnWriteResult,
)

__all__ = [
    "ConversationSession",
    "ConversationTurn",
    "DaySynopsis",
    "MemoryEpisode",
    "MemoryRecord",
    "MemoryScope",
    "MemoryRetriever",
    "MemoryQuery",
    "MemoryStore",
    "MemoryType",
    "MemoryVisibility",
    "RetrievalMatch",
    "SQLiteMemoryStore",
    "MemoryIdentity",
    "MemoryLocationError",
    "list_memory_worlds",
    "normalize_save_relative_path",
    "public_memory_world",
    "memory_root_for_settings",
    "is_context_eligible",
    "MemoryPrimitiveService",
    "normalize_event_time",
    "register_primitive",
    "StructuredFact",
    "TurnWriteResult",
]
