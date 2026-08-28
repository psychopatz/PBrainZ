"""Save-scoped conversation memory primitives.

The memory package deliberately exposes small interfaces instead of leaking
SQLite details into the bridge or provider code.  A future sqlite-vec or
remote retriever can implement the same ``MemoryRetriever`` contract.
"""

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
    "memory_root_for_settings",
    "StructuredFact",
    "TurnWriteResult",
]
