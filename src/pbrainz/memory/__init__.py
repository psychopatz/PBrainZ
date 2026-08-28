"""Save-scoped conversation memory primitives.

The memory package deliberately exposes small interfaces instead of leaking
SQLite details into the bridge or provider code.  A future sqlite-vec or
remote retriever can implement the same ``MemoryRetriever`` contract.
"""

from .sqlite import SQLiteMemoryStore
from .types import (
    ConversationSession,
    ConversationTurn,
    MemoryRecord,
    MemoryRetriever,
    MemoryScope,
    MemoryStore,
    MemoryType,
    RetrievalMatch,
)

__all__ = [
    "ConversationSession",
    "ConversationTurn",
    "MemoryRecord",
    "MemoryScope",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryType",
    "RetrievalMatch",
    "SQLiteMemoryStore",
]
