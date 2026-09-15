"""
TRACERA Memory Layer — agent-native memory modelled after Memori.

A middleware layer that intercepts every LLM call through the unified
:class:`~tracera.providers.base.LLMProvider` interface:

    from tracera.memory.layer import MemoryLayer, MemoryStore

    layer = MemoryLayer(store=MemoryStore(db_path), embed_fn=embed)
    provider = layer.register(provider)        # wrap once
    layer.attribution("user_123", "support_agent")

After that every call transparently:

  1. **recalls** the entity's top-k relevant memories into the system prompt,
  2. **forwards** the enriched request to the real provider,
  3. **returns** the response immediately,
  4. **extracts** structured memory in the background (durable queue) and
     stores it with semantic deduplication (``mention_count``).

Guardrails: no attribution → no memory read/write (logged); every query is
entity-scoped; extraction never blocks the response path.
"""

from __future__ import annotations

from tracera.memory.layer.attribution import (
    Attribution,
    AttributionError,
    current_attribution,
    current_session_id,
    reset_attribution,
    set_attribution,
    set_session_id,
)
from tracera.memory.layer.events import (
    EventPipeline,
    EventType,
    MemoryEvent,
    build_default_pipeline,
    handle_agent_decision,
    handle_file_changed,
    handle_llm_response,
    handle_repository_discovery,
    handle_tool_completed,
)
from tracera.memory.layer.extract import (
    EXTRACTION_PROMPT,
    ExtractedMemory,
    MemoryExtractor,
)
from tracera.memory.layer.facade import AgentMemory, MemoryLayer, MemoryLayerError
from tracera.memory.layer.queue import BackgroundWorker
from tracera.memory.layer.recall import RecallInjector, format_memories, format_memories_grouped
from tracera.memory.layer.reconcile import (
    MemoryReconciler,
    ReconcileAction,
    ReconcileEvent,
)
from tracera.memory.layer.store import (
    ALL_KINDS,
    FUNCTIONAL_PREDICATE_EXACT,
    FUNCTIONAL_PREDICATE_TOKENS,
    SCHEMA_VERSION,
    Job,
    MemoryKind,
    MemoryPolicy,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    VectorIndex,
    canonical_key,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)
from tracera.memory.layer.wrapper import MemoryProvider

__all__ = [
    "Attribution",
    "AttributionError",
    "BackgroundWorker",
    "EventPipeline",
    "EventType",
    "EXTRACTION_PROMPT",
    "ExtractedMemory",
    "FUNCTIONAL_PREDICATE_EXACT",
    "FUNCTIONAL_PREDICATE_TOKENS",
    "Job",
    "MemoryEvent",
    "MemoryExtractor",
    "MemoryKind",
    "MemoryReconciler",
    "ReconcileAction",
    "ReconcileEvent",
    "SCHEMA_VERSION",
    "VectorIndex",
    "canonical_key",
    "pack_embedding",
    "unpack_embedding",
    "AgentMemory",
    "MemoryLayer",
    "MemoryLayerError",
    "MemoryProvider",
    "MemoryRecord",
    "MemoryScope",
    "MemoryPolicy",
    "MemoryStore",
    "RecallInjector",
    "build_default_pipeline",
    "cosine_similarity",
    "current_attribution",
    "current_session_id",
    "format_memories",
    "format_memories_grouped",
    "handle_agent_decision",
    "handle_file_changed",
    "handle_llm_response",
    "handle_repository_discovery",
    "handle_tool_completed",
    "reset_attribution",
    "set_attribution",
    "set_session_id",
    "ALL_KINDS",
]
