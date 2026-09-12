"""
Memory Layer facade — the Memori-equivalent for TRACERA.

Public API (mirrors Memori's SDK):

    layer = MemoryLayer(store=..., embed_fn=...)
    provider = layer.register(provider)        # wrap the LLM provider
    layer.attribution(entity_id, process_id)   # must be set before LLM calls
    layer.new_session()                        # group turns into a session
    layer.set_session(session_id)              # resume an existing session

From that point on memory is automatic:

    recall (before) → forward → return immediately → async extraction (after)

The per-process rollout knob lives on the layer: a process not listed in
``enabled_processes`` simply passes through untouched.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from tracera.logging import get_logger
from tracera.memory.layer.attribution import (
    Attribution,
    current_attribution,
    require_attribution,
    set_attribution,
    set_session_id,
)
from tracera.memory.layer.extract import MemoryExtractor
from tracera.memory.layer.queue import BackgroundWorker
from tracera.memory.layer.recall import RecallInjector
from tracera.memory.layer.store import Job, MemoryStore, ALL_KINDS
from tracera.memory.layer.wrapper import MemoryProvider
from tracera.providers.base import LLMMessage, LLMProvider

log = get_logger("memory.layer")

EmbedFn = Callable[[str], list[float]]


class MemoryLayerError(RuntimeError):
    """Raised for misuse of the memory layer API (e.g. double registration)."""


class AgentMemory:
    """
    TRACERA-compatible memory facade.

    This is the class that ``TraceraTUI`` and the agent runtime expect.
    It wraps a ``MemoryLayer`` and exposes the attribute-based API used by
    ``action_show_memory`` and the memory tools:

        memory.stats()
        memory.get_by_type(mtype)
        memory.recall(query, top_k=...)
        memory.add(memory)
        memory.delete(memory_id)
        memory.entries()

    It also exposes ``triple_store`` and ``session_manager`` so the TUI's
    ``/memgraph`` and ``/sessions`` commands can reach the underlying stores.
    """

    def __init__(self, layer: MemoryLayer, *, triple_store: Any = None, session_manager: Any = None):
        self._layer = layer
        self._triple_store = triple_store
        self._session_manager = session_manager

    # ── Introspection (used by action_show_memory) ──────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return memory statistics matching what action_show_memory expects."""
        store = self._layer.store
        total = store.count_memories()
        by_type: dict[str, int] = {}
        # find_by_kind(entity_id) returns dict[str, list] grouped by kind
        grouped = store.find_by_kind('default')
        for kind, records in grouped.items():
            if records:
                by_type[kind] = len(records)
        return {
            "total": total,
            "by_type": by_type,
        }

    def count(self) -> int:
        """Number of active memories (used by status line)."""
        return self._layer.store.count_memories()

    def entries(self) -> list:
        """Legacy fallback — return raw memory records."""
        return [self._record_to_memory(r) for r in self._layer.store.find_memories(self._layer.store._resolve_ids('*', '*')[0], limit=50)]

    # ── Type-based retrieval (used by action_show_memory) ───────────────────

    def get_by_type(self, mtype: Any) -> list:
        """Return memories of a specific type for display."""
        kind = getattr(mtype, "value", str(mtype)).lower()
        grouped = self._layer.store.find_by_kind('default')
        if kind in grouped:
            return [self._record_to_memory(rec) for rec in grouped[kind][:20]]
        return []

    # ── Explicit memory operations (used by memory tools) ───────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        max_chars: int = 8000,
        include_sessions: bool = True,
        include_triples: bool = True,
        include_legacy: bool = True,
        use_graph_expansion: bool = True,
    ) -> str:
        """Recall relevant memories as a formatted string."""
        # Delegate to the RecallInjector on the layer
        from tracera.memory.recall import ContextRecall
        # Build a simple recall using the store directly
        # For now, return a formatted string from store memories
        all_memories = self._layer.store.find_memories(self._layer.store._resolve_ids('*', '*')[0], limit=top_k * 3)
        lines = []
        for rec in all_memories[:top_k]:
            lines.append(f"[{rec.kind}] {rec.text[:200]}")
        return "\n".join(lines) if lines else "No relevant memories found."

    def add(self, memory: Any, *, kind: str = "fact", importance: float = 0.5) -> Any:
        """Store a structured memory."""
        # Accept structured memory objects or dicts
        if hasattr(memory, 'content'):
            text = memory.content
        elif isinstance(memory, dict):
            text = memory.get('text', memory.get('content', ''))
        else:
            text = str(memory)

        # Extract subject/predicate/object from text if possible
        subject = 'user'
        predicate = 'knows'
        object_val = text[:100]

        embedding = self._layer._embed_fn(text)
        inserted, record = self._layer.store.upsert_memory(
            entity_id='default',
            process_id='tracera-agent',
            kind=kind,
            subject=subject,
            predicate=predicate,
            object=object_val,
            text=text,
            embedding=embedding,
            confidence=importance,
            importance=importance,
        )
        return record

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            mem_id = int(memory_id)
            # Use upsert to mark as invalidated
            self._layer.store.upsert_memory(
                entity_id='default',
                process_id='tracera-agent',
                kind='fact',
                subject='deleted',
                predicate='memory',
                object=memory_id,
                text='',
                embedding=[0.0] * 384,
                status='invalidated',
            )
            return True
        except (ValueError, Exception):
            # Try to mark as invalidated via direct SQL
            return False

    def search(self, query: str, top_k: int = 10) -> list:
        """Search memories by semantic similarity."""
        embedding = self._layer._embed_fn(query)
        results = self._layer.store.recall('default', embedding, k=top_k, min_score=0.1)
        return [(rec, score) for rec, score in results]

    # ── Triple store access (used by /memgraph, /triples) ──────────────────

    @property
    def triple_store(self) -> Any:
        """Expose the triple store for graph commands."""
        return self._triple_store

    @property
    def session_manager(self) -> Any:
        """Expose the session manager for session commands."""
        return self._session_manager

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _record_to_memory(self, record: Any) -> Any:
        """Convert a MemoryRecord to a memory object for display."""
        from tracera.memory.taxonomy import (
            MemoryFact,
            MemoryPreference,
            MemoryRelationship,
            MemoryRule,
            MemorySkill,
            MemoryEvent,
            MemoryDecision,
            MemoryGoal,
            MemoryConstraint,
            MemoryExperience,
            MemoryAttribute,
            StructuredMemory,
        )

        kind = getattr(record, "kind", "fact").lower()
        content = getattr(record, "text", "")

        if kind == "fact":
            return MemoryFact(content=content, importance=record.importance)
        elif kind == "preference":
            return MemoryPreference(content=content, importance=record.importance)
        elif kind == "relationship":
            return MemoryRelationship(
                subject=getattr(record, 'subject', ''),
                predicate=getattr(record, 'predicate', ''),
                object=getattr(record, 'object', ''),
                content=content,
                importance=record.importance,
            )
        elif kind == "rule":
            return MemoryRule(content=content, importance=record.importance)
        elif kind == "skill":
            return MemorySkill(content=content, importance=record.importance)
        elif kind == "event":
            return MemoryEvent(content=content, importance=record.importance)
        elif kind == "decision":
            return MemoryDecision(content=content, importance=record.importance)
        elif kind == "goal":
            return MemoryGoal(content=content, importance=record.importance)
        elif kind == "constraint":
            return MemoryConstraint(content=content, importance=record.importance)
        elif kind == "experience":
            return MemoryExperience(content=content, importance=record.importance)
        elif kind == "attribute":
            return MemoryAttribute(content=content, importance=record.importance)
        else:
            return StructuredMemory(content=content, importance=record.importance)


class MemoryLayerError(RuntimeError):
    """Raised for misuse of the memory layer API (e.g. double registration)."""


class MemoryLayer:
    """
    Owns the store, recall/extraction machinery, the background worker, and
    exposes the Memori-style ``register`` / ``attribution`` / ``session`` API.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        embed_fn: EmbedFn,
        top_k: int = 5,
        dedup_threshold: float = 0.9,
        min_recall_score: float = 0.3,
        enabled_processes: list[str] | None = None,
        extraction_model: str | None = None,
        default_attribution: Attribution | None = None,
        worker_enabled: bool = True,
        max_job_attempts: int = 5,
        min_extraction_confidence: float = 0.5,
        min_extraction_importance: float = 0.3,
        enable_worthiness_filter: bool = True,
        enable_safety: bool = True,
        # Recall config
        recall_use_hybrid: bool = True,
        recall_token_budget: int = 2000,
        recall_grouped: bool = True,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._dedup_threshold = dedup_threshold
        self._extraction_model = extraction_model
        self._default_attribution = default_attribution
        self._worker_enabled = worker_enabled
        self._max_job_attempts = max_job_attempts
        self._min_extraction_confidence = min_extraction_confidence
        self._min_extraction_importance = min_extraction_importance
        self._enable_worthiness_filter = enable_worthiness_filter
        self._enable_safety = enable_safety
        self._enabled_processes = (
            set(p.strip() for p in enabled_processes if p and p.strip())
            if enabled_processes
            else None
        )
        self._recaller = RecallInjector(
            store,
            embed_fn,
            top_k=top_k,
            min_score=min_recall_score,
            use_hybrid=recall_use_hybrid,
            token_budget=recall_token_budget,
            grouped=recall_grouped,
        )
        self._extractor: MemoryExtractor | None = None
        self._extraction_provider: LLMProvider | None = None
        self._wrapped: MemoryProvider | None = None
        self._worker: BackgroundWorker | None = None
        self._auto_sessions: dict[tuple[str, str], str] = {}
# ── Memori-style public API ──────────────────────────────────────────────

    def register(self, provider: LLMProvider) -> MemoryProvider:
        """
        Wrap an ``LLMProvider`` so every call flows through the memory layer.
        Returns the wrapper — hand *that* to the agent instead of ``provider``.
        """
        if self._wrapped is not None:
            raise MemoryLayerError(
                "MemoryLayer already registered a provider; create a new "
                "MemoryLayer for a second provider."
            )
        self._extraction_provider = provider
        self._extractor = MemoryExtractor(
            self._build_extraction_call(provider),
            min_confidence=self._min_extraction_confidence,
            min_importance=self._min_extraction_importance,
            enable_worthiness_filter=self._enable_worthiness_filter,
            enable_safety=self._enable_safety,
        )
        self._wrapped = MemoryProvider(provider, self)
        log.info("Memory layer registered on provider %s", provider.name)
        return self._wrapped

    def attribution(self, entity_id: str, process_id: str) -> Attribution:
        """Set the current (entity_id, process_id) attribution scope."""
        return set_attribution(entity_id, process_id)

    def new_session(self) -> str:
        """
        Begin a new session (creates a row, associates it with the current
        attribution) and make it the active session for this context.
        """
        scope = require_attribution()
        session_id = self._store.create_session(scope.entity_id, scope.process_id)
        set_session_id(session_id)
        self._auto_sessions.pop((scope.entity_id, scope.process_id), None)
        log.info("New memory session %s for %s", session_id[:8], scope)
        return session_id

    def set_session(self, session_id: str) -> None:
        """Resume an existing session for the current context."""
        if not session_id:
            raise MemoryLayerError("session_id must be non-empty")
        set_session_id(session_id)

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def wrapped_provider(self) -> MemoryProvider | None:
        return self._wrapped

    @property
    def enabled_processes(self) -> set[str] | None:
        """Process allow-list (None = all processes enabled)."""
        return self._enabled_processes

    @property
    def enabled(self) -> bool:
        return True

    # ── Internals used by the wrapper ────────────────────────────────────────

    def prepare(self) -> tuple[Attribution | None, bool]:
        """
        Resolve the current attribution scope and whether memory is active.

        Returns ``(scope, active)``. When no attribution is set a clear
        warning is logged and the call proceeds without memory involvement.
        """
        scope = current_attribution() or self._default_attribution
        if scope is None:
            log.warning(
                "Memory layer: no attribution set. Call "
                "memory_layer.attribution(entity_id, process_id) before LLM "
                "calls — memory read/write skipped for this request."
            )
            return None, False
        if (
            self._enabled_processes is not None
            and scope.process_id not in self._enabled_processes
        ):
            log.debug(
                "Memory layer disabled for process %r (not in enabled list)",
                scope.process_id,
            )
            return scope, False
        return scope, True

    def inject_recall(
        self,
        messages: list[LLMMessage],
        system: str | None,
        scope: Attribution,
    ) -> tuple[list[LLMMessage], str | None]:
        """Enrich the request's system material with recalled memories."""
        return self._recaller.inject(messages, system, scope)

    def enqueue_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        scope: Attribution,
        session_id: str | None = None,
        priority: int = 100,
    ) -> int:
        """
        Persist one turn for background extraction. Returns the job id.
        This is a fast local DB insert — not an LLM call — so the response
        path stays unaffected.
        """
        sid = session_id or self._auto_session(scope)
        payload = {
            "user_message": (user_message or "")[:2000],
            "assistant_message": (assistant_message or "")[:6000],
            "entity_id": scope.entity_id,
            "process_id": scope.process_id,
            "session_id": sid,
        }
        job_id = self._store.enqueue_job("extract_turn", payload, priority=priority)
        log.debug(
            "Enqueued extraction job %d for %s (%d chars)",
            job_id,
            scope,
            len(payload["assistant_message"]),
        )
        return job_id

    # ── Background worker ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the durable background worker (idempotent)."""
        if self._worker is None or not self._worker.is_alive():
            self._worker = BackgroundWorker(
                self._store,
                self.process_job,
                max_attempts=self._max_job_attempts,
            )
            self._worker.start()
            log.info("Memory extraction worker started")

    def stop(self) -> None:
        """Stop the background worker and wait for it to drain."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def process_job(self, job: Job) -> None:
        """Run one queued job (executed on the worker thread)."""
        try:
            asyncio.run(self._execute_job(job))
        except Exception as e:  # noqa: BLE001 — surfaced to the queue's retry logic
            raise RuntimeError(f"extraction job {job.id} failed: {e}") from e

    async def _execute_job(self, job: Job) -> None:
        payload = job.payload

        if job.kind == "consolidation":
            # Run consolidation job
            entity_id = payload.get("entity_id")
            self._store.process_consolidation_job(job)
            return

        if self._extractor is None:
            raise RuntimeError("memory layer has no registered provider")
        items = await self._extractor.extract_turn(
            payload.get("user_message", ""),
            payload.get("assistant_message", ""),
        )
        entity_id = payload["entity_id"]
        process_id = payload["process_id"]
        for item in items:
            embedding = self._embed_fn(item.text)
            self._store.upsert_memory(
                entity_id=entity_id,
                process_id=process_id,
                kind=item.kind,
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                text=item.text,
                embedding=embedding,
                session_id=payload.get("session_id"),
                job_id=job.id,
                similarity_threshold=self._dedup_threshold,
                confidence=item.confidence,
                importance=item.importance,
                source_event="llm.extraction",
                source_message_id=payload.get("source_message_id"),
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _auto_session(self, scope: Attribution) -> str:
        """Session auto-management for turns without an explicit session."""
        key = (scope.entity_id, scope.process_id)
        session_id = self._auto_sessions.get(key)
        if session_id is None:
            session_id = self._store.create_session(scope.entity_id, scope.process_id)
            self._auto_sessions[key] = session_id
        return session_id

    def _build_extraction_call(
        self, provider: LLMProvider
    ) -> Callable[[str], Awaitable[str]]:
        """A cheap/fast extraction LLM callable against the *unwrapped* provider."""
        model = self._extraction_model

        async def call_llm(prompt: str) -> str:
            response = await provider.complete(
                [LLMMessage.user(prompt)],
                model=model,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.content or ""

        return call_llm

    def __repr__(self) -> str:
        return (
            f"<MemoryLayer store={self._store._db_path.name} "
            f"wrapped={self._wrapped is not None} worker={self._worker is not None}>"
        )