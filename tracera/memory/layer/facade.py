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
from collections.abc import Awaitable, Callable
from typing import Any

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
from tracera.memory.layer.reconcile import MemoryReconciler, ReconcileEvent
from tracera.memory.layer.store import ALL_KINDS, Job, MemoryRecord, MemoryStore
from tracera.memory.layer.wrapper import MemoryProvider
from tracera.providers.base import LLMMessage, LLMProvider

log = get_logger("memory.layer")

EmbedFn = Callable[[str], list[float]]


#: Predicate used when an explicit memory carries no triple of its own.
_KIND_DEFAULT_PREDICATE = {
    "fact": "stated_fact",
    "preference": "prefers",
    "rule": "must_follow",
    "skill": "can_do",
    "attribute": "has_attribute",
    "relationship": "relates_to",
    "event": "happened",
    "decision": "decided",
    "goal": "wants",
    "constraint": "limited_by",
    "experience": "learned",
}


def _infer_predicate(kind: str) -> str:
    return _KIND_DEFAULT_PREDICATE.get(kind, "stated_fact")


def _kind_from_class(memory: Any) -> str | None:
    """
    Map a taxonomy class name (``MemoryPreference``) to its memory kind.

    Explicit ``remember_memory`` calls construct typed objects rather than
    passing a kind string, and those objects carry no ``kind`` attribute.
    """
    name = type(memory).__name__
    if not name.startswith("Memory"):
        return None
    candidate = name[len("Memory") :]
    candidate = "".join(f"_{c.lower()}" if c.isupper() else c for c in candidate).lstrip("_")
    return candidate if candidate in ALL_KINDS else None


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

    def __init__(
        self, layer: MemoryLayer, *, triple_store: Any = None, session_manager: Any = None
    ):
        self._layer = layer
        self._triple_store = triple_store
        self._session_manager = session_manager

    # ── Scope resolution ─────────────────────────────────────────────────────

    def _entity_id(self) -> str:
        """
        The entity whose memory this facade reads and writes.

        Prefers the layer's active attribution (set by the CLI/TUI), then its
        default, then ``"default"``. The previous implementation hard-coded
        ``"default"`` in ``stats``/``get_by_type``/``add``, so every explicit
        memory and every statistic landed on a different entity than the one the
        recall path actually used.
        """
        scope = current_attribution() or getattr(self._layer, "_default_attribution", None)
        if scope is not None:
            return scope.entity_id
        return "default"

    def _process_id(self) -> str:
        scope = current_attribution() or getattr(self._layer, "_default_attribution", None)
        if scope is not None:
            return scope.process_id
        return "tracera-agent"

    def _embed(self, text: str) -> list[float]:
        return self._layer._embed_fn(text)

    # ── Introspection (used by action_show_memory) ──────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return memory statistics matching what action_show_memory expects."""
        store = self._layer.store
        entity_id = self._entity_id()
        grouped = store.find_by_kind(entity_id)
        by_type = {kind: len(records) for kind, records in grouped.items() if records}
        full = store.stats()
        return {
            "total": sum(by_type.values()),
            "by_type": by_type,
            "store": full,
            "entity_id": entity_id,
        }

    def count(self) -> int:
        """Number of active memories (used by status line)."""
        return self._layer.store.count_memories()

    def entries(self) -> list:
        """Current memories, newest first, as display objects."""
        records = self._layer.store.find_memories(
            self._entity_id(), limit=50, current_only=True, order="recent"
        )
        return [self._record_to_memory(r) for r in records]

    # ── Type-based retrieval (used by action_show_memory) ───────────────────

    def get_by_type(self, mtype: Any) -> list:
        """Return memories of a specific type for display."""
        kind = getattr(mtype, "value", str(mtype)).lower()
        grouped = self._layer.store.find_by_kind(self._entity_id())
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
        as_of: float | None = None,
    ) -> str:
        """
        Recall relevant memories as a formatted string.

        The query is embedded and matched against the store — this is a genuine
        relevance search. It previously ignored ``query`` entirely and returned
        the most-mentioned memories in a fixed order, so every query produced
        the same answer.
        """
        store = self._layer.store
        entity_id = self._entity_id()
        # Respect the layer's configured relevance floor so a recall for an
        # unrelated query reports "nothing found" instead of dumping whatever
        # happens to be in the store.
        threshold = getattr(getattr(self._layer, "_recaller", None), "_min_score", 0.3)
        results = store.recall_hybrid(
            entity_id,
            query,
            self._embed(query),
            k=max(1, top_k),
            min_score=threshold,
            as_of=as_of,
            graph_expansion=use_graph_expansion,
        )
        if not results:
            return "No relevant memories found."

        lines: list[str] = []
        used = 0
        for record, score in results:
            line = f"[{record.kind}] {record.text}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        body = "\n".join(lines)

        if include_triples:
            graph = store.entity_graph(entity_id, limit=10)
            nodes = graph.get("nodes", [])[:8]
            if nodes:
                body += "\n\nRelated concepts: " + ", ".join(n["canonical"] for n in nodes)

        if include_sessions and self._session_manager is not None:
            try:
                sessions = getattr(self._session_manager, "sessions", [])[:3]
                if sessions:
                    body += "\n\nRecent sessions: " + "; ".join(
                        (getattr(s, "task", "") or "")[:60] for s in sessions
                    )
            except Exception:  # noqa: BLE001 — session recall is best-effort
                pass

        return body or "No relevant memories found."

    def add(
        self,
        memory: Any,
        *,
        kind: str | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
    ) -> Any:
        """
        Store an explicit memory.

        The triple is derived from the memory object when it carries one
        (``MemoryRelationship``) and otherwise inferred from the content, so
        distinct explicit memories no longer collide on a hard-coded
        ``user | knows | <first 100 chars>`` key.
        """
        text = ""
        if hasattr(memory, "content"):
            text = str(memory.content)
        elif isinstance(memory, dict):
            text = str(memory.get("text") or memory.get("content") or "")
        else:
            text = str(memory)

        memory_kind = kind
        if not memory_kind and isinstance(memory, dict):
            memory_kind = memory.get("kind")
        if not memory_kind:
            memory_kind = getattr(getattr(memory, "memory_type", None), "value", None)
        if not memory_kind:
            memory_kind = getattr(memory, "kind", None) or _kind_from_class(memory) or "fact"
        memory_kind = str(memory_kind).lower()
        if memory_kind not in ALL_KINDS:
            memory_kind = "fact"

        subject = "user"
        predicate = None
        obj = None
        if isinstance(memory, dict):
            subject = memory.get("subject") or "user"
            predicate = memory.get("predicate")
            obj = memory.get("object")
        else:
            subject = getattr(memory, "subject", None) or "user"
            predicate = getattr(memory, "predicate", None)
            obj = getattr(memory, "object", None)
        predicate = predicate or _infer_predicate(memory_kind)
        obj = obj or text[:120]

        embedding = self._embed(text)
        _inserted, record = self._layer.store.upsert_memory(
            entity_id=self._entity_id(),
            process_id=self._process_id(),
            kind=memory_kind,
            subject=str(subject),
            predicate=str(predicate),
            object=str(obj),
            text=text,
            embedding=embedding,
            confidence=float(getattr(memory, "confidence", confidence) or confidence),
            importance=float(getattr(memory, "importance", importance) or importance),
            source_event="explicit.remember",
            source_kind="explicit",
        )
        return record

    def delete(self, memory_id: str) -> bool:
        """Soft-delete a memory by ID (invalidated, not erased)."""
        try:
            return self._layer.store.delete_memory(int(memory_id))
        except (TypeError, ValueError):
            return False

    def update(
        self,
        memory_id: int,
        *,
        text: str,
        reason: str = "explicit update",
    ) -> Any:
        """Supersede a memory with corrected text."""
        return self._layer.store.supersede_memory(
            int(memory_id),
            new_text=text,
            new_embedding=self._embed(text),
            reason=reason,
            source_process=self._process_id(),
        )

    def feedback(self, memory_id: int, signal: str, *, query: str | None = None) -> dict[str, Any]:
        """Record whether a recalled memory was useful, harmful or irrelevant."""
        return self._layer.store.record_feedback(int(memory_id), signal, query=query)

    def timeline(self, *, subject: str | None = None, predicate: str | None = None) -> list:
        """Chronological history of a fact, including invalidated versions."""
        return self._layer.store.timeline(
            self._entity_id(), subject=subject, predicate=predicate
        )

    def entities(self, node: str | None = None) -> dict[str, Any]:
        """Canonical entities / memory graph for this scope."""
        return self._layer.store.entity_graph(self._entity_id(), node=node)

    def export(self, *, include_invalidated: bool = True) -> dict[str, Any]:
        """Export this entity's memories and graph as JSON."""
        return self._layer.store.export_entity(
            self._entity_id(), include_invalidated=include_invalidated
        )

    def import_data(self, payload: dict[str, Any]) -> dict[str, int]:
        """Import an export payload, re-embedding with the local embedder."""
        return self._layer.store.import_entity(payload, embed_fn=self._embed)

    def search(self, query: str, top_k: int = 10) -> list:
        """Search memories by semantic similarity."""
        embedding = self._embed(query)
        return self._layer.store.recall(self._entity_id(), embedding, k=top_k, min_score=0.1)

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
            MemoryAttribute,
            MemoryConstraint,
            MemoryDecision,
            MemoryEvent,
            MemoryExperience,
            MemoryFact,
            MemoryGoal,
            MemoryPreference,
            MemoryRelationship,
            MemoryRule,
            MemorySkill,
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
                subject=getattr(record, "subject", ""),
                predicate=getattr(record, "predicate", ""),
                object=getattr(record, "object", ""),
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
        recall_graph_expansion: bool = False,
        recall_graph_hops: int = 1,
        # Reconciliation config (Mem0-style ADD/UPDATE/DELETE/NOOP)
        enable_reconciliation: bool = True,
        reconciliation_candidates: int = 5,
        reconciliation_min_similarity: float = 0.55,
        # Maintenance config
        decay_half_life_days: float = 90.0,
        retention_days: int = 365,
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
        self._enable_reconciliation = enable_reconciliation
        self._reconciliation_candidates = max(1, reconciliation_candidates)
        self._reconciliation_min_similarity = reconciliation_min_similarity
        self._decay_half_life_days = decay_half_life_days
        self._retention_days = retention_days
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
            graph_expansion=recall_graph_expansion,
            graph_hops=recall_graph_hops,
        )
        self._extractor: MemoryExtractor | None = None
        self._reconciler: MemoryReconciler | None = None
        self._extraction_provider: LLMProvider | None = None
        self._wrapped: MemoryProvider | None = None
        self._worker: BackgroundWorker | None = None
        self._auto_sessions: dict[tuple[str, str], str] = {}
        #: Recent reconciliation decisions, newest last (inspection surface).
        self._reconcile_log: list[dict[str, Any]] = []

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
        self._reconciler = MemoryReconciler(
            self._build_extraction_call(provider),
            use_llm=self._enable_reconciliation,
            noop_similarity=self._dedup_threshold,
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
        if self._enabled_processes is not None and scope.process_id not in self._enabled_processes:
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
            self._store.process_consolidation_job(job)
            return

        if job.kind == "maintenance":
            self.run_maintenance(dry_run=bool(payload.get("dry_run")))
            return

        if self._extractor is None:
            raise RuntimeError("memory layer has no registered provider")
        items = await self._extractor.extract_turn(
            payload.get("user_message", ""),
            payload.get("assistant_message", ""),
        )
        entity_id = payload["entity_id"]
        process_id = payload["process_id"]
        session_id = payload.get("session_id")
        for item in items:
            await self.apply_memory(
                item,
                entity_id=entity_id,
                process_id=process_id,
                session_id=session_id,
                job_id=job.id,
                source_message_id=payload.get("source_message_id"),
            )

    async def apply_memory(
        self,
        item: Any,
        *,
        entity_id: str,
        process_id: str,
        session_id: str | None = None,
        job_id: int | None = None,
        source_message_id: str | None = None,
        source_event: str = "llm.extraction",
    ) -> MemoryRecord | None:
        """
        Reconcile one extracted memory against the store, then apply the decision.

        This is where ADD/UPDATE/DELETE/NOOP is honoured:

        * ``NOOP``   → reinforce the existing row (mention_count++, refresh text)
        * ``UPDATE`` → invalidate the superseded row, then insert the new value
        * ``DELETE`` → invalidate the row outright
        * ``ADD``    → plain insert (with the store's own dedup as a backstop)

        Returns the resulting record, or None when nothing was written.
        """
        text = item.text
        embedding = self._embed_fn(text)
        candidate = {
            "kind": item.kind,
            "subject": item.subject,
            "predicate": item.predicate,
            "object": item.object,
            "text": text,
            "confidence": item.confidence,
            "importance": item.importance,
        }

        similar: list[MemoryRecord] = []
        if self._reconciler is not None:
            hits = self._store.recall_hybrid(
                entity_id,
                text,
                embedding,
                k=self._reconciliation_candidates,
                min_score=self._reconciliation_min_similarity,
                record_hits=False,
            )
            similar = [rec for rec, _ in hits]
            action = await self._reconciler.reconcile(candidate, similar)
        else:
            # No provider registered (e.g. a store-only embedding path) — fall
            # back to the deterministic rules so behaviour stays consistent.
            action = MemoryReconciler(use_llm=False).reconcile_deterministic(candidate, similar)

        self._reconcile_log.append(
            {
                "text": text[:200],
                "event": action.event.value,
                "memory_id": action.memory_id,
                "reason": action.reason,
                "source": action.source,
                "entity_id": entity_id,
            }
        )
        if len(self._reconcile_log) > 200:
            del self._reconcile_log[:-200]

        if action.event is ReconcileEvent.NOOP and action.memory_id is not None:
            existing = self._store.get_memory(action.memory_id)
            if existing is not None:
                self._store.upsert_memory(
                    entity_id=entity_id,
                    process_id=process_id,
                    kind=existing.kind,
                    subject=existing.subject,
                    predicate=existing.predicate,
                    object=existing.object,
                    text=existing.text,
                    embedding=embedding,
                    session_id=session_id,
                    job_id=job_id,
                    similarity_threshold=1.1,  # force the reinforcement path
                    confidence=max(existing.confidence, item.confidence),
                    importance=max(existing.importance, item.importance),
                    source_event=source_event,
                    source_message_id=source_message_id,
                    source_kind="reconciliation",
                )
                return existing

        if action.event is ReconcileEvent.DELETE and action.memory_id is not None:
            self._store.delete_memory(action.memory_id, reason=f"reconciliation: {action.reason}")
            return None

        if action.event is ReconcileEvent.UPDATE and action.memory_id is not None:
            # replace_memory (not supersede_memory) keeps the previous value
            # queryable via recall_as_of — the whole point of a temporal store.
            self._store.replace_memory(
                action.memory_id,
                new_text=text,
                new_embedding=embedding,
                reason=f"reconciliation: {action.reason}",
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                kind=item.kind,
                confidence=item.confidence,
                importance=item.importance,
                session_id=session_id,
                process_id=process_id,
                job_id=job_id,
            )
            return self._store.get_memory(action.memory_id)

        _inserted, record = self._store.upsert_memory(
            entity_id=entity_id,
            process_id=process_id,
            kind=item.kind,
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            text=text,
            embedding=embedding,
            session_id=session_id,
            job_id=job_id,
            similarity_threshold=self._dedup_threshold,
            confidence=item.confidence,
            importance=item.importance,
            source_event=source_event,
            source_message_id=source_message_id,
            source_kind="reconciliation",
        )
        return record

    # ── Maintenance ──────────────────────────────────────────────────────────

    def run_maintenance(
        self,
        *,
        entity_id: str | None = None,
        dry_run: bool = False,
        consolidate: bool = True,
    ) -> dict[str, Any]:
        """
        Recompute decay, collect stale memories, and merge near-duplicates.

        Intended to run on a schedule (or via ``memory_maintenance``); it is the
        piece that keeps a long-lived store from growing without bound — the
        gap every memory server in this market currently leaves open.
        """
        report: dict[str, Any] = {
            "decay_updated": self._store.apply_decay(half_life_days=self._decay_half_life_days),
            "gc": self._store.run_gc(retention_days=self._retention_days, dry_run=dry_run),
        }
        if consolidate:
            report["consolidation"] = self._store.run_consolidation(
                entity_id=entity_id, dry_run=dry_run
            )
        log.info("Memory maintenance: %s", report)
        return report

    @property
    def reconciliation_log(self) -> list[dict[str, Any]]:
        """Recent reconciliation decisions (what the layer chose and why)."""
        return list(self._reconcile_log)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _auto_session(self, scope: Attribution) -> str:
        """Session auto-management for turns without an explicit session."""
        key = (scope.entity_id, scope.process_id)
        session_id = self._auto_sessions.get(key)
        if session_id is None:
            session_id = self._store.create_session(scope.entity_id, scope.process_id)
            self._auto_sessions[key] = session_id
        return session_id

    def _build_extraction_call(self, provider: LLMProvider) -> Callable[[str], Awaitable[str]]:
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
