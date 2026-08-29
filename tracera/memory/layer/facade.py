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
from tracera.memory.layer.store import Job, MemoryStore
from tracera.memory.layer.wrapper import MemoryProvider
from tracera.providers.base import LLMMessage, LLMProvider

log = get_logger("memory.layer")

EmbedFn = Callable[[str], list[float]]


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
        self._enabled_processes = (
            set(p.strip() for p in enabled_processes if p and p.strip())
            if enabled_processes
            else None
        )
        self._recaller = RecallInjector(
            store, embed_fn, top_k=top_k, min_score=min_recall_score
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
        job_id = self._store.enqueue_job("extract_turn", payload)
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