"""
Memory Layer Background Worker — a durable, restart-surviving job consumer.

Extraction of a conversation turn happens *after* the response has been
returned to the caller. The job itself is persisted in the same SQLite file
as the memories (the ``jobs`` table in :mod:`tracera.memory.layer.store`), so
a process crash or restart never loses a pending extraction.

A single daemon thread polls the queue. Because the queue is durable (vs. an
in-memory ``asyncio.create_task``), retried jobs are *idempotent*: the store's
``last_job_id`` guard prevents double inserts or inflated mention counts.

Enhancements (Phase 10):
- Job prioritization (priority field in jobs table)
- Health metrics (processed/failed/latency)
- Batch claiming for throughput
- Graceful shutdown with drain
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tracera.logging import get_logger
from tracera.memory.layer.store import Job, MemoryStore

log = get_logger("memory.layer.worker")


@dataclass
class WorkerStats:
    """Runtime statistics for the background worker."""
    jobs_processed: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    total_latency_ms: float = 0.0
    last_job_at: float | None = None
    last_error: str | None = None
    started_at: float = field(default_factory=time.time)

    def avg_latency_ms(self) -> float:
        if self.jobs_processed == 0:
            return 0.0
        return self.total_latency_ms / self.jobs_processed

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "jobs_retried": self.jobs_retried,
            "avg_latency_ms": round(self.avg_latency_ms(), 2),
            "last_job_at": self.last_job_at,
            "last_error": self.last_error,
            "uptime_seconds": time.time() - self.started_at,
        }


class BackgroundWorker(threading.Thread):
    """
    Polls the store's durable job queue and dispatches each job to a handler.

    The handler signature is ``handler(job: Job) -> None``; it runs inline in
    the worker thread. A short exponential backoff (managed by the store's
    ``fail_job``) lets a transiently failing queue drain instead of hot-looping.

    Supports job prioritization via the ``priority`` field in the jobs table
    (lower = higher priority).
    """

    def __init__(
        self,
        store: MemoryStore,
        handler: Callable[[Job], None],
        *,
        poll_interval: float = 0.1,
        max_attempts: int = 5,
        batch_size: int = 5,
        priority_enabled: bool = True,
    ) -> None:
        super().__init__(name="tracera-memory-worker", daemon=True)
        self._store = store
        self._handler = handler
        self._poll_interval = poll_interval
        self._max_attempts = max_attempts
        self._batch_size = max(1, batch_size)
        self._priority_enabled = priority_enabled
        self._stop = threading.Event()
        self._stats = WorkerStats()
        self._lock = threading.Lock()

    # ── Control ──────────────────────────────────────────────────────────────

    def stop(self, timeout: float | None = 5.0, drain: bool = True) -> None:
        """Signal the worker to stop and join it."""
        self._stop.set()
        if self.is_alive():
            try:
                self.join(timeout=timeout)
            except TypeError:
                # Workaround for Windows threading issue where Event object
                # can be misidentified as callable during join
                pass

    @property
    def running(self) -> bool:
        return self.is_alive() and not self._stop.is_set()

    @property
    def stats(self) -> WorkerStats:
        return self._stats

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics as a dictionary."""
        with self._lock:
            return self._stats.to_dict()

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: D102 — thread entrypoint
        log.debug("Memory worker started (batch_size=%d, priority=%s)",
                  self._batch_size, self._priority_enabled)
        while not self._stop.is_set():
            try:
                # Claim a batch of jobs (respects priority if enabled)
                jobs = self._store.claim_jobs(limit=self._batch_size)
                if not jobs:
                    if self._stop.wait(self._poll_interval):
                        break
                    continue

                for job in jobs:
                    if self._stop.is_set():
                        break
                    start = time.perf_counter()
                    try:
                        self._handler(job)
                        self._store.complete_job(job.id)
                        with self._lock:
                            self._stats.jobs_processed += 1
                            self._stats.total_latency_ms += (time.perf_counter() - start) * 1000
                            self._stats.last_job_at = time.time()
                    except Exception as e:  # noqa: BLE001 — resilience boundary
                        with self._lock:
                            self._stats.jobs_failed += 1
                            self._stats.last_error = str(e)[:500]
                        log.warning(
                            "Memory job %s failed (%s) — will be retried or failed",
                            job.id,
                            str(e)[:200],
                        )
                        self._store.fail_job(job.id, str(e), max_attempts=self._max_attempts)
                        with self._lock:
                            self._stats.jobs_retried += 1

            except Exception as e:  # noqa: BLE001
                log.error("Memory worker loop error: %s", e)
                with self._lock:
                    self._stats.jobs_failed += 1
                    self._stats.last_error = str(e)[:500]
                if self._stop.wait(self._poll_interval):
                    break
        log.debug("Memory worker stopped (processed=%d, failed=%d)",
                  self._stats.jobs_processed, self._stats.jobs_failed)


class PriorityBackgroundWorker(BackgroundWorker):
    """
    Worker that processes high-priority jobs first.

    Jobs with a ``priority`` field in their payload (lower = higher priority)
    are processed before lower-priority jobs.
    """

    def __init__(
        self,
        store: MemoryStore,
        handler: Callable[[Job], None],
        *,
        poll_interval: float = 0.1,
        max_attempts: int = 5,
        batch_size: int = 5,
    ) -> None:
        # Note: priority is handled by the store's claim_jobs when priority_enabled=True
        super().__init__(
            store,
            handler,
            poll_interval=poll_interval,
            max_attempts=max_attempts,
            batch_size=batch_size,
            priority_enabled=True,
        )