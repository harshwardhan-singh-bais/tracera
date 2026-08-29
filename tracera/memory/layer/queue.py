"""
Memory Layer Background Worker — a durable, restart-surviving job consumer.

Extraction of a conversation turn happens *after* the response has been
returned to the caller. The job itself is persisted in the same SQLite file
as the memories (the ``jobs`` table in :mod:`tracera.memory.layer.store`), so
a process crash or restart never loses a pending extraction.

A single daemon thread polls the queue. Because the queue is durable (vs. an
in-memory ``asyncio.create_task``), retried jobs are *idempotent*: the store's
``last_job_id`` guard prevents double inserts or inflated mention counts.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from tracera.logging import get_logger
from tracera.memory.layer.store import Job, MemoryStore

log = get_logger("memory.layer.worker")


class BackgroundWorker(threading.Thread):
    """
    Polls the store's durable job queue and dispatches each job to a handler.

    The handler signature is ``handler(job: Job) -> None``; it runs inline in
    the worker thread. A short exponential backoff (managed by the store's
    ``fail_job``) lets a transiently failing queue drain instead of hot-looping.
    """

    def __init__(
        self,
        store: MemoryStore,
        handler: Callable[[Job], None],
        *,
        poll_interval: float = 0.1,
        max_attempts: int = 5,
    ) -> None:
        super().__init__(name="tracera-memory-worker", daemon=True)
        self._store = store
        self._handler = handler
        self._poll_interval = poll_interval
        self._max_attempts = max_attempts
        self._stop = threading.Event()

    # ── Control ──────────────────────────────────────────────────────────────

    def stop(self, timeout: float | None = 5.0) -> None:
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

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: D102 — thread entrypoint
        log.debug("Memory worker started")
        while not self._stop.is_set():
            try:
                jobs = self._store.claim_jobs(limit=1)
                if not jobs:
                    if self._stop.wait(self._poll_interval):
                        break
                    continue
                job = jobs[0]
                try:
                    self._handler(job)
                    self._store.complete_job(job.id)
                except Exception as e:  # noqa: BLE001 — resilience boundary
                    log.warning(
                        "Memory job %s failed (%s) — will be retried or failed",
                        job.id,
                        str(e)[:200],
                    )
                    self._store.fail_job(job.id, str(e), max_attempts=self._max_attempts)
            except Exception as e:  # noqa: BLE001
                log.error("Memory worker loop error: %s", e)
                if self._stop.wait(self._poll_interval):
                    break
        log.debug("Memory worker stopped")