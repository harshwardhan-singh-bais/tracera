"""
Memory Layer Attribution — the (entity_id, process_id) scoping that governs
every memory read and write.

Attribution answers two questions about any LLM call:

  * **entity**  — *whose* memory is this? (a user / org / customer)
  * **process** — *which* agent feature/flow is doing the talking?
                  (e.g. ``"support_agent"``, ``"onboarding_flow"``)

It is a **hard requirement**: no attribution → no memory created and no memory
recalled. The wrapper logs a clear warning instead of failing silently.

The values are kept in ``contextvars`` (not globals) so concurrent requests or
agent loops running in different async tasks never leak attribution into each
other.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tracera.logging import get_logger

log = get_logger("memory.layer.attribution")

#: Raised when an operation requires attribution but none has been set.
class AttributionError(RuntimeError):
    """Raised when memory is accessed without an attribution scope."""


@dataclass(frozen=True)
class Attribution:
    """The immutable (entity_id, process_id) pair a memory operation belongs to."""

    entity_id: str
    process_id: str

    def __str__(self) -> str:
        return f"{self.entity_id}@{self.process_id}"

    def as_dict(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "process_id": self.process_id}


# ── Context-local state ────────────────────────────────────────────────────────
# ContextVars give per-async-task isolation: two concurrent calls in the same
# process carry independent attribution scopes.

_entity_var: ContextVar[str | None] = ContextVar("tracera_memory_entity", default=None)
_process_var: ContextVar[str | None] = ContextVar("tracera_memory_process", default=None)
_session_var: ContextVar[str | None] = ContextVar("tracera_memory_session_id", default=None)


def set_attribution(entity_id: str, process_id: str) -> Attribution:
    """
    Set the current attribution scope for this async task / thread.

    This must be called before any wrapped LLM call for memory to be created
    or recalled. Returns the resulting :class:`Attribution`.

    Raises:
        AttributionError: if either id is empty.
    """
    entity_id = (entity_id or "").strip()
    process_id = (process_id or "").strip()
    if not entity_id or not process_id:
        raise AttributionError(
            "attribution requires both an entity_id and a process_id"
        )
    _entity_var.set(entity_id)
    _process_var.set(process_id)
    scope = Attribution(entity_id, process_id)
    log.debug("Attribution set: %s", scope)
    return scope


def current_attribution() -> Attribution | None:
    """Return the current attribution scope, or None if never set."""
    entity = _entity_var.get()
    process = _process_var.get()
    if entity is None or process is None:
        return None
    return Attribution(entity, process)


def reset_attribution() -> None:
    """Clear the current attribution scope."""
    _entity_var.set(None)
    _process_var.set(None)


# ── Session state ──────────────────────────────────────────────────────────────

def set_session_id(session_id: str) -> None:
    """Associate the current task with a session id."""
    _session_var.set(session_id)


def current_session_id() -> str | None:
    """Return the session id of the current task, if any."""
    return _session_var.get()


def clear_session_id() -> None:
    """Clear the current session association."""
    _session_var.set(None)


def require_attribution() -> Attribution:
    """
    Return the current attribution or raise :class:`AttributionError`.
    Used internally by memory write paths that are not allowed to skip.
    """
    scope = current_attribution()
    if scope is None:
        raise AttributionError(
            "no attribution set — call attribution(entity_id, process_id) "
            "before any memory operation"
        )
    return scope