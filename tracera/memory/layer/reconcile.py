"""
Memory Layer Reconciliation — the ADD / UPDATE / DELETE / NOOP decision.

Extraction answers *"what is worth remembering?"*. Reconciliation answers the
harder question Mem0 built its reputation on: *"given what I already believe,
what should I actually do with this?"*

Without this step a store is append-only in practice. A user says "I use
PostgreSQL", then a month later "I've moved to SQLite", and both facts sit side
by side forever, each as confident as the other. Cosine similarity alone cannot
fix that — 0.86-similar sentences can still contradict each other.

Two implementations:

* :meth:`MemoryReconciler.reconcile` — asks a small/cheap LLM, given the
  candidate plus the nearest existing memories, to return one of the four
  operations with a one-line reason. This handles the cases heuristics miss
  ("moved to", "actually it's", "no longer").
* :meth:`MemoryReconciler.reconcile_deterministic` — the always-available
  fallback. Safe, explainable, and used whenever no LLM is wired up or the call
  fails. It never guesses: ambiguity resolves to ADD.

Both paths return the same :class:`ReconcileAction`, so the caller does not care
which one ran.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from tracera.logging import get_logger
from tracera.memory.layer.store import (
    MemoryRecord,
    _is_functional_predicate,
    canonical_key,
)

log = get_logger("memory.layer.reconcile")

LLMCallFn = Callable[[str], Awaitable[str]]


class ReconcileEvent(str, Enum):
    """The four operations a reconciliation decision can produce."""

    ADD = "ADD"  # new information — insert a new memory
    UPDATE = "UPDATE"  # supersedes an existing memory — invalidate + insert
    DELETE = "DELETE"  # existing memory is now wrong and has no replacement
    NOOP = "NOOP"  # already known — reinforce the existing memory


@dataclass(frozen=True)
class ReconcileAction:
    """One reconciliation decision, with the reasoning kept for the audit trail."""

    event: ReconcileEvent
    memory_id: int | None = None
    reason: str = ""
    source: str = "deterministic"  # deterministic | llm | fallback

    @property
    def is_write(self) -> bool:
        return self.event in (ReconcileEvent.ADD, ReconcileEvent.UPDATE)

    def as_dict(self) -> dict[str, object]:
        return {
            "event": self.event.value,
            "memory_id": self.memory_id,
            "reason": self.reason,
            "source": self.source,
        }


RECONCILE_PROMPT = """You maintain a long-term memory store about a user and their projects.

You are given a NEW fact extracted from the latest conversation, and the
EXISTING memories that are semantically closest to it.

Decide what to do. Reply with a single JSON object:

{{"event": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "memory_id": <id of the affected existing memory, or null>,
  "reason": "<one short sentence>"}}

Rules:
- ADD    — the new fact is genuinely new information. Use this when nothing
           existing covers it, or when the existing memories are about a
           different thing.
- UPDATE — the new fact supersedes a specific existing memory (the value
           changed: a move, a rename, a correction, a version bump). Set
           memory_id to that memory. Do NOT use UPDATE for facts that can
           legitimately accumulate (things a user likes, files a project
           contains, events that happened).
- DELETE — an existing memory is now known to be false and nothing replaces it.
- NOOP   — the existing memory already says this. Prefer NOOP over ADD whenever
           the meaning is the same, even if the wording differs.

Return only the JSON object, no prose and no code fences.

NEW FACT:
{new_fact}

EXISTING MEMORIES:
{existing}
"""


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_decision(raw: str, valid_ids: set[int]) -> ReconcileAction | None:
    """Parse the model's JSON reply into an action, rejecting anything unsafe."""
    text = _strip_code_fences(raw)
    if not text:
        return None
    # Tolerate a model that wraps the object in a list or adds trailing prose.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        event = ReconcileEvent(str(data.get("event", "")).strip().upper())
    except ValueError:
        return None
    memory_id = data.get("memory_id")
    try:
        memory_id = int(memory_id) if memory_id is not None else None
    except (TypeError, ValueError):
        memory_id = None
    # A destructive event that names a memory we never offered is untrustworthy.
    if event in (ReconcileEvent.UPDATE, ReconcileEvent.DELETE):
        if memory_id is None or memory_id not in valid_ids:
            return None
    else:
        memory_id = None
    reason = str(data.get("reason", "")).strip()[:300]
    return ReconcileAction(event=event, memory_id=memory_id, reason=reason, source="llm")


class MemoryReconciler:
    """
    Turns a candidate memory + its nearest neighbours into one action.

    ``call_llm`` is optional: without it the reconciler silently uses the
    deterministic rules, so the memory layer keeps working with no LLM budget
    and in offline tests.
    """

    def __init__(
        self,
        call_llm: LLMCallFn | None = None,
        *,
        use_llm: bool = True,
        update_confidence_margin: float = 0.1,
        noop_similarity: float = 0.9,
    ) -> None:
        self._call_llm = call_llm
        self._use_llm = use_llm and call_llm is not None
        self._margin = update_confidence_margin
        self._noop_similarity = noop_similarity

    async def reconcile(
        self,
        candidate: dict[str, object],
        similar: list[MemoryRecord],
    ) -> ReconcileAction:
        """
        Decide what to do with ``candidate`` given the ``similar`` memories.

        Falls back to the deterministic rules on any LLM failure — a broken or
        rate-limited reconciler must never stall the extraction worker.
        """
        if not similar:
            return ReconcileAction(
                ReconcileEvent.ADD, reason="no existing memory is close enough", source="deterministic"
            )
        if not self._use_llm:
            return self.reconcile_deterministic(candidate, similar)

        valid_ids = {m.id for m in similar}
        prompt = RECONCILE_PROMPT.format(
            new_fact=json.dumps(candidate, ensure_ascii=False, indent=2),
            existing=json.dumps(
                [
                    {
                        "id": m.id,
                        "kind": m.kind,
                        "text": m.text,
                        "triple": f"{m.subject} → {m.predicate} → {m.object}",
                        "confidence": round(m.confidence, 2),
                        "mentions": m.mention_count,
                        "valid_window": m.valid_window(),
                    }
                    for m in similar
                ],
                ensure_ascii=False,
                indent=2,
            ),
        )
        try:
            raw = await self._call_llm(prompt)  # type: ignore[misc]
            action = _parse_decision(raw, valid_ids)
            if action is None:
                log.debug("Reconciler reply unparseable; using deterministic rules")
                fallback = self.reconcile_deterministic(candidate, similar)
                return ReconcileAction(
                    event=fallback.event,
                    memory_id=fallback.memory_id,
                    reason=fallback.reason,
                    source="fallback",
                )
            log.debug(
                "Reconciler chose %s for %r (memory_id=%s)",
                action.event.value,
                str(candidate.get("text", ""))[:60],
                action.memory_id,
            )
            return action
        except Exception as e:  # noqa: BLE001 — reconciliation must never be fatal
            log.warning("Reconciler LLM call failed (%s); using deterministic rules", str(e)[:150])
            fallback = self.reconcile_deterministic(candidate, similar)
            return ReconcileAction(
                event=fallback.event,
                memory_id=fallback.memory_id,
                reason=fallback.reason,
                source="fallback",
            )

    def reconcile_deterministic(
        self,
        candidate: dict[str, object],
        similar: list[MemoryRecord],
    ) -> ReconcileAction:
        """
        Rule-based decision, in priority order.

        The rules are deliberately conservative — an ambiguous case becomes ADD,
        because a duplicate memory is a much cheaper mistake than a deleted
        belief.
        """
        if not similar:
            return ReconcileAction(
                ReconcileEvent.ADD, reason="nothing similar exists", source="deterministic"
            )

        new_obj = canonical_key(str(candidate.get("object", "")))
        new_pred = str(candidate.get("predicate", "")).strip().lower()
        new_subj = canonical_key(str(candidate.get("subject", "")))
        new_conf = float(candidate.get("confidence", 0.8) or 0.0)

        for mem in similar:
            same_predicate = mem.predicate.strip().lower() == new_pred
            same_subject = canonical_key(mem.subject) == new_subj
            same_object = canonical_key(mem.object) == new_obj

            # 1) Same triple, same meaning → reinforce, never duplicate.
            if same_predicate and same_subject and same_object:
                return ReconcileAction(
                    ReconcileEvent.NOOP,
                    memory_id=mem.id,
                    reason="existing memory states the same fact",
                    source="deterministic",
                )

            # 2) Same functional predicate, different value, at least as
            #    confident → the value was replaced.
            if (
                same_predicate
                and same_subject
                and not same_object
                and _is_functional_predicate(new_pred)
                and new_conf + self._margin >= mem.confidence
            ):
                return ReconcileAction(
                    ReconcileEvent.UPDATE,
                    memory_id=mem.id,
                    reason=(
                        f"single-valued relation {new_pred!r} changed value "
                        f"({mem.object!r} → {candidate.get('object')!r})"
                    ),
                    source="deterministic",
                )

        # 3) A near-identical paraphrase that is not an exact triple match is
        #    still the same belief when both subject and predicate line up.
        for mem in similar:
            if (
                canonical_key(mem.subject) == new_subj
                and mem.predicate.strip().lower() == new_pred
            ):
                return ReconcileAction(
                    ReconcileEvent.NOOP,
                    memory_id=mem.id,
                    reason="same subject and relation already recorded",
                    source="deterministic",
                )

        return ReconcileAction(
            ReconcileEvent.ADD,
            reason="no existing memory covers this fact",
            source="deterministic",
        )
