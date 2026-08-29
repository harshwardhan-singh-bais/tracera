"""
Memory Layer Recall — the read path executed *before* each LLM call.

1. Embed the outbound user message.
2. Hybrid-search the entity's stored memories (vector + keyword + metadata).
3. Take the top-k most relevant with token budget awareness.
4. Inject them into the system prompt in a clearly delimited block.

This is what makes the agent "remember" without any manual RAG code at the
call site — the wrapper handles everything.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from tracera.logging import get_logger
from tracera.memory.layer.attribution import Attribution
from tracera.memory.layer.store import MemoryRecord, MemoryStore
from tracera.providers.base import LLMMessage, Role

log = get_logger("memory.layer.recall")

#: Prompt-injection block header (Memori-style, clearly delimited).
INJECTION_HEADER = "Known context about this user:"
#: Approximate chars per token for budgeting
CHARS_PER_TOKEN = 4


EmbedFn = Callable[[str], list[float]]


def format_memories(results: list[tuple[MemoryRecord, float]]) -> str:
    """Render recalled memories as an injection block for the system prompt."""
    lines = [INJECTION_HEADER]
    for record, _score in results:
        lines.append(f"- {record.to_line()}")
    return "\n".join(lines)


def format_memories_grouped(results: list[tuple[MemoryRecord, float]]) -> str:
    """Render recalled memories grouped by kind for better readability."""
    if not results:
        return ""
    lines = [INJECTION_HEADER]
    by_kind: dict[str, list[tuple[MemoryRecord, float]]] = {}
    for record, score in results:
        by_kind.setdefault(record.kind, []).append((record, score))

    kind_order = ["fact", "preference", "rule", "decision", "constraint", "skill",
                  "attribute", "relationship", "event", "goal", "experience"]
    for kind in kind_order:
        if kind not in by_kind:
            continue
        group = by_kind[kind]
        lines.append(f"\n  [{kind.upper()}]")
        for record, score in group:
            conf = f" (conf: {record.confidence:.2f})" if record.confidence < 0.9 else ""
            lines.append(f"  - {record.text}{conf}")
    return "\n".join(lines)


def last_user_text(messages: list[LLMMessage]) -> str:
    """Extract the most recent user-role text from a message list."""
    for message in reversed(messages):
        if message.role == Role.USER and message.content:
            return message.content
    return ""


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return max(1, len(text) // CHARS_PER_TOKEN)


class RecallInjector:
    """
    Enriches a request's system material with the entity's top-k memories.

    Heavily defensive by design: retrieval problems must never break the LLM
    call, so any exception degrades to the plain (unenriched) request.
    """

    def __init__(
        self,
        store: MemoryStore,
        embed_fn: EmbedFn,
        *,
        top_k: int = 5,
        min_score: float = 0.3,
        use_hybrid: bool = True,
        token_budget: int = 2000,
        grouped: bool = True,
        debug: bool = False,
    ) -> None:
        self._store = store
        self._embed = embed_fn
        self._top_k = max(1, top_k)
        self._min_score = min_score
        self._use_hybrid = use_hybrid
        self._token_budget = token_budget
        self._grouped = grouped
        self._debug = debug

    def inject(
        self,
        messages: list[LLMMessage],
        system: str | None,
        scope: Attribution,
    ) -> tuple[list[LLMMessage], str | None]:
        """
        Return (messages, system) with the memory block appended when any
        memories match the outbound query for this entity.
        """
        query = last_user_text(messages)
        if not query:
            return messages, system

        query_embedding = self._embed(query)

        try:
            if self._use_hybrid and hasattr(self._store, "recall_hybrid"):
                results = self._store.recall_hybrid(
                    scope.entity_id,
                    query,
                    query_embedding,
                    k=self._top_k,
                    min_score=self._min_score,
                )
            else:
                results = self._store.recall(
                    scope.entity_id,
                    query_embedding,
                    k=self._top_k,
                    min_score=self._min_score,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("Memory recall failed, skipping injection: %s", e)
            return messages, system

        if not results:
            return messages, system

        # Apply token budget
        results = self._apply_token_budget(results)

        if not results:
            return messages, system

        block = format_memories_grouped(results) if self._grouped else format_memories(results)
        log.debug(
            "Recalled %d memory(ies) for entity %s into system prompt (~%d tokens)",
            len(results),
            scope.entity_id,
            estimate_tokens(block),
        )
        return self._attach(messages, system, block)

    def debug_recall(
        self,
        query: str,
        scope: Attribution,
        *,
        k: int | None = None,
    ) -> dict[str, Any]:
        """
        Debug version of recall that returns detailed scoring breakdown.

        Returns a dict with:
        - query: the input query
        - results: list of dicts with memory details and score breakdown
        - total_candidates: number of memories considered
        - timing_ms: how long the recall took
        """
        import time as _time
        k = k or self._top_k

        start = _time.perf_counter()
        query_embedding = self._embed(query)

        if self._use_hybrid and hasattr(self._store, "recall_hybrid"):
            results = self._store.recall_hybrid(
                scope.entity_id,
                query,
                query_embedding,
                k=k * 3,  # Get more for debug
                min_score=0.0,  # No threshold for debug
            )
        else:
            # For debug, we need to call the underlying store directly
            entity_pk = self._store.register_entity(scope.entity_id)
            with self._store._lock:
                conn = self._store._conn()
                rows = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? AND status = 'active'",
                    (entity_pk,),
                ).fetchall()

            results = []
            for row in rows:
                try:
                    other = json.loads(row["embedding"])
                except (TypeError, ValueError):
                    continue
                score = self._store.cosine_similarity(query_embedding, other)
                if score >= 0.0:
                    record = self._store._record_from_row(conn, row)
                    results.append((record, score))

            results.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (_time.perf_counter() - start) * 1000

        # Build detailed breakdown
        detailed = []
        for record, score in results[:k]:
            # Get hybrid score components if available
            detail = {
                "memory_id": record.id,
                "kind": record.kind,
                "text": record.text,
                "triple": f"{record.subject} → {record.predicate} → {record.object}",
                "final_score": round(score, 4),
                "mention_count": record.mention_count,
                "confidence": record.confidence,
                "importance": record.importance,
                "first_seen": record.first_seen_at,
                "last_seen": record.last_seen_at,
                "source_event": record.source_event,
            }
            if hasattr(self._store, "recall_hybrid") and self._use_hybrid:
                # Try to compute hybrid components
                detail["note"] = "Hybrid scoring active (vector + keyword + recency + importance + mentions)"
            detailed.append(detail)

        return {
            "query": query,
            "entity_id": scope.entity_id,
            "process_id": scope.process_id,
            "total_candidates": len(results),
            "returned": len(detailed),
            "timing_ms": round(elapsed_ms, 2),
            "results": detailed,
            "min_score_threshold": self._min_score,
            "top_k": k,
        }

    def _apply_token_budget(self, results: list[tuple[MemoryRecord, float]]) -> list[tuple[MemoryRecord, float]]:
        """Trim results to fit within token budget."""
        budget = self._token_budget
        kept = []
        for record, score in results:
            line = record.to_line()
            line_tokens = estimate_tokens(line)
            if line_tokens > budget:
                # Truncate the line
                max_chars = budget * CHARS_PER_TOKEN
                line = line[:max_chars] + "..."
                line_tokens = estimate_tokens(line)
            if line_tokens <= budget:
                kept.append((record, score))
                budget -= line_tokens
            else:
                break
        return kept

    @staticmethod
    def _attach(
        messages: list[LLMMessage],
        system: str | None,
        block: str,
    ) -> tuple[list[LLMMessage], str | None]:
        """
        Prefer the ``system`` kwarg; otherwise merge into an existing system
        message; otherwise prepend a new one.
        """
        if system is not None:
            return messages, system + "\n\n" + block
        for i, message in enumerate(messages):
            if message.role == Role.SYSTEM:
                merged = list(messages)
                merged[i] = LLMMessage.system(
                    f"{message.content or ''}\n\n{block}"
                )
                return merged, None
        return [LLMMessage.system(block), *messages], None


class RecallConfig:
    """Configuration for recall behavior per process/session."""

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.3,
        use_hybrid: bool = True,
        token_budget: int = 2000,
        grouped: bool = True,
        enabled: bool = True,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.token_budget = token_budget
        self.grouped = grouped
        self.enabled = enabled