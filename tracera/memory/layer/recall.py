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

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

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

    kind_order = [
        "fact",
        "preference",
        "rule",
        "decision",
        "constraint",
        "skill",
        "attribute",
        "relationship",
        "event",
        "goal",
        "experience",
    ]
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
        as_of: float | None = None,
        graph_expansion: bool = False,
        graph_hops: int = 1,
    ) -> None:
        self._store = store
        self._embed = embed_fn
        self._top_k = max(1, top_k)
        self._min_score = min_score
        self._use_hybrid = use_hybrid
        self._token_budget = token_budget
        self._grouped = grouped
        self._debug = debug
        self._as_of = as_of
        self._graph_expansion = graph_expansion
        self._graph_hops = graph_hops

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

        try:
            results = self.search(query, scope, k=self._top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("Memory recall failed, skipping injection: %s", e)
            return messages, system

        if not results:
            return messages, system

        # Apply token budget — the block must fit, so records that overflow are
        # truncated (or dropped) rather than injected whole.
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

    def search(
        self,
        query: str,
        scope: Attribution,
        *,
        k: int | None = None,
        min_score: float | None = None,
        as_of: float | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Run the recall query against the store.

        This is the single entry point for both prompt injection and the debug
        surface, so the two can never drift apart.
        """
        k = k or self._top_k
        threshold = self._min_score if min_score is None else min_score
        point_in_time = self._as_of if as_of is None else as_of
        query_embedding = self._embed(query)

        if self._use_hybrid and hasattr(self._store, "recall_hybrid"):
            return self._store.recall_hybrid(
                scope.entity_id,
                query,
                query_embedding,
                k=k,
                min_score=threshold,
                as_of=point_in_time,
                graph_expansion=self._graph_expansion,
                graph_hops=self._graph_hops,
            )
        return self._store.recall(
            scope.entity_id,
            query_embedding,
            k=k,
            min_score=threshold,
            as_of=point_in_time,
        )

    def debug_recall(
        self,
        query: str,
        scope: Attribution,
        *,
        k: int | None = None,
    ) -> dict[str, Any]:
        """
        Debug version of recall that returns the full scoring breakdown.

        Returns the query, the candidate memories with every scoring component,
        how many rows were considered, and how long it took.
        """
        k = k or self._top_k
        start = time.perf_counter()
        # Floor at -1 (the true minimum of cosine similarity) rather than 0:
        # the point of the debug view is to show what was *rejected* and why,
        # including anti-correlated candidates. `above_threshold` then reports
        # what the live recall path would actually have injected.
        results = self.search(query, scope, k=k * 3, min_score=-1.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        breakdown = (
            self._store.last_score_breakdown()
            if hasattr(self._store, "last_score_breakdown")
            else {}
        )
        detailed = []
        for record, score in results[: k * 3]:
            components = breakdown.get(record.id, {})
            detailed.append(
                {
                    "memory_id": record.id,
                    "kind": record.kind,
                    "text": record.text,
                    "triple": f"{record.subject} → {record.predicate} → {record.object}",
                    "final_score": round(score, 4),
                    "above_threshold": score >= self._min_score,
                    "components": components,
                    "mention_count": record.mention_count,
                    "confidence": record.confidence,
                    "importance": record.importance,
                    "valid_window": record.valid_window(),
                    "first_seen": record.first_seen_at,
                    "last_seen": record.last_seen_at,
                    "source_event": record.source_event,
                    "why_recalled": self._explain(record, score, components),
                }
            )

        return {
            "query": query,
            "entity_id": scope.entity_id,
            "process_id": scope.process_id,
            "total_candidates": len(results),
            # `returned` is everything inspected (this is a debug surface, so
            # rejected candidates are shown too, with the reason they lost).
            "returned": len(detailed),
            # `would_inject` is what the real recall path would actually send.
            "would_inject": len([d for d in detailed if d["above_threshold"]]),
            "timing_ms": round(elapsed_ms, 2),
            "results": detailed,
            "min_score_threshold": self._min_score,
            "top_k": k,
            "as_of": self._as_of,
        }

    @staticmethod
    def _explain(
        record: MemoryRecord, score: float, components: dict[str, float]
    ) -> str:
        """Human-readable reason this memory ranked where it did."""
        if components:
            parts = []
            if components.get("vector", 0) >= 0.5:
                parts.append(f"strong semantic match ({components['vector']:.2f})")
            elif components.get("vector", 0) > 0:
                parts.append(f"weak semantic match ({components['vector']:.2f})")
            if components.get("keyword", 0) > 0:
                parts.append(f"keyword overlap ({components['keyword']:.2f})")
            if components.get("exact_boost", 0) > 0:
                parts.append("exact triple match in query")
            if components.get("quality", 0) >= 0.7:
                parts.append("high quality (recent / important / reinforced)")
            if not parts:
                parts.append("matched on metadata only")
            return "; ".join(parts)
        if score >= 0.7:
            return "high overall relevance"
        if score >= 0.4:
            return "moderate relevance"
        return "low relevance"

    def _apply_token_budget(
        self, results: list[tuple[MemoryRecord, float]]
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Trim results to fit the token budget.

        The header counts against the budget, and a memory that would overflow
        is truncated in place (its *text* is shortened, and the shortened record
        is what gets injected) rather than silently kept at full length — the
        previous version computed a truncated line and then threw it away.
        """
        budget = self._token_budget - estimate_tokens(INJECTION_HEADER)
        if budget <= 0:
            return []
        kept: list[tuple[MemoryRecord, float]] = []
        for record, score in results:
            line_tokens = estimate_tokens(record.to_line())
            if line_tokens <= budget:
                kept.append((record, score))
                budget -= line_tokens
                continue
            # Truncate this record to whatever budget remains, then stop.
            # The "[kind] " prefix and the ellipsis both count against the
            # budget, so the text is trimmed until the *rendered* line fits.
            prefix_len = len(f"[{record.kind}] ")
            available = max(8, budget * CHARS_PER_TOKEN - prefix_len - 1)
            trimmed = replace(record, text=record.text[:available])
            while estimate_tokens(trimmed.to_line() + "…") > budget and len(trimmed.text) > 8:
                trimmed = replace(trimmed, text=trimmed.text[: max(4, len(trimmed.text) - 8)])
            if estimate_tokens(trimmed.to_line() + "…") > budget:
                break  # no room left for even a stub
            kept.append((replace(trimmed, text=trimmed.text.rstrip() + "…"), score))
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
                merged[i] = LLMMessage.system(f"{message.content or ''}\n\n{block}")
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
        as_of: float | None = None,
        graph_expansion: bool = False,
        graph_hops: int = 1,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.token_budget = token_budget
        self.grouped = grouped
        self.enabled = enabled
        self.as_of = as_of
        self.graph_expansion = graph_expansion
        self.graph_hops = graph_hops
