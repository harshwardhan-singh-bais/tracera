"""
Phase 60 — Observability telemetry.

A single :class:`Telemetry` accumulator records:

  - LLM calls (count, input/output tokens, latency, cost, model, errors)
  - Tool calls (count, per-tool, failures, latency)
  - Retrieval calls (deduped category: BM25 / dense / hybrid / graph)
  - Agent iterations
  - Errors

The hot path only does a few dict updates and appends to bounded ring buffers;
it must never raise into the agent loop.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

#: Rough per-1M-token pricing used for the cost estimate (USD). Providers all
#: expose OpenAI-compatible usage so a single default is fine; this is an
#: estimate, not billing.
_INPUT_COST_PER_1M = 0.30
_OUTPUT_COST_PER_1M = 1.20

#: Bounded history kept per category so snapshots stay small.
_MAX_EVENTS = 500

#: Tool names tracked as "retrieval" (separate counter, not a new category).
_RETRIEVAL_TOOLS = {
    "search_code",
    "search_symbols",
    "find_symbol",
    "find_definition",
    "get_context",
    "get_ranked_context",
    "find_references",
    "find_implementations",
    "get_dependencies",
    "find_importers",
    "get_repo_map",
    "assemble_code_context",
    "assemble_task_context",
    "plan_turn",
}


@dataclass
class _Counters:
    llm_calls: int = 0
    llm_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_latency_ms: float = 0.0

    tool_calls: int = 0
    tool_failures: int = 0
    tool_latency_ms: float = 0.0
    per_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    retrieval_calls: int = 0
    retrieval_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    iterations: int = 0
    errors: int = 0


class Telemetry:
    """Process-wide telemetry accumulator (thread-safe)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._c = _Counters()
        self._started_at = time.time()
        self._llm_history: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
        self._tool_history: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
        self._error_history: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_llm(
        self,
        *,
        provider: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._c.llm_calls += 1
            if error:
                self._c.llm_errors += 1
            self._c.prompt_tokens += max(0, int(prompt_tokens))
            self._c.completion_tokens += max(0, int(completion_tokens))
            self._c.total_tokens += max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
            self._c.llm_latency_ms += max(0.0, float(latency_ms))
            self._llm_history.append(
                {
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                    "error": error,
                    "at": time.time(),
                }
            )

    def record_tool(
        self,
        *,
        name: str,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        with self._lock:
            self._c.tool_calls += 1
            if not success:
                self._c.tool_failures += 1
            self._c.tool_latency_ms += max(0.0, float(duration_ms))
            self._c.per_tool[name] += 1
            self._tool_history.append(
                {"name": name, "duration_ms": duration_ms, "success": success, "at": time.time()}
            )

    def record_retrieval(self, *, kind: str = "hybrid") -> None:
        with self._lock:
            self._c.retrieval_calls += 1
            self._c.retrieval_counts[kind] += 1

    def record_iteration(self) -> None:
        with self._lock:
            self._c.iterations += 1

    def record_error(self, message: str, *, category: str = "general") -> None:
        with self._lock:
            self._c.errors += 1
            self._error_history.append(
                {"category": category, "message": message[:200], "at": time.time()}
            )

    # ── Classification helpers ────────────────────────────────────────────────

    @staticmethod
    def classify_tool(name: str) -> str:
        """Map a tool name to its telemetry category."""
        if name in _RETRIEVAL_TOOLS:
            return "retrieval"
        if name.startswith(("find_", "get_", "search_", "assemble_", "plan_")):
            return "analysis"
        if name in ("read_file", "list_dir", "grep"):
            return "read"
        if name in ("write_file", "edit_file", "delete_file"):
            return "write"
        if name in ("run_command", "test"):
            return "execute"
        return "other"

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for the CLI / TUI."""
        with self._lock:
            c = self._c
            elapsed = max(0.0, time.time() - self._started_at)
            cost = (
                c.prompt_tokens / 1_000_000 * _INPUT_COST_PER_1M
                + c.completion_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
            )
            return {
                "elapsed_seconds": round(elapsed, 2),
                "llm": {
                    "calls": c.llm_calls,
                    "errors": c.llm_errors,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "total_latency_ms": round(c.llm_latency_ms, 2),
                    "avg_latency_ms": round(
                        c.llm_latency_ms / c.llm_calls, 2
                    ) if c.llm_calls else 0.0,
                },
                "tools": {
                    "calls": c.tool_calls,
                    "failures": c.tool_failures,
                    "total_latency_ms": round(c.tool_latency_ms, 2),
                    "per_tool": dict(sorted(c.per_tool.items(), key=lambda kv: -kv[1])),
                    "by_category": self._category_breakdown(c.per_tool),
                },
                "retrieval": {
                    "calls": c.retrieval_calls,
                    "by_kind": dict(c.retrieval_counts),
                },
                "agent": {
                    "iterations": c.iterations,
                    "errors": c.errors,
                },
                "cost": {
                    "estimate_usd": round(cost, 6),
                    "input_cost_per_1m": _INPUT_COST_PER_1M,
                    "output_cost_per_1m": _OUTPUT_COST_PER_1M,
                },
                "recent": {
                    "llm": list(self._llm_history)[-10:],
                    "tools": list(self._tool_history)[-10:],
                    "errors": list(self._error_history)[-10:],
                },
            }

    def _category_breakdown(self, per_tool: dict[str, int]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for name, count in per_tool.items():
            out[self.classify_tool(name)] += count
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def reset(self) -> None:
        with self._lock:
            self._c = _Counters()
            self._started_at = time.time()
            self._llm_history.clear()
            self._tool_history.clear()
            self._error_history.clear()


_telemetry: Telemetry | None = None


def get_telemetry() -> Telemetry:
    """Return the process-wide Telemetry singleton."""
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry()
    return _telemetry


def reset_telemetry() -> None:
    """Reset the process-wide Telemetry singleton (used by tests)."""
    global _telemetry
    if _telemetry is not None:
        _telemetry.reset()
    else:
        _telemetry = Telemetry()