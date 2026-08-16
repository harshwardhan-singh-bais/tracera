"""
Phase 49 — Agent benchmark.

Evaluates complete coding tasks end-to-end through the agent:

    task → agent → implementation → tests

Metrics collected per task:

    - success: did the agent finish without error?
    - tests_passed: did the task's test suite pass afterwards?
    - iterations, tool_calls, retrieval_calls (search_code / find_*)
    - tokens (in + out), latency, estimated cost

Aggregate report: task success rate, mean iterations, mean tool calls,
mean tokens, mean latency, mean cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from tracera.logging import get_logger

log = get_logger("evaluation.agent_benchmark")

#: Tool names counted as retrieval calls.
RETRIEVAL_TOOLS = {
    "search_code", "find_symbol", "find_definition", "find_references",
    "get_context", "get_dependencies",
}

#: Rough cost per 1M tokens (USD) — configurable; used for estimates only.
DEFAULT_COST_PER_1M_IN = 0.30
DEFAULT_COST_PER_1M_OUT = 1.20


@dataclass
class AgentTaskResult:
    """Outcome of one benchmark task run through the agent."""

    task: str
    success: bool = False
    tests_passed: bool | None = None
    error: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    retrieval_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "success": self.success,
            "tests_passed": self.tests_passed,
            "error": self.error,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "retrieval_calls": self.retrieval_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class AgentBenchmarkReport:
    """Aggregate results for the whole benchmark run."""

    name: str
    results: list[AgentTaskResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # ── Aggregates ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return (sum(1 for r in self.results if r.success) / self.n) if self.n else 0.0

    @property
    def tests_passed_rate(self) -> float:
        measured = [r for r in self.results if r.tests_passed is not None]
        if not measured:
            return 0.0
        return sum(1 for r in measured if r.tests_passed) / len(measured)

    def _mean(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attr) for r in self.results) / len(self.results)

    @property
    def mean_iterations(self) -> float:
        return self._mean("iterations")

    @property
    def mean_tool_calls(self) -> float:
        return self._mean("tool_calls")

    @property
    def mean_retrieval_calls(self) -> float:
        return self._mean("retrieval_calls")

    @property
    def mean_tokens(self) -> float:
        return self._mean("total_tokens")

    @property
    def mean_latency_ms(self) -> float:
        return self._mean("latency_ms")

    @property
    def mean_cost_usd(self) -> float:
        return self._mean("cost_usd")

    # ── Output ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "success_rate": round(self.success_rate, 4),
            "tests_passed_rate": round(self.tests_passed_rate, 4),
            "mean_iterations": round(self.mean_iterations, 2),
            "mean_tool_calls": round(self.mean_tool_calls, 2),
            "mean_retrieval_calls": round(self.mean_retrieval_calls, 2),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "mean_cost_usd": round(self.mean_cost_usd, 6),
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Agent benchmark: {self.name}",
            "",
            f"- Tasks: **{self.n}**",
            f"- Success rate: **{self.success_rate:.0%}**",
            f"- Tests passed: **{self.tests_passed_rate:.0%}**",
            f"- Mean iterations: {self.mean_iterations:.1f}",
            f"- Mean tool calls: {self.mean_tool_calls:.1f} "
            f"(retrieval: {self.mean_retrieval_calls:.1f})",
            f"- Mean tokens: {self.mean_tokens:,.0f}",
            f"- Mean latency: {self.mean_latency_ms:.0f} ms",
            f"- Mean cost: ${self.mean_cost_usd:.4f}",
            "",
        ]
        return "\n".join(lines)


class AgentBenchmark:
    """
    Runs coding tasks through an agent runner and scores the results.

    The runner is injected so benchmarks work with any agent implementation
    (ReActAgent.ask, the orchestrator, ...):

        async def runner(task: str) -> AgentRunOutcome: ...

    where AgentRunOutcome is a dict with keys: output, success, iterations,
    tool_calls, tokens_in, tokens_out, latency_ms.
    """

    def __init__(
        self,
        runner: Callable[[str], Awaitable[dict[str, Any]]],
        *,
        tasks: list[str] | None = None,
        verify_tests: Callable[[], bool] | None = None,
        name: str = "agent-benchmark",
        cost_per_1m_in: float = DEFAULT_COST_PER_1M_IN,
        cost_per_1m_out: float = DEFAULT_COST_PER_1M_OUT,
    ) -> None:
        self.runner = runner
        self.tasks = list(tasks or [])
        #: Optional callable that runs the task's test suite and returns
        #: whether it passed (checked after each agent run).
        self.verify_tests = verify_tests
        self.name = name
        self.cost_per_1m_in = cost_per_1m_in
        self.cost_per_1m_out = cost_per_1m_out

    async def run_task(self, task: str) -> AgentTaskResult:
        result = AgentTaskResult(task=task)
        t0 = time.perf_counter()
        try:
            outcome = await self.runner(task)
        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.success = bool(outcome.get("success", True)) and not outcome.get("error")
        result.error = outcome.get("error")
        result.iterations = int(outcome.get("iterations", 0))
        result.tool_calls = int(outcome.get("tool_calls", 0))
        result.tokens_in = int(outcome.get("tokens_in", 0))
        result.tokens_out = int(outcome.get("tokens_out", 0))

        tool_names = set(outcome.get("tool_names") or [])
        result.retrieval_calls = sum(
            1 for n in tool_names if n in RETRIEVAL_TOOLS
        ) if tool_names else 0
        # Fallback: count from the raw output if tool_names is empty.
        if not tool_names and result.tool_calls:
            result.retrieval_calls = int(outcome.get("retrieval_calls", 0))

        result.cost_usd = (
            result.tokens_in / 1_000_000 * self.cost_per_1m_in
            + result.tokens_out / 1_000_000 * self.cost_per_1m_out
        )

        if self.verify_tests is not None:
            try:
                result.tests_passed = bool(self.verify_tests())
            except Exception as e:
                log.warning("Test verification failed: %s", e)
                result.tests_passed = False

        log.info(
            "Task %-40s success=%s iter=%d tools=%d tokens=%d lat=%.0fms",
            task[:40], result.success, result.iterations,
            result.tool_calls, result.total_tokens, result.latency_ms,
        )
        return result

    async def run(self) -> AgentBenchmarkReport:
        report = AgentBenchmarkReport(name=self.name)
        for task in self.tasks:
            report.results.append(await self.run_task(task))
        log.info(
            "Agent benchmark %s complete: %d tasks, success %.0f%%",
            self.name, report.n, report.success_rate * 100,
        )
        return report
