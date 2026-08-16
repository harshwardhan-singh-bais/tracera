"""
Phase 50 — Ablation framework.

Turn retrieval components on and off to get engineering evidence about
which parts of the pipeline actually matter:

    Agent                     (no retrieval tools)
    Agent + BM25              (lexical search only)
    Agent + Dense             (semantic search only)
    Agent + Hybrid            (BM25 + dense fusion)
    Agent + Hybrid + Reranker (reranked hybrid)
    Agent + Hybrid + Graph    (graph-backed context)

Each configuration builds an agent with the matching retrieval tool set
and runs it through the same benchmark tasks, so the only variable is the
retrieval component. Results are aggregated into a comparison report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from tracera.evaluation.agent_benchmark import (
    AgentBenchmark,
    AgentBenchmarkReport,
    AgentTaskResult,
)
from tracera.logging import get_logger

log = get_logger("evaluation.ablation")


# ── Configuration model ───────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """One ablation arm: which retrieval components are enabled."""

    name: str
    bm25: bool = False
    dense: bool = False
    hybrid: bool = False
    reranker: bool = False
    graph: bool = False

    @property
    def label(self) -> str:
        parts = ["Agent"]
        if self.bm25 and not self.hybrid:
            parts.append("BM25")
        if self.dense and not self.hybrid:
            parts.append("Dense")
        if self.hybrid:
            parts.append("Hybrid")
        if self.reranker:
            parts.append("Reranker")
        if self.graph:
            parts.append("Graph")
        return " + ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "bm25": self.bm25,
            "dense": self.dense,
            "hybrid": self.hybrid,
            "reranker": self.reranker,
            "graph": self.graph,
        }


def default_ablation_configs() -> list[AblationConfig]:
    """The roadmap's canonical arm set."""
    return [
        AblationConfig(name="agent"),
        AblationConfig(name="agent+bm25", bm25=True),
        AblationConfig(name="agent+dense", dense=True),
        AblationConfig(name="agent+hybrid", hybrid=True),
        AblationConfig(name="agent+hybrid+reranker", hybrid=True, reranker=True),
        AblationConfig(name="agent+hybrid+graph", hybrid=True, graph=True),
    ]


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class AblationReport:
    """Comparison of benchmark results across ablation arms."""

    task_count: int
    arms: dict[str, AgentBenchmarkReport] = field(default_factory=dict)
    ran_at: float = field(default_factory=time.time)

    def best_arm(self, metric: str = "success_rate") -> str | None:
        if not self.arms:
            return None
        return max(self.arms, key=lambda n: getattr(self.arms[n], metric))

    def to_markdown(self) -> str:
        lines = ["# Ablation study\n", f"- Tasks per arm: **{self.task_count}**\n"]
        lines.append(
            "| Configuration | Success | Tests | Iterations | Tool calls | "
            "Retrieval | Tokens | Latency |"
        )
        lines.append("|---" * 8 + "|")
        for name, rep in self.arms.items():
            lines.append(
                f"| {name} | {rep.success_rate:.0%} | {rep.tests_passed_rate:.0%} | "
                f"{rep.mean_iterations:.1f} | {rep.mean_tool_calls:.1f} | "
                f"{rep.mean_retrieval_calls:.1f} | {rep.mean_tokens:,.0f} | "
                f"{rep.mean_latency_ms:.0f}ms |"
            )
        best = self.best_arm()
        if best:
            lines.append(f"\n**Best arm: {best}**")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "ran_at": self.ran_at,
            "arms": {name: rep.to_dict() for name, rep in self.arms.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        return path


# ── Framework ─────────────────────────────────────────────────────────────────

class AblationFramework:
    """
    Runs an agent benchmark per ablation config.

    The agent builder is injected:

        def build_agent(config: AblationConfig) -> Awaitable[Callable[[str], Awaitable[dict]]]:
            # returns an async runner that runs one task through an agent
            # built with the retrieval components in config

    This keeps the framework decoupled from a specific agent factory.
    """

    def __init__(
        self,
        tasks: list[str],
        build_agent: Callable[[AblationConfig], Awaitable[Callable[[str], Awaitable[dict[str, Any]]]]],
        configs: list[AblationConfig] | None = None,
        *,
        verify_tests: Callable[[], bool] | None = None,
        name: str = "ablation",
    ) -> None:
        self.tasks = list(tasks)
        self.build_agent = build_agent
        self.configs = configs or default_ablation_configs()
        self.verify_tests = verify_tests
        self.name = name

    async def run(self) -> AblationReport:
        report = AblationReport(task_count=len(self.tasks))
        for config in self.configs:
            log.info("Ablation arm: %s", config.label)
            runner = await self.build_agent(config)
            bench = AgentBenchmark(
                runner,
                tasks=self.tasks,
                verify_tests=self.verify_tests,
                name=f"{self.name}:{config.name}",
            )
            report.arms[config.name] = await bench.run()
        return report
