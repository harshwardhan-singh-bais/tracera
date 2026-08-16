"""
Phases 46-48 — Retrieval benchmark.

Runs an EvaluationDataset through every available retrieval strategy and
reports, per strategy and per query:

    - Recall@1 / Recall@5 / Recall@10
    - Precision@5
    - MRR
    - nDCG@5
    - latency (ms, mean + total)
    - context size (bytes of content returned, mean + total)

This is the numbers the roadmap asks for (Phase 46), the strategy
comparison (Phase 47: BM25 vs dense vs hybrid vs hybrid+reranker) and the
grep baseline comparison (Phase 48: accuracy, latency, context size).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.evaluation.dataset import EvaluationDataset
from tracera.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tracera.evaluation.strategies import RetrievalStrategy
from tracera.logging import get_logger

log = get_logger("evaluation.retrieval_benchmark")


@dataclass
class StrategyScores:
    """Aggregate metrics for one strategy over the whole dataset."""

    name: str
    kind: str
    recall_1: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0
    precision_5: float = 0.0
    mrr: float = 0.0
    ndcg_5: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_total: float = 0.0
    context_bytes_mean: float = 0.0
    context_bytes_total: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "recall@1": round(self.recall_1, 4),
            "recall@5": round(self.recall_5, 4),
            "recall@10": round(self.recall_10, 4),
            "precision@5": round(self.precision_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_5, 4),
            "latency_ms_mean": round(self.latency_ms_mean, 2),
            "latency_ms_total": round(self.latency_ms_total, 2),
            "context_bytes_mean": round(self.context_bytes_mean, 1),
            "context_bytes_total": self.context_bytes_total,
        }


@dataclass
class RetrievalBenchmarkReport:
    """Full benchmark output for one dataset × strategy set."""

    dataset: str
    strategies: dict[str, StrategyScores] = field(default_factory=dict)
    ran_at: float = field(default_factory=time.time)

    def best_strategy(self, metric: str = "recall@5") -> str | None:
        if not self.strategies:
            return None
        return max(
            self.strategies,
            key=lambda n: getattr(self.strategies[n], {
                "recall@1": "recall_1",
                "recall@5": "recall_5",
                "recall@10": "recall_10",
                "mrr": "mrr",
                "ndcg@5": "ndcg_5",
            }.get(metric, "recall_5")),
        )

    def to_markdown(self) -> str:
        lines = [f"# Retrieval benchmark: {self.dataset}\n"]
        header = (
            "| Strategy | R@1 | R@5 | R@10 | P@5 | MRR | nDCG@5 | "
            "lat (ms) | ctx (B) |"
        )
        lines.append(header)
        lines.append("|---" * 9 + "|")
        for scores in self.strategies.values():
            d = scores.to_dict()
            lines.append(
                f"| {scores.name} | {d['recall@1']:.3f} | {d['recall@5']:.3f} | "
                f"{d['recall@10']:.3f} | {d['precision@5']:.3f} | {d['mrr']:.3f} | "
                f"{d['ndcg@5']:.3f} | {d['latency_ms_mean']:.1f} | "
                f"{d['context_bytes_mean']:.0f} |"
            )
        best = self.best_strategy()
        if best:
            lines.append(f"\n**Best overall: {best}** (by recall@5)")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "dataset": self.dataset,
                "ran_at": self.ran_at,
                "strategies": {
                    name: s.to_dict() for name, s in self.strategies.items()
                },
            }, indent=2),
            encoding="utf-8",
        )
        return path


class RetrievalBenchmark:
    """Runs an EvaluationDataset against a dict of strategies."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        strategies: dict[str, RetrievalStrategy],
        *,
        k_values: tuple[int, ...] = (1, 5, 10),
    ) -> None:
        self.dataset = dataset
        self.strategies = strategies
        self.k_values = k_values

    def run(self) -> RetrievalBenchmarkReport:
        report = RetrievalBenchmarkReport(dataset=self.dataset.name)
        n = max(1, len(self.dataset))

        for name, strategy in self.strategies.items():
            per_query: list[dict[str, Any]] = []
            total_latency = 0.0
            total_context = 0
            r1 = r5 = r10 = p5 = mrr_sum = ndcg_sum = 0.0

            for query in self.dataset:
                gt = query.ground_truth
                hits = strategy.retrieve(query.query, k=max(self.k_values))
                total_latency += strategy.last_latency_ms
                total_context += strategy.last_result_bytes

                r1 += recall_at_k(hits, gt, k=1)
                r5 += recall_at_k(hits, gt, k=5)
                r10 += recall_at_k(hits, gt, k=10)
                p5 += precision_at_k(hits, gt, k=5)
                mrr_sum += reciprocal_rank(hits, gt)
                ndcg_sum += ndcg_at_k(hits, gt, k=5)

                per_query.append({
                    "query": query.query,
                    "hits": [h.to_dict() for h in hits[:10]],
                    "recall@5": recall_at_k(hits, gt, k=5),
                })

            scores = StrategyScores(name=name, kind=strategy.kind)
            scores.recall_1 = r1 / n
            scores.recall_5 = r5 / n
            scores.recall_10 = r10 / n
            scores.precision_5 = p5 / n
            scores.mrr = mrr_sum / n
            scores.ndcg_5 = ndcg_sum / n
            scores.latency_ms_mean = total_latency / n
            scores.latency_ms_total = total_latency
            scores.context_bytes_mean = total_context / n
            scores.context_bytes_total = total_context
            scores.per_query = per_query
            report.strategies[name] = scores

            log.info(
                "Strategy %-16s R@5=%.3f MRR=%.3f lat=%.1fms ctx=%.0fB",
                name, scores.recall_5, scores.mrr,
                scores.latency_ms_mean, scores.context_bytes_mean,
            )

        return report


def build_report(
    dataset: EvaluationDataset,
    strategies: dict[str, RetrievalStrategy],
) -> RetrievalBenchmarkReport:
    return RetrievalBenchmark(dataset, strategies).run()
