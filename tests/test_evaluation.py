"""
Tests for Phases 45-50 — evaluation dataset, metrics, strategies,
retrieval benchmark, agent benchmark, ablation framework.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracera.evaluation.agent_benchmark import AgentBenchmark, AgentTaskResult
from tracera.evaluation.dataset import EvalQuery, EvaluationDataset, example_dataset
from tracera.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tracera.evaluation.retrieval_benchmark import RetrievalBenchmark
from tracera.evaluation.strategies import (
    BM25Strategy,
    RetrievalHit,
    build_strategies,
)
from tracera.evaluation.ablation import (
    AblationConfig,
    AblationFramework,
    default_ablation_configs,
)


# ════════════════════════════════════════════════════════════════════════════
# Phase 45 — dataset
# ════════════════════════════════════════════════════════════════════════════


def test_dataset_add_and_persist(tmp_path):
    ds = EvaluationDataset("test")
    ds.add_query("Where is JWT auth?", files=["auth/middleware.py"], symbols=["validate_token"])
    assert len(ds) == 1

    path = ds.save(tmp_path / "ds.json")
    loaded = EvaluationDataset.load(path)
    assert loaded.name == "test"
    assert loaded[0].query == "Where is JWT auth?"
    assert "auth/middleware.py" in loaded[0].ground_truth


def test_example_dataset_has_queries():
    ds = example_dataset()
    assert len(ds) >= 3
    assert all(q.query for q in ds)


def test_ground_truth_case_insensitive():
    q = EvalQuery(query="q", files=["Auth/Middleware.py"])
    assert "auth/middleware.py" in q.ground_truth


# ════════════════════════════════════════════════════════════════════════════
# Phase 46 — metrics
# ════════════════════════════════════════════════════════════════════════════


def _hits(*paths):
    return [RetrievalHit(doc_id=f"d{i}", score=1.0, file_path=p) for i, p in enumerate(paths)]


def test_recall_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["b.py", "c.py", "z.py"]
    assert recall_at_k(hits, gt, k=1) == 0.0
    assert recall_at_k(hits, gt, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(hits, gt, k=3) == pytest.approx(2 / 3)


def test_precision_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["a.py"]
    assert precision_at_k(hits, gt, k=2) == pytest.approx(0.5)


def test_reciprocal_rank():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["b.py"]
    assert reciprocal_rank(hits, gt) == pytest.approx(0.5)
    assert reciprocal_rank(hits, ["zzz.py"]) == 0.0


def test_ndcg_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["a.py", "c.py"]
    # Relevant at ranks 1 and 3: DCG = 1 + 1/log2(3) ≈ 1.631
    # Ideal: 1 + 1/log2(2) = 2
    assert ndcg_at_k(hits, gt, k=3) == pytest.approx(1.631 / 2.0, abs=0.01)


def test_mrr_mean():
    rankings = [
        (_hits("a.py", "b.py"), ["b.py"]),   # RR 0.5
        (_hits("a.py"), ["a.py"]),           # RR 1.0
        (_hits("a.py"), ["zzz.py"]),         # RR 0.0
    ]
    assert mean_reciprocal_rank(rankings) == pytest.approx(0.5)


# ════════════════════════════════════════════════════════════════════════════
# Phases 47-48 — strategies + benchmark
# ════════════════════════════════════════════════════════════════════════════


def _make_bm25():
    from tracera.retrieval.bm25 import BM25Index
    bm25 = BM25Index()
    bm25.add_document("d1", "def validate_token(): jwt middleware", {"file_path": "auth/token.py"})
    bm25.add_document("d2", "class AuthMiddleware: handles authentication", {"file_path": "auth/middleware.py"})
    bm25.add_document("d3", "def retry_db_call(): database retries", {"file_path": "db/retry.py"})
    return bm25


def test_bm25_strategy_returns_hits():
    bm25 = _make_bm25()
    strategy = BM25Strategy(bm25)
    hits = strategy.retrieve("authentication middleware", k=5)
    assert hits
    assert hits[0].doc_id in {"d1", "d2"}
    assert strategy.last_latency_ms >= 0.0


def test_grep_strategy_baseline(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def authenticate(): pass\nJWT token validation with authentication flow here"
    )
    (tmp_path / "other.txt").write_text("noise")
    strategy = build_strategies(workspace=tmp_path)["grep"]
    hits = strategy.retrieve("JWT authentication", k=5)
    assert hits
    assert any("auth.py" in (h.file_path or "") for h in hits)


def test_retrieval_benchmark_full_run(tmp_path):
    bm25 = _make_bm25()
    dataset = EvaluationDataset("t", [
        EvalQuery(query="authentication middleware", docs=["d1", "d2"]),
        EvalQuery(query="database retry", docs=["d3"]),
    ])
    strategies = build_strategies(workspace=tmp_path, bm25=bm25)
    report = RetrievalBenchmark(dataset, strategies).run()

    assert "grep" in report.strategies
    assert "bm25" in report.strategies
    bm25_scores = report.strategies["bm25"]
    assert 0.0 <= bm25_scores.recall_5 <= 1.0
    assert bm25_scores.mrr >= 0.0
    md = report.to_markdown()
    assert "R@5" in md or "recall" in md.lower()


# ════════════════════════════════════════════════════════════════════════════
# Phase 49 — agent benchmark
# ════════════════════════════════════════════════════════════════════════════


async def _fake_runner(task: str) -> dict:
    return {
        "success": True,
        "output": "done",
        "iterations": 3,
        "tool_calls": 5,
        "tool_names": {"search_code", "read_file", "edit_file"},
        "tokens_in": 1000,
        "tokens_out": 500,
    }


async def test_agent_benchmark_measures_all_metrics():
    bench = AgentBenchmark(_fake_runner, tasks=["task one", "task two"])
    report = await bench.run()

    assert report.n == 2
    assert report.success_rate == 1.0
    assert report.mean_iterations == 3.0
    assert report.mean_tool_calls == 5.0
    assert report.mean_retrieval_calls == 1.0  # only search_code counts
    assert report.mean_tokens == 1500.0
    assert report.mean_latency_ms >= 0.0
    assert report.mean_cost_usd > 0.0
    assert "Success rate" in report.to_markdown()


async def test_agent_benchmark_failure_and_test_verification():
    async def flaky_runner(task: str) -> dict:
        if "bad" in task:
            return {"success": False, "error": "boom"}
        return {"success": True, "output": "ok"}

    bench = AgentBenchmark(
        flaky_runner,
        tasks=["good task", "bad task"],
        verify_tests=lambda: True,
    )
    report = await bench.run()
    assert report.success_rate == 0.5
    assert report.tests_passed_rate == 1.0


# ════════════════════════════════════════════════════════════════════════════
# Phase 50 — ablation framework
# ════════════════════════════════════════════════════════════════════════════


def test_default_ablation_configs_cover_roadmap():
    configs = default_ablation_configs()
    labels = [c.label for c in configs]
    assert "Agent" in labels
    assert "Agent + BM25" in labels
    assert "Agent + Dense" in labels
    assert "Agent + Hybrid" in labels
    assert "Agent + Hybrid + Reranker" in labels
    assert "Agent + Hybrid + Graph" in labels


async def test_ablation_framework_runs_arms():
    async def build_agent(config: AblationConfig):
        async def runner(task: str) -> dict:
            return {"success": True, "output": "ok", "tool_names": set()}
        return runner

    framework = AblationFramework(
        ["task"], build_agent, configs=default_ablation_configs()
    )
    report = await framework.run()
    assert len(report.arms) == 6
    assert report.best_arm() is not None
    md = report.to_markdown()
    assert "Configuration" in md
