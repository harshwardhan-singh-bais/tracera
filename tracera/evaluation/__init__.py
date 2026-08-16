"""
TRACERA Evaluation — Phases 45-50.

    dataset.py           — Phase 45: retrieval evaluation dataset (queries +
                            ground-truth symbols/files), load/save.
    metrics.py           — Phase 46: Recall@k, MRR, nDCG@k, precision@k.
    strategies.py        — Phase 47/48: retrievable strategy wrappers
                            (grep, BM25, dense, hybrid, hybrid+reranker).
    retrieval_benchmark.py — Phases 46-48: run a dataset against strategies
                            and report accuracy / latency / context size.
    agent_benchmark.py   — Phase 49: end-to-end coding-task benchmark
                            (success rate, tests, iterations, tokens, cost).
    ablation.py          — Phase 50: turn retrieval components on/off and
                            compare agent performance.
"""

from tracera.evaluation.dataset import EvalQuery, EvaluationDataset
from tracera.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tracera.evaluation.strategies import (
    BM25Strategy,
    DenseStrategy,
    GrepStrategy,
    HybridStrategy,
    RetrievalHit,
    RerankedHybridStrategy,
    build_strategies,
)
from tracera.evaluation.retrieval_benchmark import RetrievalBenchmark
from tracera.evaluation.agent_benchmark import AgentBenchmark, AgentTaskResult
from tracera.evaluation.ablation import AblationFramework

__all__ = [
    "EvalQuery",
    "EvaluationDataset",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "RetrievalHit",
    "GrepStrategy",
    "BM25Strategy",
    "DenseStrategy",
    "HybridStrategy",
    "RerankedHybridStrategy",
    "build_strategies",
    "RetrievalBenchmark",
    "AgentBenchmark",
    "AgentTaskResult",
    "AblationFramework",
]
