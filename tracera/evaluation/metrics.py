"""
Phase 46 — Retrieval metrics.

Standard information-retrieval metrics over ranked retrieval results:

    Recall@k    — how much of the ground truth appears in the top k.
    Precision@k — how many of the top k are relevant.
    MRR         — mean reciprocal rank of the first relevant result.
    nDCG@k      — discounted cumulative gain, normalised by the ideal
                  ranking (binary relevance, per the roadmap's ground-truth
                  file/symbol matching).

Relevance is binary: a result is either in the ground truth or it isn't.
"""

from __future__ import annotations

import math
from typing import Iterable


def is_relevant(hit: object, ground_truth: Iterable[str]) -> bool:
    """
    Match a retrieval hit against the ground truth.

    A hit is relevant when any of its identity fields (doc_id, file_path,
    symbol) matches a ground-truth token (case-insensitive).
    """
    gt = {str(t).lower() for t in ground_truth}
    if not gt:
        return False
    fields = []
    for attr in ("doc_id", "file_path", "symbol", "content"):
        value = getattr(hit, attr, None)
        if value:
            fields.append(value)
    # Also fall back to dict access for plain-dict hits.
    if not fields and isinstance(hit, dict):
        for key in ("id", "file_path", "symbol", "content"):
            value = hit.get(key)
            if value:
                fields.append(value)
    return any(str(f).lower() in gt for f in fields)


def _relevant_mask(hits: list, ground_truth: Iterable[str]) -> list[bool]:
    return [is_relevant(h, ground_truth) for h in hits]


def recall_at_k(
    hits: list,
    ground_truth: Iterable[str],
    k: int | None = None,
) -> float:
    """
    Recall@k: |relevant ∩ top-k| / |ground truth|.
    If ground truth is empty, returns 0.0 (nothing to recall).
    """
    gt = list(ground_truth)
    if not gt:
        return 0.0
    top = hits[:k] if k is not None else hits
    relevant = sum(1 for h in top if is_relevant(h, gt))
    return relevant / len(gt)


def precision_at_k(
    hits: list,
    ground_truth: Iterable[str],
    k: int | None = None,
) -> float:
    """Precision@k: |relevant ∩ top-k| / k."""
    gt = list(ground_truth)
    top = hits[:k] if k is not None else hits
    if not top:
        return 0.0
    relevant = sum(1 for h in top if is_relevant(h, gt))
    return relevant / len(top)


def reciprocal_rank(
    hits: list,
    ground_truth: Iterable[str],
) -> float:
    """
    RR: 1/rank of the first relevant hit (0.0 when none is found).
    Rank is 1-based.
    """
    gt = list(ground_truth)
    if not gt:
        return 0.0
    for i, hit in enumerate(hits):
        if is_relevant(hit, gt):
            return 1.0 / (i + 1)
    return 0.0


def _dcg(relevance: list[int], k: int) -> float:
    """DCG@k with binary relevance."""
    total = 0.0
    for i in range(min(k, len(relevance))):
        if i == 0:
            total += relevance[i]
        else:
            total += relevance[i] / math.log2(i + 1)
    return total


def ndcg_at_k(
    hits: list,
    ground_truth: Iterable[str],
    k: int | None = None,
) -> float:
    """
    nDCG@k with binary relevance: DCG of the retrieved ranking divided by
    the DCG of the ideal ranking (all relevant items first).
    """
    gt = list(ground_truth)
    if not gt:
        return 0.0
    k = k if k is not None else len(hits)
    relevance = [1 if is_relevant(h, gt) else 0 for h in hits[:k]]
    dcg = _dcg(relevance, k)
    n_relevant = min(sum(relevance), k)
    ideal = [1] * n_relevant + [0] * (k - n_relevant)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(
    rankings: list[tuple[list, list]],
) -> float:
    """MRR across queries: mean of per-query reciprocal ranks."""
    if not rankings:
        return 0.0
    total = sum(reciprocal_rank(hits, gt) for hits, gt in rankings)
    return total / len(rankings)


def mean_ndcg(
    rankings: list[tuple[list, list]],
    k: int | None = None,
) -> float:
    """Mean nDCG@k across queries."""
    if not rankings:
        return 0.0
    total = sum(ndcg_at_k(hits, gt, k) for hits, gt in rankings)
    return total / len(rankings)
