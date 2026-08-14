"""
Phase 20 — Hybrid Retrieval (BM25 + Dense).

Fuses BM25 lexical scores with dense vector similarity using
Reciprocal Rank Fusion (RRF) for a configurable final ranking.
"""

from __future__ import annotations

from tracera.retrieval.bm25 import BM25Index
from tracera.retrieval.dense import DenseRetriever
from tracera.logging import get_logger

log = get_logger("retrieval.hybrid")

# RRF rank constant (higher = less penalisation of lower ranks)
_RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 lexical and dense vector search
    using Reciprocal Rank Fusion (RRF).

    This consistently outperforms either approach alone because:
    - BM25 excels at exact keyword matches (function names, error codes).
    - Dense excels at semantic similarity (paraphrasing, concept queries).

    Usage:
        retriever = HybridRetriever(bm25_index, dense_retriever, bm25_weight=0.5)
        results = retriever.search("authentication middleware", k=10)
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fetch_k: int = 50,
    ) -> None:
        self._bm25 = bm25_index
        self._dense = dense_retriever
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._fetch_k = fetch_k

    def search(
        self,
        query: str,
        k: int = 10,
        language: str | None = None,
    ) -> list[dict]:
        """
        Fused hybrid search.

        Returns:
            List of merged result dicts, sorted by descending RRF score.
            Each dict has an extra '_rrf_score' field.
        """
        # --- BM25 results ---
        bm25_hits = self._bm25.search(query, k=self._fetch_k)
        bm25_scores: dict[str, float] = {}
        for rank, (doc_id, score) in enumerate(bm25_hits):
            bm25_scores[doc_id] = _rrf_score(rank) * self._bm25_weight

        # --- Dense results ---
        dense_hits = self._dense.search(query, k=self._fetch_k, language=language)
        dense_scores: dict[str, float] = {}
        dense_docs: dict[str, dict] = {}
        for rank, row in enumerate(dense_hits):
            doc_id = row["id"]
            dense_scores[doc_id] = _rrf_score(rank) * self._dense_weight
            dense_docs[doc_id] = row

        # --- Merge ---
        all_ids = set(bm25_scores) | set(dense_scores)
        fused: list[tuple[str, float]] = []
        for doc_id in all_ids:
            score = bm25_scores.get(doc_id, 0.0) + dense_scores.get(doc_id, 0.0)
            fused.append((doc_id, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in fused[:k]]

        # --- Build result dicts ---
        results = []
        for doc_id, rrf_score in fused[:k]:
            if doc_id in dense_docs:
                row = dict(dense_docs[doc_id])
            else:
                # BM25-only hit — reconstruct partial record
                text = self._bm25.get_document(doc_id) or ""
                row = {"id": doc_id, "content": text, "file_path": "", "language": ""}

            row["_rrf_score"] = rrf_score
            row["_bm25_score"] = bm25_scores.get(doc_id, 0.0)
            row["_dense_score"] = dense_scores.get(doc_id, 0.0)
            row["_source"] = "hybrid"
            results.append(row)

        log.debug(
            "Hybrid retrieval: query=%r k=%d → %d results "
            "(bm25_weight=%.2f dense_weight=%.2f)",
            query[:40], k, len(results),
            self._bm25_weight, self._dense_weight,
        )
        return results
