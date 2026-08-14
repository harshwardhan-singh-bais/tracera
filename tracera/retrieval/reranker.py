"""
Phase 23 — Cross-Encoder Reranker.

After BM25+Dense hybrid retrieval generates a candidate pool,
a cross-encoder model scores each (query, chunk) pair jointly,
producing a highly accurate final ranking.

Cross-encoders are expensive (one forward pass per candidate) but
are only applied to the top-N candidates after hybrid retrieval,
keeping latency manageable.
"""

from __future__ import annotations

from typing import Any

from tracera.logging import get_logger

log = get_logger("retrieval.reranker")


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    The model is loaded lazily. Falls back to a relevance-score
    passthrough if cross-encoder is unavailable.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        top_n: int = 5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._top_n = top_n
        self._model: Any = None

    def _load_model(self) -> bool:
        """Lazy load. Returns True if model loaded successfully."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder: %s", self._model_name)
            self._model = CrossEncoder(self._model_name, device=self._device)
            return True
        except Exception as e:
            log.warning("Cross-encoder not available: %s — skipping reranking", e)
            return False

    def rerank(self, query: str, results: list[dict], k: int | None = None) -> list[dict]:
        """
        Rerank results using cross-encoder.

        Args:
            query: The original search query.
            results: Hybrid retrieval results (list of dicts with 'content' key).
            k: Final number of results to return. Defaults to self.top_n.

        Returns:
            Reranked and trimmed results with added '_rerank_score' field.
        """
        if not results:
            return results

        final_k = k or self._top_n

        if not self._load_model():
            # Passthrough — just truncate to final_k
            return results[:final_k]

        # Build (query, chunk_content) pairs
        pairs = [(query, r.get("content", "")[:2000]) for r in results]

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            log.error("Cross-encoder prediction failed: %s", e)
            return results[:final_k]

        # Attach scores and sort
        for result, score in zip(results, scores):
            result["_rerank_score"] = float(score)

        results.sort(key=lambda r: r.get("_rerank_score", 0.0), reverse=True)

        log.debug(
            "Reranker: %d candidates → top %d (best=%.3f worst=%.3f)",
            len(results), final_k,
            results[0].get("_rerank_score", 0.0) if results else 0.0,
            results[min(final_k - 1, len(results) - 1)].get("_rerank_score", 0.0) if results else 0.0,
        )

        return results[:final_k]
