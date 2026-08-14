"""
Phase 19 — Dense Retrieval.

Converts a query string into an embedding and retrieves the top-k
most semantically similar CodeChunks from the LanceDB vector store.
"""

from __future__ import annotations

from tracera.retrieval.embedder import EmbeddingPipeline
from tracera.retrieval.vector_store import VectorStore
from tracera.logging import get_logger

log = get_logger("retrieval.dense")


class DenseRetriever:
    """
    Dense (semantic) retrieval over indexed CodeChunks.

    Flow:
        query string
            ↓ embed (local model)
            ↓ nearest-neighbour search (LanceDB)
            ↓ top-k results
    """

    def __init__(self, embedder: EmbeddingPipeline, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._store = vector_store

    def search(
        self,
        query: str,
        k: int = 10,
        language: str | None = None,
        symbol_type: str | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k semantically similar code chunks.

        Args:
            query: Natural language or code query.
            k: Number of results.
            language: Optional language filter.
            symbol_type: Optional symbol type filter (class/function/method).

        Returns:
            List of result dicts with keys: id, content, file_path, language,
            symbol, symbol_type, parent, start_line, end_line, _relevance_score.
        """
        embedding = self._embedder.embed_single(query)
        raw = self._store.search(embedding, k=k, language=language, symbol_type=symbol_type)

        results = []
        for row in raw:
            result = dict(row)
            # LanceDB returns _distance (lower is closer for L2); we convert to score
            distance = result.pop("_distance", 0.0)
            result["_relevance_score"] = 1.0 / (1.0 + distance)
            result["_source"] = "dense"
            results.append(result)

        log.debug("Dense retrieval: query=%r k=%d → %d results", query[:40], k, len(results))
        return results
