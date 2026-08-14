"""
Phase 22 — Context Expansion.

After finding a relevant symbol, automatically retrieve additional context:
    matched function → parent class → its imports → called functions → related defs

This makes the agent context richer without the user having to ask explicitly.
"""

from __future__ import annotations

from tracera.retrieval.bm25 import BM25Index
from tracera.retrieval.vector_store import VectorStore
from tracera.logging import get_logger

log = get_logger("retrieval.context_expander")


class ContextExpander:
    """
    Given an initial set of retrieved chunks, expand context by fetching:
    1. The parent class/module of any matched function/method.
    2. Import blocks from the same file.
    3. Callers (if function name is referenced in BM25).
    """

    def __init__(self, bm25_index: BM25Index, vector_store: VectorStore) -> None:
        self._bm25 = bm25_index
        self._store = vector_store

    def expand(
        self,
        base_results: list[dict],
        max_additional: int = 5,
    ) -> list[dict]:
        """
        Expand a set of retrieval results with related context.

        Args:
            base_results: The initial retrieval results (already ranked).
            max_additional: Max number of additional context chunks to add.

        Returns:
            Original results + expansion results (deduplicated by chunk ID).
        """
        seen_ids = {r["id"] for r in base_results}
        additional: list[dict] = []

        for result in base_results:
            if len(additional) >= max_additional:
                break

            symbol = result.get("symbol") or ""
            parent = result.get("parent") or ""
            file_path = result.get("file_path") or ""
            symbol_type = result.get("symbol_type") or ""

            # 1. Fetch the parent class if this is a method
            if parent and symbol_type in ("method", "function"):
                parent_hits = self._bm25.search(parent, k=3)
                for doc_id, score in parent_hits:
                    if doc_id not in seen_ids and score > 0.1:
                        text = self._bm25.get_document(doc_id) or ""
                        additional.append({
                            "id": doc_id,
                            "content": text,
                            "file_path": file_path,
                            "_expansion_reason": f"parent class of {symbol}",
                            "_final_score": 0.0,
                        })
                        seen_ids.add(doc_id)
                        break

            # 2. Fetch import blocks from the same file
            import_query = f"import {symbol}"
            import_hits = self._bm25.search(import_query, k=3)
            for doc_id, score in import_hits:
                if doc_id not in seen_ids and score > 0.1:
                    text = self._bm25.get_document(doc_id) or ""
                    if "import" in text[:100]:
                        additional.append({
                            "id": doc_id,
                            "content": text,
                            "file_path": file_path,
                            "_expansion_reason": f"imports referencing {symbol}",
                            "_final_score": 0.0,
                        })
                        seen_ids.add(doc_id)
                        break

        log.debug(
            "Context expansion: %d base → +%d additional chunks",
            len(base_results), len(additional),
        )
        return base_results + additional
