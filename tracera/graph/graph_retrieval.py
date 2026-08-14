"""
Phase 26 — Dependency-Aware Graph Retrieval.

Instead of just returning a matched function, traverses the symbol graph
to return the full dependency chain:
    authentication() → AuthMiddleware → UserService → UserRepository
"""

from __future__ import annotations

from tracera.graph.symbol_graph import SymbolGraph
from tracera.retrieval.bm25 import BM25Index
from tracera.logging import get_logger

log = get_logger("graph.graph_retrieval")


class GraphRetriever:
    """
    Graph-aware retrieval: expands a matched symbol by traversing the
    symbol relationship graph for its callers, callees, and parent context.
    """

    def __init__(self, symbol_graph: SymbolGraph, bm25_index: BM25Index) -> None:
        self._graph = symbol_graph
        self._bm25 = bm25_index

    @property
    def graph(self) -> SymbolGraph:
        """The underlying symbol graph (used by the agent's graph tools)."""
        return self._graph

    def expand_with_graph(
        self,
        base_results: list[dict],
        max_depth: int = 2,
        max_total: int = 15,
    ) -> list[dict]:
        """
        For each base result, traverse the symbol graph to add:
        - Parent class (if symbol is a method)
        - Callers of this symbol (who uses it?)
        - Callees of this symbol (what does it call?)

        Args:
            base_results: Initial retrieval results.
            max_depth: How deep to traverse in the graph.
            max_total: Hard limit on total results returned.

        Returns:
            Augmented results with graph-traversal additions.
        """
        seen_ids = {r["id"] for r in base_results}
        expanded = list(base_results)

        for result in base_results:
            if len(expanded) >= max_total:
                break

            symbol_name = result.get("symbol") or ""
            file_path = result.get("file_path") or ""
            if not symbol_name or not file_path:
                continue

            node_id = f"{file_path}::{symbol_name}"

            # Traverse: ancestors (callers) + descendants (callees)
            related_node_ids = (
                self._graph.get_ancestors(node_id, max_depth=max_depth)
                + self._graph.get_descendants(node_id, max_depth=max_depth)
            )

            for related_id in related_node_ids:
                if len(expanded) >= max_total:
                    break

                node_data = self._graph.get_node(related_id)
                if not node_data:
                    continue

                related_symbol = node_data.get("name") or ""
                related_file = node_data.get("file_path") or ""

                # Find the BM25 chunk for this related symbol
                hits = self._bm25.search(related_symbol, k=3)
                for chunk_id, score in hits:
                    if chunk_id in seen_ids or score < 0.1:
                        continue
                    text = self._bm25.get_document(chunk_id) or ""
                    expanded.append({
                        "id": chunk_id,
                        "content": text,
                        "file_path": related_file,
                        "symbol": related_symbol,
                        "_expansion_reason": f"graph neighbor of {symbol_name}",
                        "_final_score": 0.0,
                        "_source": "graph",
                    })
                    seen_ids.add(chunk_id)
                    break

        log.debug(
            "Graph retrieval: %d base → %d total (max_depth=%d)",
            len(base_results), len(expanded), max_depth,
        )
        return expanded
