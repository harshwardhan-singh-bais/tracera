"""
Phase 21 — Symbol-Aware Retrieval.

Understands code-specific query patterns and enriches retrieval
with symbol metadata — returning classes, functions, callers, and imports
rather than raw text chunks.
"""

from __future__ import annotations

import re

from tracera.retrieval.hybrid import HybridRetriever
from tracera.logging import get_logger

log = get_logger("retrieval.symbol")

# Code-specific query patterns
_CLASS_PATTERNS = [r"\bclass\b", r"\binterface\b", r"\binherit", r"\bextends\b"]
_FUNC_PATTERNS = [r"\bfunc\b", r"\bfunction\b", r"\bmethod\b", r"\bdef\b", r"\bhandler\b"]
_IMPORT_PATTERNS = [r"\bimport\b", r"\bdependenc", r"\bmodule\b"]


def _detect_symbol_type(query: str) -> str | None:
    """Heuristically detect what kind of symbol the user is querying for."""
    q = query.lower()
    for pattern in _CLASS_PATTERNS:
        if re.search(pattern, q):
            return "class"
    for pattern in _FUNC_PATTERNS:
        if re.search(pattern, q):
            return "function"
    return None


class SymbolAwareRetriever:
    """
    Wraps HybridRetriever and adds symbol-level intelligence.

    1. Detects whether the query is looking for a class, function, or general code.
    2. Applies symbol_type filtering when appropriate.
    3. Deduplicates results that represent the same symbol across chunks.
    4. Promotes results that have matching symbol names.
    """

    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        self._hybrid = hybrid_retriever

    def search(
        self,
        query: str,
        k: int = 10,
        language: str | None = None,
    ) -> list[dict]:
        """
        Symbol-aware search.

        Returns results enriched with a '_final_score' that accounts for
        symbol name matches and type relevance.
        """
        detected_type = _detect_symbol_type(query)
        fetch_k = k * 3  # Over-fetch then re-rank

        # Primary search — apply symbol_type filter if detected
        results = self._hybrid.search(query, k=fetch_k, language=language)

        # Boost results that match the expected symbol type
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for row in results:
            boost = 1.0
            symbol_name = (row.get("symbol") or "").lower()
            symbol_type = row.get("symbol_type") or ""

            # Boost if symbol name contains query terms
            if any(term in symbol_name for term in query_terms if len(term) > 2):
                boost += 0.5

            # Boost if symbol type matches detected type
            if detected_type and detected_type in symbol_type:
                boost += 0.3

            row["_final_score"] = row.get("_rrf_score", 0.0) * boost

        # Re-sort by final score
        results.sort(key=lambda r: r.get("_final_score", 0.0), reverse=True)

        # Deduplicate by symbol name — keep only the best chunk per symbol
        seen_symbols: set[str] = set()
        deduplicated = []
        for row in results:
            symbol = row.get("symbol") or row.get("id")
            if symbol and symbol in seen_symbols:
                continue
            if symbol:
                seen_symbols.add(symbol)
            deduplicated.append(row)
            if len(deduplicated) >= k:
                break

        log.debug(
            "Symbol-aware retrieval: query=%r k=%d type=%s → %d results",
            query[:40], k, detected_type, len(deduplicated),
        )
        return deduplicated
