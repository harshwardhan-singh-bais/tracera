"""
Phase 30 — Context Compression.

When retrieved context is too large for the LLM's token budget,
applies staged compression:
  1. Relevance filtering — drop lowest-score chunks
  2. Deduplication — remove near-identical content
  3. Symbol compression — summarize verbose chunks (imports, docstrings)
  4. Hard truncation — last-resort per-chunk content trimming
"""

from __future__ import annotations

import re

from tracera.logging import get_logger

log = get_logger("agent.compressor")

_CHARS_PER_TOKEN = 4


class ContextCompressor:
    """
    Compresses a pool of retrieved chunks to fit within a token budget.

    Strategy (in order):
      1. Drop chunks below relevance threshold.
      2. Remove near-duplicate chunks (Jaccard similarity > 0.85).
      3. Trim imports and docstrings to first N lines.
      4. Hard-truncate remaining chunks to fit.
    """

    def __init__(
        self,
        target_tokens: int = 15_000,
        min_score: float = 0.01,
        near_dup_threshold: float = 0.85,
    ) -> None:
        self._target_chars = target_tokens * _CHARS_PER_TOKEN
        self._min_score = min_score
        self._near_dup_threshold = near_dup_threshold

    def _token_count(self, text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN

    def _jaccard(self, a: str, b: str) -> float:
        """Approximate Jaccard similarity using word sets."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _compress_chunk(self, content: str, sym_type: str) -> str:
        """
        Compress a single chunk's content by:
        - Stripping docstrings from functions/classes.
        - Keeping only the first 30 lines of import blocks.
        """
        lines = content.split("\n")

        # For import blocks, keep top 30 lines
        if sym_type and "import" in sym_type:
            return "\n".join(lines[:30]) + ("\n... [imports truncated]" if len(lines) > 30 else "")

        # Strip triple-quote docstrings
        content = re.sub(r'""".*?"""', '"""[docstring omitted]"""', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", "'''[docstring omitted]'''", content, flags=re.DOTALL)

        return content

    def compress(self, chunks: list[dict]) -> list[dict]:
        """
        Compress chunk list to fit within target token budget.

        Args:
            chunks: Retrieved chunk dicts.

        Returns:
            Compressed list of chunk dicts.
        """
        if not chunks:
            return chunks

        # --- 1. Relevance filtering ---
        before = len(chunks)
        chunks = [
            c for c in chunks
            if c.get("_final_score", c.get("_rrf_score", 0.99)) >= self._min_score
        ]
        log.debug("Relevance filter: %d → %d chunks", before, len(chunks))

        # --- 2. Near-duplicate removal ---
        deduped = []
        for chunk in chunks:
            content = chunk.get("content", "")
            is_dup = any(
                self._jaccard(content, d.get("content", "")) >= self._near_dup_threshold
                for d in deduped
            )
            if not is_dup:
                deduped.append(chunk)
        log.debug("Dedup: %d → %d chunks", len(chunks), len(deduped))
        chunks = deduped

        # --- 3. Symbol compression ---
        for chunk in chunks:
            sym_type = chunk.get("symbol_type") or ""
            chunk["content"] = self._compress_chunk(chunk.get("content", ""), sym_type)

        # --- 4. Measure and hard-truncate if still over budget ---
        total_chars = sum(len(c.get("content", "")) for c in chunks)
        if total_chars > self._target_chars:
            log.debug(
                "Over budget (%d > %d chars) — truncating",
                total_chars, self._target_chars,
            )
            budget_per_chunk = self._target_chars // max(len(chunks), 1)
            for chunk in chunks:
                content = chunk.get("content", "")
                if len(content) > budget_per_chunk:
                    chunk["content"] = content[:budget_per_chunk] + "\n... [truncated]"

        final_chars = sum(len(c.get("content", "")) for c in chunks)
        log.info(
            "Context compressed: %d chunks, ~%d tokens",
            len(chunks), final_chars // _CHARS_PER_TOKEN,
        )
        return chunks
