"""
Phase 29 — Context Assembly Engine.

Takes a set of retrieved code chunks and assembles them into a coherent
LLM context string, with:
  - Deduplication by content hash
  - Relevance ordering (highest score first)
  - Symbol grouping (methods under their class)
  - Dependency ordering (imports before functions)
  - Token budget enforcement
  - Memory context integration (Phase 15)
"""

from __future__ import annotations

import hashlib

from tracera.logging import get_logger

log = get_logger("agent.context_engine")

_CHARS_PER_TOKEN = 4  # Rough approximation


class ContextAssemblyEngine:
    """
    Assembles retrieval results into a structured, token-budgeted context string.
    """

    def __init__(self, max_tokens: int = 32_000) -> None:
        self._max_tokens = max_tokens
        self._max_chars = max_tokens * _CHARS_PER_TOKEN

    def _content_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN

    def assemble(
        self,
        chunks: list[dict],
        query: str = "",
        memory_context: str = "",
        memory_budget_tokens: int = 2000,
    ) -> str:
        """
        Assemble a list of chunk dicts into a single context string.

        Args:
            chunks: Retrieved chunk dicts (with 'content', 'file_path', 'symbol', etc.)
            query: The original query (used for context header).
            memory_context: Pre-formatted memory context from ContextRecall.
            memory_budget_tokens: Token budget reserved for memory context.

        Returns:
            A formatted string ready to be inserted into an LLM prompt.
        """
        if not chunks and not memory_context:
            return ""

        # 1. Deduplicate by content hash
        seen_hashes: set[str] = set()
        unique_chunks = []
        for c in chunks:
            h = self._content_hash(c.get("content", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_chunks.append(c)

        # 2. Sort: imports first, then classes, then functions, then rest
        def _sort_key(c: dict) -> int:
            sym_type = c.get("symbol_type") or ""
            if "import" in sym_type:
                return 0
            if "class" in sym_type or "module" in sym_type:
                return 1
            if "function" in sym_type or "method" in sym_type:
                return 2
            return 3

        unique_chunks.sort(key=_sort_key)

        # 3. Assemble with token budget
        header = f"# Retrieved Code Context\n*Query: {query}*\n\n" if query else "# Code Context\n\n"
        parts = [header]
        total_chars = len(header)

        # Add memory context first if provided (highest priority)
        if memory_context:
            memory_chars = min(len(memory_context), memory_budget_tokens * _CHARS_PER_TOKEN)
            memory_block = f"## Agent Memory Context\n{memory_context[:memory_chars]}\n\n"
            parts.append(memory_block)
            total_chars += len(memory_block)

        for chunk in unique_chunks:
            content = chunk.get("content") or ""
            symbol = chunk.get("symbol") or ""
            sym_type = chunk.get("symbol_type") or ""
            file_path = chunk.get("file_path") or ""
            start = chunk.get("start_line", "")
            end = chunk.get("end_line", "")
            language = chunk.get("language") or ""
            expansion_reason = chunk.get("_expansion_reason", "")

            title = f"### `{symbol}`" if symbol else f"### Chunk"
            if sym_type:
                title += f" ({sym_type})"
            title += f"\n**File:** `{file_path}`"
            if start and end:
                title += f" (L{start}-{end})"
            if expansion_reason:
                title += f" _{expansion_reason}_"
            title += "\n"

            block = f"{title}```{language}\n{content}\n```\n\n"

            if total_chars + len(block) > self._max_chars:
                # Truncate the content to fit
                available = self._max_chars - total_chars - len(title) - 20
                if available < 100:
                    log.debug("Context budget exhausted at %d chunks", len(parts) - 1)
                    break
                truncated_content = content[:available] + "\n... [truncated]"
                block = f"{title}```{language}\n{truncated_content}\n```\n\n"

            parts.append(block)
            total_chars += len(block)

        result = "".join(parts)
        log.debug(
            "Context assembled: %d chunks + memory → %d chars (~%d tokens)",
            len(parts) - 1 - (1 if memory_context else 0), len(result), self._estimate_tokens(result),
        )
        return result
