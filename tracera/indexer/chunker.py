"""
TRACERA Symbol-Aware Code Chunker (Phase 14).

Slices files into `CodeChunk`s, usually grouped by class or function.
"""

from __future__ import annotations

import hashlib

from tracera.indexer.schema import CodeChunk, LineRange, Symbol, SymbolType


class SymbolAwareChunker:
    """Generates semantic chunks from extracted symbols and file content."""

    def __init__(self, max_tokens: int = 256, chars_per_token: float = 2.0) -> None:
        """
        Args:
            max_tokens: Per-chunk budget. **Must not exceed the embedding
                model's ``max_seq_length``** — the default 256 matches
                ``all-MiniLM-L6-v2``, whose window is exactly that. A chunk
                larger than the window is stored whole but only *represented*
                by its first N tokens, so its tail is unsearchable.
            chars_per_token: Characters per token. The usual "~4 chars" figure
                is a natural-language heuristic and badly undercounts code,
                which tokenizes into many short subword pieces. Measured on
                this repo, Python lands near 2.0; being conservative here makes
                chunks smaller, never silently truncated.
        """
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def _estimate_tokens(self, content: str) -> int:
        return int(len(content) / self.chars_per_token)

    def _bounded_pieces(
        self, content: str, start_line: int
    ) -> list[tuple[str, int, int]]:
        """
        Split ``content`` into pieces no larger than ``max_tokens``.

        Embedding models truncate silently at their own context window, so an
        oversized chunk is stored whole but only *represented* by its first few
        hundred tokens — the tail becomes unreachable by search while still
        inflating recalled context. Bounding at chunk time avoids that.

        A single line longer than the budget is emitted whole: splitting
        mid-line would corrupt code, and one pathological line is preferable to
        nonsense. Files already under the budget return a single piece, so
        small-file behaviour is unchanged.
        """
        # Count characters, not per-line tokens: the joined piece also carries a
        # separator per line, and ignoring those overshoots the budget.
        budget_chars = int(self.max_tokens * self.chars_per_token)
        lines = content.split("\n")
        pieces: list[tuple[str, int, int]] = []
        current: list[str] = []
        current_chars = 0
        piece_start = start_line

        def flush() -> None:
            nonlocal current, current_chars, piece_start
            if not current:
                return
            end = piece_start + len(current) - 1
            pieces.append(("\n".join(current), piece_start, end))
            piece_start = end + 1
            current = []
            current_chars = 0

        for line in lines:
            added = len(line) + (1 if current else 0)
            if current and current_chars + added > budget_chars:
                flush()
                added = len(line)
            current.append(line)
            current_chars += added
        flush()
        return pieces

    def chunk_file(
        self,
        file_path: str,
        language: str,
        content: str,
        symbols: list[Symbol],
    ) -> list[CodeChunk]:
        """
        Split a file into semantically meaningful chunks based on symbols.
        """
        chunks: list[CodeChunk] = []
        code_lines = content.split("\n")
        total_lines = len(code_lines)

        if not symbols:
            # Fallback for plain text or unstructured code (and every language
            # without a tree-sitter grammar): line-based chunking, bounded so a
            # large file does not collapse into one un-embeddable chunk.
            chunk_content = "\n".join(code_lines)
            for content, start, end in self._bounded_pieces(chunk_content, 0):
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, start, end),
                        file_path=file_path,
                        language=language,
                        content=content,
                        range=LineRange(start_line=start, end_line=end),
                        tokens=self._estimate_tokens(content),
                    )
                )
            return chunks

        # Extract major symbols (Classes, Functions, Methods)
        major_symbols = [
            s
            for s in symbols
            if s.type in (SymbolType.CLASS, SymbolType.FUNCTION, SymbolType.METHOD)
        ]

        # Sort by start line
        major_symbols.sort(key=lambda s: s.range.start_line)

        # Track which lines have been covered by major symbols
        covered_lines = set()

        for sym in major_symbols:
            for i in range(sym.range.start_line, sym.range.end_line + 1):
                covered_lines.add(i)

            # Create a chunk for this symbol. A very large symbol is split, but
            # the symbol metadata is carried by every piece so symbol search
            # still finds it.
            pieces = self._bounded_pieces(sym.content, sym.range.start_line)
            if len(pieces) == 1:
                pieces = [(sym.content, sym.range.start_line, sym.range.end_line)]
            for content, start, end in pieces:
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, start, end),
                        file_path=file_path,
                        language=language,
                        content=content,
                        range=LineRange(start_line=start, end_line=end),
                        primary_symbol=sym.name,
                        symbol_type=sym.type,
                        parent_symbol=sym.parent_symbol,
                        tokens=self._estimate_tokens(content),
                    )
                )

        # Find gaps (loose code, imports, globals)
        uncovered_blocks = []
        in_block = False
        start = 0

        for i in range(total_lines):
            if i not in covered_lines and not in_block:
                in_block = True
                start = i
            elif i in covered_lines and in_block:
                in_block = False
                end = i - 1
                if end >= start:
                    uncovered_blocks.append((start, end))

        if in_block:
            uncovered_blocks.append((start, total_lines - 1))

        # Add uncovered blocks as generic chunks (imports, globals, loose code)
        for start_line, end_line in uncovered_blocks:
            block_content = "\n".join(code_lines[start_line : end_line + 1])
            if not block_content.strip():
                continue

            for content, start, end in self._bounded_pieces(block_content, start_line):
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, start, end),
                        file_path=file_path,
                        language=language,
                        content=content,
                        range=LineRange(start_line=start, end_line=end),
                        tokens=self._estimate_tokens(content),
                    )
                )

        # Sort all chunks by line number
        chunks.sort(key=lambda c: c.range.start_line)
        return chunks

    def _generate_id(self, file_path: str, start_line: int, end_line: int) -> str:
        """Generate a deterministic chunk ID."""
        key = f"{file_path}:{start_line}-{end_line}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
