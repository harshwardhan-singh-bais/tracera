"""
TRACERA Symbol-Aware Code Chunker (Phase 14).

Slices files into `CodeChunk`s, usually grouped by class or function.
"""

from __future__ import annotations

import hashlib

from tracera.indexer.schema import CodeChunk, LineRange, Symbol, SymbolType


class SymbolAwareChunker:
    """Generates semantic chunks from extracted symbols and file content."""

    def __init__(self, max_tokens: int = 500) -> None:
        self.max_tokens = max_tokens

    def _estimate_tokens(self, content: str) -> int:
        """Rough heuristic: ~4 chars per token."""
        return len(content) // 4

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
            # Fallback for plain text or unstructured code: line-based chunking
            # For simplicity, we just chunk it all into 1 piece for now
            # A true implementation would split by max_tokens.
            chunk_content = "\n".join(code_lines)
            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, 0, total_lines),
                    file_path=file_path,
                    language=language,
                    content=chunk_content,
                    range=LineRange(start_line=0, end_line=total_lines - 1),
                    tokens=self._estimate_tokens(chunk_content),
                )
            )
            return chunks

        # Extract major symbols (Classes, Functions, Methods)
        major_symbols = [
            s for s in symbols 
            if s.type in (SymbolType.CLASS, SymbolType.FUNCTION, SymbolType.METHOD)
        ]
        
        # Sort by start line
        major_symbols.sort(key=lambda s: s.range.start_line)

        # Track which lines have been covered by major symbols
        covered_lines = set()

        for sym in major_symbols:
            for i in range(sym.range.start_line, sym.range.end_line + 1):
                covered_lines.add(i)

            # Create a chunk for this symbol
            chunk_id = self._generate_id(file_path, sym.range.start_line, sym.range.end_line)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=file_path,
                    language=language,
                    content=sym.content,
                    range=sym.range,
                    primary_symbol=sym.name,
                    symbol_type=sym.type,
                    parent_symbol=sym.parent_symbol,
                    tokens=self._estimate_tokens(sym.content),
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

        # Add uncovered blocks as generic chunks
        for start_line, end_line in uncovered_blocks:
            block_content = "\n".join(code_lines[start_line:end_line + 1])
            if not block_content.strip():
                continue

            chunk_id = self._generate_id(file_path, start_line, end_line)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=file_path,
                    language=language,
                    content=block_content,
                    range=LineRange(start_line=start_line, end_line=end_line),
                    tokens=self._estimate_tokens(block_content),
                )
            )

        # Sort all chunks by line number
        chunks.sort(key=lambda c: c.range.start_line)
        return chunks

    def _generate_id(self, file_path: str, start_line: int, end_line: int) -> str:
        """Generate a deterministic chunk ID."""
        key = f"{file_path}:{start_line}-{end_line}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
