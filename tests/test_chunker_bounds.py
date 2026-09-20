"""
Chunk-size bounds.

``SymbolAwareChunker(max_tokens=...)`` stored its budget and never used it —
the code even said so ("A true implementation would split by max_tokens"). Every
path could emit an arbitrarily large chunk, and since embedding models truncate
silently at their own context window, the tail of such a chunk was stored but
never *searchable*, while still inflating recalled context.

These tests pin the bound without changing what happens to small files.
"""

from __future__ import annotations

from tracera.indexer.chunker import SymbolAwareChunker
from tracera.indexer.extractor import SymbolExtractor
from tracera.indexer.parser import LanguageParser
from tracera.indexer.schema import SymbolType


def _chunks(content: str, language: str = "python", max_tokens: int = 100):
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)
    symbols = extractor.extract_symbols(content.encode("utf-8"), language)
    return SymbolAwareChunker(max_tokens=max_tokens).chunk_file(
        "f.py", language, content, symbols
    )


def test_a_small_file_still_produces_one_chunk():
    """The common case must not change."""
    content = "def a():\n    return 1\n"
    chunks = _chunks(content)
    assert len(chunks) == 1, [c.primary_symbol for c in chunks]


def test_a_large_file_without_symbols_is_split():
    """
    Whole-file fallback used to emit one chunk regardless of size.

    This is also what happens to every language with no tree-sitter grammar
    (.go, .rs, .java, …), so it is the path most likely to hit huge files.
    """
    lines = [f"value_{i} = {i}" for i in range(4000)]  # ~4000 lines
    chunks = _chunks("\n".join(lines), language="go")
    assert len(chunks) > 1, "a 4000-line file must not be one chunk"


def test_every_chunk_stays_within_the_token_budget():
    lines = [f"value_{i} = {i}" for i in range(4000)]
    chunks = _chunks("\n".join(lines), language="go")
    for chunk in chunks:
        assert chunk.tokens <= 100, (chunk.tokens, chunk.range)


def test_pieces_cover_the_file_with_no_gaps_or_overlaps():
    lines = [f"value_{i} = {i}" for i in range(500)]
    chunks = _chunks("\n".join(lines), language="go")
    covered = [line for c in chunks for line in range(c.range.start_line, c.range.end_line + 1)]
    assert covered == list(range(500)), "pieces must tile the file exactly"


def test_reassembled_content_matches_the_original():
    lines = [f"value_{i} = {i}" for i in range(500)]
    original = "\n".join(lines)
    chunks = _chunks(original, language="go")
    assert "\n".join(c.content for c in chunks) == original


def test_an_oversized_symbol_is_split_but_keeps_its_identity():
    """
    A single huge function must be split, and every piece must still carry the
    symbol metadata — otherwise symbol search would lose it.
    """
    body = "\n".join(f"    x_{i} = {i}" for i in range(4000))
    content = f"def huge():\n{body}\n"
    chunks = _chunks(content)

    assert len(chunks) > 1
    symbol_chunks = [c for c in chunks if c.primary_symbol == "huge"]
    assert len(symbol_chunks) > 1, "all pieces carry the symbol name"
    for chunk in symbol_chunks:
        assert chunk.symbol_type == SymbolType.FUNCTION


def test_a_line_longer_than_the_budget_is_not_split_mid_line():
    """
    Splitting mid-line would corrupt code. One pathological line is better than
    nonsense.
    """
    content = "X" * 5000 + "\nY = 2\n"
    chunks = _chunks(content, language="go")
    assert any("X" * 5000 in c.content for c in chunks)
    assert all(c.content.count("\n") <= 2 for c in chunks)


def test_the_budget_is_configurable():
    lines = [f"value_{i} = {i}" for i in range(1000)]
    narrow = _chunks("\n".join(lines), language="go", max_tokens=25)
    wide = _chunks("\n".join(lines), language="go", max_tokens=250)
    assert len(narrow) > len(wide)


def test_the_default_budget_fits_the_embedding_models_window():
    """
    The budget must not exceed the embedder's max_seq_length.

    The default model is all-MiniLM-L6-v2 (window 256). A chunk over the window
    is stored whole but only represented by its leading tokens, so this is the
    assertion that keeps the tail of every chunk searchable.
    """
    chunker = SymbolAwareChunker()
    assert chunker.max_tokens <= 256, (
        f"default budget {chunker.max_tokens} exceeds all-MiniLM-L6-v2's 256-token window"
    )


def test_the_estimate_is_code_aware_not_natural_language():
    """
    "~4 chars per token" is a prose heuristic and undercounts code by about 2x.

    Measured on this repo: 500-token chunks at chars/4 produced real MiniLM
    token counts up to 976, so 29% of chunks were silently truncated.
    """
    chunker = SymbolAwareChunker()
    code = "def auth_middleware(request, handler):\n    return handler(request)\n"
    estimated = chunker._estimate_tokens(code)
    assert estimated >= len(code) // 3, (
        "the estimate must be conservative enough for code, not prose"
    )
