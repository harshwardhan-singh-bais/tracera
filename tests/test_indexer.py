"""
Tests for the Tracera Indexer Pipeline (Phases 11-15).
"""

from pathlib import Path
import pytest

from tracera.indexer.schema import SymbolType
from tracera.indexer.scanner import RepositoryScanner
from tracera.indexer.parser import LanguageParser
from tracera.indexer.extractor import SymbolExtractor
from tracera.indexer.chunker import SymbolAwareChunker


def test_scanner(tmp_path: Path):
    """Test repository scanning and exclusion logic."""
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "data.bin").write_bytes(b"hello\x00world")
    
    # Create gitignore
    (tmp_path / ".gitignore").write_text("ignore_me.py\n")
    (tmp_path / "ignore_me.py").write_text("print('ignored')")
    
    # Nested directory
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "utils.py").write_text("def x(): pass")

    scanner = RepositoryScanner(workspace_root=tmp_path)
    files = list(scanner.scan())
    
    paths = {f.path for f in files}
    assert "main.py" in paths
    assert "nested/utils.py" in paths
    assert "data.bin" not in paths
    assert "ignore_me.py" not in paths


def test_parser_and_extractor():
    """Test tree-sitter parsing and symbol extraction."""
    code = b'''
import os

class MyClass:
    def method_a(self):
        pass

def my_func():
    return True
'''
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)
    
    symbols = extractor.extract_symbols(code, "python")
    
    # Verify symbols
    assert len(symbols) >= 3
    
    names = {s.name: s for s in symbols}
    
    assert "MyClass" in names
    assert names["MyClass"].type == SymbolType.CLASS
    
    assert "method_a" in names
    assert names["method_a"].type == SymbolType.METHOD
    assert names["method_a"].parent_symbol == "MyClass"
    
    assert "my_func" in names
    assert names["my_func"].type == SymbolType.FUNCTION


def test_chunker():
    """Test that the chunker splits files properly based on symbols."""
    content = '''
import os

class MyClass:
    def method_a(self):
        pass

def my_func():
    return True
'''
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)
    symbols = extractor.extract_symbols(content.encode("utf-8"), "python")
    
    chunker = SymbolAwareChunker()
    chunks = chunker.chunk_file("test.py", "python", content, symbols)
    
    assert len(chunks) >= 3
    
    # At least one chunk for MyClass and one for my_func
    primary_symbols = {c.primary_symbol for c in chunks if c.primary_symbol}
    assert "MyClass" in primary_symbols
    assert "my_func" in primary_symbols
