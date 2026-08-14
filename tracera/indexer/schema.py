"""
TRACERA Code Metadata Schema (Phase 15).

Canonical representation for indexed code units.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    """The type of symbol extracted from source code."""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    DECORATOR = "decorator"


class LineRange(BaseModel):
    """0-indexed line range."""
    start_line: int
    end_line: int


class Symbol(BaseModel):
    """
    A semantic code symbol extracted from an AST.
    """
    name: str
    type: SymbolType
    range: LineRange
    content: str
    parent_symbol: str | None = None
    children: list["Symbol"] = Field(default_factory=list)


class FileMetadata(BaseModel):
    """
    Metadata about a file in the workspace.
    """
    path: str  # Relative to workspace root
    language: str | None = None
    size_bytes: int
    sha256: str


class CodeChunk(BaseModel):
    """
    A semantic chunk of code ready to be indexed.
    Usually represents a single class or function.
    """
    id: str
    file_path: str
    language: str
    content: str
    range: LineRange
    
    # Metadata for retrieval
    primary_symbol: str | None = None
    symbol_type: SymbolType | None = None
    parent_symbol: str | None = None
    
    # Number of tokens (calculated by tokenizer later)
    tokens: int | None = None


class IndexDocument(BaseModel):
    """
    The final document stored in the vector database and BM25 index.
    """
    id: str
    content: str
    metadata: dict[str, Any]
