"""
TRACERA Parser Abstraction — Phase 12.

A language-agnostic parser interface backed by Tree-sitter.

    parser = get_parser("python")
    tree = parser.parse(source_bytes)
    symbols = extract_symbols(tree, source_bytes)

Add new languages by registering a Tree-sitter language:
    register_language("go", language_go)

Design inspired by jCodeMunch's parser layer — reimplmented natively.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from tracera.logging import get_logger

log = get_logger("parser")

# ── Language registry ─────────────────────────────────────────────────────────

_LANGUAGES: dict[str, Any] = {}


def register_language(name: str, language: Any) -> None:
    """Register a Tree-sitter language by name."""
    _LANGUAGES[name] = language
    log.debug("Registered parser language: %s", name)


def get_language(name: str) -> Any:
    """Get a registered Tree-sitter language by name."""
    if name not in _LANGUAGES:
        # Try to load common languages dynamically
        _try_load_language(name)
    if name not in _LANGUAGES:
        raise ValueError(f"Unknown language: {name}. Register it first.")
    return _LANGUAGES[name]


def _try_load_language(name: str) -> None:
    """Attempt to load a Tree-sitter language dynamically."""
    try:
        import tree_sitter_python as tspy
        if name == "python":
            _LANGUAGES[name] = tspy.language()
            log.debug("Dynamically loaded python parser")
    except ImportError:
        pass

    try:
        import tree_sitter_typescript as tst
        if name in ("typescript", "javascript"):
            _LANGUAGES[name] = tst.language()
            log.debug("Dynamically loaded %s parser", name)
    except ImportError:
        pass

    try:
        import tree_sitter_go as tsgo
        if name == "go":
            _LANGUAGES[name] = tsgo.language()
            log.debug("Dynamically loaded go parser")
    except ImportError:
        pass

    try:
        import tree_sitter_rust as tsrust
        if name == "rust":
            _LANGUAGES[name] = tsrust.language()
            log.debug("Dynamically loaded rust parser")
    except ImportError:
        pass

    try:
        import tree_sitter_java as tsjava
        if name == "java":
            _LANGUAGES[name] = tsjava.language()
            log.debug("Dynamically loaded java parser")
    except ImportError:
        pass


def list_languages() -> list[str]:
    """Return all known/registered language names."""
    if not _LANGUAGES:
        _try_load_language("python")
    return sorted(_LANGUAGES.keys())


# ── Parser interface ──────────────────────────────────────────────────────────

class Parser:
    """A Tree-sitter parser for a specific language."""

    def __init__(self, language_name: str) -> None:
        self.language_name = language_name
        self._language = get_language(language_name)
        try:
            import tree_sitter as ts
            self._parser = ts.Parser()
            # tree-sitter 0.26+ uses property assignment for language
            self._parser.language = ts.Language(self._language)
        except ImportError:
            raise RuntimeError(
                "tree-sitter Python package not installed. "
                "Run: pip install tree-sitter"
            )

    def parse(self, source: bytes | str) -> Any:
        """Parse source code and return a Tree-sitter Tree."""
        if isinstance(source, str):
            source = source.encode("utf-8")
        return self._parser.parse(source)

    def language(self) -> str:
        """Return the language name."""
        return self.language_name


# ── Symbol extraction ─────────────────────────────────────────────────────────

def extract_symbols(tree: Any, source: bytes | str) -> list[dict[str, Any]]:
    """
    Extract symbols from a Tree-sitter parse tree.

    Returns a list of dicts with keys:
        name, type, file_path, start_byte, end_byte,
        start_line, end_line, parent, children
    """
    if isinstance(source, str):
        source = source.encode("utf-8")

    symbols: list[dict[str, Any]] = []
    root_node = tree.root_node

    _extract_nodes(root_node, source, symbols, parent_name="")

    return symbols


def _extract_nodes(
    node: Any,
    source: bytes,
    symbols: list[dict[str, Any]],
    parent_name: str = "",
) -> None:
    """Recursively extract symbol nodes from the tree."""
    node_type = node.type

    # Symbol-like node types by language (common across Tree-sitter grammars)
    symbol_types = {
        "function_declaration",
        "class_declaration",
        "class_definition",
        "function_definition",
        "method_declaration",
        "method_definition",
        "function",
        "class",
        "module",
        "import_declaration",
        "import_statement",
        "import",
        "variable_declaration",
        "constant_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "decorator",
    }

    name = ""
    if node_type in symbol_types and node.child_count > 0:
        # Get the name child (first identifier-like child)
        name_node = _find_name_node(node)
        if name_node is not None:
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

        if name:
            symbols.append({
                "name": name,
                "type": _normalize_symbol_type(node_type),
                "file_path": "",
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
                "parent": parent_name,
                "children": [],
            })

    # Recurse into children
    for child in node.children:
        _extract_nodes(child, source, symbols, parent_name=name if name else parent_name)


def _find_name_node(node: Any) -> Any | None:
    """Find the name identifier node within a declaration node."""
    for child in node.children:
        if child.type in ("identifier", "name", "string"):
            return child
        # Check deeper
        result = _find_name_node(child)
        if result is not None:
            return result
    return None


def _normalize_symbol_type(node_type: str) -> str:
    """Normalize Tree-sitter node types to common symbol types."""
    mapping = {
        "function_declaration": "function",
        "function": "function",
        "method_declaration": "method",
        "class_declaration": "class",
        "class": "class",
        "interface_declaration": "interface",
        "struct_declaration": "struct",
        "enum_declaration": "enum",
        "type_alias_declaration": "type",
        "variable_declaration": "variable",
        "constant_declaration": "constant",
        "import_declaration": "import",
        "import": "import",
        "module": "module",
        "decorator": "decorator",
    }
    return mapping.get(node_type, node_type)


# ── Convenience API ───────────────────────────────────────────────────────────

def get_parser(language: str) -> Parser:
    """Get a parser for the given language, loading it if needed."""
    return Parser(language)


def parse_file(file_path: str | Path, language: str | None = None) -> dict[str, Any]:
    """
    Parse a file and extract its symbols.

    If *language* is None, attempt to detect from file extension.
    """
    path = Path(file_path)
    if language is None:
        language = _detect_language(path.suffix)

    parser = get_parser(language)
    source = path.read_bytes()
    tree = parser.parse(source)
    symbols = extract_symbols(tree, source)

    # Annotate with file path
    for sym in symbols:
        sym["file_path"] = str(path)

    return {
        "file_path": str(path),
        "language": language,
        "symbols": symbols,
        "tree": tree,
    }


def _detect_language(suffix: str) -> str:
    """Detect language from file extension."""
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
    }
    return mapping.get(suffix.lower(), "unknown")


# ── Auto-register common languages on import ──────────────────────────────────

try:
    import tree_sitter_python as tspy
    register_language("python", tspy.language())
except ImportError:
    pass

try:
    import tree_sitter_typescript as tst
    # tree-sitter-typescript may expose language() differently per version
    lang_fn = getattr(tst, 'language', None)
    if callable(lang_fn):
        register_language("typescript", lang_fn())
        register_language("javascript", lang_fn())
except (ImportError, AttributeError):
    pass
