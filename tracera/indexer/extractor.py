"""
TRACERA Symbol Extraction (Phase 13).

Uses tree-sitter queries to extract semantic symbols from ASTs.
"""

from __future__ import annotations

import tree_sitter

from tracera.indexer.parser import LanguageParser
from tracera.indexer.schema import LineRange, Symbol, SymbolType
from tracera.logging import get_logger

log = get_logger("indexer.extractor")


class SymbolExtractor:
    """Extracts symbols from source code files using tree-sitter."""

    # Tree-sitter queries for supported languages
    QUERIES = {
        "python": """
            (class_definition name: (identifier) @name) @class
            (function_definition name: (identifier) @name) @function
            (import_statement name: (dotted_name) @name) @import
            (import_from_statement module_name: (dotted_name) @name) @import
        """,
        "javascript": """
            (class_declaration name: (identifier) @name) @class
            (function_declaration name: (identifier) @name) @function
            (method_definition name: (property_identifier) @name) @method
            (import_statement (import_clause (identifier) @name)) @import
        """,
        "typescript": """
            (class_declaration name: (identifier) @name) @class
            (function_declaration name: (identifier) @name) @function
            (method_definition name: (property_identifier) @name) @method
            (interface_declaration name: (identifier) @name) @interface
            (import_statement (import_clause (identifier) @name)) @import
        """,
    }

    def __init__(self, parser_mgr: LanguageParser) -> None:
        self.parser_mgr = parser_mgr

    def extract_symbols(self, code: bytes, lang_name: str) -> list[Symbol]:
        """Parse code and extract symbols."""
        if lang_name not in self.QUERIES:
            return []

        tree = self.parser_mgr.parse(code, lang_name)
        if not tree:
            return []

        lang = self.parser_mgr._get_language(lang_name)
        if not lang:
            return []

        try:
            query = tree_sitter.Query(lang, self.QUERIES[lang_name])
            cursor = tree_sitter.QueryCursor(query)
            matches = cursor.matches(tree.root_node)
        except Exception as e:
            log.warning("Failed to run tree-sitter query for %s: %s", lang_name, e)
            return []

        symbols = []
        code_lines = code.split(b"\n")

        for match_id, capture_dict in matches:
            # capture_dict maps capture names to lists of nodes in new python bindings
            # e.g., {'name': [node1], 'class': [node2]}
            
            # We expect a @name capture and a @<type> capture (e.g. @class)
            name_nodes = capture_dict.get("name")
            if not name_nodes:
                continue
            name_node = name_nodes[0]

            symbol_name = code[name_node.start_byte:name_node.end_byte].decode("utf-8")
            
            node = None
            sym_type = None

            for capture_type, captured_nodes in capture_dict.items():
                if capture_type != "name" and captured_nodes:
                    sym_type = capture_type
                    node = captured_nodes[0]
                    break
            
            if not node or not sym_type:
                continue

            # Convert capture name to SymbolType
            try:
                if sym_type == "class":
                    stype = SymbolType.CLASS
                elif sym_type == "function":
                    stype = SymbolType.FUNCTION
                elif sym_type == "method":
                    stype = SymbolType.METHOD
                elif sym_type == "interface":
                    stype = SymbolType.INTERFACE
                elif sym_type == "import":
                    stype = SymbolType.IMPORT
                else:
                    stype = SymbolType.VARIABLE
            except ValueError:
                continue

            start_line = node.start_point.row
            end_line = node.end_point.row

            # Extract content from lines
            # If start and end are the same, just get that line
            if start_line == end_line:
                content = code_lines[start_line].decode("utf-8")
            else:
                content = b"\n".join(code_lines[start_line:end_line+1]).decode("utf-8")

            # Basic parent determination (by nesting)
            parent = None
            parent_node = node.parent
            while parent_node:
                if parent_node.type in ("class_definition", "class_declaration"):
                    # Finding the name of the parent class is tricky without doing another query,
                    # but we can try to extract it from the child identifier.
                    for child in parent_node.children:
                        if child.type == "identifier":
                            parent = code[child.start_byte:child.end_byte].decode("utf-8")
                            if sym_type == "function":
                                stype = SymbolType.METHOD
                            break
                    break
                parent_node = parent_node.parent

            symbols.append(
                Symbol(
                    name=symbol_name,
                    type=stype,
                    range=LineRange(start_line=start_line, end_line=end_line),
                    content=content,
                    parent_symbol=parent,
                )
            )

        # Remove duplicates while preserving order
        unique_symbols = []
        seen = set()
        for sym in symbols:
            key = (sym.name, sym.range.start_line, sym.range.end_line)
            if key not in seen:
                seen.add(key)
                unique_symbols.append(sym)

        return unique_symbols
