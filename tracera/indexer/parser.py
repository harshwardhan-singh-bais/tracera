"""
TRACERA Language Parsing Layer (Phase 12).

Integrates tree-sitter to parse source files into ASTs.
"""

from __future__ import annotations

from typing import Dict, Optional

import tree_sitter

from tracera.logging import get_logger

log = get_logger("indexer.parser")


class LanguageParser:
    """Manages tree-sitter parsers for different languages."""

    def __init__(self) -> None:
        self._parsers: Dict[str, tree_sitter.Parser] = {}
        self._languages: Dict[str, tree_sitter.Language] = {}

    def _get_language(self, lang_name: str) -> Optional[tree_sitter.Language]:
        """Lazy load the tree-sitter language."""
        if lang_name in self._languages:
            return self._languages[lang_name]

        try:
            if lang_name == "python":
                import tree_sitter_python as ts_lang
                lang = tree_sitter.Language(ts_lang.language())
            elif lang_name == "javascript":
                import tree_sitter_javascript as ts_lang
                lang = tree_sitter.Language(ts_lang.language())
            elif lang_name == "typescript":
                import tree_sitter_typescript as ts_lang
                # Typescript actually has ts and tsx, we use typescript
                lang = tree_sitter.Language(ts_lang.language_typescript())
            else:
                return None
            
            self._languages[lang_name] = lang
            return lang
        except ImportError:
            log.warning("Tree-sitter package for %s not installed.", lang_name)
            return None
        except Exception as e:
            log.error("Failed to load tree-sitter language %s: %s", lang_name, e)
            return None

    def get_parser(self, lang_name: str) -> Optional[tree_sitter.Parser]:
        """Get a configured parser for the given language."""
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        lang = self._get_language(lang_name)
        if not lang:
            return None

        parser = tree_sitter.Parser(lang)
        self._parsers[lang_name] = parser
        return parser

    def parse(self, code: bytes, lang_name: str) -> Optional[tree_sitter.Tree]:
        """Parse source code into an AST."""
        parser = self.get_parser(lang_name)
        if not parser:
            return None
        return parser.parse(code)
