"""
TRACERA Language Parsing Layer (Phase 12).

Integrates tree-sitter to parse source files into ASTs.
"""

from __future__ import annotations

from collections.abc import Sequence

import tree_sitter

from tracera.logging import get_logger

log = get_logger("indexer.parser")


# Umbrella names a user or an agent types at the CLI, mapped to the language
# keys the index actually stores. ``.tsx`` is indexed under its own key because
# TSX is a distinct tree-sitter grammar (only it understands JSX), so without
# this alias ``--lang typescript`` silently returns zero frontend results.
# ``.jsx`` needs no entry: the scanner already maps it to ``javascript``.
LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "ts": ("typescript", "tsx"),
    "typescript": ("typescript", "tsx"),
}


def expand_language_filter(name: str | None) -> list[str] | None:
    """
    Resolve a user-supplied ``--lang`` value into the language keys to match.

    Returns ``None`` for "no filter" so callers can keep passing the value
    straight through, and a one-element list for a language with no alias.
    """
    if not name:
        return None
    key = name.strip().lower()
    return list(LANGUAGE_ALIASES.get(key, (key,)))


class LanguageParser:
    """Manages tree-sitter parsers for different languages."""

    def __init__(self) -> None:
        self._parsers: dict[str, tree_sitter.Parser] = {}
        self._languages: dict[str, tree_sitter.Language] = {}

    def _get_language(self, lang_name: str) -> tree_sitter.Language | None:
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

                lang = tree_sitter.Language(ts_lang.language_typescript())
            elif lang_name == "tsx":
                import tree_sitter_typescript as ts_lang

                # TSX is a separate grammar: only it understands JSX, and
                # parsing a component file with the plain TS grammar yields a
                # tree full of errors.
                lang = tree_sitter.Language(ts_lang.language_tsx())
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

    def get_parser(self, lang_name: str) -> tree_sitter.Parser | None:
        """Get a configured parser for the given language."""
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        lang = self._get_language(lang_name)
        if not lang:
            return None

        parser = tree_sitter.Parser(lang)
        self._parsers[lang_name] = parser
        return parser

    def parse(self, code: bytes, lang_name: str) -> tree_sitter.Tree | None:
        """Parse source code into an AST."""
        parser = self.get_parser(lang_name)
        if not parser:
            return None
        return parser.parse(code)

    def languages(self) -> list[str]:
        """Return list of languages that have been loaded."""
        return list(self._languages.keys())
