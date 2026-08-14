"""
Phase 27 — Code-Search Agent Tools.

Exposes the retrieval pipeline as native tools that the ReAct agent can call:
    search_code      — hybrid BM25+Dense search
    find_symbol      — find a specific class/function by name
    find_references  — find all places a symbol is used
    find_definition  — find where something is defined
    get_context      — get full context for a symbol
    get_dependencies — get the dependency chain for a symbol
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import BaseTool, ToolResult

log = get_logger("tools.code_search")


class SearchCodeTool(BaseTool):
    """Search the codebase using hybrid BM25+Dense retrieval."""

    name = "search_code"
    description = (
        "Search the indexed codebase using a natural language or keyword query. "
        "Returns the most relevant code chunks from across the repository."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (e.g. 'authentication middleware', 'database retry logic').",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default: 5).",
                "default": 5,
            },
            "language": {
                "type": "string",
                "description": "Optional language filter (python, javascript, typescript, etc.).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    async def execute(self, query: str, k: int = 5, language: str | None = None) -> ToolResult:
        try:
            results = self._retriever.search(query, k=k, language=language)
            if not results:
                return ToolResult(
                    success=True,
                    output="No results found for the given query.",
                )

            output_parts = [f"## Search Results for: '{query}'\n"]
            for i, r in enumerate(results, 1):
                symbol = r.get("symbol") or "—"
                sym_type = r.get("symbol_type") or ""
                file_path = r.get("file_path") or "unknown"
                start = r.get("start_line", "?")
                end = r.get("end_line", "?")
                content = r.get("content", "")[:800]

                output_parts.append(
                    f"### [{i}] `{symbol}` ({sym_type}) in `{file_path}` (lines {start}-{end})\n"
                    f"```\n{content}\n```\n"
                )

            return ToolResult(success=True, output="\n".join(output_parts))
        except Exception as e:
            log.error("search_code failed: %s", e)
            return ToolResult(success=False, error=str(e))


class FindSymbolTool(BaseTool):
    """Find a specific symbol by exact or partial name."""

    name = "find_symbol"
    description = (
        "Find the definition of a specific class, function, or method by name. "
        "More precise than search_code when you know the exact symbol name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The exact or partial name of the symbol to find.",
            },
            "symbol_type": {
                "type": "string",
                "enum": ["class", "function", "method", "any"],
                "description": "Type of symbol to look for.",
                "default": "any",
            },
        },
        "required": ["name"],
    }

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    async def execute(self, name: str, symbol_type: str = "any") -> ToolResult:
        try:
            query = f"definition of {name}"
            if symbol_type != "any":
                query = f"{symbol_type} {name}"

            results = self._retriever.search(query, k=5)
            # Prioritize exact symbol name matches
            results.sort(
                key=lambda r: (r.get("symbol") or "").lower() == name.lower(),
                reverse=True,
            )

            if not results:
                return ToolResult(success=True, output=f"Symbol '{name}' not found in the index.")

            r = results[0]
            content = r.get("content", "")
            file_path = r.get("file_path") or "unknown"
            start = r.get("start_line", "?")
            end = r.get("end_line", "?")

            output = (
                f"## Symbol: `{name}`\n"
                f"**File:** `{file_path}` (lines {start}-{end})\n"
                f"**Type:** {r.get('symbol_type') or 'unknown'}\n\n"
                f"```\n{content}\n```"
            )
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetContextTool(BaseTool):
    """Get full symbol context including parent, imports, and related code."""

    name = "get_context"
    description = (
        "Get the full context for a code symbol: its definition, parent class, "
        "and the most relevant related symbols."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The symbol name to get context for.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retriever: Any, expander: Any) -> None:
        self._retriever = retriever
        self._expander = expander

    async def execute(self, symbol: str) -> ToolResult:
        try:
            results = self._retriever.search(symbol, k=5)
            expanded = self._expander.expand(results, max_additional=3)

            if not expanded:
                return ToolResult(success=True, output=f"No context found for '{symbol}'.")

            output_parts = [f"## Context for `{symbol}`\n"]
            for r in expanded:
                sym = r.get("symbol") or r.get("id") or "chunk"
                fp = r.get("file_path") or "unknown"
                reason = r.get("_expansion_reason", "")
                content = r.get("content", "")[:600]
                tag = f" _{reason}_" if reason else ""
                output_parts.append(f"### `{sym}` in `{fp}`{tag}\n```\n{content}\n```\n")

            return ToolResult(success=True, output="\n".join(output_parts))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
