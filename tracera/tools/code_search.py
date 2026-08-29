"""
Phase 27 — Code-Search Agent Tools.

Exposes the retrieval pipeline as native tools that the ReAct agent can call:
    search_code      — hybrid BM25+Dense search
    find_symbol      — find a specific class/function by name
    find_references  — find all places a symbol is used (graph-backed)
    get_context      — get full context for a symbol (incl. graph neighbours)
    get_dependencies — get the dependency chain for a symbol (graph-backed)
"""

from __future__ import annotations

from typing import Any

from tracera.graph.symbol_graph import SymbolGraph
from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.code_search")


class SearchCodeTool(Tool):
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

    def __init__(
        self,
        retriever: Any,
        compressor: Any | None = None,
        context_engine: Any | None = None,
        context_recall: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._compressor = compressor
        self._context_engine = context_engine
        self._context_recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, query: str, k: int = 5, language: str | None = None) -> ToolResult:
        try:
            results = self._retriever.search(query, k=k, language=language)
            if not results:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output="No results found for the given query.",
                    query=query,
                    count=0,
                )

            # Phases 29/30: assemble + compress the retrieved pool into an
            # LLM-ready context block instead of dumping raw chunks.
            if self._context_engine is not None or self._compressor is not None:
                if self._compressor is not None:
                    results = self._compressor.compress(results)

                # Phase 15: Get memory context for the query
                memory_context = ""
                if self._context_recall is not None:
                    try:
                        memory_context = self._context_recall.recall(
                            query, k=10, max_chars=4000,
                            include_sessions=True, include_triples=True,
                        )
                    except Exception:
                        pass  # Memory recall is optional

                if self._context_engine is not None:
                    assembled = self._context_engine.assemble(
                        results, query=f"Search: {query}",
                        memory_context=memory_context,
                        memory_budget_tokens=2000,
                    )
                    return ToolResult.ok(
                        tool_name=self.name,
                        tool_call_id="",
                        output=assembled,
                        query=query,
                        count=len(results),
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

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(output_parts),
                query=query,
                count=len(results),
            )
        except Exception as e:
            log.error("search_code failed: %s", e)
            return ToolResult.fail(self.name, "", str(e), query=query)


class FindDefinitionTool(Tool):
    """Find the exact definition (full source) of a symbol by name."""

    name = "find_definition"
    description = (
        "Find the full definition source of a specific class, function, or method "
        "by name. Returns the complete implementation, not just a snippet."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The exact name of the symbol whose definition to find.",
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

    def __init__(
        self, retriever: Any, compressor: Any | None = None, context_recall: Any | None = None
    ) -> None:
        self._retriever = retriever
        self._compressor = compressor
        self._context_recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, name: str, symbol_type: str = "any") -> ToolResult:
        try:
            query = f"definition of {name}"
            if symbol_type != "any":
                query = f"{symbol_type} {name}"

            results = self._retriever.search(query, k=8)
            # Prioritize exact symbol name matches
            results.sort(
                key=lambda r: (r.get("symbol") or "").lower() == name.lower(),
                reverse=True,
            )
            results = [r for r in results if r.get("content")]

            if not results:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Definition of '{name}' not found in the index.",
                    symbol=name,
                )

            # Phase 15: Get memory context for the query
            memory_context = ""
            if self._context_recall is not None:
                try:
                    memory_context = self._context_recall.recall(
                        query, k=5, max_chars=2000,
                        include_sessions=True, include_triples=True,
                    )
                except Exception:
                    pass

            # Phase 30: compress oversized results before returning to the LLM
            if self._compressor is not None:
                results = self._compressor.compress(results)

            parts = [f"## Definition of `{name}`\n"]
            if memory_context:
                parts.append(f"## Agent Memory Context\n{memory_context}\n\n")

            for r in results:
                fp = r.get("file_path") or "unknown"
                start = r.get("start_line", "?")
                end = r.get("end_line", "?")
                content = r.get("content", "")
                parts.append(f"### `{fp}` (lines {start}-{end})\n```\n{content}\n```\n")
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(parts),
                symbol=name,
                matches=len(results),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), symbol=name)


class FindSymbolTool(Tool):
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

    def __init__(self, retriever: Any, context_recall: Any | None = None) -> None:
        self._retriever = retriever
        self._context_recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

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
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Symbol '{name}' not found in the index.",
                    symbol=name,
                )

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
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                symbol=name,
                file_path=file_path,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), symbol=name)


class GetContextTool(Tool):
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

    def __init__(
        self,
        retriever: Any,
        expander: Any,
        graph_retriever: Any = None,
        compressor: Any | None = None,
        context_engine: Any | None = None,
        context_recall: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        self._graph_retriever = graph_retriever
        self._compressor = compressor
        self._context_engine = context_engine
        self._context_recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, symbol: str) -> ToolResult:
        try:
            results = self._retriever.search(symbol, k=5)
            expanded = self._expander.expand(results, max_additional=3)

            # Phase 26: additionally pull in graph neighbours (parent class,
            # callers, callees) when the knowledge graph is available.
            if self._graph_retriever is not None and expanded:
                expanded = self._graph_retriever.expand_with_graph(
                    expanded, max_depth=1, max_total=10
                )

            if not expanded:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"No context found for '{symbol}'.",
                    symbol=symbol,
                )

            # Phases 29/30: compress + assemble into a single context block.
            if self._compressor is not None or self._context_engine is not None:
                if self._compressor is not None:
                    expanded = self._compressor.compress(expanded)

                # Phase 15: Get memory context for the query
                memory_context = ""
                if self._context_recall is not None:
                    try:
                        memory_context = self._context_recall.recall(
                            f"Context for: {symbol}", k=5, max_chars=2000,
                            include_sessions=True, include_triples=True,
                        )
                    except Exception:
                        pass

                if self._context_engine is not None:
                    assembled = self._context_engine.assemble(
                        expanded, query=f"Context for: {symbol}",
                        memory_context=memory_context,
                        memory_budget_tokens=2000,
                    )
                    return ToolResult.ok(
                        tool_name=self.name,
                        tool_call_id="",
                        output=assembled,
                        symbol=symbol,
                        chunks=len(expanded),
                    )

            output_parts = [f"## Context for `{symbol}`\n"]
            for r in expanded:
                sym = r.get("symbol") or r.get("id") or "chunk"
                fp = r.get("file_path") or "unknown"
                reason = r.get("_expansion_reason", "")
                content = r.get("content", "")[:600]
                tag = f" _{reason}_" if reason else ""
                output_parts.append(f"### `{sym}` in `{fp}`{tag}\n```\n{content}\n```\n")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(output_parts),
                symbol=symbol,
                chunks=len(expanded),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), symbol=symbol)


class FindReferencesTool(Tool):
    """Find everywhere a symbol is referenced or called (Phase 27, graph-backed)."""

    name = "find_references"
    description = (
        "Find everywhere a class, function, or method is referenced or called, "
        "using the repository's symbol relationship graph. Returns caller locations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Name of the symbol to find references for.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, symbol: str) -> ToolResult:
        try:
            node_ids = self._graph.find_by_name(symbol)
            if not node_ids:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"No references found for '{symbol}'.",
                    symbol=symbol,
                )

            lines: list[str] = []
            for node_id in node_ids[:10]:
                node = self._graph.get_node(node_id)
                where = (
                    f"{node['file_path']}:{node['start_line']}"
                    if node else node_id
                )
                callers = self._graph.get_callers(node_id)
                if not callers:
                    lines.append(f"### `{symbol}` defined at `{where}` — no recorded callers.")
                    continue

                lines.append(f"### `{symbol}` defined at `{where}` — referenced by:")
                for caller_id in callers[:15]:
                    cnode = self._graph.get_node(caller_id)
                    if cnode:
                        lines.append(
                            f"- `{cnode['name']}` ({cnode['symbol_type']}) "
                            f"in `{cnode['file_path']}:{cnode['start_line']}`"
                        )

            if not lines:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"No references found for '{symbol}'.",
                    symbol=symbol,
                )
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                symbol=symbol,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), symbol=symbol)


class GetDependenciesTool(Tool):
    """Get the dependency chain for a symbol (Phase 27, graph-backed)."""

    name = "get_dependencies"
    description = (
        "Get the dependency chain of a symbol: what it contains/calls and "
        "what code uses it, via graph traversal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Name of the symbol to inspect.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, graph: SymbolGraph) -> None:
        self._graph = graph

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, symbol: str) -> ToolResult:
        try:
            node_ids = self._graph.find_by_name(symbol)
            if not node_ids:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"No graph entries found for '{symbol}'.",
                    symbol=symbol,
                )

            node_id = node_ids[0]
            node = self._graph.get_node(node_id)
            header = (
                f"## `{symbol}` ({node['symbol_type']}) "
                f"in `{node['file_path']}:{node['start_line']}`"
                if node else f"## `{symbol}`"
            )

            lines = [header]

            descendants = self._graph.get_descendants(node_id, max_depth=2)
            if descendants:
                lines.append("\n**Contains / calls:**")
                for did in descendants[:15]:
                    n = self._graph.get_node(did)
                    if n:
                        lines.append(
                            f"- `{n['name']}` ({n['symbol_type']}) "
                            f"in `{n['file_path']}:{n['start_line']}`"
                        )

            ancestors = self._graph.get_ancestors(node_id, max_depth=2)
            if ancestors:
                lines.append("\n**Used by:**")
                for aid in ancestors[:15]:
                    n = self._graph.get_node(aid)
                    if n:
                        lines.append(
                            f"- `{n['name']}` ({n['symbol_type']}) "
                            f"in `{n['file_path']}:{n['start_line']}`"
                        )

            if len(lines) == 1:
                lines.append("No recorded relationships in the graph.")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                symbol=symbol,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), symbol=symbol)
