"""
Session & Context Assembly Tools — Inspired by jCodeMunch MCP.

  assemble_task_context — one-call task orchestration
  plan_turn             — confidence-guided routing before first read
  get_ranked_context    — token-budgeted context pack
  get_session_stats     — session economics and token savings
  get_repo_map          — cold-start orientation map
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.session_tools")


def _get_graph(retrieval_pipeline=None):
    if retrieval_pipeline is None or len(retrieval_pipeline) < 10:
        return None
    graph_retriever = retrieval_pipeline[-1]
    if graph_retriever is not None and hasattr(graph_retriever, "graph"):
        return graph_retriever.graph
    return None


# ── assemble_task_context ────────────────────────────────────────────────────

class AssembleTaskContextTool(Tool):
    """One-call task orchestration — classify intent, extract anchors, run tools."""

    name = "assemble_task_context"
    description = (
        "Takes a natural-language task and returns a source-attributed context "
        "capsule under a token budget. Auto-classifies the task intent (explore, "
        "debug, refactor, extend, audit, review), extracts anchor symbols, and "
        "runs the appropriate retrieval sequence end-to-end."
    )
    _params = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language task description.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for the context capsule (default: 8000).",
                "default": 8000,
            },
        },
        "required": ["task"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, task: str, max_tokens: int = 8000) -> ToolResult:
        if self._pipeline is None or len(self._pipeline) < 2:
            return ToolResult.ok(self.name, "", "Retrieval pipeline not available. Run `tracera index` first.")

        symbol_retriever = self._pipeline[1]
        expander = self._pipeline[2] if len(self._pipeline) > 2 else None
        compressor = self._pipeline[5] if len(self._pipeline) > 5 else None
        context_engine = self._pipeline[4] if len(self._pipeline) > 4 else None
        graph_retriever = self._pipeline[-1] if self._pipeline else None

        # Classify intent
        intent = self._classify_intent(task)
        # Extract anchor symbols
        anchors = self._extract_anchors(task)

        lines = [f"## Task Context Assembly\n"]
        lines.append(f"**Task:** {task[:200]}")
        lines.append(f"**Intent:** {intent}")
        lines.append(f"**Anchors:** {', '.join(anchors) if anchors else 'none detected'}\n")

        # Run retrieval based on intent
        all_results: list[dict] = []

        if intent == "explore":
            # Search for overview symbols
            for anchor in anchors[:3]:
                results = symbol_retriever.search(anchor, k=5)
                all_results.extend(results)
        elif intent == "debug":
            # Focus on error-prone areas
            for anchor in anchors[:3]:
                results = symbol_retriever.search(f"{anchor} error handling", k=5)
                all_results.extend(results)
        elif intent == "refactor":
            # Find callers and dependencies
            for anchor in anchors[:3]:
                results = symbol_retriever.search(anchor, k=5)
                all_results.extend(results)
        elif intent == "extend":
            # Find similar patterns to follow
            for anchor in anchors[:3]:
                results = symbol_retriever.search(anchor, k=5)
                all_results.extend(results)
        else:
            # General: just search
            results = symbol_retriever.search(task, k=8)
            all_results.extend(results)

        # Expand context
        if expander and all_results:
            all_results = expander.expand(all_results, max_additional=3)

        # Graph expansion
        if graph_retriever and all_results:
            all_results = graph_retriever.expand_with_graph(all_results, max_depth=1, max_total=12)

        # Deduplicate
        seen = set()
        unique = []
        for r in all_results:
            key = r.get("symbol") or r.get("id") or id(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        all_results = unique

        # Compress if available
        if compressor and all_results:
            all_results = compressor.compress(all_results)

        # Format output
        token_estimate = 0
        for r in all_results:
            content = r.get("content", "")
            tokens = len(content) // 4  # rough estimate
            if token_estimate + tokens > max_tokens:
                lines.append(f"\n_[Token budget ({max_tokens}) reached — {len(all_results)} results truncated]_")
                break
            token_estimate += tokens

            symbol = r.get("symbol") or "chunk"
            sym_type = r.get("symbol_type") or ""
            fp = r.get("file_path") or "unknown"
            start = r.get("start_line", "?")
            end = r.get("end_line", "?")
            reason = r.get("_expansion_reason", "")

            tag = f" _{reason}_" if reason else ""
            lines.append(f"### `{symbol}` ({sym_type}) `{fp}` L{start}-{end}{tag}")
            lines.append(f"```{content[:600]}```\n")

        if not all_results:
            lines.append("_No relevant context found. Try indexing the workspace first._")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            intent=intent, anchors=anchors, tokens_used=token_estimate,
        )

    @staticmethod
    def _classify_intent(task: str) -> str:
        """Classify task intent from keywords."""
        t = task.lower()
        if any(w in t for w in ["error", "bug", "fix", "debug", "crash", "fail", "broken"]):
            return "debug"
        if any(w in t for w in ["refactor", "rename", "move", "extract", "reorganize", "cleanup"]):
            return "refactor"
        if any(w in t for w in ["add", "implement", "create", "extend", "new feature", "build"]):
            return "extend"
        if any(w in t for w in ["audit", "review", "security", "lint", "check", "verify"]):
            return "audit"
        if any(w in t for w in ["explore", "understand", "explain", "overview", "what does", "how does"]):
            return "explore"
        if any(w in t for w in ["test", "assert", "coverage", "spec"]):
            return "review"
        return "explore"

    @staticmethod
    def _extract_anchors(task: str) -> list[str]:
        """Extract likely symbol names from the task."""
        import re
        # Find CamelCase identifiers
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', task)
        # Find snake_case identifiers (likely function/variable names)
        snake = re.findall(r'\b[a-z]+(?:_[a-z]+){1,}\b', task)
        # Filter out common English words
        stopwords = {"the", "this", "that", "with", "from", "have", "been", "does", "should", "would"}
        anchors = [a for a in camel + snake if a.lower() not in stopwords]
        return anchors[:5]


# ── plan_turn ────────────────────────────────────────────────────────────────

class PlanTurnTool(Tool):
    """Analyze query and return confidence-guided routing before first read."""

    name = "plan_turn"
    description = (
        "Analyzes a query against the index and returns a confidence-guided "
        "route: which tools to call, on which symbols, under a turn budget. "
        "Low confidence means the symbol probably doesn't exist."
    )
    _params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user query to analyze.",
            },
            "budget": {
                "type": "integer",
                "description": "Token budget for this turn (default: 4000).",
                "default": 4000,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, query: str, budget: int = 4000) -> ToolResult:
        if self._pipeline is None or len(self._pipeline) < 2:
            return ToolResult.ok(self.name, "", "Retrieval pipeline not available.")

        symbol_retriever = self._pipeline[1]
        anchors = AssembleTaskContextTool._extract_anchors(query)
        intent = AssembleTaskContextTool._classify_intent(query)

        # Probe search to assess confidence
        results = symbol_retriever.search(query, k=3)
        top_score = results[0].get("_final_score", 0) if results else 0
        confidence = min(1.0, top_score * 2)  # rough calibration

        # Build route
        route = []
        if intent == "explore":
            route.append("get_repo_outline → search_symbols → get_file_outline")
        elif intent == "debug":
            route.append("search_symbols(get_symbol_source) → find_references → get_call_hierarchy")
        elif intent == "refactor":
            route.append("search_symbols → find_references → check_edit_safe → plan_refactoring")
        elif intent == "extend":
            route.append("search_symbols(similar patterns) → get_context → get_dependencies")
        elif intent == "audit":
            route.append("search_ast(all) → find_dead_code → get_hotspots")
        else:
            route.append("search_symbols → get_symbol_source")

        lines = [
            f"## Turn Plan\n",
            f"**Query:** {query[:200]}",
            f"**Intent:** {intent}",
            f"**Anchors:** {', '.join(anchors[:5]) if anchors else 'none'}",
            f"**Confidence:** {confidence:.2f} {'🟢' if confidence > 0.5 else '🟡' if confidence > 0.2 else '🔴'}",
            f"**Budget:** {budget} tokens\n",
        ]

        if confidence < 0.1:
            lines.append("**⚠️ Low confidence** — this symbol likely doesn't exist in the index.")
            lines.append("Consider: is the codebase indexed? Is the symbol name correct?\n")

        lines.append("### Recommended route:")
        for step in route:
            lines.append(f"1. `{step}`")

        # Estimated token consumption
        est_tokens = min(budget, 500 + len(results) * 200)
        lines.append(f"\n**Estimated consumption:** ~{est_tokens} tokens ({len(results)} results)")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            intent=intent, confidence=confidence, anchors=anchors,
            estimated_tokens=est_tokens, route=route,
        )


# ── get_ranked_context ───────────────────────────────────────────────────────

class GetRankedContextTool(Tool):
    """Pack most relevant symbols into a fixed token budget."""

    name = "get_ranked_context"
    description = (
        "Returns the most relevant symbols for a query, packed into a fixed "
        "token budget. Uses BM25 + PageRank-style ranking for relevance."
    )
    _params = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget (default: 6000).",
                "default": 6000,
            },
            "k": {
                "type": "integer",
                "description": "Max symbols to return (default: 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, query: str, max_tokens: int = 6000, k: int = 10) -> ToolResult:
        if self._pipeline is None or len(self._pipeline) < 2:
            return ToolResult.ok(self.name, "", "Retrieval pipeline not available.")

        symbol_retriever = self._pipeline[1]
        compressor = self._pipeline[5] if len(self._pipeline) > 5 else None

        results = symbol_retriever.search(query, k=k)

        if compressor and results:
            results = compressor.compress(results)

        # Build token-budgeted output
        lines = [f"## Ranked Context for: `{query}`\n"]
        tokens_used = 0
        included = 0

        for r in results:
            content = r.get("content", "")
            tokens = len(content) // 4
            if tokens_used + tokens > max_tokens:
                break
            tokens_used += tokens
            included += 1

            symbol = r.get("symbol") or "chunk"
            fp = r.get("file_path") or "unknown"
            score = r.get("_final_score", 0)
            start = r.get("start_line", "?")
            end = r.get("end_line", "?")

            lines.append(f"### [{included}] `{symbol}` — `{fp}` L{start}-{end} (score: {score:.3f})")
            lines.append(f"```{content[:500]}```\n")

        lines.append(f"_Tokens used: {tokens_used}/{max_tokens}, symbols: {included}/{len(results)}_")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            query=query, tokens_used=tokens_used, symbols=included,
        )


# ── get_session_stats ────────────────────────────────────────────────────────

class GetSessionStatsTool(Tool):
    """Session economics — token usage, savings, and tool breakdown."""

    name = "get_session_stats"
    description = (
        "Report session-level statistics: tokens served, savings vs. naive "
        "file reading, tool usage breakdown, and context efficiency."
    )
    _params = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, session_manager=None) -> None:
        self._session_manager = session_manager

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self) -> ToolResult:
        stats = {
            "tool_calls": 0,
            "total_tokens": 0,
            "tools_used": {},
            "files_read": 0,
            "estimated_naive_tokens": 0,
        }

        if self._session_manager and self._session_manager.active_session:
            session = self._session_manager.active_session
            stats["tool_calls"] = len(session.tool_calls) if hasattr(session, "tool_calls") else 0
            stats["total_tokens"] = getattr(session, "total_tokens", 0)

            if hasattr(session, "tool_calls"):
                for tc in session.tool_calls:
                    name = tc.get("tool_name", "unknown")
                    stats["tools_used"][name] = stats["tools_used"].get(name, 0) + 1
                    if name in ("read_file", "get_file_content"):
                        stats["files_read"] += 1

        # Estimate savings
        # Naive approach would read entire files (~500 lines avg = ~2000 tokens each)
        naive = stats["files_read"] * 2000
        stats["estimated_naive_tokens"] = naive
        savings = max(0, naive - stats["total_tokens"])
        savings_pct = (savings / naive * 100) if naive > 0 else 0

        lines = [
            "## Session Stats\n",
            f"**Tool calls:** {stats['tool_calls']}",
            f"**Total tokens served:** {stats['total_tokens']}",
            f"**Files read:** {stats['files_read']}",
            f"**Estimated naive tokens (full-file reads):** {naive}",
            f"**Estimated savings:** {savings} tokens ({savings_pct:.1f}%)\n",
        ]

        if stats["tools_used"]:
            lines.append("### Tool breakdown:")
            for tool, count in sorted(stats["tools_used"].items(), key=lambda x: -x[1]):
                lines.append(f"- `{tool}`: {count} calls")

        return ToolResult.ok(self.name, "", "\n".join(lines), **stats)


# ── get_repo_map ─────────────────────────────────────────────────────────────

class GetRepoMapTool(Tool):
    """Cold-start orientation map — query-less repo overview ranked by centrality."""

    name = "get_repo_map"
    description = (
        "Generate a token-budgeted, signature-only overview of the repository "
        "structure, ranked by architectural centrality (PageRank on import graph). "
        "No query needed — gives you the lay of the land."
    )
    _params = {
        "type": "object",
        "properties": {
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for the map (default: 4000).",
                "default": 4000,
            },
        },
        "required": [],
    }

    def __init__(self, retrieval_pipeline=None, workspace=None) -> None:
        self._pipeline = retrieval_pipeline
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, max_tokens: int = 4000) -> ToolResult:
        graph = _get_graph(self._pipeline)

        lines = ["## Repository Map\n"]
        tokens_used = 0

        if graph:
            # PageRank centrality
            try:
                import networkx as nx
                pagerank = nx.pagerank(graph._g)
                # Sort by centrality
                ranked = sorted(pagerank.items(), key=lambda x: -x[1])
            except Exception:
                ranked = [(n, 0) for n in graph._g.nodes()]

            # File-level summary
            by_file: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for node_id, score in ranked:
                node = graph.get_node(node_id)
                if node:
                    fp = node.get("file_path", "?")
                    name = node.get("name", "?")
                    stype = node.get("symbol_type", "?")
                    by_file[fp].append((f"{stype} {name}", score))

            for fp in sorted(by_file, key=lambda f: -max(s for _, s in by_file[f])):
                if tokens_used > max_tokens:
                    break
                file_line = f"### `{fp}`"
                lines.append(file_line)
                tokens_used += len(file_line) // 4

                for label, score in by_file[fp][:8]:  # Top 8 per file
                    if tokens_used > max_tokens:
                        break
                    entry = f"- {label} (centrality: {score:.4f})"
                    lines.append(entry)
                    tokens_used += len(entry) // 4

                lines.append("")  # blank line
                tokens_used += 1

        else:
            # Fallback: file listing
            lines.append("_Symbol graph not available. Showing file structure._\n")
            if self._workspace:
                from tracera.indexer.scanner import FileScanner
                scanner = FileScanner(str(self._workspace.root))
                files = scanner.scan()
                for fp in sorted(files)[:50]:
                    entry = f"- `{fp}`"
                    lines.append(entry)
                    tokens_used += len(entry) // 4
                    if tokens_used > max_tokens:
                        break

        lines.append(f"\n_Tokens used: {tokens_used}/{max_tokens}_")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            tokens_used=tokens_used,
        )
