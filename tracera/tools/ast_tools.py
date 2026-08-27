"""
AST-Aware Structural Analysis Tools — Inspired by jCodeMunch MCP.

Provides structural queries that go beyond grep:
  find_importers       — what imports a file
  get_blast_radius     — what breaks if a symbol changes
  get_call_hierarchy   — trace callers/callees N levels deep
  find_dead_code       — symbols unreachable from entry points
  get_changed_symbols  — map git diff to affected symbols
  get_hotspots         — risky code by complexity × churn
  search_ast           — cross-language AST pattern matching
  get_class_hierarchy  — traverse inheritance chains
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.ast_tools")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_graph(retrieval_pipeline=None):
    """Extract the SymbolGraph from a retrieval pipeline tuple."""
    if retrieval_pipeline is None or len(retrieval_pipeline) < 10:
        return None
    graph_retriever = retrieval_pipeline[-1]
    if graph_retriever is not None and hasattr(graph_retriever, "graph"):
        return graph_retriever.graph
    return None


def _resolve_symbol(graph, symbol_name: str) -> str | None:
    """Find the canonical node ID for a symbol name in the graph."""
    node_ids = graph.find_by_name(symbol_name)
    if node_ids:
        return node_ids[0]
    return None


def _node_label(graph, node_id: str) -> str:
    """Human-readable label for a graph node."""
    node = graph.get_node(node_id)
    if not node:
        return node_id
    return (
        f"{node.get('symbol_type', '?')} `{node['name']}` "
        f"in `{node.get('file_path', '?')}`:{node.get('start_line', '?')}"
    )


# ── find_importers ───────────────────────────────────────────────────────────

class FindImportersTool(Tool):
    """Find all files/symbols that import a given file or module."""

    name = "find_importers"
    description = (
        "Find all files and symbols that import a given file or module. "
        "Returns the import graph entry points — what would break if you changed this file."
    )
    _params = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to find importers for (e.g. 'src/auth.py').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default: 20).",
                "default": 20,
            },
        },
        "required": ["path"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, path: str, max_results: int = 20) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available. Run `tracera index` first.")

        # Find all nodes in the target file
        target_nodes = graph.find_by_file(path)
        if not target_nodes:
            # Try partial match
            all_files = set()
            for _, data in graph._g.nodes(data=True):
                fp = data.get("file_path", "")
                if path in fp or fp.endswith(path):
                    all_files.add(fp)
            if all_files:
                target_file = list(all_files)[0]
                target_nodes = graph.find_by_file(target_file)

        if not target_nodes:
            return ToolResult.ok(self.name, "", f"No symbols found for '{path}' in the graph.")

        # Collect all importers (ancestors) of these nodes
        importers: set[str] = set()
        for node_id in target_nodes:
            ancestors = graph.get_ancestors(node_id, max_depth=2)
            importers.update(ancestors)

        # Format output
        lines = [f"## Importers of `{path}`\n"]
        count = 0
        for imp_id in sorted(importers):
            if count >= max_results:
                lines.append(f"\n... and {len(importers) - max_results} more.")
                break
            node = graph.get_node(imp_id)
            if node:
                lines.append(
                    f"- `{node.get('name', '?')}` ({node.get('symbol_type', '?')}) "
                    f"in `{node.get('file_path', '?')}`:{node.get('start_line', '?')}"
                )
                count += 1

        if count == 0:
            lines.append("No importers found.")

        return ToolResult.ok(self.name, "", "\n".join(lines), importers=count)


# ── get_blast_radius ─────────────────────────────────────────────────────────

class GetBlastRadiusTool(Tool):
    """Compute blast radius — what breaks if a symbol changes."""

    name = "get_blast_radius"
    description = (
        "Compute the blast radius of changing a symbol: all downstream symbols "
        "that depend on it, with depth-weighted risk scores."
    )
    _params = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name to compute blast radius for.",
            },
            "depth": {
                "type": "integer",
                "description": "Max traversal depth (default: 3).",
                "default": 3,
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, symbol: str, depth: int = 3) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in graph.")

        # BFS blast radius with depth tracking
        import networkx as nx
        reverse_g = graph._g.reverse()
        affected: dict[str, int] = {}  # node_id → depth

        bfs_queue = [(node_id, 0)]
        visited = {node_id}
        while bfs_queue:
            current, d = bfs_queue.pop(0)
            if d >= depth:
                continue
            for predecessor in reverse_g.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    affected[predecessor] = d + 1
                    bfs_queue.append((predecessor, d + 1))

        # Score: closer = higher risk
        lines = [f"## Blast Radius for `{symbol}` (depth={depth})\n"]
        lines.append(f"**Total affected symbols:** {len(affected)}\n")

        if not affected:
            lines.append("No downstream dependencies found.")
        else:
            # Group by depth
            by_depth: dict[int, list[str]] = defaultdict(list)
            for nid, d in sorted(affected.items(), key=lambda x: x[1]):
                by_depth[d].append(nid)

            for d in sorted(by_depth):
                risk_label = "🔴 HIGH" if d == 1 else "🟡 MEDIUM" if d == 2 else "🟢 LOW"
                lines.append(f"\n### Depth {d} — {risk_label}")
                for nid in by_depth[d]:
                    lines.append(f"- {_node_label(graph, nid)}")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            symbol=symbol, affected=len(affected), depth=depth,
        )


# ── get_call_hierarchy ───────────────────────────────────────────────────────

class GetCallHierarchyTool(Tool):
    """Trace callers and callees N levels deep."""

    name = "get_call_hierarchy"
    description = (
        "Trace the call hierarchy of a symbol: who calls it (callers) and "
        "what it calls (callees), up to N levels deep."
    )
    _params = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name to trace.",
            },
            "depth": {
                "type": "integer",
                "description": "Traversal depth (default: 2).",
                "default": 2,
            },
            "direction": {
                "type": "string",
                "enum": ["callers", "callees", "both"],
                "description": "Which direction to trace (default: both).",
                "default": "both",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(
        self, symbol: str, depth: int = 2, direction: str = "both"
    ) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found.")

        lines = [f"## Call Hierarchy for `{symbol}`\n"]

        callers = graph.get_ancestors(node_id, max_depth=depth) if direction in ("callers", "both") else []
        callees = graph.get_descendants(node_id, max_depth=depth) if direction in ("callees", "both") else []

        if direction in ("callers", "both"):
            lines.append("### Callers (who calls this)")
            if not callers:
                lines.append("No callers found.")
            for cid in callers[:20]:
                lines.append(f"- {_node_label(graph, cid)}")

        if direction in ("callees", "both"):
            lines.append("\n### Callees (what this calls)")
            if not callees:
                lines.append("No callees found.")
            for cid in callees[:20]:
                lines.append(f"- {_node_label(graph, cid)}")

        return ToolResult.ok(self.name, "", "\n".join(lines), symbol=symbol)


# ── find_dead_code ───────────────────────────────────────────────────────────

class FindDeadCodeTool(Tool):
    """Find symbols unreachable from entry points."""

    name = "find_dead_code"
    description = (
        "Find symbols and files that are unreachable from any entry point "
        "(main functions, CLI handlers, API routes, test files)."
    )
    _params = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum dead code results (default: 30).",
                "default": 30,
            },
        },
        "required": [],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, max_results: int = 30) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        import networkx as nx

        g = graph._g
        entry_keywords = {"main", "__main__", "app", "cli", "handler", "route", "setup"}
        entry_nodes: set[str] = set()

        # Find entry points by name or file pattern
        for node_id, data in g.nodes(data=True):
            name = (data.get("name") or "").lower()
            fp = (data.get("file_path") or "").lower()
            stype = (data.get("symbol_type") or "").lower()

            if any(kw in name for kw in entry_keywords):
                entry_nodes.add(node_id)
            elif "test" in fp or "__main__" in fp:
                entry_nodes.add(node_id)
            elif stype in ("method", "function") and name in ("main", "app", "cli"):
                entry_nodes.add(node_id)

        # BFS from all entry points
        reachable: set[str] = set()
        queue = list(entry_nodes)
        visited = set(entry_nodes)
        while queue:
            current = queue.pop(0)
            reachable.add(current)
            for successor in g.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        # Dead code = all nodes - reachable
        all_nodes = set(g.nodes())
        dead = all_nodes - reachable

        lines = [f"## Dead Code Analysis\n"]
        lines.append(f"**Total symbols:** {len(all_nodes)}")
        lines.append(f"**Reachable from entries:** {len(reachable)}")
        lines.append(f"**Potentially dead:** {len(dead)}\n")

        if not dead:
            lines.append("No dead code found — all symbols are reachable from entry points.")
        else:
            # Group by file
            by_file: dict[str, list[str]] = defaultdict(list)
            for nid in sorted(dead):
                node = g.nodes[nid]
                fp = node.get("file_path", "unknown")
                label = _node_label(graph, nid)
                by_file[fp].append(label)

            count = 0
            for fp in sorted(by_file):
                if count >= max_results:
                    lines.append(f"\n... and {len(dead) - max_results} more symbols.")
                    break
                lines.append(f"\n### `{fp}`")
                for label in by_file[fp]:
                    if count >= max_results:
                        break
                    lines.append(f"- {label}")
                    count += 1

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            total=len(all_nodes), reachable=len(reachable), dead=len(dead),
        )


# ── get_changed_symbols ──────────────────────────────────────────────────────

class GetChangedSymbolsTool(Tool):
    """Map a git diff to the exact symbols that changed."""

    name = "get_changed_symbols"
    description = (
        "Map a git diff (current uncommitted changes or a commit range) to "
        "the exact symbols that were added, modified, or removed."
    )
    _params = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": (
                    "Git ref or range: 'HEAD' for uncommitted, "
                    "'HEAD~1' for last commit, or 'abc123..def456' for a range."
                ),
                "default": "HEAD",
            },
        },
        "required": [],
    }

    def __init__(self, workspace=None, retrieval_pipeline=None) -> None:
        self._workspace = workspace
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, ref: str = "HEAD") -> ToolResult:
        import subprocess
        workspace_root = str(self._workspace.root) if self._workspace else "."

        try:
            if ref == "HEAD":
                # Uncommitted changes
                diff = subprocess.run(
                    ["git", "diff", "--name-only"],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )
            else:
                # Commit range
                diff = subprocess.run(
                    ["git", "diff", "--name-only", ref],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )

            changed_files = [
                f.strip() for f in diff.stdout.strip().split("\n") if f.strip()
            ]
        except Exception as e:
            return ToolResult.fail(self.name, "", f"Git diff failed: {e}")

        if not changed_files:
            return ToolResult.ok(self.name, "", "No changed files detected.")

        # Try to map changed files to symbols via the graph
        graph = _get_graph(self._pipeline)
        lines = [f"## Changed Symbols (ref: `{ref}`)\n"]
        lines.append(f"**Changed files:** {len(changed_files)}\n")

        for fp in changed_files:
            lines.append(f"### `{fp}`")
            if graph:
                node_ids = graph.find_by_file(fp)
                if node_ids:
                    for nid in node_ids[:10]:
                        lines.append(f"- {_node_label(graph, nid)}")
                else:
                    lines.append("  _No symbols in graph for this file._")
            else:
                lines.append("  _Symbol graph not available._")

        return ToolResult.ok(self.name, "", "\n".join(lines), changed_files=changed_files)


# ── get_hotspots ─────────────────────────────────────────────────────────────

class GetHotspotsTool(Tool):
    """Surface risky code by complexity × churn."""

    name = "get_hotspots"
    description = (
        "Find the riskiest code locations by combining cyclomatic complexity "
        "with git churn (commit frequency). Hotspots indicate areas that are "
        "both complex and frequently changed — the highest-risk code."
    )
    _params = {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Number of hotspots to return (default: 10).",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, workspace=None, retrieval_pipeline=None) -> None:
        self._workspace = workspace
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, top_n: int = 10) -> ToolResult:
        import subprocess
        workspace_root = str(self._workspace.root) if self._workspace else "."

        # Get churn data from git log
        churn: dict[str, int] = defaultdict(int)
        try:
            result = subprocess.run(
                ["git", "log", "--pretty=format:", "--name-only", "--since=90 days"],
                cwd=workspace_root, capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    churn[line] += 1
        except Exception:
            pass

        # Get complexity from graph (if available)
        graph = _get_graph(self._pipeline)
        candidates: list[dict] = []

        if graph:
            for node_id, data in graph._g.nodes(data=True):
                fp = data.get("file_path", "")
                name = data.get("name", "")
                start = data.get("start_line", 0)
                end = data.get("end_line", 0)
                complexity = max(1, (end - start) if end and start else 10)
                churn_count = churn.get(fp, 0)
                score = complexity * (1 + churn_count * 0.1)
                candidates.append({
                    "symbol": name,
                    "file_path": fp,
                    "start_line": start,
                    "end_line": end,
                    "complexity": complexity,
                    "churn": churn_count,
                    "score": score,
                })

        candidates.sort(key=lambda c: c["score"], reverse=True)

        lines = [f"## Code Hotspots (complexity × churn)\n"]
        if not candidates:
            lines.append("No hotspot data available. Run `tracera index` first.")
        else:
            for i, c in enumerate(candidates[:top_n], 1):
                risk = "🔴" if c["score"] > 500 else "🟡" if c["score"] > 100 else "🟢"
                lines.append(
                    f"{i}. {risk} **{c['symbol']}** in `{c['file_path']}` "
                    f"(L{c['start_line']}-{c['end_line']}) — "
                    f"complexity={c['complexity']}, churn={c['churn']}, score={c['score']:.0f}"
                )

        return ToolResult.ok(self.name, "", "\n".join(lines), hotspots=len(candidates[:top_n]))


# ── search_ast ───────────────────────────────────────────────────────────────

class SearchAstTool(Tool):
    """Cross-language AST pattern matching with preset anti-patterns."""

    name = "search_ast"
    description = (
        "Search code using AST-aware patterns. Supports preset anti-patterns "
        "(empty_catch, bare_except, deeply_nested, nested_loops, god_function, "
        "eval_exec, hardcoded_secret, todo_fixme, magic_number, reassigned_param) "
        "and custom queries via a mini-DSL: 'call:*.unwrap', 'string:/password/i', "
        "'nesting:5+'."
    )
    _params = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Preset anti-pattern name or custom DSL query. "
                    "Presets: empty_catch, bare_except, deeply_nested, nested_loops, "
                    "god_function, eval_exec, hardcoded_secret, todo_fixme, "
                    "magic_number, reassigned_param, or 'all' for a full sweep."
                ),
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional file glob filter (e.g. '*.py').",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results (default: 20).",
                "default": 20,
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace=None) -> None:
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(
        self, pattern: str, file_pattern: str | None = None, max_results: int = 20
    ) -> ToolResult:
        workspace_root = Path(self._workspace.root) if self._workspace else Path(".")
        results: list[dict] = []

        # Define regex-based preset detectors
        presets = {
            "empty_catch": {
                "regex": re.compile(r"except.*:\s*\n\s*(pass|\.\.\.)\s*$", re.MULTILINE),
                "category": "error_handling",
                "description": "Silently swallowed exceptions",
            },
            "bare_except": {
                "regex": re.compile(r"except\s*:", re.MULTILINE),
                "category": "error_handling",
                "description": "Catch-all exception handlers",
            },
            "todo_fixme": {
                "regex": re.compile(r"(TODO|FIXME|HACK|XXX|TEMP)\b", re.IGNORECASE),
                "category": "maintenance",
                "description": "Unfinished work markers",
            },
            "hardcoded_secret": {
                "regex": re.compile(
                    r"""(password|secret|api_key|token|credential)\s*=\s*['"][^'"]{8,}['"]""",
                    re.IGNORECASE,
                ),
                "category": "security",
                "description": "Hardcoded credentials in strings",
            },
            "eval_exec": {
                "regex": re.compile(r"\b(eval|exec)\s*\(", re.MULTILINE),
                "category": "security",
                "description": "Dynamic code execution (injection risk)",
            },
            "magic_number": {
                "regex": re.compile(r"(?<!=\s)(?<![\w.])\b(?:[2-9]\d{2,}|[1-9]\d{3,})\b(?!\s*[:\]])"),
                "category": "maintenance",
                "description": "Unexplained numeric constants",
            },
        }

        # Determine which presets to run
        if pattern == "all":
            active_presets = presets
        elif pattern in presets:
            active_presets = {pattern: presets[pattern]}
        else:
            # Custom DSL parsing
            return await self._custom_search(pattern, file_pattern, max_results, workspace_root)

        # Scan files
        from tracera.indexer.scanner import FileScanner
        scanner = FileScanner(str(workspace_root))
        source_files = scanner.scan()

        if file_pattern:
            import fnmatch
            source_files = [f for f in source_files if fnmatch.fnmatch(f, file_pattern)]

        for fp in source_files[:500]:  # Cap to avoid runaway
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for preset_name, preset in active_presets.items():
                for match in preset["regex"].finditer(content):
                    line_no = content[:match.start()].count("\n") + 1
                    context_start = max(0, match.start() - 40)
                    context_end = min(len(content), match.end() + 40)
                    context = content[context_start:context_end].replace("\n", " ").strip()
                    results.append({
                        "file": fp,
                        "line": line_no,
                        "preset": preset_name,
                        "category": preset["category"],
                        "description": preset["description"],
                        "match": match.group().strip()[:100],
                        "context": context[:200],
                    })

        # Deduplicate and sort
        results = results[:max_results]

        lines = [f"## AST Pattern Search: `{pattern}`\n"]
        lines.append(f"**Matches found:** {len(results)}\n")

        if not results:
            lines.append("No matches found.")
        else:
            for r in results:
                lines.append(
                    f"- **[{r['preset']}]** `{r['file']}`:{r['line']} — {r['description']}"
                )
                lines.append(f"  ```{r['context']}```")

        return ToolResult.ok(self.name, "", "\n".join(lines), matches=len(results))

    async def _custom_search(
        self, pattern: str, file_pattern: str | None,
        max_results: int, workspace_root: Path,
    ) -> ToolResult:
        """Handle custom DSL queries like 'call:*.unwrap', 'string:/password/i'."""
        results: list[dict] = []

        # Parse DSL: type:value
        match_type = "text"
        match_value = pattern
        if ":" in pattern:
            parts = pattern.split(":", 1)
            match_type = parts[0].lower()
            match_value = parts[1]

        # Build regex
        if match_value.startswith("/") and match_value.endswith("/"):
            regex = re.compile(match_value[1:-1], re.IGNORECASE)
        else:
            escaped = re.escape(match_value).replace(r"\*", ".*")
            regex = re.compile(escaped, re.IGNORECASE)

        from tracera.indexer.scanner import FileScanner
        scanner = FileScanner(str(workspace_root))
        source_files = scanner.scan()

        if file_pattern:
            import fnmatch
            source_files = [f for f in source_files if fnmatch.fnmatch(f, file_pattern)]

        for fp in source_files[:500]:
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for match in regex.finditer(content):
                line_no = content[:match.start()].count("\n") + 1
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                if line_end == -1:
                    line_end = len(content)
                line_text = content[line_start:line_end].strip()

                results.append({
                    "file": fp,
                    "line": line_no,
                    "match": match.group()[:100],
                    "line_text": line_text[:200],
                })
                if len(results) >= max_results:
                    break

        lines = [f"## Custom AST Search: `{pattern}`\n"]
        lines.append(f"**Matches:** {len(results)}\n")

        for r in results:
            lines.append(f"- `{r['file']}`:{r['line']}")
            lines.append(f"  ```{r['line_text']}```")

        return ToolResult.ok(self.name, "", "\n".join(lines), matches=len(results))


# ── get_class_hierarchy ──────────────────────────────────────────────────────

class GetClassHierarchyTool(Tool):
    """Traverse inheritance chains for a class."""

    name = "get_class_hierarchy"
    description = (
        "Traverse the inheritance hierarchy of a class: base classes, subclasses, "
        "and implemented interfaces."
    )
    _params = {
        "type": "object",
        "properties": {
            "class_name": {
                "type": "string",
                "description": "Name of the class to inspect.",
            },
        },
        "required": ["class_name"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, class_name: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, class_name)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Class '{class_name}' not found.")

        lines = [f"## Class Hierarchy: `{class_name}`\n"]

        # Parents (inheritance edges)
        parents = []
        for src, dst, data in graph._g.in_edges(node_id, data=True):
            if data.get("relation") in ("inherits", "implements"):
                parents.append(src)

        if parents:
            lines.append("### Extends / Implements")
            for pid in parents:
                lines.append(f"- {_node_label(graph, pid)}")
        else:
            lines.append("### Extends / Implements")
            lines.append("_No base classes found._")

        # Children (subclasses)
        children = []
        for src, dst, data in graph._g.out_edges(node_id, data=True):
            if data.get("relation") in ("inherits", "implements"):
                children.append(dst)

        if children:
            lines.append("\n### Subclasses")
            for cid in children:
                lines.append(f"- {_node_label(graph, cid)}")
        else:
            lines.append("\n### Subclasses")
            lines.append("_No subclasses found._")

        # Methods
        methods = graph.get_children(node_id)
        if methods:
            lines.append(f"\n### Methods ({len(methods)})")
            for mid in methods[:30]:
                lines.append(f"- {_node_label(graph, mid)}")

        return ToolResult.ok(self.name, "", "\n".join(lines), class_name=class_name)
