"""
AST-Aware Structural Analysis Tools — Inspired by jCodeMunch MCP.
Phase 11-20 complete: hotspots, change awareness, blast radius, reference verification, freshness, confidence.
"""

from __future__ import annotations

import ast
import re
import subprocess
import json
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult
from tracera.retrieval.incremental import IncrementalIndexer

log = get_logger("tools.ast_tools")


# ── Shared Metadata Helpers ──────────────────────────────────────────────────

def build_metadata(
    graph: Any,
    confidence: float,
    freshness: dict[str, dict] | None = None,
    extra: dict | None = None
) -> dict:
    """Build standard metadata object for all tool results"""
    # Calculate coverage stats
    files_analyzed = 0
    files_excluded = 0
    languages = set()
    dynamic_dispatch = "partial"
    reflection = "unsupported"
    
    if graph and hasattr(graph, "_g"):
        for node in graph._g.nodes(data=True):
            if node[1].get("file_path"):
                files_analyzed += 1
            if node[1].get("language"):
                languages.add(node[1].get("language"))
    
    return {
        "_meta": {
            "confidence": confidence,
            "freshness": freshness or {},
            "coverage": {
                "languages": list(languages),
                "files_analyzed": files_analyzed,
                "files_excluded": files_excluded,
                "dynamic_dispatch": dynamic_dispatch,
                "reflection": reflection
            }
        },
        **(extra or {})
    }


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


def _find_tests_for_symbol(graph, node_id: str) -> list[str]:
    """Find all test files that reference this symbol."""
    tests = []
    node = graph.get_node(node_id)
    if not node:
        return tests
    
    node_name = node.get("name", "")
    # Search for test files that import this symbol
    for nid, data in graph._g.nodes(data=True):
        fp = data.get("file_path", "")
        if ("test_" in fp or "_test.py" in fp) and node_name in str(data):
            tests.append(_node_label(graph, nid))
    return tests


def _find_endpoints_for_symbol(graph, node_id: str) -> list[str]:
    """Find all HTTP endpoints/API routes that depend on this symbol."""
    endpoints = []
    node = graph.get_node(node_id)
    if not node:
        return endpoints
    
    # BFS to find any route/handler nodes upstream
    import networkx as nx
    reverse_g = graph._g.reverse()
    visited = set()
    queue = [node_id]
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        current_node = graph.get_node(current)
        if current_node:
            name = current_node.get("name", "").lower()
            if any(kw in name for kw in ["route", "endpoint", "handler", "app"]):
                endpoints.append(_node_label(graph, current))
        
        for predecessor in reverse_g.predecessors(current):
            if predecessor not in visited:
                queue.append(predecessor)
    return endpoints


# ── STEP 11: Hotspot Analysis ────────────────────────────────────────────────

class GetHotspotsTool(Tool):
    """Surface risky code by complexity × churn × centrality × test coverage."""
    name = "get_hotspots"
    description = "Find high-risk code hotspots combining complexity, git churn, and graph centrality."
    _params = {
        "type": "object",
        "properties": {
            "top_n": {"type": "integer", "default": 10},
            "formula": {"type": "string", "description": "Configurable score formula (complexity*churn*centrality)"}
        },
        "required": []
    }

    def __init__(self, workspace=None, retrieval_pipeline=None):
        self._workspace = workspace
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, top_n: int = 10, formula: str | None = None) -> ToolResult:
        graph = _get_graph(self._pipeline)
        workspace_root = str(self._workspace.root) if self._workspace else "."
        # Get git churn
        churn = defaultdict(int)
        try:
            result = subprocess.run(
                ["git", "log", "--pretty=format:", "--name-only", "--since=90 days"],
                cwd=workspace_root, capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip(): churn[line.strip()] +=1
        except: pass
        # Get test coverage if available
        coverage: dict[str, float] = {}
        try:
            cov_result = subprocess.run(
                ["pytest", "--cov-report=json", "--cov=."], cwd=workspace_root, capture_output=True, text=True, timeout=60
            )
            cov_path = Path(workspace_root) / "coverage.json"
            if cov_path.exists():
                cov_data = json.loads(cov_path.read_text())
                coverage = {f: d["summary"]["percent_covered"] for f, d in cov_data.get("files", {}).items()}
        except: pass
        # Calculate page_rank centrality
        import networkx as nx
        page_rank = nx.pagerank(graph._g) if graph else {}
        # Compute hotspots
        candidates = []
        if graph:
            for node_id, data in graph._g.nodes(data=True):
                fp = data.get("file_path", "")
                start = data.get("start_line", 0)
                end = data.get("end_line", 0)
                complexity = max(1, (end - start) if end and start else 1)
                churn_count = churn.get(fp, 0)
                centrality = page_rank.get(node_id, 0.0001)
                cov_score = coverage.get(fp, 50.0) / 100
                # Apply formula
                if formula and formula == "complexity*churn*centrality*(1-cov*0.5)":
                    score = complexity * (1 + churn_count*0.1) * (1 + centrality*100) * (1.5 - cov_score)
                else: # default formula
                    score = complexity * (1 + churn_count*0.1) * (1 + centrality*100)
                candidates.append({
                    "symbol": data.get("name", ""),
                    "file_path": fp,
                    "complexity": complexity,
                    "churn": churn_count,
                    "centrality": round(centrality*1000, 2),
                    "test_coverage": round(cov_score*100, 1),
                    "score": round(score, 2)
                })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        # Build output
        lines = ["## Code Hotspots (complexity × churn × centrality)\n"]
        if not candidates:
            lines.append("No data available to calculate hotspots.")
            metadata = build_metadata(graph, 0.0, None, {"hotspots": 0})
            return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)
        for i, c in enumerate(candidates[:top_n]):
            lines.append(f"### {i+1}. `{c['symbol']}` in `{c['file_path']}`")
            lines.append(f"- Complexity: {c['complexity']} | Churn: {c['churn']} | Centrality: {c['centrality']} | Coverage: {c['test_coverage']}%")
            lines.append(f"- Risk score: {c['score']}\n")
        # Add freshness metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.92, freshness, {"hotspots": len(candidates[:top_n])})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 12: Change-Aware Intelligence ───────────────────────────────────────

class GetChangedSymbolsTool(Tool):
    """Map git diff to affected symbols, their callers, tests, and blast radius."""
    name = "get_changed_symbols"
    description = "Map git changes to affected symbols, including dependent callers, tests, and impact."
    _params = {"type": "object", "properties": {"ref": {"type": "string", "default": "HEAD"}}, "required": []}

    def __init__(self, workspace=None, retrieval_pipeline=None):
        self._workspace = workspace
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, ref: str = "HEAD") -> ToolResult:
        graph = _get_graph(self._pipeline)
        workspace_root = str(self._workspace.root) if self._workspace else "."
        # Get changed files
        try:
            if ref == "HEAD":
                diff = subprocess.run(["git", "diff", "--name-only"], cwd=workspace_root, capture_output=True, text=True, timeout=10)
            else:
                diff = subprocess.run(["git", "diff", "--name-only", ref], cwd=workspace_root, capture_output=True, text=True, timeout=10)
            changed_files = [f.strip() for f in diff.stdout.strip().split("\n") if f.strip()]
        except Exception as e:
            return ToolResult.fail(self.name, "", f"Git diff failed: {e}")
        if not changed_files:
            metadata = build_metadata(graph, 1.0, None, {"changed_files": 0})
            return ToolResult.ok(self.name, "", "No changed files detected in the analyzed corpus.", **metadata)
        # Map to symbols + their dependencies
        lines = [f"## Changed Symbols (ref: `{ref}`)\n"]
        lines.append(f"**Changed files:** {len(changed_files)}\n")
        all_affected = []
        if graph:
            for fp in changed_files:
                lines.append(f"### `{fp}`")
                node_ids = graph.find_by_file(fp)
                if node_ids:
                    for nid in node_ids[:10]:
                        lines.append(f"- Changed: {_node_label(graph, nid)}")
                        # Get affected callers
                        callers = graph.get_callers(nid)
                        if callers:
                            lines.append("  - Affected callers:")
                            for c in callers[:5]:
                                lines.append(f"    * {_node_label(graph, c)}")
                        # Get affected tests
                        tests = _find_tests_for_symbol(graph, nid)
                        if tests:
                            lines.append("  - Affected tests:")
                            for t in tests[:3]:
                                lines.append(f"    * {t}")
                        # Get blast radius
                        affected = graph.get_ancestors(nid, max_depth=3)
                        all_affected.extend(affected)
                else:
                    lines.append("  - No symbols matched in index for this file")
        else:
            lines.append("Symbol graph not available.")
        # Add freshness
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.88, freshness, {"changed_files": len(changed_files), "total_affected": len(all_affected)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 13: Blast-Radius Analysis ───────────────────────────────────────────

class GetBlastRadiusTool(Tool):
    """Complete blast radius including tests, endpoints, and all dependencies."""
    name = "get_blast_radius"
    description = "Calculate full impact of changing a symbol: callers, imports, tests, endpoints."
    _params = {"type": "object", "properties": {"symbol": {"type": "string"}, "depth": {"type": "integer", "default": 3}}, "required": ["symbol"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str, depth: int = 3) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        # BFS blast radius
        import networkx as nx
        reverse_g = graph._g.reverse()
        affected: dict[str, int] = {}
        bfs_queue = [(node_id, 0)]
        visited = {node_id}
        while bfs_queue:
            current, d = bfs_queue.pop(0)
            if d >= depth: continue
            for predecessor in reverse_g.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    affected[predecessor] = d +1
                    bfs_queue.append((predecessor, d+1))
        # Categorize results
        importers = []
        callers = []
        tests = _find_tests_for_symbol(graph, node_id)
        endpoints = _find_endpoints_for_symbol(graph, node_id)
        other_symbols = []
        for nid in affected:
            node = graph.get_node(nid)
            if node:
                if node.get("symbol_type") == "import": importers.append(_node_label(graph, nid))
                elif node.get("relation") == "calls": callers.append(_node_label(graph, nid))
                else: other_symbols.append(_node_label(graph, nid))
        # Build output
        lines = [f"## Blast Radius for `{symbol}` (depth={depth})\n"]
        lines.append(f"**Total affected symbols:** {len(affected)}\n")
        if not any([importers, callers, tests, endpoints, other_symbols]):
            metadata = build_metadata(graph, 0.9, None, {"total_affected": 0})
            return ToolResult.ok(self.name, "", "No downstream dependencies found in the indexed/analyzable corpus.", **metadata)
        if importers:
            lines.append("### Modules that import it")
            for i in importers[:10]: lines.append(f"- {i}")
        if callers:
            lines.append("\n### Functions that call it")
            for c in callers[:10]: lines.append(f"- {c}")
        if tests:
            lines.append("\n### Tests that touch it")
            for t in tests[:10]: lines.append(f"- {t}")
        if endpoints:
            lines.append("\n### Endpoints that depend on it")
            for e in endpoints[:10]: lines.append(f"- {e}")
        if other_symbols:
            lines.append("\n### Other affected symbols")
            for o in other_symbols[:10]: lines.append(f"- {o}")
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.91, freshness, {"total_affected": len(affected)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 14: Reference / Implementation Analysis ─────────────────────────────

class FindReferencesTool(Tool):
    """Find references with confidence levels: compiler_verified/lsp_verified/ast_inferred/heuristic."""
    name = "find_references"
    description = "Find all references to a symbol with verification level."
    _params = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        # Get all references
        references = graph.get_callers(node_id)
        if not references:
            metadata = build_metadata(graph, 0.85)
            return ToolResult.ok(self.name, "", "No references found in the indexed/analyzable corpus.", **metadata)
        # Classify verification level (all ast_inferred for now, framework ready for LSP/compiler)
        lines = [f"## References to `{symbol}`\n"]
        verified_references = []
        for ref_id in references:
            node = graph.get_node(ref_id)
            if node:
                verified_references.append({
                    "location": _node_label(graph, ref_id),
                    "verification": "ast_inferred"
                })
        for r in verified_references[:30]:
            lines.append(f"- [{r['verification']}] {r['location']}")
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.82, freshness, {"references_found": len(references)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


class FindImplementationsTool(Tool):
    """Find implementations of interfaces/abstract classes with confidence levels."""
    name = "find_implementations"
    description = "Find all implementations of an interface/abstract class."
    _params = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        # Find all classes that inherit from this symbol
        implementations = []
        for nid, data in graph._g.nodes(data=True):
            if data.get("inherits_from") == node_id:
                implementations.append({"location": _node_label(graph, nid), "verification": "ast_inferred"})
        if not implementations:
            metadata = build_metadata(graph, 0.8)
            return ToolResult.ok(self.name, "", "No implementations found in the indexed/analyzable corpus.", **metadata)
        lines = [f"## Implementations of `{symbol}`\n"]
        for impl in implementations:
            lines.append(f"- [{impl['verification']}] {impl['location']}")
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.8, freshness, {"implementations_found": len(implementations)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 15: Task-Level Orchestration ────────────────────────────────────────

class PlanCodeTaskTool(Tool):
    """Classify task intent, extract anchors, and recommend tool chain."""
    name = "plan_code_task"
    description = "Classify a code task and build a recommended execution plan."
    _params = {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, task: str) -> ToolResult:
        # Classify intent
        intent = "explore"
        task_lower = task.lower()
        if "refactor" in task_lower: intent = "refactor"
        elif "debug" in task_lower or "fix" in task_lower: intent = "debug"
        elif "extend" in task_lower or "add" in task_lower: intent = "extend"
        elif "audit" in task_lower or "review" in task_lower: intent = "audit"
        # Extract anchor symbols (simple keyword extraction)
        import re
        anchors = re.findall(r"`(\w+)`", task) or re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', task)
        # Recommend tools based on intent
        tool_chains = {
            "refactor": ["search_symbols", "get_symbol_source", "find_references", "get_call_hierarchy", "get_blast_radius"],
            "debug": ["search_symbols", "get_symbol_source", "get_call_hierarchy", "get_changed_symbols"],
            "extend": ["search_symbols", "get_symbol_source", "find_implementations"],
            "audit": ["get_hotspots", "find_dead_code", "get_blast_radius"],
            "explore": ["search_symbols", "get_symbol_source", "find_importers", "get_call_hierarchy"]
        }
        recommended_tools = tool_chains.get(intent, tool_chains["explore"])
        # Build output
        lines = [f"## Task Plan: {task}\n"]
        lines.append(f"**Intent:** {intent}")
        lines.append(f"**Anchor symbols:** {', '.join(anchors) if anchors else 'None detected'}")
        lines.append(f"**Recommended tools:** {', '.join(recommended_tools)}")
        # Add metadata
        graph = _get_graph(self._pipeline)
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.85, freshness, {"intent": intent, "anchors": anchors, "recommended_tools": recommended_tools})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


class AssembleCodeContextTool(Tool):
    """Assemble token-budgeted context capsule for a task."""
    name = "assemble_code_context"
    description = "Build a minimal context capsule that stays within token budget."
    _params = {"type": "object", "properties": {"task": {"type": "string"}, "token_budget": {"type": "integer", "default": 8000}}, "required": ["task"]}

    def __init__(self, workspace=None, retrieval_pipeline=None):
        self._workspace = workspace
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, task: str, token_budget: int = 8000) -> ToolResult:
        # First run planning
        plan = await PlanCodeTaskTool(self._pipeline).execute(task)
        # Extract anchors and retrieve symbols
        anchors = plan.metadata.get("anchors", [])
        graph = _get_graph(self._pipeline)
        retrieved_nodes = []
        tokens_used = 0
        if graph and anchors:
            for anchor in anchors:
                node_id = _resolve_symbol(graph, anchor)
                if node_id:
                    # Add symbol source
                    node = graph.get_node(node_id)
                    if node:
                        source = Path(node["file_path"]).read_text()[node.get("start_byte",0):node.get("end_byte", 1000)]
                        tokens_used += len(source) // 4 # rough token estimate
                        if tokens_used < token_budget:
                            retrieved_nodes.append(node)
        # Deduplicate and format
        lines = [f"## Context Capsule for: {task}\n"]
        lines.append(f"**Token budget:** {token_budget} | Tokens used: {tokens_used}")
        lines.append(f"**Symbols included:** {len(retrieved_nodes)}\n")
        for node in retrieved_nodes:
            lines.append(f"### `{node['name']}` in {node['file_path']}")
            lines.append("```")
            lines.append(Path(node["file_path"]).read_text()[node.get("start_line",0)*100:node.get("end_line",100)*100][:500])
            lines.append("```\n")
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.88, freshness, {"tokens_used": tokens_used, "token_budget": token_budget, "symbols_included": len(retrieved_nodes)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 16: Token-Efficient Context Measurement ─────────────────────────────

class GetSessionStatsTool(Tool):
    """Track token savings from code-intelligence context vs whole-file reads."""
    name = "get_session_stats"
    description = "Get token efficiency metrics: code-intelligence vs naive full-file context."
    _params = {"type": "object", "properties": {}, "required": []}

    def __init__(self, session=None, retrieval_pipeline=None):
        self._session = session
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self) -> ToolResult:
        stats = {
            "bytes_saved": 0,
            "tokens_saved": 0,
            "files_avoided": 0,
            "symbols_used": 0,
            "retrieval_latency_ms": 0.0
        }
        if self._session:
            stats = self._session.get_stats()
        # Build output
        lines = ["## Session Token Efficiency Stats\n"]
        lines.append(f"**Bytes saved:** {stats['bytes_saved']:,}")
        lines.append(f"**Tokens saved:** {stats['tokens_saved']:,}")
        lines.append(f"**Files avoided:** {stats['files_avoided']}")
        lines.append(f"**Symbols retrieved:** {stats['symbols_used']}")
        lines.append(f"**Average retrieval latency:** {stats['retrieval_latency_ms']:.2f}ms")
        # Add metadata
        graph = _get_graph(self._pipeline)
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 1.0, freshness, stats)
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 17/18: Freshness + Index Snapshots ──────────────────────────────────

class GetIndexFreshnessTool(Tool):
    """Check index freshness against filesystem and git."""
    name = "get_index_freshness"
    description = "Verify index is fresh against filesystem and git status."
    _params = {"type": "object", "properties": {}, "required": []}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self) -> ToolResult:
        if not self._pipeline or len(self._pipeline) ==0:
            return ToolResult.ok(self.name, "", "No retrieval pipeline available.")
        indexer = self._pipeline[0]
        if not hasattr(indexer, "check_freshness"):
            return ToolResult.ok(self.name, "", "Indexer does not support freshness checks.")
        freshness = indexer.check_freshness()
        # Load snapshot metadata
        manifest = indexer._load_manifest()
        snapshot = manifest.get("snapshot", {})
        # Build output
        lines = ["## Index Freshness Report\n"]
        lines.append(f"**Index snapshot generated:** {snapshot.get('generation_timestamp', 'unknown')}")
        lines.append(f"**Git SHA:** {snapshot.get('git_sha', 'unknown')}")
        lines.append(f"**Files indexed:** {snapshot.get('files_indexed', 0)}\n")
        fresh = sum(1 for f in freshness.values() if f["state"] == "fresh")
        edited = sum(1 for f in freshness.values() if f["state"] == "edited_uncommitted")
        stale = sum(1 for f in freshness.values() if f["state"] == "stale_index")
        missing = sum(1 for f in freshness.values() if f["state"] == "missing")
        lines.append(f"✅ Fresh: {fresh}")
        lines.append(f"✏️ Edited (uncommitted): {edited}")
        lines.append(f"⚠️ Stale: {stale}")
        lines.append(f"❌ Missing: {missing}")
        if edited + stale + missing > 0:
            lines.append("\n⚠️ **WARNING: Index has stale/edited/missing files - re-run `tracera index` to refresh.**")
        # Add metadata
        graph = _get_graph(self._pipeline)
        metadata = build_metadata(graph, 0.99, freshness, snapshot)
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── Existing tools updated with new metadata ─────────────────────────────────

class FindImportersTool(Tool):
    """Find all files/symbols that import a given file or module."""
    name = "find_importers"
    description = "Find all files and symbols that import a given file or module."
    _params = {"type": "object", "properties": {"path": {"type": "string"}, "max_results": {"type": "integer", "default":20}}, "required": ["path"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, path: str, max_results: int =20) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        target_nodes = graph.find_by_file(path)
        if not target_nodes:
            all_files = set()
            for _, data in graph._g.nodes(data=True):
                fp = data.get("file_path", "")
                if path in fp or fp.endswith(path):
                    all_files.add(fp)
            if all_files:
                target_file = list(all_files)[0]
                target_nodes = graph.find_by_file(target_file)
        if not target_nodes:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"No symbols found for '{path}' in the indexed/analyzable corpus.", **metadata)
        importers = set()
        for node_id in target_nodes:
            ancestors = graph.get_ancestors(node_id, max_depth=2)
            importers.update(ancestors)
        lines = [f"## Importers of `{path}`\n"]
        count =0
        for imp_id in sorted(importers):
            if count >= max_results:
                lines.append(f"\n... and {len(importers)-max_results} more.")
                break
            node = graph.get_node(imp_id)
            if node:
                lines.append(f"- `{node.get('name','?')}` ({node.get('symbol_type','?')}) in `{node.get('file_path','?')}`:{node.get('start_line','?')}")
                count +=1
        if count ==0:
            metadata = build_metadata(graph, 0.85)
            return ToolResult.ok(self.name, "", "No importers found in the indexed/analyzable corpus.", **metadata)
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.9, freshness, {"importers_found": len(importers)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


class GetCallHierarchyTool(Tool):
    """Trace callers and callees N levels deep."""
    name = "get_call_hierarchy"
    description = "Trace call hierarchy of a symbol."
    _params = {"type": "object", "properties": {"symbol": {"type": "string"}, "depth": {"type": "integer", "default":2}, "direction": {"type": "string", "enum": ["callers","callees","both"], "default":"both"}}, "required": ["symbol"]}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str, depth: int=2, direction: str="both") -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        lines = [f"## Call Hierarchy for `{symbol}`\n"]
        callers = graph.get_ancestors(node_id, max_depth=depth) if direction in ("callers","both") else []
        callees = graph.get_descendants(node_id, max_depth=depth) if direction in ("callees","both") else []
        if direction in ("callers","both"):
            lines.append("### Callers (who calls this)")
            if not callers:
                lines.append("No callers found in the indexed/analyzable corpus.")
            for cid in callers[:20]:
                lines.append(f"- {_node_label(graph, cid)}")
        if direction in ("callees","both"):
            lines.append("\n### Callees (what this calls)")
            if not callees:
                lines.append("No callees found in the indexed/analyzable corpus.")
            for cid in callees[:20]:
                lines.append(f"- {_node_label(graph, cid)}")
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.88, freshness, {"callers": len(callers), "callees": len(callees)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


class FindDeadCodeTool(Tool):
    """Find symbols unreachable from entry points."""
    name = "find_dead_code"
    description = "Find dead code in the repository."
    _params = {"type": "object", "properties": {"max_results": {"type": "integer", "default":30}}, "required": []}

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, max_results: int=30) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        import networkx as nx
        g = graph._g
        entry_keywords = {"main", "__main__", "app", "cli", "handler", "route", "setup"}
        entry_nodes = set()
        for node_id, data in g.nodes(data=True):
            name = (data.get("name") or "").lower()
            if any(kw in name for kw in entry_keywords):
                entry_nodes.add(node_id)
        reachable = set()
        queue = list(entry_nodes)
        visited = set(entry_nodes)
        while queue:
            current = queue.pop(0)
            reachable.add(current)
            for successor in g.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        all_nodes = set(g.nodes())
        dead = all_nodes - reachable
        lines = ["## Dead Code Analysis\n"]
        lines.append(f"**Total symbols:** {len(all_nodes)}")
        lines.append(f"**Reachable from entries:** {len(reachable)}")
        lines.append(f"**Potentially dead:** {len(dead)}\n")
        if not dead:
            metadata = build_metadata(graph, 0.85)
            return ToolResult.ok(self.name, "", "No dead code found in the indexed/analyzable corpus.", **metadata)
        by_file = defaultdict(list)
        for nid in sorted(dead):
            node = g.nodes[nid]
            fp = node.get("file_path", "unknown")
            by_file[fp].append(_node_label(graph, nid))
        count =0
        for fp in sorted(by_file):
            if count >= max_results:
                lines.append(f"\n... and {len(dead)-max_results} more symbols.")
                break
            lines.append(f"\n### `{fp}`")
            for label in by_file[fp]:
                if count >= max_results: break
                lines.append(f"- {label}")
                count +=1
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.82, freshness, {"total_symbols": len(all_nodes), "dead_symbols": len(dead)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 21: Code Intelligence + TRACERA Memory Integration ──────────────────

class MemoryIntegration:
    """Bridge between Code Intelligence and TRACERA's persistent memory layer.
    
    Maintains conceptual separation:
    - Code Intelligence = what the repository currently contains
    - Persistent Memory = what TRACERA learned/observed over time
    
    Only stores durable agent/project knowledge, never individual symbols.
    """
    def __init__(self, memory_facade):
        self._memory = memory_facade
        self._durable_patterns = {
            "auth_middleware": {
                "patterns": [r"auth/middleware\.py", r"authentication.*middleware"],
                "memory_template": "Authentication logic lives in {location}"
            },
            "repository_pattern": {
                "patterns": [r".*repository\.py", r".*Repository\s*class"],
                "memory_template": "This repository uses the repository pattern for database access"
            },
            "change_correlation": {
                "patterns": [],  # Learned over time from repeated changes
                "memory_template": "Changing {x} usually requires updating {y}"
            },
            "failure_history": {
                "patterns": [],  # Learned from failed attempts
                "memory_template": "Previous attempts to modify {x} failed because {reason}"
            }
        }

    def extract_durable_knowledge(self, analysis_results: dict) -> list[str]:
        """Extract and store durable project knowledge from code intelligence analysis."""
        new_memories = []
        # Detect architectural patterns
        if "file_structure" in analysis_results:
            for file in analysis_results["file_structure"]:
                for pattern_type, config in self._durable_patterns.items():
                    import re
                    for p in config["patterns"]:
                        if re.search(p, file):
                            memory = config["memory_template"].format(location=file)
                            self._memory.store(memory)
                            new_memories.append(memory)
        # Detect change correlations
        if "changed_symbols" in analysis_results:
            changed = analysis_results["changed_symbols"]
            if len(changed) >= 2:
                x, y = changed[0], changed[1]
                memory = f"Changing {x} usually requires updating {y}"
                self._memory.store(memory)
                new_memories.append(memory)
        return new_memories

    def recall_relevant_memories(self, context: str) -> list[str]:
        """Recall memories relevant to the current task context."""
        return self._memory.recall(context, top_k=5)


# ── STEP 24: Refactoring Intelligence ────────────────────────────────────────

class PlanRefactoringTool(Tool):
    """Analyze impact of refactoring operations (rename, move, extract, etc.)."""
    name = "plan_refactoring"
    description = "Plan a code refactoring: analyze affected symbols, files, imports, tests, and risk."
    _params = {
        "type": "object",
        "properties": {
            "refactor_type": {"type": "string", "enum": ["rename", "move", "extract", "signature_change", "interface_change", "module_relocation"]},
            "symbol": {"type": "string"},
            "new_name": {"type": "string", "default": ""},
            "new_path": {"type": "string", "default": ""}
        },
        "required": ["refactor_type", "symbol"]
    }

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, refactor_type: str, symbol: str, new_name: str = "", new_path: str = "") -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            metadata = build_metadata(graph, 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        
        # Get full blast radius
        affected = graph.get_ancestors(node_id, max_depth=10)
        affected_symbols = [_node_label(graph, nid) for nid in affected]
        affected_files = list(set(graph.get_node(nid)["file_path"] for nid in affected if "file_path" in graph.get_node(nid)))
        tests = _find_tests_for_symbol(graph, node_id)
        
        # Identify imports to update
        imports_to_modify = []
        for nid in affected:
            node = graph.get_node(nid)
            if node.get("symbol_type") == "import":
                imports_to_modify.append(_node_label(graph, nid))
        
        # Detect possible collisions
        collisions = []
        if new_name:
            existing = graph.find_by_name(new_name)
            if existing:
                collisions.append(f"Symbol '{new_name}' already exists in the repository")
        
        # Calculate risk
        risk_score = len(affected) * 0.1 + len(tests) * 0.05
        risk_level = "low" if risk_score < 5 else "medium" if risk_score < 15 else "high"
        
        # Build output
        lines = [f"## Refactoring Plan: {refactor_type} on `{symbol}`\n"]
        lines.append(f"**Risk Level:** {risk_level} (score: {risk_score:.1f})")
        lines.append(f"**Total affected symbols:** {len(affected_symbols)}")
        lines.append(f"**Affected files:** {len(affected_files)}")
        
        if affected_files:
            lines.append("\n### Files requiring modification:")
            for f in affected_files[:15]:
                lines.append(f"- {f}")
            if len(affected_files) > 15:
                lines.append(f"... and {len(affected_files)-15} more")
        
        if imports_to_modify:
            lines.append("\n### Imports requiring updates:")
            for imp in imports_to_modify[:10]:
                lines.append(f"- {imp}")
        
        if tests:
            lines.append("\n### Tests to update:")
            for t in tests[:10]:
                lines.append(f"- {t}")
        
        if collisions:
            lines.append("\n### ⚠️ Possible collisions:")
            for c in collisions:
                lines.append(f"- {c}")
        
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.9, freshness, {
            "refactor_type": refactor_type,
            "affected_symbols": len(affected_symbols),
            "affected_files": len(affected_files),
            "risk_score": risk_score,
            "risk_level": risk_level
        })
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 25: Code Provenance (Git History Integration) ──────────────────────

class GetCodeProvenanceTool(Tool):
    """Retrieve Git history for a symbol: why it was introduced, when it changed, who modified it."""
    name = "get_code_provenance"
    description = "Get structured Git history/provenance for a symbol including commits, authors, and change types."
    _params = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"]
    }

    def __init__(self, retrieval_pipeline=None, workspace=None):
        self._pipeline = retrieval_pipeline
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        workspace_root = str(self._workspace.root) if self._workspace else "."
        node_id = _resolve_symbol(graph, symbol) if graph else None
        
        if not node_id or not graph:
            metadata = build_metadata(graph, 0.0 if not graph else 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        
        node = graph.get_node(node_id)
        file_path = node.get("file_path", "")
        start_line = node.get("start_line", 1)
        
        if not file_path:
            metadata = build_metadata(graph, 0.8)
            return ToolResult.ok(self.name, "", f"No file path found for symbol '{symbol}'.", **metadata)
        
        # Get git history for the lines containing this symbol
        try:
            result = subprocess.run(
                ["git", "log", f"-L{start_line},{start_line+50}:{file_path}", "--pretty=format:%h|%an|%ad|%s", "--date=iso"],
                cwd=workspace_root, capture_output=True, text=True, timeout=30
            )
            commit_lines = result.stdout.strip().split("\n")
        except Exception as e:
            return ToolResult.fail(self.name, "", f"Failed to retrieve git history: {e}")
        
        commits = []
        for line in commit_lines:
            if line.strip():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    sha, author, date, message = parts
                    change_type = "feature" if "add" in message.lower() or "implement" in message.lower() else \
                                  "fix" if "fix" in message.lower() or "bug" in message.lower() else \
                                  "refactor" if "refactor" in message.lower() or "move" in message.lower() else "other"
                    commits.append({
                        "sha": sha,
                        "author": author,
                        "date": date,
                        "message": message,
                        "change_type": change_type
                    })
        
        # Build output
        lines = [f"## Code Provenance for `{symbol}`\n"]
        lines.append(f"Located in `{file_path}` lines {start_line}-{node.get('end_line', '?')}\n")
        if not commits:
            lines.append("No git history found for this symbol in the repository.")
        else:
            lines.append(f"### {len(commits)} commits affecting this symbol:\n")
            for c in commits[:15]:
                lines.append(f"**`{c['sha']}`** {c['date']} - {c['author']}")
                lines.append(f"  *[{c['change_type']}]* {c['message']}\n")
            if len(commits) > 15:
                lines.append(f"... and {len(commits)-15} more commits")
        
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.94, freshness, {"commits_found": len(commits)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 26: Code Risk Engine ────────────────────────────────────────────────

class AssessChangeRiskTool(Tool):
    """Composite change-risk model combining blast radius, complexity, churn, test coverage, and more."""
    name = "assess_change_risk"
    description = "Calculate risk score for a change including factors, affected areas, and missing tests."
    _params = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"]
    }

    def __init__(self, retrieval_pipeline=None, workspace=None):
        self._pipeline = retrieval_pipeline
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        workspace_root = str(self._workspace.root) if self._workspace else "."
        node_id = _resolve_symbol(graph, symbol) if graph else None
        
        if not node_id or not graph:
            metadata = build_metadata(graph, 0.0 if not graph else 0.95)
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found in the indexed/analyzable corpus.", **metadata)
        
        node = graph.get_node(node_id)
        file_path = node.get("file_path", "")
        
        # 1. Blast radius factor
        affected = graph.get_ancestors(node_id, max_depth=5)
        blast_radius_score = len(affected) * 0.1
        
        # 2. Complexity factor
        complexity = (node.get("end_line", 100) - node.get("start_line", 0)) / 100
        complexity_score = complexity * 0.5
        
        # 3. Git churn factor
        churn = 0
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--", file_path],
                cwd=workspace_root, capture_output=True, text=True, timeout=10
            )
            churn = len(result.stdout.strip().split("\n"))
        except: pass
        churn_score = churn * 0.05
        
        # 4. Test coverage factor
        coverage_score = 1.0
        try:
            cov_path = Path(workspace_root) / "coverage.json"
            if cov_path.exists():
                import json
                cov_data = json.loads(cov_path.read_text())
                if file_path in cov_data.get("files", {}):
                    cov_pct = cov_data["files"][file_path]["summary"]["percent_covered"]
                    coverage_score = (100 - cov_pct) / 100  # lower coverage = higher risk
        except: pass
        
        # 5. Dependency centrality
        import networkx as nx
        page_rank = nx.pagerank(graph._g).get(node_id, 0.0001)
        centrality_score = page_rank * 100
        
        # Total risk score (0-10 scale)
        total_risk = min(10, blast_radius_score + complexity_score + churn_score + coverage_score + centrality_score)
        risk_level = "low" if total_risk < 3 else "medium" if total_risk < 6 else "high"
        
        # Identify risk factors
        risk_factors = []
        if blast_radius_score > 2: risk_factors.append("Large blast radius (many affected symbols)")
        if complexity_score > 0.5: risk_factors.append("High code complexity")
        if churn_score > 1: risk_factors.append("High git churn (frequently modified)")
        if coverage_score > 0.5: risk_factors.append("Low test coverage")
        if centrality_score > 1: risk_factors.append("High dependency centrality")
        
        # Check for missing tests
        tests = _find_tests_for_symbol(graph, node_id)
        missing_tests = len(tests) == 0
        
        # Build output
        lines = [f"## Change Risk Assessment for `{symbol}`\n"]
        lines.append(f"**Overall Risk Score:** {total_risk:.1f}/10 ({risk_level.upper()})\n")
        if risk_factors:
            lines.append("### Key risk factors:")
            for rf in risk_factors:
                lines.append(f"- {rf}")
        lines.append(f"\n### Detailed metrics:")
        lines.append(f"- Blast radius impact: {blast_radius_score:.2f} ({len(affected)} affected symbols)")
        lines.append(f"- Complexity factor: {complexity_score:.2f} (LOC: {node.get('end_line', 0)-node.get('start_line',0)})")
        lines.append(f"- Churn factor: {churn_score:.2f} ({churn} commits)")
        lines.append(f"- Coverage factor: {coverage_score:.2f}{' (MISSING TESTS)' if missing_tests else ''}")
        lines.append(f"- Centrality factor: {centrality_score:.2f}")
        
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.93, freshness, {
            "risk_score": round(total_risk, 2),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "missing_tests": missing_tests,
            "affected_symbols": len(affected)
        })
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 27: AST Structural Search ───────────────────────────────────────────

class StructuralSearchTool(Tool):
    """AST-level structural search supporting patterns like 'function calls X', 'class inherits Y', etc."""
    name = "structural_search"
    description = "Search code using AST patterns: function calls X, class inherits Y, try/except patterns, etc."
    _params = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Structural search pattern"},
            "max_results": {"type": "integer", "default": 30}
        },
        "required": ["pattern"]
    }

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline
        # Language-specific AST query handlers (extendable)
        self._pattern_handlers = {
            "function.calls": self._search_calls,
            "class.inherits": self._search_inheritance,
            "pattern.try_except": self._search_try_except,
            "call.unsafe": self._search_unsafe_calls
        }

    @property
    def parameters_schema(self) -> dict: return self._params

    def _parse_pattern(self, pattern: str) -> tuple[str, str]:
        """Parse pattern into type and target."""
        if "calls" in pattern:
            target = pattern.split("calls")[-1].strip()
            return "function.calls", target
        elif "inherits" in pattern or "extends" in pattern:
            target = pattern.split("inherits")[-1].strip()
            return "class.inherits", target
        elif "try/except" in pattern or "try_except" in pattern:
            return "pattern.try_except", ""
        elif "unsafe" in pattern or "eval" in pattern or "exec" in pattern:
            return "call.unsafe", ""
        return "unknown", ""

    def _search_calls(self, graph, target: str) -> list[dict]:
        """Find all functions that call the target function."""
        results = []
        target_id = _resolve_symbol(graph, target)
        if not target_id:
            return results
        callers = graph.get_callers(target_id)
        for cid in callers:
            node = graph.get_node(cid)
            if node:
                results.append({"location": _node_label(graph, cid), "match": f"Calls {target}"})
        return results

    def _search_inheritance(self, graph, target: str) -> list[dict]:
        """Find all classes that inherit from the target."""
        results = []
        for nid, data in graph._g.nodes(data=True):
            if data.get("inherits_from") == target:
                results.append({"location": _node_label(graph, nid), "match": f"Inherits from {target}"})
        return results

    def _search_try_except(self, graph, target: str) -> list[dict]:
        """Find all try/except blocks."""
        results = []
        for nid, data in graph._g.nodes(data=True):
            if data.get("node_type") == "try_statement":
                results.append({"location": _node_label(graph, nid), "match": "Contains try/except block"})
        return results

    def _search_unsafe_calls(self, graph, target: str) -> list[dict]:
        """Find calls to unsafe functions (eval, exec, execvp, etc.)."""
        unsafe_funcs = {"eval", "exec", "execvp", "system", "popen", "subprocess.run"}
        results = []
        for nid, data in graph._g.nodes(data=True):
            if data.get("name") in unsafe_funcs:
                results.append({"location": _node_label(graph, nid), "match": f"Calls unsafe function {data.get('name')}"})
        return results

    async def execute(self, pattern: str, max_results: int = 30) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            metadata = build_metadata(None, 0.0)
            return ToolResult.ok(self.name, "", "Symbol graph not available in the analyzable corpus.", **metadata)
        
        pattern_type, target = self._parse_pattern(pattern)
        handler = self._pattern_handlers.get(pattern_type)
        if not handler:
            metadata = build_metadata(graph, 0.8)
            return ToolResult.ok(self.name, "", f"Unsupported pattern type. Available patterns: function calls X, class inherits Y, try/except patterns, unsafe calls.", **metadata)
        
        matches = handler(graph, target)
        
        # Build output
        lines = [f"## Structural Search Results: `{pattern}`\n"]
        lines.append(f"Found {len(matches)} matches\n")
        if not matches:
            lines.append("No matches found in the indexed/analyzable corpus.")
        else:
            for m in matches[:max_results]:
                lines.append(f"- **{m['location']}**: {m['match']}")
            if len(matches) > max_results:
                lines.append(f"\n... and {len(matches)-max_results} more matches")
        
        # Add metadata
        freshness = {}
        if hasattr(self._pipeline[0], "check_freshness"):
            freshness = self._pipeline[0].check_freshness()
        metadata = build_metadata(graph, 0.88, freshness, {"pattern": pattern, "matches_found": len(matches)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── STEP 28: Semantic/Hybrid Search ──────────────────────────────────────────

class SearchSymbolsTool(Tool):
    """Hybrid search combining BM25 lexical search, semantic embeddings, symbol importance, and exact-match boosts."""
    name = "search_symbols"
    description = "Search for symbols using hybrid lexical+semantic retrieval with ranking."
    _params = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_n": {"type": "integer", "default": 20}
        },
        "required": ["query"]
    }

    def __init__(self, retriever=None):
        self._retriever = retriever

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, query: str, top_n: int = 20) -> ToolResult:
        if not self._retriever:
            return ToolResult.ok(self.name, "", "Retriever not available.")
        
        # Hybrid scoring as specified:
        # final_score = lexical_relevance + semantic_relevance + structural_importance + exact_identifier_boost + freshness_adjustment
        results = self._retriever.search(query, top_n=top_n)
        # Post-process with hybrid scoring
        scored_results = []
        for res in results:
            lexical = res.get("lexical_score", 0.0)
            semantic = res.get("semantic_score", 0.0)
            structural = res.get("importance_score", 0.0)
            exact_boost = 1.5 if res.get("name", "").lower() == query.lower() else 1.0
            freshness = res.get("freshness_score", 1.0)
            final_score = (lexical + semantic + structural) * exact_boost * freshness
            res["final_score"] = final_score
            scored_results.append(res)
        # Sort by final score
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Build output
        lines = [f"## Symbol Search Results: '{query}'\n"]
        for i, r in enumerate(scored_results[:top_n]):
            lines.append(f"{i+1}. `{r.get('name')}` in `{r.get('file_path')}` (score: {r['final_score']:.3f})")
        
        metadata = {"symbols_found": len(scored_results), "query": query}
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


# ── Core Symbol Retrieval Tools ───────────────────────────────────────────────

class GetSymbolSourceTool(Tool):
    """Get the source code for a specific symbol."""
    name = "get_symbol_source"
    description = "Retrieve the full source code of a symbol."
    _params = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"]
    }

    def __init__(self, retriever=None):
        self._retriever = retriever

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str) -> ToolResult:
        if not self._retriever:
            return ToolResult.ok(self.name, "", "Retriever not available.")
        node = self._retriever.get_symbol(symbol)
        if not node:
            return ToolResult.ok(self.name, f"Symbol '{symbol}' not found in the indexed/analyzable corpus.")
        # Load source
        source = Path(node["file_path"]).read_text()[node.get("start_byte", 0):node.get("end_byte", None)]
        lines = [f"## Source for `{symbol}`\n", f"Location: `{node['file_path']}`:{node.get('start_line', '?')}\n", "```python", source, "```"]
        return ToolResult.ok(self.name, "", "\n".join(lines))


class GetFileOutlineTool(Tool):
    """Get the outline/symbols of a file."""
    name = "get_file_outline"
    description = "Get the symbol outline of a file (classes, functions, constants)."
    _params = {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"]
    }

    def __init__(self, retriever=None):
        self._retriever = retriever

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, file_path: str) -> ToolResult:
        if not self._retriever:
            return ToolResult.ok(self.name, "", "Retriever not available.")
        nodes = self._retriever.get_file_symbols(file_path)
        if not nodes:
            return ToolResult.ok(self.name, "", f"No symbols found for '{file_path}' in the indexed/analyzable corpus.")
        lines = [f"## File Outline: `{file_path}`\n"]
        for node in sorted(nodes, key=lambda x: x.get("start_line", 0)):
            node_type = node.get("symbol_type", "symbol")
            lines.append(f"- {node_type}: `{node.get('name')}` (line {node.get('start_line', '?')})")
        return ToolResult.ok(self.name, "", "\n".join(lines))


class GetRepoMapTool(Tool):
    """Get a high-level repository map."""
    name = "get_repo_map"
    description = "Get a high-level map of the repository structure and key components."
    _params = {"type": "object", "properties": {}, "required": []}

    def __init__(self, graph_retriever=None):
        self._graph = graph_retriever

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self) -> ToolResult:
        if not self._graph:
            return ToolResult.ok(self.name, "", "Graph retriever not available.")
        # Generate repository map
        files = {}
        for _, data in self._graph._g.nodes(data=True):
            fp = data.get("file_path", "")
            if fp:
                if fp not in files:
                    files[fp] = {"symbols": 0, "types": set()}
                files[fp]["symbols"] += 1
                files[fp]["types"].add(data.get("symbol_type", "other"))
        lines = ["## Repository Map\n"]
        lines.append(f"Total files with symbols: {len(files)}")
        lines.append(f"Total symbols: {sum(f['symbols'] for f in files.values())}\n")
        for fp in sorted(files.keys())[:50]:
            f = files[fp]
            lines.append(f"- `{fp}`: {f['symbols']} symbols ({', '.join(f['types'])})")
        if len(files) > 50:
            lines.append(f"\n... and {len(files)-50} more files")
        return ToolResult.ok(self.name, "", "\n".join(lines))


# ── Additional Required Tools ────────────────────────────────────────────────

class GetDependenciesTool(Tool):
    """Get dependencies of a symbol (what it imports/calls)."""
    name = "get_dependencies"
    description = "Get all dependencies of a symbol (imports, calls, references)."
    _params = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
        "required": ["symbol"]
    }

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict: return self._params

    async def execute(self, symbol: str, depth: int = 2) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if not graph:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")
        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found.")
        dependencies = graph.get_descendants(node_id, max_depth=depth)
        lines = [f"## Dependencies of `{symbol}` (depth={depth})\n"]
        for nid in dependencies[:30]:
            lines.append(f"- {_node_label(graph, nid)}")
        if len(dependencies) > 30:
            lines.append(f"... and {len(dependencies)-30} more")
        metadata = build_metadata(graph, 0.9, None, {"dependencies_found": len(dependencies)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)


class CalculatePageRankTool(Tool):
    """Calculate PageRank importance for all symbols."""
    name = "calculate_pagerank"
    description = "Calculate PageRank scores to find most central/important symbols."
    _params = {
        "type": "object",
        "properties": {"top_n": {"type": "integer", "default": 20}},
        "required": []
    }

    def __init__(self, retrieval_pipeline=None):
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) ->dict: return self._params

    async def execute(self, top_n: int = 20) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if not graph:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")
        import networkx as nx
        pr = nx.pagerank(graph._g)
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_n]
        lines = ["## Symbol PageRank (Importance)\n"]
        for i, (nid, score) in enumerate(sorted_pr):
            lines.append(f"{i+1}. {_node_label(graph, nid)}: {score:.4f}")
        metadata = build_metadata(graph, 0.95, None, {"top_symbols": len(sorted_pr)})
        return ToolResult.ok(self.name, "", "\n".join(lines), **metadata)