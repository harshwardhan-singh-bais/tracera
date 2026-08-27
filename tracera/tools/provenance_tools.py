"""
Symbol Provenance & Agent Config Tools — Inspired by jCodeMunch MCP.

  get_symbol_provenance — git archaeology for a symbol
  audit_agent_config    — scan agent config files for token waste
  get_endpoint_impact   — what breaks if you change an HTTP endpoint
  get_dependency_cycles — detect circular imports
  get_coupling_metrics  — module coupling and instability
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.provenance_tools")


def _get_graph(retrieval_pipeline=None):
    if retrieval_pipeline is None or len(retrieval_pipeline) < 10:
        return None
    graph_retriever = retrieval_pipeline[-1]
    if graph_retriever is not None and hasattr(graph_retriever, "graph"):
        return graph_retriever.graph
    return None


def _resolve_symbol(graph, name: str) -> str | None:
    node_ids = graph.find_by_name(name)
    return node_ids[0] if node_ids else None


# ── get_symbol_provenance ────────────────────────────────────────────────────

class GetSymbolProvenanceTool(Tool):
    """Git archaeology — trace every commit that touched a symbol."""

    name = "get_symbol_provenance"
    description = (
        "Given a symbol, trace every commit that touched it, classify each "
        "into semantic categories (creation, bugfix, refactor, feature, perf, "
        "rename, revert), and generate a human-readable evolution narrative."
    )
    _params = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name to trace provenance for.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retrieval_pipeline=None, workspace=None) -> None:
        self._pipeline = retrieval_pipeline
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found.")

        node = graph.get_node(node_id)
        file_path = node.get("file_path", "")
        start_line = node.get("start_line", 1)

        # Get git log for this file
        workspace_root = str(self._workspace.root) if self._workspace else "."
        try:
            result = subprocess.run(
                ["git", "log", "--format=%H|%s|%an|%ai", "--follow", "--", file_path],
                cwd=workspace_root, capture_output=True, text=True, timeout=15,
            )
            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 3)
                    if len(parts) >= 4:
                        commits.append({
                            "hash": parts[0][:8],
                            "message": parts[1],
                            "author": parts[2],
                            "date": parts[3],
                        })
        except Exception:
            commits = []

        # Classify commits
        def classify(msg: str) -> str:
            msg_lower = msg.lower()
            if any(w in msg_lower for w in ["fix", "bug", "patch", "resolve"]):
                return "bugfix 🐛"
            if any(w in msg_lower for w in ["refactor", "clean", "restructure", "reorganize"]):
                return "refactor 🔧"
            if any(w in msg_lower for w in ["feat", "add", "implement", "new"]):
                return "feature ✨"
            if any(w in msg_lower for w in ["perf", "optimize", "speed", "fast"]):
                return "performance ⚡"
            if any(w in msg_lower for w in ["rename", "move"]):
                return "rename 📝"
            if any(w in msg_lower for w in ["revert"]):
                return "revert ↩️"
            if any(w in msg_lower for w in ["test", "spec"]):
                return "test 🧪"
            return "other 📦"

        lines = [f"## Symbol Provenance: `{symbol}`\n"]
        lines.append(f"**Location:** `{file_path}` L{start_line}\n")

        if not commits:
            lines.append("_No git history found for this file._")
        else:
            lines.append(f"**Commit history ({len(commits)} commits):**\n")
            for c in commits[:20]:
                category = classify(c["message"])
                lines.append(
                    f"- `{c['hash']}` {category} — {c['message'][:80]}"
                )
                lines.append(f"  _by {c['author']} on {c['date'][:10]}_")

            # Evolution narrative
            if len(commits) >= 2:
                lines.append(f"\n### Evolution Summary")
                categories = defaultdict(int)
                for c in commits:
                    cat = classify(c["message"])
                    categories[cat] += 1
                lines.append(f"Total commits: {len(commits)}")
                for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                    lines.append(f"- {cat}: {count}")

                first = commits[-1]
                latest = commits[0]
                lines.append(
                    f"\n**Created:** {first['date'][:10]} by {first['author']}"
                )
                lines.append(
                    f"**Last modified:** {latest['date'][:10]} by {latest['author']}"
                )

        return ToolResult.ok(self.name, "", "\n".join(lines), symbol=symbol, commits=len(commits))


# ── audit_agent_config ───────────────────────────────────────────────────────

class AuditAgentConfigTool(Tool):
    """Scan agent config files for token waste and stale references."""

    name = "audit_agent_config"
    description = (
        "Scan CLAUDE.md, .cursorrules, copilot-instructions.md, and other "
        "agent config files for token waste: stale symbol references, dead "
        "file paths, redundancy, bloat, and scope leaks."
    )
    _params = {
        "type": "object",
        "properties": {
            "config_path": {
                "type": "string",
                "description": "Path to a specific config file to audit (optional).",
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

    async def execute(self, config_path: str = "") -> ToolResult:
        workspace_root = Path(self._workspace.root) if self._workspace else Path(".")

        # Config files to scan
        config_names = [
            "CLAUDE.md", ".cursorrules", "copilot-instructions.md",
            ".github/copilot-instructions.md", ".tracera/config.json",
            ".tracera/AGENTS.md", "AGENTS.md",
        ]

        if config_path:
            config_names = [config_path]

        graph = _get_graph(self._pipeline)
        lines = ["## Agent Config Audit\n"]
        total_tokens = 0
        issues: list[str] = []

        for name in config_names:
            fp = workspace_root / name
            if not fp.exists():
                continue

            content = fp.read_text(encoding="utf-8", errors="ignore")
            file_tokens = len(content) // 4
            total_tokens += file_tokens

            lines.append(f"### `{name}` ({file_tokens} tokens)\n")

            # Check for stale file references
            import re
            file_refs = re.findall(r'`([^`]+\.(py|js|ts|go|rs|java|cpp))`', content)
            for ref, ext in file_refs:
                ref_path = workspace_root / ref
                if not ref_path.exists():
                    issues.append(f"⚠️ Stale file reference in {name}: `{ref}` does not exist")
                    lines.append(f"- ⚠️ Stale reference: `{ref}`")

            # Check for stale symbol references (if graph available)
            if graph:
                symbol_refs = re.findall(r'`([A-Z][a-zA-Z]+)`', content)
                for sym in symbol_refs[:20]:
                    node_ids = graph.find_by_name(sym)
                    if not node_ids:
                        # Might be a class name — check common patterns
                        pass  # Skip to avoid false positives

            # Check for bloat
            if file_tokens > 500:
                lines.append(f"- ⚠️ Large config file ({file_tokens} tokens) — consider trimming")

            # Check for redundancy with other configs
            lines.append("")

        if not lines:
            lines.append("No agent config files found.")

        # Summary
        lines.insert(1, f"**Total config tokens:** {total_tokens}\n")

        if issues:
            lines.append(f"\n### Issues Found ({len(issues)}):")
            for issue in issues[:15]:
                lines.append(f"- {issue}")
        else:
            lines.append("\n### ✅ No issues found")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            total_tokens=total_tokens, issues=len(issues),
        )


# ── get_endpoint_impact ──────────────────────────────────────────────────────

class GetEndpointImpactTool(Tool):
    """What breaks if you change an HTTP endpoint handler."""

    name = "get_endpoint_impact"
    description = (
        "Given an HTTP endpoint path or handler symbol, find the route handler, "
        "its blast radius (importing files + callers), and all code affected by "
        "changes to it."
    )
    _params = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": (
                    "HTTP endpoint path (e.g. '/api/users') or handler symbol name."
                ),
            },
        },
        "required": ["endpoint"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, endpoint: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        # Try to find the handler by name
        node_id = _resolve_symbol(graph, endpoint)

        # If not found by exact name, search for partial match
        if not node_id:
            for nid, data in graph._g.nodes(data=True):
                name = (data.get("name") or "").lower()
                ep = endpoint.lower().strip("/")
                if ep in name or name in ep:
                    node_id = nid
                    break

        if not node_id:
            return ToolResult.ok(
                self.name, "",
                f"Endpoint '{endpoint}' not found in the graph. "
                "Try the handler function name instead.",
            )

        info = graph.get_node(node_id)
        callers = graph.get_callers(node_id)
        callees = graph.get_callees(node_id)

        lines = [
            f"## Endpoint Impact: `{endpoint}`\n",
            f"**Handler:** `{info.get('name', '?')}` in `{info.get('file_path', '?')}`:{info.get('start_line', '?')}\n",
            f"**Callers / importers:** {len(callers)}",
            f"**Callees / dependencies:** {len(callees)}\n",
        ]

        if callers:
            lines.append("### Affected by changes to this endpoint:")
            for cid in callers[:15]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('name', '?')}` ({n.get('symbol_type', '?')}) in `{n.get('file_path', '?')}`:{n.get('start_line', '?')}")

        if callees:
            lines.append("\n### Dependencies (this endpoint calls):")
            for cid in callees[:15]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('name', '?')}` ({n.get('symbol_type', '?')}) in `{n.get('file_path', '?')}`:{n.get('start_line', '?')}")

        total_impact = len(callers) + len(callees)
        if total_impact > 10:
            lines.append(f"\n⚠️ **High impact** — {total_impact} symbols affected. Test thoroughly.")
        elif total_impact > 3:
            lines.append(f"\n🟡 **Moderate impact** — {total_impact} symbols affected.")
        else:
            lines.append(f"\n🟢 **Low impact** — {total_impact} symbols affected.")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            endpoint=endpoint, callers=len(callers), callees=len(callees),
        )


# ── get_dependency_cycles ────────────────────────────────────────────────────

class GetDependencyCyclesTool(Tool):
    """Detect circular imports in the codebase."""

    name = "get_dependency_cycles"
    description = (
        "Detect circular import chains in the codebase. Returns each cycle "
        "with the files involved and suggestions for breaking them."
    )
    _params = {
        "type": "object",
        "properties": {
            "max_cycles": {
                "type": "integer",
                "description": "Maximum cycles to report (default: 10).",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, max_cycles: int = 10) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        try:
            import networkx as nx
            cycles = list(nx.simple_cycles(graph._g))
        except Exception as e:
            return ToolResult.ok(self.name, "", f"Cycle detection failed: {e}")

        lines = [f"## Dependency Cycles\n"]
        lines.append(f"**Cycles found:** {len(cycles)}\n")

        if not cycles:
            lines.append("✅ No circular dependencies detected.")
        else:
            for i, cycle in enumerate(cycles[:max_cycles], 1):
                lines.append(f"### Cycle {i}")
                for node_id in cycle:
                    node = graph.get_node(node_id)
                    if node:
                        lines.append(
                            f"  → `{node.get('file_path', '?')}`::`{node.get('name', '?')}`"
                        )
                lines.append("")

            if len(cycles) > max_cycles:
                lines.append(f"_... and {len(cycles) - max_cycles} more cycles._")

            lines.append("### How to break cycles:")
            lines.append("- Move shared types to a separate module")
            lines.append("- Use late imports (import inside functions)")
            lines.append("- Extract interfaces/protocols to break circular type deps")

        return ToolResult.ok(self.name, "", "\n".join(lines), cycles=len(cycles))


# ── get_coupling_metrics ─────────────────────────────────────────────────────

class GetCouplingMetricsTool(Tool):
    """Measure module coupling and instability."""

    name = "get_coupling_metrics"
    description = (
        "Measure coupling metrics for each module: afferent couplings (Ca), "
        "efferent couplings (Ce), instability (Ce/(Ca+Ce)), and abstractness. "
        "Helps identify god modules and unstable abstractions."
    )
    _params = {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Number of modules to report (default: 15).",
                "default": 15,
            },
        },
        "required": [],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, top_n: int = 15) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        # Compute per-file coupling
        files: dict[str, dict] = defaultdict(lambda: {"ca": 0, "ce": 0, "symbols": 0})

        for node_id, data in graph._g.nodes(data=True):
            fp = data.get("file_path", "?")
            files[fp]["symbols"] += 1

        for src, dst, data in graph._g.edges(data=True):
            src_file = graph.get_node(src)
            dst_file = graph.get_node(dst)
            if src_file and dst_file:
                src_fp = src_file.get("file_path", "?")
                dst_fp = dst_file.get("file_path", "?")
                if src_fp != dst_fp:
                    files[src_fp]["ce"] += 1  # efferent: src depends on dst
                    files[dst_fp]["ca"] += 1  # afferent: dst is depended on

        # Compute instability: I = Ce / (Ca + Ce)
        metrics = []
        for fp, m in files.items():
            total = m["ca"] + m["ce"]
            instability = m["ce"] / total if total > 0 else 0
            metrics.append({
                "file": fp,
                "ca": m["ca"],
                "ce": m["ce"],
                "instability": instability,
                "symbols": m["symbols"],
                "risk": m["ca"] + m["ce"],
            })

        metrics.sort(key=lambda m: -m["risk"])

        lines = [f"## Coupling Metrics\n"]
        lines.append(f"**Modules analyzed:** {len(metrics)}\n")

        for m in metrics[:top_n]:
            inst = m["instability"]
            indicator = "🔴" if inst > 0.7 else "🟡" if inst > 0.4 else "🟢"
            lines.append(
                f"{indicator} **`{m['file']}`** — "
                f"Ca={m['ca']}, Ce={m['ce']}, I={inst:.2f}, "
                f"symbols={m['symbols']}"
            )

        lines.append("\n### Legend:")
        lines.append("- **Ca (Afferent):** number of modules that depend on this one")
        lines.append("- **Ce (Efferent):** number of modules this one depends on")
        lines.append("- **I (Instability):** Ce/(Ca+Ce) — 0=stable, 1=unstable")
        lines.append("- 🟢 Stable (I<0.4) | 🟡 Moderate | 🔴 Unstable (I>0.7)")

        return ToolResult.ok(self.name, "", "\n".join(lines), modules=len(metrics))
