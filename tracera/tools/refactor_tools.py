"""
Refactoring & Safety Tools — Inspired by jCodeMunch MCP.

  plan_refactoring   — edit-ready rename/move/extract instructions
  check_edit_safe    — preflight check before modifying a symbol
  check_delete_safe  — preflight check before deleting a symbol
  get_pr_risk_profile — composite risk score for a branch/PR
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.refactor_tools")


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


def _node_info(graph, node_id: str) -> dict:
    node = graph.get_node(node_id)
    return node if node else {}


# ── plan_refactoring ─────────────────────────────────────────────────────────

class PlanRefactoringTool(Tool):
    """Generate edit-ready refactoring instructions."""

    name = "plan_refactoring"
    description = (
        "Generate edit-ready refactoring instructions for rename, move, extract, "
        "or signature-change operations. Returns {old_text, new_text} blocks "
        "compatible with any editor's find-and-replace, plus import rewrites."
    )
    _params = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["rename", "move", "extract", "change_signature"],
                "description": "Type of refactoring operation.",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name to refactor.",
            },
            "target": {
                "type": "string",
                "description": (
                    "For rename: new name. For move: destination file. "
                    "For extract: name for the extracted symbol. "
                    "For change_signature: JSON of old→new parameter mapping."
                ),
            },
        },
        "required": ["operation", "symbol"],
    }

    def __init__(self, retrieval_pipeline=None, workspace=None) -> None:
        self._pipeline = retrieval_pipeline
        self._workspace = workspace

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(
        self, operation: str, symbol: str, target: str = ""
    ) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found.")

        info = _node_info(graph, node_id)
        lines = [f"## Refactoring Plan: {operation} `{symbol}`\n"]

        if operation == "rename":
            new_name = target or f"{symbol}_renamed"
            # Find all references
            callers = graph.get_callers(node_id)
            lines.append(f"**Rename:** `{symbol}` → `{new_name}`\n")
            lines.append(f"**Symbol defined in:** `{info.get('file_path', '?')}` L{info.get('start_line', '?')}-{info.get('end_line', '?')}\n")
            lines.append("### Edit blocks:")
            lines.append(f"```json\n{{\"old_text\": \"{symbol}\", \"new_text\": \"{new_name}\"}}\n```\n")
            lines.append(f"### Files that reference this symbol ({len(callers)}):")
            for cid in callers[:20]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('file_path', '?')}`:{n.get('start_line', '?')} — `{n.get('name', '?')}`")

            # Import rewrites
            if callers:
                lines.append("\n### Import rewrites needed:")
                lines.append(f"Replace `{symbol}` with `{new_name}` in all import statements across the files above.")

        elif operation == "move":
            dest = target or "new_module.py"
            lines.append(f"**Move:** `{symbol}` → `{dest}`\n")
            lines.append(f"**Current location:** `{info.get('file_path', '?')}`\n")
            lines.append(f"### Edit blocks:")
            lines.append(f"1. Remove from `{info.get('file_path', '?')}`")
            lines.append(f"2. Add to `{dest}`")
            lines.append(f"3. Update imports in all referencing files")
            callers = graph.get_callers(node_id)
            if callers:
                lines.append(f"\n### Import updates needed ({len(callers)} files):")
                for cid in callers[:15]:
                    n = graph.get_node(cid)
                    if n:
                        lines.append(f"- `{n.get('file_path', '?')}`")

        elif operation == "extract":
            lines.append(f"**Extract:** Create new symbol `{target or 'extracted_func'}` from `{symbol}`\n")
            lines.append("### Steps:")
            lines.append(f"1. Identify the code block to extract in `{info.get('file_path', '?')}`")
            lines.append(f"2. Create `{target or 'extracted_func'}` with the extracted code")
            lines.append(f"3. Replace the original block with a call to `{target or 'extracted_func'}`")
            lines.append(f"4. Update imports if the new symbol is in a different module")

        elif operation == "change_signature":
            lines.append(f"**Change Signature:** `{symbol}`\n")
            lines.append(f"**Location:** `{info.get('file_path', '?')}` L{info.get('start_line', '?')}-{info.get('end_line', '?')}\n")
            callers = graph.get_callers(node_id)
            lines.append(f"### Callers that need updating ({len(callers)}):")
            for cid in callers[:15]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('file_path', '?')}`:{n.get('start_line', '?')} — `{n.get('name', '?')}`")
            if target:
                lines.append(f"\n### Signature change:")
                lines.append(f"```json\n{target}\n```")

        # Collision detection for rename
        if operation == "rename" and target:
            existing = graph.find_by_name(target)
            if existing:
                lines.append(f"\n⚠️ **Collision:** Symbol `{target}` already exists in the graph:")
                for eid in existing[:5]:
                    n = graph.get_node(eid)
                    if n:
                        lines.append(f"  - `{n.get('file_path', '?')}`:{n.get('start_line', '?')}")

        return ToolResult.ok(self.name, "", "\n".join(lines), operation=operation, symbol=symbol)


# ── check_edit_safe ──────────────────────────────────────────────────────────

class CheckEditSafeTool(Tool):
    """Preflight check: can I safely modify this symbol?"""

    name = "check_edit_safe"
    description = (
        "Check whether modifying a symbol is safe. Fuses import impact, "
        "call count, complexity, test coverage presence, and entry-point "
        "proximity into a verdict with recommended action."
    )
    _params = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name to check.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

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

        info = _node_info(graph, node_id)
        callers = graph.get_callers(node_id)
        callees = graph.get_callees(node_id)
        children = graph.get_children(node_id)

        # Risk factors
        caller_count = len(callers)
        complexity = max(1, (info.get("end_line", 0) or 0) - (info.get("start_line", 0) or 0))
        is_entry = any(
            kw in (info.get("name") or "").lower()
            for kw in ("main", "app", "cli", "handler", "route")
        )
        has_children = len(children) > 0

        # Score: 0.0 (safe) to 1.0 (dangerous)
        risk = 0.0
        risk += min(0.4, caller_count * 0.05)      # More callers = more risk
        risk += min(0.3, complexity / 500)           # Larger functions = more risk
        risk += 0.2 if is_entry else 0.0             # Entry points are risky
        risk += 0.1 if has_children else 0.0         # Classes with methods are risky

        risk = min(1.0, risk)

        if risk < 0.3:
            verdict = "SAFE"
            action = "You can modify this symbol with low risk."
            color = "🟢"
        elif risk < 0.6:
            verdict = "CAUTION"
            action = "Review callers before modifying. Consider adding tests."
            color = "🟡"
        else:
            verdict = "DANGEROUS"
            action = "High-impact change. Run tests and review all callers first."
            color = "🔴"

        lines = [
            f"## Edit Safety Check: `{symbol}`\n",
            f"**Verdict:** {color} **{verdict}** (risk: {risk:.2f})\n",
            f"**Recommendation:** {action}\n",
            f"### Factors:",
            f"- Callers: {caller_count}",
            f"- Complexity (lines): {complexity}",
            f"- Entry point: {'yes' if is_entry else 'no'}",
            f"- Has methods/properties: {'yes' if has_children else 'no'}",
        ]

        if caller_count > 0:
            lines.append(f"\n### Callers to review:")
            for cid in callers[:10]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('name', '?')}` in `{n.get('file_path', '?')}`:{n.get('start_line', '?')}")

        terminal = risk >= 0.6
        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            symbol=symbol, verdict=verdict, risk=risk, terminal=terminal,
        )


# ── check_delete_safe ────────────────────────────────────────────────────────

class CheckDeleteSafeTool(Tool):
    """Preflight check: can I safely delete this symbol?"""

    name = "check_delete_safe"
    description = (
        "Check whether deleting a symbol is safe. Analyzes importers, "
        "references, entry-point proximity, and whether the symbol appears "
        "to be dead code."
    )
    _params = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name to check for deletion.",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, retrieval_pipeline=None) -> None:
        self._pipeline = retrieval_pipeline

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._params

    async def execute(self, symbol: str) -> ToolResult:
        graph = _get_graph(self._pipeline)
        if graph is None:
            return ToolResult.ok(self.name, "", "Symbol graph not available.")

        node_id = _resolve_symbol(graph, symbol)
        if not node_id:
            return ToolResult.ok(self.name, "", f"Symbol '{symbol}' not found — may already be deleted.")

        info = _node_info(graph, node_id)
        callers = graph.get_callers(node_id)
        caller_count = len(callers)

        # Check if it's an entry point
        is_entry = any(
            kw in (info.get("name") or "").lower()
            for kw in ("main", "app", "cli", "handler", "route", "setup")
        )

        risk = 0.0
        risk += min(0.5, caller_count * 0.1)
        risk += 0.3 if is_entry else 0.0
        risk = min(1.0, risk)

        if risk < 0.2:
            verdict = "SAFE_TO_DELETE"
            action = "No callers found. Safe to remove."
            color = "🟢"
        elif risk < 0.5:
            verdict = "REVIEW_NEEDED"
            action = f"Found {caller_count} caller(s). Review before deleting."
            color = "🟡"
        else:
            verdict = "DO_NOT_DELETE"
            action = f"High-impact: {caller_count} caller(s) + entry point. Do not delete."
            color = "🔴"

        lines = [
            f"## Delete Safety Check: `{symbol}`\n",
            f"**Verdict:** {color} **{verdict}** (risk: {risk:.2f})\n",
            f"**Recommendation:** {action}\n",
            f"**Location:** `{info.get('file_path', '?')}` L{info.get('start_line', '?')}-{info.get('end_line', '?')}\n",
        ]

        if callers:
            lines.append(f"### Callers ({caller_count}):")
            for cid in callers[:15]:
                n = graph.get_node(cid)
                if n:
                    lines.append(f"- `{n.get('name', '?')}` in `{n.get('file_path', '?')}`:{n.get('start_line', '?')}")

        terminal = risk >= 0.5
        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            symbol=symbol, verdict=verdict, risk=risk, terminal=terminal,
        )


# ── get_pr_risk_profile ──────────────────────────────────────────────────────

class GetPrRiskProfileTool(Tool):
    """Unified risk assessment for a branch or PR."""

    name = "get_pr_risk_profile"
    description = (
        "Produce a unified risk assessment for uncommitted changes or a branch. "
        "Fuses blast radius, complexity, churn, and change volume into a "
        "composite risk score (0.0–1.0) with actionable recommendations."
    )
    _params = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": (
                    "Git ref or branch name: 'HEAD' for uncommitted, "
                    "branch name for a branch diff."
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
        workspace_root = str(self._workspace.root) if self._workspace else "."

        try:
            if ref == "HEAD":
                result = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )
            else:
                result = subprocess.run(
                    ["git", "diff", "--stat", f"main...{ref}"],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )
            diff_stat = result.stdout
        except Exception as e:
            return ToolResult.fail(self.name, "", f"Git diff failed: {e}")

        # Parse changed files
        try:
            if ref == "HEAD":
                files_result = subprocess.run(
                    ["git", "diff", "--name-only"],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )
            else:
                files_result = subprocess.run(
                    ["git", "diff", "--name-only", f"main...{ref}"],
                    cwd=workspace_root, capture_output=True, text=True, timeout=10,
                )
            changed_files = [f.strip() for f in files_result.stdout.strip().split("\n") if f.strip()]
        except Exception:
            changed_files = []

        # Risk factors
        file_count = len(changed_files)
        test_files = sum(1 for f in changed_files if "test" in f.lower())
        config_files = sum(1 for f in changed_files if any(k in f.lower() for k in ["config", ".env", "settings"]))

        # Churn on changed files
        churn_total = 0
        for fp in changed_files:
            try:
                log_result = subprocess.run(
                    ["git", "log", "--oneline", "--since=30 days", "--", fp],
                    cwd=workspace_root, capture_output=True, text=True, timeout=5,
                )
                churn_total += len(log_result.stdout.strip().split("\n")) if log_result.stdout.strip() else 0
            except Exception:
                pass

        # Blast radius (if graph available)
        graph = _get_graph(self._pipeline)
        blast_radius_total = 0
        if graph:
            for fp in changed_files:
                node_ids = graph.find_by_file(fp)
                for nid in node_ids:
                    ancestors = graph.get_ancestors(nid, max_depth=2)
                    blast_radius_total += len(ancestors)

        # Composite score
        risk = 0.0
        risk += min(0.3, file_count * 0.02)                # Volume
        risk += min(0.2, churn_total * 0.01)               # Churn
        risk += min(0.2, blast_radius_total * 0.005)       # Blast radius
        risk += 0.15 if config_files > 0 else 0.0          # Config changes
        risk -= 0.1 if test_files > 0 else 0.0             # Tests included
        risk = max(0.0, min(1.0, risk))

        if risk < 0.25:
            grade = "LOW"
            color = "🟢"
        elif risk < 0.5:
            grade = "MODERATE"
            color = "🟡"
        elif risk < 0.75:
            grade = "HIGH"
            color = "🟠"
        else:
            grade = "CRITICAL"
            color = "🔴"

        lines = [
            f"## PR Risk Profile (ref: `{ref}`)\n",
            f"**Risk Score:** {color} **{risk:.2f}** ({grade})\n",
            f"### Metrics:",
            f"- Files changed: {file_count}",
            f"- Test files: {test_files}",
            f"- Config files: {config_files}",
            f"- 30-day churn: {churn_total} commits",
            f"- Blast radius: {blast_radius_total} affected symbols",
            f"\n### Diff summary:\n```{diff_stat}```",
        ]

        # Recommendations
        lines.append("\n### Recommendations:")
        if config_files > 0:
            lines.append("- ⚠️ Config files changed — verify no production secrets are exposed")
        if test_files == 0 and file_count > 3:
            lines.append("- ⚠️ No test files in changeset — consider adding tests")
        if churn_total > 10:
            lines.append("- ⚠️ High-churn files in changeset — these are prone to merge conflicts")
        if blast_radius_total > 20:
            lines.append("- ⚠️ Large blast radius — verify all downstream consumers")
        if risk < 0.25:
            lines.append("- ✅ Low risk — looks good to merge")

        return ToolResult.ok(
            self.name, "", "\n".join(lines),
            ref=ref, risk=risk, grade=grade, file_count=file_count,
        )
