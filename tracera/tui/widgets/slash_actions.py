"""
Unified slash-command surface.

Every capability that exists in TRACERA (registered tool or TUI handler) is
reachable as a slash command so nothing is hidden behind a CLI-only path:

    /tool <name> [key=value ...]   run ANY registered tool by its registry name
    /<alias> [arg]                 short alias for the common tools (see below)
    /features                      list every feature grouped by area

``SLASH_TOOLS`` is the single source of truth for aliases; ``command_registry``
merges it into autocomplete and ``app`` uses it for dispatch, so adding a tool
alias here makes it appear and work everywhere with no other change.
"""

from __future__ import annotations

import shlex
from typing import Any

#: slash alias -> tool registry name. Names that already have a dedicated TUI
#: command (search, deps, plan, …) are still listed; the explicit handler wins
#: because it is matched first in the app's command chain.
SLASH_TOOLS: dict[str, str] = {
    # ── retrieval / code intelligence ───────────────────────────────────────
    "search": "search_code",
    "symbol": "find_symbol",
    "symbols": "search_symbols",
    "source": "get_symbol_source",
    "definition": "find_definition",
    "context": "get_context",
    "deps": "get_dependencies",
    "outline": "get_file_outline",
    "repomap": "get_repo_map",
    "assemble": "assemble_code_context",
    "refs": "find_references",
    "callers": "get_call_hierarchy",
    "blast": "get_blast_radius",
    "changed": "get_changed_symbols",
    "freshness": "get_index_freshness",
    "importers": "find_importers",
    "classhier": "get_class_hierarchy",
    "cycles": "get_dependency_cycles",
    "coupling": "get_coupling_metrics",
    "endpoint": "get_endpoint_impact",
    "deadcode": "find_dead_code",
    "hotspots": "get_hotspots",
    "pagerank": "calculate_pagerank",
    "refactor": "plan_refactoring",
    "provenance": "get_code_provenance",
    "risk": "assess_change_risk",
    "ast": "structural_search",
    "sessionstats": "get_session_stats",
    "plantask": "plan_code_task",
    "impls": "find_implementations",
    "editsafe": "check_edit_safe",
    "deletesafe": "check_delete_safe",
    "prrisk": "get_pr_risk_profile",
    "auditconfig": "audit_agent_config",
    # ── memory ──────────────────────────────────────────────────────────────
    "recall": "recall_memory",
    "remember": "remember_memory",
    "forget": "forget_memory",
    "sessions": "list_sessions",
    "memstats": "memory_stats",
    "consolidate": "memory_consolidate",
    "memgraph2": "memory_graph",
    "memworker": "memory_worker_status",
    "memsearch": "search_memory",
    "triples": "get_memory_graph",
    # ── session / task context ──────────────────────────────────────────────
    "planturn": "plan_turn",
    "ranked": "get_ranked_context",
    "taskcontext": "assemble_task_context",
    # ── operations ──────────────────────────────────────────────────────────
    "tests": "run_tests",
    "inspectrepo": "inspect_repository",
    "git": "git",
    "run": "run_command",
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "ls": "list_dir",
    "grep": "grep",
}

#: Human-facing catalog rendered by /features. Grouped by capability area so a
#: new user can see the whole surface at a glance.
FEATURE_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Code retrieval": [
        ("/search <query>", "hybrid (BM25 + dense) code search"),
        ("/symbol <name>", "find a symbol by name"),
        ("/symbols <query>", "search symbols (exact/prefix/fuzzy)"),
        ("/source <symbol>", "exact source of a symbol (byte/line range)"),
        ("/definition <name>", "jump to a definition"),
        ("/outline <file>", "file outline — signatures only"),
        ("/repomap", "repository overview ranked by centrality"),
        ("/context <symbol>", "expanded context for a symbol"),
        ("/assemble <task>", "one-call task context capsule"),
    ],
    "Structural analysis": [
        ("/refs <symbol>", "find references"),
        ("/callers <symbol>", "call hierarchy (callers/callees)"),
        ("/blast <symbol>", "blast radius — what breaks if you change it"),
        ("/importers <file>", "what imports a file"),
        ("/classhier <class>", "inheritance chain"),
        ("/cycles", "circular import / dependency cycles"),
        ("/coupling", "module coupling + instability metrics"),
        ("/endpoint <route>", "what breaks if an endpoint changes"),
    ],
    "Quality & risk": [
        ("/deadcode", "symbols unreachable from entry points"),
        ("/hotspots", "risky code by complexity x churn"),
        ("/pagerank", "symbol importance (graph centrality)"),
        ("/risk <target>", "composite change-risk score"),
        ("/prrisk", "PR risk profile"),
        ("/ast <pattern>", "cross-language AST pattern search"),
        ("/auditconfig", "scan config files for token waste"),
    ],
    "Refactoring": [
        ("/refactor <symbol>", "edit-ready rename/move/extract plan"),
        ("/editsafe <symbol>", "preflight before modifying a symbol"),
        ("/deletesafe <symbol>", "preflight before deleting a symbol"),
        ("/impls <symbol>", "find implementations"),
    ],
    "Git & provenance": [
        ("/changed", "map git diff to affected symbols"),
        ("/provenance <symbol>", "git archaeology for a symbol"),
        ("/git <subcommand>", "run a git operation"),
    ],
    "Memory": [
        ("/memory", "show stored memories"),
        ("/recall <query>", "recall relevant memories"),
        ("/remember <text>", "store a memory"),
        ("/forget <text>", "forget a memory"),
        ("/memsearch <query>", "search the memory store"),
        ("/memstats", "memory statistics"),
        ("/consolidate", "merge near-duplicate memories"),
        ("/memgraph", "memory knowledge graph"),
        ("/memgraph2", "memory knowledge graph (tool form)"),
        ("/triples", "semantic triples in the knowledge graph"),
        ("/memworker", "background memory worker stats"),
        ("/sessions", "list sessions"),
    ],
    "Task & session": [
        ("/plan <task>", "decompose a task into steps"),
        ("/planturn <query>", "confidence-guided routing for a query"),
        ("/ranked <query>", "token-budgeted ranked context"),
        ("/taskcontext <task>", "full task context assembly"),
        ("/plantask", "plan a code task (intent + anchors + route)"),
        ("/sessionstats", "session economics + token savings"),
    ],
    "Multi-agent": [
        ("/delegate <task>", "decompose a task across sub-agents (Phases 42-44)"),
        ("/agents", "sub-agent fleet overview"),
    ],
    "Index & repo": [
        ("/index", "index the workspace (incremental)"),
        ("/freshness", "index freshness vs the filesystem"),
        ("/inspectrepo", "repository overview"),
        ("/status", "system status"),
    ],
    "Testing & review": [
        ("/test", "run the project test suite"),
        ("/review", "ask the agent to review current changes"),
        ("/tests <framework>", "run tests (pytest/unittest/npm/cargo)"),
        ("/fix <task>", "autonomous fix loop: plan → retrieve → edit → test (Phase 36)"),
        ("/selfreview", "independent LLM review of uncommitted changes (Phase 37)"),
        ("/regression", "baseline vs current test comparison (Phase 38)"),
    ],
    "Providers & UI": [
        ("/model <id>", "switch model"),
        ("/models", "list provider/model options"),
        ("/theme", "cycle accent theme"),
        ("/observability", "live telemetry"),
        ("/dashboard", "system overview panel"),
        ("/cost", "session token/cost estimate"),
        ("/files", "recently touched files"),
        ("/tools", "list available tools"),
        ("/mcp", "MCP status & config"),
        ("/phases", "phase map + verified checklist"),
    ],
    "Direct tool access": [
        ("/tool <name> [k=v ...]", "run any registry tool directly"),
        ("/read <path>", "read a file"),
        ("/write <path>", "write a file (content via editor)"),
        ("/edit <path>", "edit a file"),
        ("/ls <path>", "list a directory"),
        ("/grep <pattern>", "regex search file contents"),
        ("/run <command>", "run a shell command"),
    ],
}


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_tool_invocation(spec: str) -> tuple[str, dict[str, str]]:
    """
    Split ``"name key=value key2='a b'"`` into ``("name", {k: v})``.

    Raises ValueError on an empty spec or malformed tokens.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Usage: /tool <name> [key=value ...]")
    try:
        tokens = shlex.split(spec)
    except ValueError as e:
        raise ValueError(f"Could not parse arguments: {e}") from e
    if not tokens:
        raise ValueError("Usage: /tool <name> [key=value ...]")
    name = tokens[0]
    args: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            args.setdefault("query", token)
            continue
        key, value = token.split("=", 1)
        args[key.strip()] = value
    return name, args


def _coerce_one(value: str, json_type: str | None) -> Any:
    if json_type == "integer":
        return int(value)
    if json_type == "number":
        return float(value)
    if json_type == "boolean":
        return value.strip().lower() in ("1", "true", "yes", "on")
    if json_type == "array":
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def coerce_args(tool: Any, raw_args: dict[str, str]) -> dict[str, Any]:
    """Coerce string args to the types declared by the tool's JSON schema."""
    schema = getattr(tool, "parameters_schema", {}) or {}
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for key, value in raw_args.items():
        json_type = props.get(key, {}).get("type")
        try:
            out[key] = _coerce_one(value, json_type)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key}={value!r} is not a valid {json_type}") from e
    return out


#: Fallback positional parameter names, most-specific first.
_POSITIONAL_CANDIDATES = (
    "query", "symbol", "name", "path", "pattern", "command",
    "task", "file", "instruction", "text",
)


def single_arg_kwargs(tool: Any, value: str) -> dict[str, Any]:
    """
    Map a single positional slash argument to the right tool parameter.

    Prefers the schema's sole required property; otherwise the first matching
    well-known name; otherwise gives up (returns {} so the tool uses defaults).
    """
    value = (value or "").strip()
    if not value:
        return {}
    schema = getattr(tool, "parameters_schema", {}) or {}
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if len(required) == 1:
        param = required[0]
        return {param: _coerce_one(value, props.get(param, {}).get("type"))}
    for candidate in _POSITIONAL_CANDIDATES:
        if candidate in props:
            return {candidate: _coerce_one(value, props[candidate].get("type"))}
    return {}


__all__ = [
    "SLASH_TOOLS",
    "FEATURE_GROUPS",
    "parse_tool_invocation",
    "coerce_args",
    "single_arg_kwargs",
]
