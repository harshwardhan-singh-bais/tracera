"""
Execution harness — runs EVERY code-intelligence tool against a real
SymbolGraph, exactly the way the app wires them.

Why this exists: the slash-command guard tests proved that every alias maps
to a registered tool, but several tools still crashed at execution time
because they received the wrong object type (a pipeline tuple where a
SymbolGraph was expected, a retriever without the method they call, …).
Wiring tests check that a name resolves; only EXECUTION checks that the
feature actually works.

The pipeline tuple shape used here matches `_build_retrieval_pipeline` in
main.py: [indexer, retriever, expander, ?, context_engine, compressor,
?, ?, ?, graph_retriever].
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from tracera.graph.symbol_graph import RelationType, SymbolGraph
from tracera.indexer.schema import LineRange, Symbol, SymbolType
from tracera.tools.base import ToolResult
from tracera.tools.registry import (
    ToolRegistry,
    create_default_registry,
    extend_registry_with_ast_tools,
    extend_registry_with_retrieval,
)


def _sym(
    name: str,
    stype: SymbolType,
    start: int,
    end: int,
    parent: str | None = None,
) -> Symbol:
    return Symbol(
        name=name,
        type=stype,
        range=LineRange(start_line=start, end_line=end),
        content=f"# {name}",
        parent_symbol=parent,
    )


def _rich_graph() -> SymbolGraph:
    """A small but complete graph: classes, methods, calls, imports, inheritance."""
    g = SymbolGraph()
    g.add_symbol("tracera/tools/base.py", _sym("Tool", SymbolType.CLASS, 1, 40))
    g.add_symbol("tracera/tools/base.py", _sym("ToolResult", SymbolType.CLASS, 41, 80))
    g.add_symbol("tracera/tools/registry.py", _sym("ToolRegistry", SymbolType.CLASS, 1, 60, parent=None))
    g.add_symbol("tracera/tools/registry.py", _sym("execute", SymbolType.METHOD, 20, 40, parent="ToolRegistry"))
    g.add_symbol("tracera/agent/react_loop.py", _sym("ReActAgent", SymbolType.CLASS, 1, 90))
    g.add_symbol("tracera/agent/react_loop.py", _sym("run", SymbolType.METHOD, 30, 80, parent="ReActAgent"))

    # calls: run -> execute; ReActAgent -> ToolRegistry
    agent_id = "tracera/agent/react_loop.py::ReActAgent"
    registry_id = "tracera/tools/registry.py::ToolRegistry"
    g.add_relation(agent_id, registry_id, RelationType.CALLS)
    g.add_relation(
        "tracera/agent/react_loop.py::run",
        "tracera/tools/registry.py::execute",
        RelationType.CALLS,
    )
    # inheritance: ReActAgent inherits Tool? no — use a dedicated pair
    g.add_symbol("tests/fake.py", _sym("Base", SymbolType.CLASS, 1, 10))
    g.add_symbol("tests/fake.py", _sym("Child", SymbolType.CLASS, 11, 20))
    g.add_relation("tests/fake.py::Child", "tests/fake.py::Base", RelationType.INHERITS)
    return g


def _fake_indexer() -> SimpleNamespace:
    """Minimal indexer stand-in exposing check_freshness (used by many tools)."""
    return SimpleNamespace(
        check_freshness=lambda: {},
        _load_manifest=lambda: {"snapshot": {}},
    )


def _build_registry(tmp_path: Path) -> ToolRegistry:
    graph = _rich_graph()
    graph_retriever = SimpleNamespace(graph=graph)

    class FakeRetriever:
        def search(self, query, k=5, language=None):
            return []

        def find(self, name, symbol_type="any"):
            return []

    class FakeExpander:
        def expand(self, hits, **kwargs):
            return hits

    pipeline = (
        _fake_indexer(),   # 0  indexer
        FakeRetriever(),   # 1  symbol retriever
        FakeExpander(),    # 2  expander
        None,              # 3  reranker
        None,              # 4  context engine
        None,              # 5  compressor
        None,              # 6  embedder
        None,              # 7  vector store
        None,              # 8  bm25
        graph_retriever,   # 9  graph retriever
    )

    from tracera.workspace.sandbox import WorkspaceSandbox
    workspace = WorkspaceSandbox(tmp_path)
    registry = create_default_registry(workspace)
    extend_registry_with_retrieval(
        registry,
        pipeline[1],
        pipeline[2],
        graph_retriever,
        retrieval_pipeline=pipeline,
        workspace=workspace,
        tool_profile="advanced",
    )
    extend_registry_with_ast_tools(registry, retrieval_pipeline=pipeline, workspace=workspace)
    return registry


# ── Argument map: one entry per registered tool ──────────────────────────────
# Names/arg-keys mirror the JSON schemas the LLM actually sees, so a mismatch
# between schema and execute() signature fails here.

ARGS: dict[str, dict] = {
    "read_file": {"path": "_harness_tmp.txt"},
    "write_file": {"path": "_harness_tmp.txt", "content": "harness"},
    "edit_file": {"path": "_harness_tmp.txt", "old_text": "harness", "new_text": "harness2"},
    "list_dir": {"path": "."},
    "grep": {"pattern": "def", "path": "."},
    "run_command": {"command": "python --version"},
    "git": {"operation": "status"},
    "search_code": {"query": "registry"},
    "find_symbol": {"name": "ToolRegistry"},
    "find_definition": {"name": "ToolRegistry"},
    "get_context": {"symbol": "ToolRegistry"},
    "get_dependencies": {"symbol": "ToolRegistry"},
    "get_file_outline": {"file_path": "tracera/tools/base.py"},
    "get_repo_map": {},
    "assemble_code_context": {"task": "fix the registry"},
    "find_references": {"symbol": "ToolRegistry"},
    "get_call_hierarchy": {"symbol": "ToolRegistry"},
    "get_class_hierarchy": {"class_name": "Child"},
    "get_blast_radius": {"symbol": "ToolRegistry"},
    "get_changed_symbols": {},
    "get_index_freshness": {},
    "find_dead_code": {},
    "get_hotspots": {},
    "calculate_pagerank": {},
    "plan_refactoring": {"refactor_type": "rename", "symbol": "ToolRegistry", "new_name": "ToolRegistry2"},
    "get_code_provenance": {"symbol": "ToolRegistry"},
    "assess_change_risk": {"symbol": "ToolRegistry"},
    "structural_search": {"pattern": "class $NAME"},
    "get_session_stats": {},
    "plan_code_task": {"task": "add caching"},
    "find_implementations": {"symbol": "Tool"},
    "find_importers": {"path": "tracera/tools/base.py"},
    "check_edit_safe": {"symbol": "ToolRegistry"},
    "check_delete_safe": {"symbol": "ToolRegistry"},
    "get_pr_risk_profile": {},
    "get_symbol_provenance": {"symbol": "ToolRegistry"},
    "get_dependency_cycles": {},
    "get_coupling_metrics": {},
    "get_endpoint_impact": {"endpoint": "/api/x"},
    "audit_agent_config": {},
}


def test_every_tool_executes_successfully(tmp_path: Path) -> None:
    """THE harness: every registered tool must execute without crashing.

    A tool that raises, or returns success=False for valid input, means a
    broken feature — wiring tests alone cannot catch this.
    """
    registry = _build_registry(tmp_path)
    assert len(registry) > 20, "registry should contain the code-intelligence tools"

    # Pre-create files the write/edit tools touch.
    (tmp_path / "_harness_tmp.txt").write_text("harness", encoding="utf-8")

    failures: list[str] = []
    for name in sorted(registry.names):
        if name not in ARGS:
            continue  # run_tests/inspect_repository/memory tools covered elsewhere
        result: ToolResult = asyncio.run(registry.execute(name, "harness", dict(ARGS[name])))
        if not result.success:
            failures.append(f"{name}: {result.error}")
    # git is exercised in the repo suite (tests/test_git_tool.py) against a real
    # repository; tmp_path is not one, so skip it here.
    failures = [f for f in failures if not f.startswith("git:")]

    assert not failures, (
        "Tools that failed execution against a live-shaped pipeline:\n  "
        + "\n  ".join(failures)
    )


def test_tool_schema_required_args_match_harness(tmp_path: Path) -> None:
    """Every ARGS entry's keys must be declared in the tool's JSON schema.

    Catches drift between the harness and the schemas — and, indirectly,
    schemas that advertise parameters execute() does not accept.
    """
    registry = _build_registry(tmp_path)
    problems: list[str] = []
    for name, args in ARGS.items():
        if not registry.has(name):
            problems.append(f"{name}: not registered")
            continue
        props = set(registry.get(name).parameters_schema.get("properties", {}))
        unknown = set(args) - props
        if unknown:
            problems.append(f"{name}: harness args {sorted(unknown)} not in schema")
    assert not problems, "\n".join(problems)


def test_schema_required_params_have_harness_coverage(tmp_path: Path) -> None:
    """Every tool schema's required params are satisfiable by single-arg dispatch.

    Mirrors the TUI contract: single_arg_kwargs maps one positional arg to the
    sole required property. If a required property were misspelled relative to
    execute(), coercion would pass it and execution would crash — exactly the
    find_symbol/find_definition bug class fixed alongside this harness.
    """
    registry = _build_registry(tmp_path)
    for name in registry.names:
        tool = registry.get(name)
        required = tool.parameters_schema.get("required", [])
        props = tool.parameters_schema.get("properties", {})
        for req in required:
            assert req in props, f"{name}: required param '{req}' missing from schema properties"
