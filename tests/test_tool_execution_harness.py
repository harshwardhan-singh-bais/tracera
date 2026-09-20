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
from tracera.tools.base import ToolResult, _accepted_kwargs
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
    g.add_symbol(
        "tracera/tools/registry.py", _sym("ToolRegistry", SymbolType.CLASS, 1, 60, parent=None)
    )
    g.add_symbol(
        "tracera/tools/registry.py",
        _sym("execute", SymbolType.METHOD, 20, 40, parent="ToolRegistry"),
    )
    g.add_symbol("tracera/agent/react_loop.py", _sym("ReActAgent", SymbolType.CLASS, 1, 90))
    g.add_symbol(
        "tracera/agent/react_loop.py", _sym("run", SymbolType.METHOD, 30, 80, parent="ReActAgent")
    )

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
        _fake_indexer(),  # 0  indexer
        FakeRetriever(),  # 1  symbol retriever
        FakeExpander(),  # 2  expander
        None,  # 3  reranker
        None,  # 4  context engine
        None,  # 5  compressor
        None,  # 6  embedder
        None,  # 7  vector store
        None,  # 8  bm25
        graph_retriever,  # 9  graph retriever
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
    "plan_refactoring": {
        "refactor_type": "rename",
        "symbol": "ToolRegistry",
        "new_name": "ToolRegistry2",
    },
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
        "Tools that failed execution against a live-shaped pipeline:\n  " + "\n  ".join(failures)
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


def test_every_schema_property_is_accepted_by_execute(tmp_path: Path) -> None:
    """
    No tool may advertise a parameter its own ``execute()`` rejects.

    ``test_tool_schema_required_args_match_harness`` covers only *required*
    parameters, and only for tools present in ARGS. An **optional** rejected
    parameter is the same trap: the model reads the schema, sends the argument,
    and the call dies with a TypeError that looks like a broken feature rather
    than a schema bug.
    """
    registry = _build_registry(tmp_path)
    problems: list[str] = []
    examined = 0
    for name in registry.names:
        tool = registry.get(name)
        props = set(tool.parameters_schema.get("properties", {}))
        examined += len(props)
        accepted = _accepted_kwargs(tool.execute)
        if accepted is None:
            continue  # declares **kwargs — accepts whatever it is handed
        unknown = props - accepted
        if unknown:
            problems.append(f"{name}: schema advertises {sorted(unknown)} not accepted by execute()")

    # Assert the floor before the property: a walk that examined nothing would
    # satisfy `not problems` while proving nothing at all.
    assert examined >= 40, f"only {examined} schema properties examined — the walk is broken"
    assert not problems, "\n".join(problems)


def test_safe_execute_drops_surplus_arguments_instead_of_failing(tmp_path: Path) -> None:
    """
    A hallucinated parameter must not cost the whole turn.

    Models confuse the argument names of similar tools — a live run passed
    grep's ``file_extensions``/``max_results``/``path`` to ``search_code``.
    Discarding the surplus keeps the call alive, because the arguments it did
    send were the ones the tool wanted.

    The other half matters just as much: a *missing required* argument must
    still fail, or this would be hiding real misuse instead of tolerating noise.
    """
    registry = _build_registry(tmp_path)

    surplus = asyncio.run(
        registry.execute("find_symbol", "harness", {"name": "ToolRegistry", "totally_made_up": 1})
    )
    assert surplus.success, f"a surplus argument killed the call: {surplus.error}"

    missing = asyncio.run(registry.execute("find_symbol", "harness", {}))
    assert not missing.success, "a missing required argument must still fail"


def test_search_code_accepts_the_argument_names_models_actually_send(tmp_path: Path) -> None:
    """
    Regression for the exact call a live agent made, which used to raise.

    ``search_code`` and ``grep`` both search the repository, so the model mixes
    their parameter names up. The retrieval must still happen — filtered — rather
    than fail.
    """
    registry = _build_registry(tmp_path)
    result = asyncio.run(
        registry.execute(
            "search_code",
            "harness",
            {
                "query": "registry",
                "file_extensions": [".py"],
                "max_results": 20,
                "path": "",
            },
        )
    )
    assert result.success, f"the live failing call still fails: {result.error}"


def test_search_code_extension_filter_normalises_and_actually_filters() -> None:
    """
    ``py`` and ``.py`` must both work, and the filter must really narrow.

    Asserting only that the call *succeeds* would pass even if the filter were
    silently ignored. The returned set is what proves it did something.
    """
    from tracera.tools.code_search import SearchCodeTool

    class Pool:
        def search(self, query, k=5, language=None):
            return [
                {"file_path": "src/a.py", "content": "a", "symbol": "a"},
                {"file_path": "src/b.tsx", "content": "b", "symbol": "b"},
                {"file_path": "docs/c.md", "content": "c", "symbol": "c"},
            ]

    tool = SearchCodeTool(Pool())

    bare = asyncio.run(tool.execute("q", file_extensions=["py"]))
    assert bare.success, bare.error
    assert "src/a.py" in bare.output
    assert "src/b.tsx" not in bare.output, "extension filter was ignored"

    dotted = asyncio.run(tool.execute("q", file_extensions=[".tsx"]))
    assert dotted.success, dotted.error
    assert "src/b.tsx" in dotted.output
    assert "src/a.py" not in dotted.output, "extension filter was ignored"

    # `max_results` is grep's name for the same idea and must map onto k.
    limited = asyncio.run(tool.execute("q", max_results=1))
    assert limited.success, limited.error
    assert limited.metadata.get("count", 1) == 1 or limited.output.count("### [") == 1


def test_search_code_caps_chunks_per_file_by_default() -> None:
    """
    Without a cap, one file that matches often takes the entire window.

    The tool defaults to 2 chunks per file, so k=5 spans three files. Asserting
    only that the call succeeds would pass with no cap at all, so the returned
    file set is what gets checked.
    """
    import re

    from tracera.tools.code_search import SearchCodeTool

    class Clustered:
        def search(self, query, k=5, language=None):
            pool = [
                {"file_path": f"src/{letter}.py", "content": "x" * 50, "symbol": letter}
                for letter in ("a", "b", "c")
                for _ in range(10)
            ]
            return pool[:k]

    tool = SearchCodeTool(Clustered())

    result = asyncio.run(tool.execute("q"))
    assert result.success, result.error
    files = re.findall(r"in `([^`]+)`", result.output)
    assert len(files) == 5, f"the window was not filled: {files}"
    assert len(set(files)) == 3, f"one file monopolised the window: {files}"

    uncapped = asyncio.run(tool.execute("q", max_per_file=0))
    assert uncapped.success, uncapped.error
    uncapped_files = re.findall(r"in `([^`]+)`", uncapped.output)
    assert len(set(uncapped_files)) == 1, "max_per_file=0 must disable the cap"
