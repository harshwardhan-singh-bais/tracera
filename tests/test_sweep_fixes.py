"""
Regression tests for fixes from the full slash-command sweep (2026-09-12).

Covers:
1. Sub-agent sandbox — mutating tools/git unreachable from every role,
   and run_command rejects destructive git invocations.
2. TestRunner timeout is configurable via TRACERA_TEST_TIMEOUT.
3. GetHotspotsTool does not run pytest --cov unless with_coverage=True.
4. MemoryGraphTool handles an argless call (overview, not TypeError).
5. SearchSymbolsTool unwraps a retrieval-pipeline tuple.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tracera.agent.subagents import (
    ROLE_TOOL_SETS,
    SubAgentRole,
    assert_command_allowed,
    build_sub_agent,
)
from tracera.tools.registry import ToolRegistry


# ── 1. Sub-agent sandbox ──────────────────────────────────────────────────────


def test_no_role_has_git_tool():
    for role, tools in ROLE_TOOL_SETS.items():
        assert "git" not in tools, f"{role} must not have the git tool"


def test_read_only_roles_have_no_mutation_tools():
    write_tools = {"write_file", "edit_file", "delete_file"}
    for role in (SubAgentRole.RESEARCHER, SubAgentRole.TESTER, SubAgentRole.REVIEWER):
        assert not (ROLE_TOOL_SETS[role] & write_tools), f"{role} must be read-only"


@pytest.mark.asyncio
async def test_run_command_guard_rejects_git_commit():
    with pytest.raises(PermissionError):
        assert_command_allowed("git commit -m 'oops'")
    with pytest.raises(PermissionError):
        assert_command_allowed("cd repo && GIT_AUTHOR_NAME=x git push origin main")
    # Non-git commands pass
    assert_command_allowed("pytest tests/ -q")
    assert_command_allowed("python -m pytest -k slash")


@pytest.mark.asyncio
async def test_built_sub_agent_run_command_is_guarded():
    registry = ToolRegistry()
    cmd_tool = MagicMock()
    cmd_tool.name = "run_command"
    called = []

    async def _exec(*a, **k):
        called.append(a)
        return "ok"

    cmd_tool.execute = _exec
    registry.register(cmd_tool)

    spec = build_sub_agent(
        SubAgentRole.TESTER,
        provider=MagicMock(),
        registry=registry,
    )
    tool = spec.registry.get("run_command")
    with pytest.raises(PermissionError):
        await tool.execute("git commit -m nope")
    assert not called  # underlying execute never ran


# ── 2. TestRunner timeout configurable ────────────────────────────────────────


def test_test_runner_timeout_from_settings(tmp_path, monkeypatch):
    from tracera.config import settings as settings_mod
    from tracera.tools import test_runner as tr

    monkeypatch.setenv("TRACERA_TEST_TIMEOUT", "777")
    settings_mod.reset_settings()  # drop the cached singleton
    try:
        runner = tr.TestRunner(tmp_path)
        assert runner._timeout == 777
    finally:
        monkeypatch.delenv("TRACERA_TEST_TIMEOUT")
        settings_mod.reset_settings()  # restore for other tests


# ── 3. /hotspots coverage opt-in ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hotspots_skips_pytest_cov_by_default(tmp_path, monkeypatch):
    import tracera.tools.ast_tools as at

    ran_pytest = {"count": 0}

    class _FakeProc:
        stdout = ""
        returncode = 0

    def _fake_run(*a, **k):
        if a and a[0] and a[0][0] == "pytest":
            ran_pytest["count"] += 1
        return _FakeProc()

    monkeypatch.setattr(at.subprocess, "run", _fake_run)

    graph = MagicMock()
    graph._g.nodes.return_value = []
    monkeypatch.setattr(at, "_get_graph", lambda pipeline: graph)
    monkeypatch.setattr("networkx.pagerank", lambda g: {})

    tool = at.GetHotspotsTool(workspace=tmp_path, retrieval_pipeline=None)
    await tool.execute(top_n=5)
    assert ran_pytest["count"] == 0, "pytest --cov must not run by default"


# ── 4. MemoryGraphTool argless overview ───────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_graph_argless_returns_overview():
    from tracera.tools.memory_tools import MemoryGraphTool

    store = MagicMock()
    store.triple_count = 3
    store.get_central_concepts.return_value = [("concept_a", 2)]
    result = await MemoryGraphTool(store).execute()
    assert result.success
    assert "Overview" in result.output


# ── 5. SearchSymbolsTool pipeline unwrapping ──────────────────────────────────


@pytest.mark.asyncio
async def test_search_symbols_unwraps_pipeline_tuple():
    from tracera.tools.ast_tools import SearchSymbolsTool

    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "symbol": "foo",
            "symbol_type": "function",
            "file_path": "a.py",
            "start_line": 3,
            "_bm25_score": 1.0,
            "_dense_score": 0.5,
            "_rrf_score": 0.2,
        }
    ]
    tool = SearchSymbolsTool((None, retriever, None))  # pipeline-shaped tuple
    result = await tool.execute("foo")
    assert result.success
    assert "`foo`" in result.output
    retriever.search.assert_called_once()
