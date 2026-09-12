"""
Integration smoke test: every slash command dispatches without crashing.

Creates a TraceraTUI with a mock agent (no real LLM, no index, no memory layer)
and calls ``await app._handle_command("/xxx ...")`` for every registered slash
command.  Catches any exception — the goal is that the TUI itself never crashes
on malformed input or missing prerequisites, only shows error messages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tracera.tui.app import TraceraTUI
from tracera.tui.widgets.command_registry import SLASH_COMMANDS
from tracera.tui.widgets.slash_actions import SLASH_TOOLS


@pytest.fixture
def mock_agent():
    """A minimal ReActAgent stub with a registry containing the built-in tools."""
    agent = MagicMock()
    agent.provider = MagicMock()
    agent.provider.name = "mock"
    agent.provider.default_model = "mock-model"
    agent.provider.has_vision = False
    agent.model = "mock-model"
    agent.max_iterations = 12
    agent.max_tool_calls = 30
    agent.registry = MagicMock()
    agent.registry.has.return_value = False
    agent.registry.get.return_value = MagicMock()
    agent.registry.execute.return_value = _make_fake_result()
    agent._enhanced_memory = None
    agent._triple_store = None
    agent._session_manager = None
    agent._recent_files = []
    agent.decomposer = None
    return agent


@pytest.fixture
def mock_memory():
    m = MagicMock()
    m.count = 0
    m.entries.return_value = []
    return m


@pytest.fixture
def app(mock_agent, mock_memory, tmp_path):
    """Build a TraceraTUI that can have _handle_command called without a running event loop."""
    from tracera.conversation.state import ConversationState

    a = TraceraTUI(
        agent=mock_agent,
        memory=mock_memory,
        workspace_path=tmp_path,
        retrieval_pipeline=None,
        banner=None,
    )
    a._conversation = ConversationState()
    a._recent_files = []
    # _panel, _status_line need the app to be mounted.  Plant a minimal stub
    # panel so _handle_command's panel.add_* calls don't hit query_one.
    stub_panel = MagicMock()
    stub_panel.add_assistant_message = MagicMock()
    stub_panel.add_error = MagicMock()
    stub_panel.add_info_row = MagicMock()
    stub_panel.add_meta = MagicMock()
    stub_panel.add_user_message = MagicMock()
    stub_panel.clear = MagicMock()
    stub_panel.attachments = []
    stub_panel.verbose = False
    stub_panel.add_attachment = MagicMock()
    stub_panel.clear_attachments = MagicMock()
    stub_panel.type_message = MagicMock()
    stub_panel.add_banner = MagicMock()
    stub_panel.freeze_phase = MagicMock()
    stub_panel._append = MagicMock()
    stub_panel.query = MagicMock(return_value=[])
    stub_panel.set_feature_status = MagicMock()
    stub_panel.add_phase = MagicMock()
    stub_panel.add_thinking_disclosure = MagicMock()
    # Wire call_directly so _panel()/_status_line() return stubs.
    a._panel = lambda: stub_panel
    a._status_line = lambda: stub_panel
    a._plan_row = None
    a._running_worker = None
    a.query_one = MagicMock(side_effect=RuntimeError("no screens"))  # not used

    # ``@work(exclusive=False)`` methods return a Worker and schedule via
    # ``run_worker``.  Textual's ``@work`` decorator wraps the coroutine in
    # ``functools.partial`` before passing it to ``run_worker``, so we accept
    # that wrapper and extract the underlying coroutine.
    async def _inline_worker(coro, **kwargs):
        # ``coro`` may be a functools.partial wrapping the real coroutine.
        import functools
        if isinstance(coro, functools.partial):
            coro = coro.func(*coro.args, **coro.keywords)
        if asyncio.iscoroutine(coro) or asyncio.isfuture(coro):
            await coro

    a.run_worker = _inline_worker
    return a


def _all_slash_commands():
    """Every command that lives in SLASH_COMMANDS — the autocomplete surface."""
    return sorted(SLASH_COMMANDS.keys())


def _make_fake_result() -> MagicMock:
    """Build a ToolResult-shaped mock.

    ``registry.execute`` is a MagicMock whose return value does not need to be
    a real awaitable in this test: the ``@work`` decorator wraps the call in
    ``functools.partial``, and the ``run_worker`` mock unwraps and awaits
    the inner coroutine.  The ``execute`` return is only reached for
    non-@work tool execution paths (``_execute_tool_inline``).
    """
    result = MagicMock()
    result.success = True
    result.output = "mock result"
    result.duration_ms = 1.0
    result.error = None
    return result


def _command_with_dummy_arg(name: str) -> str:
    """Return a slash command string with a plausible dummy arg for tests."""
    arg_map = {
        "model": " mock-model",
        "search": "test",
        "debug": "test",
        "plan": "do something",
        "code": "do something",
        "ask": "do something",
        "symbol": "Foo",
        "symbols": "Foo",
        "source": "Foo",
        "definition": "Foo",
        "context": "Foo",
        "deps": "Foo",
        "outline": "foo.py",
        "repomap": "",
        "assemble": "do something",
        "refs": "Foo",
        "callers": "Foo",
        "blast": "Foo",
        "classhier": "Foo",
        "endpoint": "/api/test",
        "impls": "Foo",
        "provenance": "Foo",
        "risk": "Foo",
        "refactor": "Foo",
        "editsafe": "Foo",
        "deletesafe": "Foo",
        "importers": "foo.py",
        "git": "status",
        "inspectrepo": "",
        "tests": "pytest",
        "read": "pyproject.toml",
        "write": "out.txt",
        "edit": "pyproject.toml",
        "ls": ".",
        "grep": "test",
        "run": "echo hi",
        "recall": "test",
        "remember": "test memory",
        "forget": "test",
        "memsearch": "test",
        "planturn": "test",
        "ranked": "test",
        "taskcontext": "do something",
        "delegate": "do something",
        "tool": "read_file",
    }
    arg = arg_map.get(name, "")
    return f"/{name}{arg}" if arg else f"/{name}"


@pytest.mark.asyncio
async def test_every_slash_command_dispatches_without_crash(app):
    """
    Call _handle_command for every registered slash command.

    The app has no index, no memory layer, and no LLM provider, so most commands
    will show an error row or a 'not available' message — that is expected and
    fine.  The invariant is that *no exception propagates out of
    _handle_command*.
    """
    handled = 0
    failed = []

    for name in _all_slash_commands():
        cmd_text = _command_with_dummy_arg(name)
        try:
            await app._handle_command(cmd_text)
            handled += 1
        except Exception as e:
            failed.append((name, str(e)))

    assert not failed, f"{len(failed)} commands crashed: {failed}"
    assert handled == len(SLASH_COMMANDS), (
        f"Handled {handled} but SLASH_COMMANDS has {len(SLASH_COMMANDS)} entries"
    )


@pytest.mark.asyncio
async def test_tool_aliases_dispatch_without_crash(app):
    """Every alias in SLASH_TOOLS also dispatches cleanly (via the fallback path)."""
    # The fallback ``cmd[1:] in SLASH_TOOLS`` path is exercised when the command
    # is NOT an explicit ``elif cmd == "/x"`` branch.
    import inspect
    import re

    source = inspect.getsource(app._handle_command)
    explicit = set(re.findall(r'cmd == "\(/?([^"]+)"\)', source))
    for group in re.findall(r'cmd in \(([^)]*)\)', source):
        explicit.update(re.findall(r'"([^"]+)"', group))

    aliases_without_explicit_branch = sorted(
        name for name in SLASH_TOOLS if name not in explicit
    )
    assert aliases_without_explicit_branch, (
        "Expected some SLASH_TOOLS aliases to lack an explicit branch; "
        "if this fails, either all aliases have explicit handlers (good!) "
        "or the regex is out of date."
    )

    handled = 0
    failed = []

    for alias in aliases_without_explicit_branch:
        cmd_text = f"/{alias} Foo"
        try:
            await app._handle_command(cmd_text)
            handled += 1
        except Exception as e:
            failed.append((alias, str(e)))

    assert not failed, f"{len(failed)} aliases crashed: {failed}"
    assert handled == len(aliases_without_explicit_branch)
