"""
Verification: every slash command resolves to a registered tool or a TUI handler.

Two sources of truth must agree:
  1. ``command_registry.SLASH_COMMANDS`` (autocomplete/help surface)
  2. ``app._handle_command`` dispatch + ``slash_actions.SLASH_TOOLS`` (execution)

Dispatch paths in ``app._handle_command``:
  * an explicit ``cmd == "/name"`` branch, or
  * the ``cmd[1:] in SLASH_TOOLS`` alias fallback (all tool aliases at once).

A command listed in autocomplete but with no dispatch path silently fails at
runtime; a handler with no autocomplete entry is undiscoverable. This pins both.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make sure the repo root is importable when pytest is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_handled_commands() -> tuple[set[str], bool]:
    """
    Extract commands dispatched in TraceraTUI._handle_command from app.py.

    Returns (explicit commands without the leading slash, has_alias_fallback)
    where has_alias_fallback is True when the body dispatches through the
    ``SLASH_TOOLS`` alias map.
    """
    source = (REPO_ROOT / "tracera" / "tui" / "app.py").read_text(encoding="utf-8")
    match = re.search(r"def _handle_command\(self.*?(?=\n    def |\nclass )", source, re.S)
    assert match, "_handle_command not found in tracera/tui/app.py"
    body = match.group(0)
    # Both forms: cmd == "/x"  and  cmd in ("/x", "/y").
    found = set(re.findall(r'cmd == \(?"([^"]+)"\)?', body))
    for group in re.findall(r'cmd in \(([^)]*)\)', body):
        found.update(re.findall(r'"([^"]+)"', group))
    has_fallback = "cmd[1:] in SLASH_TOOLS" in body
    return {c.lstrip("/").lower() for c in found}, has_fallback


def _all_tool_names() -> set[str]:
    """Collect every tool name, including property-based and factory-built ones."""
    import tempfile

    import tracera.main  # noqa: F401  (registers nested tool classes)
    for mod_name in (
        "tracera.tools.ast_tools",
        "tracera.tools.code_search",
        "tracera.tools.memory_tools",
        "tracera.tools.session_tools",
        "tracera.tools.provenance_tools",
        "tracera.tools.refactor_tools",
    ):
        __import__(mod_name, fromlist=["*"])

    from tracera.tools.base import Tool
    from tracera.tools.registry import create_default_registry
    from tracera.workspace.sandbox import WorkspaceSandbox

    names: set[str] = set()
    # Class-level ``name`` attributes (most code-intelligence tools).
    stack = list(Tool.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        value = getattr(cls, "name", None)
        if isinstance(value, str):
            names.add(value)

    # Property-based ``name`` (builtin coding tools) need instances — the
    # default registry instantiates exactly those; the main.py factory tools
    # (run_tests / inspect_repository) are built the same way the app does.
    with tempfile.TemporaryDirectory() as td:
        ws = WorkspaceSandbox(Path(td))
        names.update(create_default_registry(ws).names)
        from tracera.main import _make_inspect_repository_tool, _make_run_tests_tool
        names.add(_make_run_tests_tool(ws).name)
        names.add(_make_inspect_repository_tool(ws).name)
    return names


def test_slash_tools_aliases_resolve_to_known_tools():
    from tracera.tui.widgets.slash_actions import SLASH_TOOLS

    known = _all_tool_names()
    missing = {alias: tool for alias, tool in SLASH_TOOLS.items() if tool not in known}
    assert not missing, f"Slash aliases pointing at non-existent tools: {missing}"


def test_dispatched_commands_match_autocomplete_registry():
    from tracera.tui.widgets.command_registry import SLASH_COMMANDS
    from tracera.tui.widgets.slash_actions import SLASH_TOOLS

    handled, has_alias_fallback = _parse_handled_commands()
    assert handled, "No dispatched commands found — did _handle_command change?"

    effective = set(handled) | (set(SLASH_TOOLS) if has_alias_fallback else set())

    # Autocomplete entries with no dispatch path (except the generic /tool).
    undispatched = sorted(set(SLASH_COMMANDS) - effective - {"tool"})
    assert not undispatched, (
        f"Commands in autocomplete with no _handle_command branch: {undispatched}"
    )

    # Dispatched commands with no autocomplete entry (undiscoverable).
    unlisted = sorted(handled - set(SLASH_COMMANDS) - {"ask"})
    assert not unlisted, f"Commands dispatched but missing from autocomplete: {unlisted}"


def test_feature_groups_reference_known_commands_or_aliases():
    from tracera.tui.widgets.slash_actions import FEATURE_GROUPS, SLASH_TOOLS
    from tracera.tui.widgets.command_registry import SLASH_COMMANDS

    for group, items in FEATURE_GROUPS.items():
        assert items, f"Feature group '{group}' is empty"
        for usage, desc in items:
            assert desc, f"Feature '{usage}' in '{group}' has no description"
            name = usage.split()[0].lstrip("/")
            # Must be either a documented command or a tool alias.
            assert name in SLASH_COMMANDS or name in SLASH_TOOLS, (
                f"/{name} in FEATURE_GROUPS ('{group}') is neither a documented "
                "command nor a tool alias"
            )


def test_command_registry_merges_aliases():
    from tracera.tui.widgets.command_registry import SLASH_COMMANDS, COMMAND_ORDER
    from tracera.tui.widgets.slash_actions import SLASH_TOOLS

    for alias in SLASH_TOOLS:
        assert alias in SLASH_COMMANDS, f"Alias '{alias}' missing from autocomplete registry"
    assert len(COMMAND_ORDER) == len(set(COMMAND_ORDER)), "Duplicate entries in COMMAND_ORDER"
