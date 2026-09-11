"""Unified slash-command surface — aliases, parsing, and dispatch wiring."""

from __future__ import annotations

import pytest

from tracera.tui.widgets.command_registry import (
    COMMAND_ORDER,
    SLASH_COMMANDS,
    SLASH_TOOLS,
)
from tracera.tui.widgets.slash_actions import (
    FEATURE_GROUPS,
    coerce_args,
    parse_tool_invocation,
    single_arg_kwargs,
)


def test_every_dedicated_command_is_ordered():
    for name in SLASH_COMMANDS:
        assert name in COMMAND_ORDER, f"{name} missing from COMMAND_ORDER"


def test_features_and_tool_present():
    assert "features" in SLASH_COMMANDS
    assert "tool" in SLASH_COMMANDS
    assert "features" in COMMAND_ORDER
    assert "tool" in COMMAND_ORDER


def test_tool_aliases_merged_into_registry():
    assert SLASH_TOOLS
    for alias, tool in SLASH_TOOLS.items():
        assert alias in SLASH_COMMANDS
        if SLASH_COMMANDS[alias] == f"Tool: {tool}":
            continue
        # A reserved alias (search/deps/…) keeps its dedicated description;
        # that is intentional — the explicit handler wins at dispatch time.
        assert alias in ("search", "deps", "memory", "test", "review", "tools")


def test_dedicated_commands_not_overwritten_by_alias():
    assert SLASH_COMMANDS["search"] == "Search the code index (hybrid): /search <query>"
    assert SLASH_COMMANDS["deps"] == "Symbol dependency chain: /deps <symbol>"


def test_feature_groups_cover_all_aliases():
    listed = {
        usage.split()[0].lstrip("/")
        for items in FEATURE_GROUPS.values()
        for usage, _ in items
    }
    reserved = {"search", "deps", "memory", "test", "review", "tools"}
    missing = set(SLASH_TOOLS) - listed - reserved
    assert not missing, f"aliases missing from /features: {sorted(missing)}"


def test_parse_tool_invocation_key_values():
    name, args = parse_tool_invocation("get_context symbol=Foo k=5")
    assert name == "get_context"
    assert args == {"symbol": "Foo", "k": "5"}


def test_parse_tool_invocation_positional_becomes_query():
    name, args = parse_tool_invocation("find_symbol AuthMiddleware")
    assert name == "find_symbol"
    assert args == {"query": "AuthMiddleware"}


def test_parse_tool_invocation_quoted_values():
    name, args = parse_tool_invocation('search_code query="auth middleware"')
    assert name == "search_code"
    assert args == {"query": "auth middleware"}


def test_parse_tool_invocation_empty_raises():
    with pytest.raises(ValueError, match="Usage"):
        parse_tool_invocation("   ")


class _FakeTool:
    def __init__(self, schema: dict):
        self.parameters_schema = schema


def test_coerce_args_types():
    tool = _FakeTool(
        {
            "type": "object",
            "properties": {
                "k": {"type": "integer"},
                "score": {"type": "number"},
                "flag": {"type": "boolean"},
                "paths": {"type": "array"},
                "query": {"type": "string"},
            },
        }
    )
    out = coerce_args(
        tool,
        {"k": "5", "score": "0.5", "flag": "true", "paths": "a,b", "query": "x"},
    )
    assert out == {"k": 5, "score": 0.5, "flag": True, "paths": ["a", "b"], "query": "x"}


def test_coerce_args_bad_integer_raises():
    tool = _FakeTool({"type": "object", "properties": {"k": {"type": "integer"}}})
    with pytest.raises(ValueError, match="not a valid integer"):
        coerce_args(tool, {"k": "notanint"})


def test_single_arg_kwargs_prefers_sole_required():
    tool = _FakeTool(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        }
    )
    assert single_arg_kwargs(tool, "auth") == {"query": "auth"}


def test_single_arg_kwargs_uses_well_known_name():
    tool = _FakeTool(
        {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "depth": {"type": "integer"}},
        }
    )
    assert single_arg_kwargs(tool, "Foo") == {"symbol": "Foo"}


def test_single_arg_kwargs_empty_value():
    tool = _FakeTool({"type": "object", "properties": {"query": {"type": "string"}}})
    assert single_arg_kwargs(tool, "  ") == {}
