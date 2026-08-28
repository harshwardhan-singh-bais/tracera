"""
Tests for the MCP layer — comprehensive coverage.

Phase 39: the MCP *server* exposes TRACERA's full capabilities (40+ tools).
Phase 40: the MCP *client* connects to external servers over the real
          stdio protocol (including the official filesystem server).
Phase 41: the *unified registry* merges native + MCP tools into one list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tracera.mcp.server import (
    ALL_MCP_TOOLS,
    CODE_INTELLIGENCE_TOOLS,
    CONTEXT_TOOLS,
    MEMORY_TOOLS,
    SAFETY_TOOLS,
    REPOSITORY_TOOLS,
    TraceraMCPServer,
)

EXPECTED_CODE_INTELLIGENCE = {
    "search_code", "find_symbol", "find_references", "get_context",
    "get_dependencies", "find_importers", "get_blast_radius",
    "get_call_hierarchy", "find_dead_code", "get_changed_symbols",
    "get_hotspots", "search_ast", "get_class_hierarchy",
    "get_dependency_cycles", "get_coupling_metrics", "get_endpoint_impact",
}

EXPECTED_CONTEXT = {
    "assemble_task_context", "get_ranked_context", "plan_turn",
    "get_session_stats", "get_repo_map",
}

EXPECTED_MEMORY = {
    "recall_memory", "remember_memory", "forget_memory",
    "list_sessions", "search_memory", "get_memory_graph",
}

EXPECTED_SAFETY = {
    "check_edit_safe", "check_delete_safe", "plan_refactoring",
    "get_pr_risk_profile", "get_symbol_provenance", "audit_agent_config",
}

EXPECTED_REPOSITORY = {
    "run_tests", "inspect_repository",
}

ALL_EXPECTED = (
    EXPECTED_CODE_INTELLIGENCE
    | EXPECTED_CONTEXT
    | EXPECTED_MEMORY
    | EXPECTED_SAFETY
    | EXPECTED_REPOSITORY
)


@pytest.fixture
def mcp_settings(tmp_path, monkeypatch):
    """Settings isolated in a temp workspace with no code index."""
    from tracera.config.settings import Settings
    monkeypatch.setenv("TRACERA_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TRACERA_DATA_DIR", str(tmp_path / ".tracera"))
    return Settings()


def _npx_cmd(pkg: str, *args: str) -> list[str]:
    """Platform-aware npx invocation (cmd shim on Windows)."""
    cmd = ["npx", "-y", pkg, *args]
    if os.name == "nt":
        return ["cmd", "/c", *cmd]
    return cmd


# ════════════════════════════════════════════════════════════════════════════
# Phase 39 — MCP server: tool catalog
# ════════════════════════════════════════════════════════════════════════════


def test_tool_catalog_lists_are_complete():
    """All category lists are defined and non-overlapping."""
    assert len(CODE_INTELLIGENCE_TOOLS) == 16
    assert len(CONTEXT_TOOLS) == 5
    assert len(MEMORY_TOOLS) == 6
    assert len(SAFETY_TOOLS) == 6
    assert len(REPOSITORY_TOOLS) == 2
    assert len(ALL_MCP_TOOLS) == 35


def test_no_duplicate_tools_across_categories():
    """No tool appears in more than one category."""
    all_lists = [
        CODE_INTELLIGENCE_TOOLS,
        CONTEXT_TOOLS,
        MEMORY_TOOLS,
        SAFETY_TOOLS,
        REPOSITORY_TOOLS,
    ]
    seen = set()
    for tool_list in all_lists:
        for name in tool_list:
            assert name not in seen, f"Duplicate tool: {name}"
            seen.add(name)


async def test_server_exposes_all_tools(mcp_settings, tmp_path):
    server = TraceraMCPServer(mcp_settings, tmp_path)
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == ALL_EXPECTED


async def test_server_tool_schemas_are_valid(mcp_settings, tmp_path):
    server = TraceraMCPServer(mcp_settings, tmp_path)
    tools = await server.mcp.list_tools()
    by_name = {t.name: t for t in tools}
    # Check a few key schemas
    schema = by_name["search_code"].inputSchema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "query" in schema["required"]

    inspect_schema = by_name["inspect_repository"].inputSchema
    assert inspect_schema["type"] == "object"

    # Memory tools should have proper schemas
    recall_schema = by_name["recall_memory"].inputSchema
    assert recall_schema["type"] == "object"
    assert "query" in recall_schema["properties"]

    remember_schema = by_name["remember_memory"].inputSchema
    assert remember_schema["type"] == "object"
    assert "content" in remember_schema["properties"]

    # Safety tools
    check_edit_schema = by_name["check_edit_safe"].inputSchema
    assert check_edit_schema["type"] == "object"
    assert "symbol" in check_edit_schema["properties"]


async def test_server_search_code_without_index_returns_hint(mcp_settings, tmp_path):
    """No index → a clear hint, not a crash (graceful degradation)."""
    server = TraceraMCPServer(mcp_settings, tmp_path)
    content, _ = await server.mcp.call_tool(
        "search_code", {"query": "authentication middleware"}
    )
    text = content[0].text
    assert "tracera index" in text


async def test_server_inspect_repository(mcp_settings, tmp_path):
    (tmp_path / "hello.py").write_text("def greet():\n    return 'hi'\n")
    server = TraceraMCPServer(mcp_settings, tmp_path)
    content, _ = await server.mcp.call_tool("inspect_repository", {})
    text = content[0].text
    assert "Repository:" in text
    assert "python" in text  # language detected for hello.py


async def test_server_run_tests(mcp_settings, tmp_path):
    """Exercises the real Phase 33 test execution path (pytest subprocess)."""
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n    assert 1 == 1\n"
    )
    server = TraceraMCPServer(mcp_settings, tmp_path)
    content, _ = await server.mcp.call_tool("run_tests", {})
    text = content[0].text
    assert "1/1 passed" in text or "1 passed" in text


async def test_server_find_symbol_without_index_returns_hint(mcp_settings, tmp_path):
    server = TraceraMCPServer(mcp_settings, tmp_path)
    content, _ = await server.mcp.call_tool("find_symbol", {"name": "Foo"})
    assert "tracera index" in content[0].text


async def test_server_memory_tools_graceful_without_memory(mcp_settings, tmp_path):
    """Memory tools should work (even if empty) without crashing."""
    server = TraceraMCPServer(mcp_settings, tmp_path)
    # recall should work (returns "no memories" or empty)
    content, _ = await server.mcp.call_tool("recall_memory", {"query": "auth"})
    assert len(content) > 0

    # list_sessions should work
    content, _ = await server.mcp.call_tool("list_sessions", {})
    assert len(content) > 0

    # search_memory should work
    content, _ = await server.mcp.call_tool("search_memory", {"query": "auth"})
    assert len(content) > 0

    # get_memory_graph should work
    content, _ = await server.mcp.call_tool("get_memory_graph", {})
    assert len(content) > 0


async def test_server_safety_tools_graceful_without_index(mcp_settings, tmp_path):
    """Safety tools should return graceful messages when index is missing."""
    server = TraceraMCPServer(mcp_settings, tmp_path)
    content, _ = await server.mcp.call_tool("check_edit_safe", {"symbol": "Foo"})
    text = content[0].text
    assert "unavailable" in text.lower() or "index" in text.lower() or "error" in text.lower()


async def test_server_structural_tools_graceful_without_index(mcp_settings, tmp_path):
    """Structural analysis tools should be graceful without an index."""
    server = TraceraMCPServer(mcp_settings, tmp_path)
    # find_importers takes 'path', the rest take 'symbol'
    tool_args = {
        "find_importers": {"path": "foo.py"},
        "get_blast_radius": {"symbol": "Foo"},
        "get_call_hierarchy": {"symbol": "Foo"},
        "find_dead_code": {},
    }
    for tool_name in ["find_importers", "get_blast_radius", "get_call_hierarchy", "find_dead_code"]:
        content, _ = await server.mcp.call_tool(tool_name, tool_args[tool_name])
        assert len(content) > 0


# ════════════════════════════════════════════════════════════════════════════
# Phase 39 — real wire protocol (stdio subprocess → ClientSession)
# ════════════════════════════════════════════════════════════════════════════


async def test_server_over_real_stdio_protocol(tmp_path):
    """Spawn the actual server as a subprocess and speak real MCP to it."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tracera.mcp.server"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            names = {t.name for t in result.tools}
            assert names == ALL_EXPECTED

            call = await session.call_tool("inspect_repository", {})
            assert any(
                "Repository:" in str(getattr(c, "text", ""))
                for c in call.content
            )


# ════════════════════════════════════════════════════════════════════════════
# Phase 40 — MCP client
# ════════════════════════════════════════════════════════════════════════════


async def test_client_connects_to_tracera_server(tmp_path):
    """The client connects to an external server and lists/calls its tools."""
    from tracera.mcp.client import MCPClient

    repo_root = Path(__file__).resolve().parent.parent
    async with MCPClient(
        "tracera",
        sys.executable,
        ["-m", "tracera.mcp.server"],
        cwd=str(repo_root),
    ) as client:
        assert client.connected
        tools = await client.list_tools()
        assert {t["name"] for t in tools} == ALL_EXPECTED

        text = await client.call_tool("inspect_repository", {})
        assert "Repository:" in text


async def test_client_interops_with_official_filesystem_server(tmp_path):
    """
    Connect to the official Model Context Protocol filesystem server (npm).

    This proves TRACERA's client speaks the real MCP protocol against a
    third-party server with no auth required. Skipped when npx/network is
    unavailable.
    """
    from tracera.mcp.client import MCPClient

    (tmp_path / "note.txt").write_text("hello filesystem")

    cmd = _npx_cmd("@modelcontextprotocol/server-filesystem", str(tmp_path))
    try:
        client = MCPClient("filesystem", cmd[0], cmd[1:])
        await __import__("asyncio").wait_for(client.connect(), timeout=120)
    except Exception as e:
        pytest.skip(f"Filesystem MCP server unavailable: {e}")

    try:
        tools = await client.list_tools()
        names = {t["name"] for t in tools}
        assert "read_text_file" in names
        assert "write_file" in names

        text = await client.call_tool(
            "read_text_file", {"path": str(tmp_path / "note.txt")}
        )
        assert "hello filesystem" in text
    finally:
        await client.disconnect()


async def test_mcp_tool_adapter(mcp_settings, tmp_path):
    """Remote tool definitions adapt into native Tools the agent can call."""
    from tracera.mcp.client import MCPClient

    repo_root = Path(__file__).resolve().parent.parent
    async with MCPClient(
        "tracera",
        sys.executable,
        ["-m", "tracera.mcp.server"],
        cwd=str(repo_root),
    ) as client:
        tool_defs = await client.list_tools()
        native = client.to_native_tools(tool_defs)
        assert len(native) == len(ALL_EXPECTED)

        # Remote tools are server-prefixed in the unified registry
        by_name = {t.name: t for t in native}
        assert "tracera_search_code" in by_name
        assert "tracera_inspect_repository" in by_name
        assert "tracera_remember_memory" in by_name
        assert "tracera_check_edit_safe" in by_name
        search = by_name["tracera_search_code"]
        assert search.parameters_schema["type"] == "object"

        result = await by_name["tracera_inspect_repository"].execute()
        assert result.success
        assert "Repository:" in result.output


# ════════════════════════════════════════════════════════════════════════════
# Phase 41 — Unified Tool Registry (native + MCP side by side)
# ════════════════════════════════════════════════════════════════════════════


async def test_unified_registry_merges_native_and_mcp_tools(mcp_settings, tmp_path):
    """Native tools and remote MCP tools live in one flat registry."""
    from tracera.mcp.manager import MCPManager, MCPServerConfig
    from tracera.tools.registry import ToolRegistry

    repo_root = Path(__file__).resolve().parent.parent
    manager = MCPManager([
        MCPServerConfig(
            name="tracera-a",
            command=sys.executable,
            args=["-m", "tracera.mcp.server"],
            cwd=str(repo_root),
        ),
        MCPServerConfig(
            name="tracera-b",
            command=sys.executable,
            args=["-m", "tracera.mcp.server"],
            cwd=str(repo_root),
        ),
    ])

    registry = ToolRegistry()

    async with manager:
        merged = await manager.connect_all()
        assert set(merged.keys()) == {"tracera-a", "tracera-b"}
        for tools in merged.values():
            assert {t["name"] for t in tools} == ALL_EXPECTED

        added = await manager.register(merged, registry)

        # A remote tool executes through the unified registry path while
        # the connection is live
        from tracera.workspace.sandbox import WorkspaceSandbox
        ws = WorkspaceSandbox(repo_root)
        result = await registry.execute(
            "tracera-b_inspect_repository", "mcp-call-1", {}
        )
        assert result.success
        assert "Repository:" in result.output

    assert added == len(ALL_EXPECTED) * 2  # all tools × 2 servers
    assert len(registry) == len(ALL_EXPECTED) * 2
    # Server-prefixed names keep both servers' tools distinct in the registry
    assert "tracera-a_search_code" in registry.names
    assert "tracera-b_inspect_repository" in registry.names
    assert "tracera-a_remember_memory" in registry.names
    assert "tracera-b_check_edit_safe" in registry.names


def test_manager_loads_config_from_file(tmp_path):
    from tracera.mcp.manager import MCPManager
    config_file = tmp_path / "mcp_servers.json"
    config_file.write_text(
        '[{"name": "filesystem", "command": "npx", '
        '"args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}]'
    )
    manager = MCPManager.from_file(config_file)
    assert manager.configs[0].name == "filesystem"
    assert manager.configs[0].args[0] == "-y"
