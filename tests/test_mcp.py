"""
Tests for the MCP layer — Phases 39-41.

Phase 39: the MCP *server* exposes TRACERA's 7 existing capabilities.
Phase 40: the MCP *client* connects to external servers over the real
          stdio protocol (including the official filesystem server).
Phase 41: the *unified registry* merges native + MCP tools into one list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tracera.mcp.server import EXPOSED_TOOLS, TraceraMCPServer

EXPECTED_TOOLS = {
    "search_code",
    "find_symbol",
    "find_references",
    "get_context",
    "get_dependencies",
    "run_tests",
    "inspect_repository",
}


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
# Phase 39 — MCP server
# ════════════════════════════════════════════════════════════════════════════


async def test_server_exposes_all_7_tools(mcp_settings, tmp_path):
    server = TraceraMCPServer(mcp_settings, tmp_path)
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS


async def test_server_tool_schemas_are_valid(mcp_settings, tmp_path):
    server = TraceraMCPServer(mcp_settings, tmp_path)
    tools = await server.mcp.list_tools()
    by_name = {t.name: t for t in tools}
    schema = by_name["search_code"].inputSchema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "query" in schema["required"]

    inspect_schema = by_name["inspect_repository"].inputSchema
    assert inspect_schema["type"] == "object"


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
            assert names == EXPECTED_TOOLS

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
        assert {t["name"] for t in tools} == EXPECTED_TOOLS

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
        assert len(native) == 7

        # Remote tools are server-prefixed in the unified registry
        by_name = {t.name: t for t in native}
        assert "tracera_search_code" in by_name
        assert "tracera_inspect_repository" in by_name
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
            assert {t["name"] for t in tools} == EXPECTED_TOOLS

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

    assert added == 14  # 7 tools × 2 servers
    assert len(registry) == 14
    # Server-prefixed names keep both servers' tools distinct in the registry
    assert "tracera-a_search_code" in registry.names
    assert "tracera-b_inspect_repository" in registry.names


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
