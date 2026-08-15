"""
TRACERA MCP Layer — Phases 39-41.

Exposes TRACERA's existing code-intelligence capabilities over the Model
Context Protocol (MCP):

    mcp/server.py   — Phase 39: an MCP *server* that adapts the 7 existing
                      capabilities (search_code, find_symbol, find_references,
                      get_context, get_dependencies, run_tests,
                      inspect_repository) for any MCP client.
    mcp/client.py   — Phase 40: an MCP *client* that connects to external MCP
                      servers, lists their tools, and adapts them into native
                      TRACERA Tool objects.
    mcp/manager.py  — Phase 41: manages multiple MCP server connections and
                      merges their tools into the unified ToolRegistry.
"""

from tracera.mcp.server import TraceraMCPServer, build_mcp_server
from tracera.mcp.client import MCPClient, MCPTool
from tracera.mcp.manager import MCPServerConfig, MCPManager

__all__ = [
    "TraceraMCPServer",
    "build_mcp_server",
    "MCPClient",
    "MCPTool",
    "MCPServerConfig",
    "MCPManager",
]
