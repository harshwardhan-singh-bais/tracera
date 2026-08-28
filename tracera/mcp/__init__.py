"""
TRACERA MCP Layer — Code Intelligence + Memory over Model Context Protocol.

    mcp/server.py   — Comprehensive MCP server exposing 40+ tools across
                      Code Intelligence, Context, Memory, Safety, and
                      Repository categories. Thin adapter over the same
                      engine used by TRACERA's CLI agent.
    mcp/client.py   — MCP client that connects to external servers, lists
                      their tools, and adapts them into native TRACERA Tool
                      objects (GitHub, Postgres, filesystem, Slack, etc.).
    mcp/manager.py  — Manages multiple MCP server connections and merges
                      their tools into the unified ToolRegistry.
"""

from tracera.mcp.server import (
    TraceraMCPServer,
    build_mcp_server,
    ALL_MCP_TOOLS,
    CODE_INTELLIGENCE_TOOLS,
    CONTEXT_TOOLS,
    MEMORY_TOOLS,
    SAFETY_TOOLS,
    REPOSITORY_TOOLS,
)
from tracera.mcp.client import MCPClient, MCPTool
from tracera.mcp.manager import MCPServerConfig, MCPManager

__all__ = [
    "TraceraMCPServer",
    "build_mcp_server",
    "ALL_MCP_TOOLS",
    "CODE_INTELLIGENCE_TOOLS",
    "CONTEXT_TOOLS",
    "MEMORY_TOOLS",
    "SAFETY_TOOLS",
    "REPOSITORY_TOOLS",
    "MCPClient",
    "MCPTool",
    "MCPServerConfig",
    "MCPManager",
]
