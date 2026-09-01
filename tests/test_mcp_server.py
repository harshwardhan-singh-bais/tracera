"""
MCP Server Integration Tests (Phase 5)
Verify that TRACERA's MCP server works correctly for external clients.
Tests cover:
- Server startup and lifecycle
- Tool registration and discovery
- Authentication
- Workspace boundary enforcement
- Structured response format
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from tracera.mcp.server import TraceraMCPServer, MCP_API_VERSION, MCP_MIN_CLIENT_VERSION
from tracera.config.settings import Settings


@pytest.mark.asyncio
async def test_server_lifecycle():
    """Test server startup and shutdown sequence works correctly."""
    settings = Settings()
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    
    # Server should not be running initially
    assert not server.is_running
    
    # Test startup
    await server.startup()
    
    # Verify properties are accessible
    assert server.api_version == MCP_API_VERSION
    assert server.workspace_path == Path(".").resolve()
    
    # Test shutdown
    await server.shutdown()
    
    # All components should be cleaned up
    assert server._pipeline is None
    assert len(server._retrieval_tools) == 0


@pytest.mark.asyncio
async def test_server_status_tool():
    """Test the get_server_status diagnostic tool returns valid structured data."""
    settings = Settings()
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    await server.startup()
    
    # Call the status tool
    status_str = await server.get_server_status(client_version="0.1.0")
    status = json.loads(status_str)
    
    # Verify all required fields are present
    assert status["success"] is True
    assert "server" in status
    assert status["server"]["api_version"] == MCP_API_VERSION
    assert status["server"]["min_client_version"] == MCP_MIN_CLIENT_VERSION
    assert status["server"]["compatible"] is True
    
    assert "workspace" in status
    assert status["workspace"]["exists"] is True
    
    assert "capabilities" in status
    assert "tools" in status
    assert "total_registered" in status["tools"]
    assert status["tools"]["total_registered"] >= 35  # We have at least 35 tools
    
    assert "timestamp" in status
    
    await server.shutdown()


@pytest.mark.asyncio
async def test_version_compatibility_check():
    """Test that client version compatibility is correctly validated."""
    settings = Settings()
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    await server.startup()
    
    # Test incompatible client version
    status_str = await server.get_server_status(client_version="0.0.1")
    status = json.loads(status_str)
    assert status["server"]["compatible"] is False
    assert "incompatible" in status["server"]["version_warning"]
    
    # Test invalid version string
    status_str = await server.get_server_status(client_version="invalid")
    status = json.loads(status_str)
    assert "Could not parse client version" in status["server"]["version_warning"]
    
    await server.shutdown()


def test_authentication_disabled_by_default():
    """Test that authentication is disabled by default for stdio transport."""
    settings = Settings()
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    
    # Auth should be disabled by default
    assert server._auth_enabled is False
    auth_ok, _ = server.validate_authentication()
    assert auth_ok is True


def test_authentication_with_api_key():
    """Test that authentication works correctly when API keys are configured."""
    settings = Settings(
        tracera_mcp_api_key="test-secret-key-123",
        tracera_mcp_enforce_workspace_boundaries=True
    )
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    
    # Auth should be enabled
    assert server._auth_enabled is True
    assert len(server._api_keys) == 1
    assert "test-secret-key-123" in server._api_keys
    
    # Missing headers should fail
    auth_ok, error = server.validate_authentication()
    assert auth_ok is False
    assert "no headers provided" in error
    
    # Invalid header format
    auth_ok, error = server.validate_authentication({"authorization": "wrong-format"})
    assert auth_ok is False
    assert "Invalid authorization header format" in error
    
    # Wrong API key
    auth_ok, error = server.validate_authentication({"authorization": "Bearer wrong-key"})
    assert auth_ok is False
    assert "Invalid API key" in error
    
    # Correct API key
    auth_ok, _ = server.validate_authentication({"authorization": "Bearer test-secret-key-123"})
    assert auth_ok is True


def test_workspace_boundary_enforcement():
    """Test that file access is restricted to the workspace."""
    workspace = Path(__file__).parent.parent  # tests/.. = project root
    settings = Settings(tracera_mcp_enforce_workspace_boundaries=True)
    server = TraceraMCPServer(settings=settings, workspace_path=workspace)
    
    # Path within workspace should be allowed
    allowed_path = workspace / "src" / "main.py"
    access_ok, _ = server.validate_file_access(allowed_path)
    assert access_ok is True
    
    # Path outside workspace should be blocked
    outside_path = Path("/etc/passwd")
    access_ok, error = server.validate_file_access(outside_path)
    assert access_ok is False
    assert "outside workspace is forbidden" in error
    
    # Parent directory traversal should be blocked
    traversal_path = workspace / ".." / "etc" / "passwd"
    access_ok, error = server.validate_file_access(traversal_path)
    assert access_ok is False


def test_workspace_boundaries_disabled():
    """Test that boundaries can be disabled in settings."""
    workspace = Path(__file__).parent
    settings = Settings(tracera_mcp_enforce_workspace_boundaries=False)
    server = TraceraMCPServer(settings=settings, workspace_path=workspace)
    
    # Any path should be allowed when boundaries are disabled
    outside_path = Path("/etc/passwd")
    access_ok, _ = server.validate_file_access(outside_path)
    assert access_ok is True


@pytest.mark.asyncio
async def test_all_tools_registered():
    """Verify that all expected tools are registered with the MCP server."""
    settings = Settings()
    server = TraceraMCPServer(settings=settings, workspace_path=Path("."))
    
    # Get list of tools from FastMCP
    tools = await server.mcp.list_tools()
    tool_names = [t.name for t in tools]
    
    # Core code intelligence tools should be present
    core_tools = [
        "search_code", "find_symbol", "find_references", 
        "get_blast_radius", "get_call_hierarchy", "search_ast",
        "get_repo_map"
    ]
    for tool in core_tools:
        assert tool in tool_names, f"Missing core tool: {tool}"
    
    # Memory tools should be present
    memory_tools = ["remember_memory", "recall_memory", "forget_memory"]
    for tool in memory_tools:
        assert tool in tool_names, f"Missing memory tool: {tool}"
    
    # Diagnostic tool should be present
    assert "get_server_status" in tool_names
    
    # We should have all our tools
    assert len(tools) >= 35, f"Expected at least 35 tools, found {len(tools)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])