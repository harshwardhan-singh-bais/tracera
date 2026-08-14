"""Tests for TRACERA coding tools."""

import pytest
from pathlib import Path

from tracera.workspace.sandbox import WorkspaceSandbox
from tracera.tools.read_file import ReadFileTool
from tracera.tools.write_file import WriteFileTool
from tracera.tools.edit_file import EditFileTool
from tracera.tools.list_dir import ListDirTool
from tracera.tools.grep import GrepTool
from tracera.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture
def workspace(tmp_path):
    return WorkspaceSandbox(tmp_path)


@pytest.fixture
def registry(workspace):
    return create_default_registry(workspace)


# ── ReadFileTool ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_ok(workspace, tmp_path):
    (tmp_path / "hello.py").write_text("print('hello')\n")
    tool = ReadFileTool(workspace)
    result = await tool.execute(path="hello.py")
    assert result.success
    assert "print('hello')" in result.output


@pytest.mark.asyncio
async def test_read_file_with_lines(workspace, tmp_path):
    (tmp_path / "multi.py").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadFileTool(workspace)
    result = await tool.execute(path="multi.py", start_line=2, end_line=3)
    assert result.success
    assert "line2" in result.output
    assert "line1" not in result.output


@pytest.mark.asyncio
async def test_read_file_missing(workspace):
    tool = ReadFileTool(workspace)
    result = await tool.execute(path="missing.py")
    assert not result.success
    assert "ERROR" in result.output


# ── WriteFileTool ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_creates(workspace, tmp_path):
    tool = WriteFileTool(workspace)
    result = await tool.execute(path="new_file.py", content="x = 1\n")
    assert result.success
    assert (tmp_path / "new_file.py").exists()
    assert (tmp_path / "new_file.py").read_text() == "x = 1\n"


@pytest.mark.asyncio
async def test_write_file_updates(workspace, tmp_path):
    (tmp_path / "existing.py").write_text("old")
    tool = WriteFileTool(workspace)
    result = await tool.execute(path="existing.py", content="new")
    assert result.success
    assert "updated" in result.output


# ── EditFileTool ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_file_ok(workspace, tmp_path):
    (tmp_path / "app.py").write_text("def greet():\n    pass\n")
    tool = EditFileTool(workspace)
    result = await tool.execute(
        path="app.py",
        old_text="    pass",
        new_text="    return 'hello'",
    )
    assert result.success
    content = (tmp_path / "app.py").read_text()
    assert "return 'hello'" in content


@pytest.mark.asyncio
async def test_edit_file_not_found(workspace, tmp_path):
    (tmp_path / "app.py").write_text("hello world")
    tool = EditFileTool(workspace)
    result = await tool.execute(
        path="app.py",
        old_text="MISSING",
        new_text="replacement",
    )
    assert not result.success


# ── ListDirTool ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_dir_ok(workspace, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "README.md").write_text("y")
    tool = ListDirTool(workspace)
    result = await tool.execute(path=".", max_depth=2)
    assert result.success
    assert "README.md" in result.output
    assert "src" in result.output


# ── GrepTool ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grep_tool_finds(workspace, tmp_path):
    (tmp_path / "auth.py").write_text("class AuthMiddleware:\n    pass\n")
    tool = GrepTool(workspace)
    result = await tool.execute(pattern="AuthMiddleware")
    assert result.success
    assert "auth.py" in result.output
    assert "AuthMiddleware" in result.output


@pytest.mark.asyncio
async def test_grep_tool_no_results(workspace, tmp_path):
    (tmp_path / "code.py").write_text("x = 1")
    tool = GrepTool(workspace)
    result = await tool.execute(pattern="NOTEXIST")
    assert result.success
    assert "No matches" in result.output


# ── Tool Registry ─────────────────────────────────────────────────────────────

def test_registry_has_all_tools(registry):
    names = registry.names
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "list_dir" in names
    assert "grep" in names
    assert "run_command" in names
    assert "git" in names


def test_registry_schemas(registry):
    schemas = registry.schemas()
    assert len(schemas) == 7
    for s in schemas:
        assert s.name
        assert s.description
        assert "type" in s.parameters


@pytest.mark.asyncio
async def test_registry_execute_tool(registry, tmp_path):
    (tmp_path / "test.py").write_text("hello")
    result = await registry.execute("read_file", "call-1", {"path": "test.py"})
    assert result.success
    assert "hello" in result.output


def test_registry_tool_not_found(registry):
    from tracera.errors import ToolNotFoundError
    with pytest.raises(ToolNotFoundError):
        registry.get("nonexistent_tool")
