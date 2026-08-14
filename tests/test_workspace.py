"""Tests for TRACERA workspace sandbox."""

import pytest
from pathlib import Path

from tracera.workspace.sandbox import WorkspaceSandbox
from tracera.errors import PathTraversalError, FileNotFoundInWorkspaceError


@pytest.fixture
def workspace(tmp_path):
    return WorkspaceSandbox(tmp_path)


# ── Path resolution ───────────────────────────────────────────────────────────

def test_resolve_relative(workspace, tmp_path):
    resolved = workspace.resolve("foo/bar.txt")
    assert resolved == tmp_path / "foo" / "bar.txt"


def test_resolve_absolute_within(workspace, tmp_path):
    resolved = workspace.resolve(str(tmp_path / "sub" / "file.py"))
    assert resolved == tmp_path / "sub" / "file.py"


def test_resolve_traversal_rejected(workspace, tmp_path):
    with pytest.raises(PathTraversalError):
        workspace.resolve("../../etc/passwd")


def test_resolve_absolute_outside_rejected(workspace, tmp_path):
    with pytest.raises(PathTraversalError):
        workspace.resolve("/etc/passwd")


# ── Read / Write ──────────────────────────────────────────────────────────────

def test_read_write_sync(workspace, tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("hello world")
    content = workspace.read_text_sync("hello.txt")
    assert content == "hello world"


def test_write_creates_dirs(workspace, tmp_path):
    workspace.write_text_sync("a/b/c/file.txt", "nested")
    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


def test_read_missing_file(workspace):
    with pytest.raises(FileNotFoundInWorkspaceError):
        workspace.read_text_sync("does_not_exist.txt")


@pytest.mark.asyncio
async def test_async_read_write(workspace, tmp_path):
    await workspace.write_text("async_file.txt", "async content")
    content = await workspace.read_text("async_file.txt")
    assert content == "async content"


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_file(workspace, tmp_path):
    await workspace.write_text("to_delete.txt", "bye")
    assert workspace.exists("to_delete.txt")
    await workspace.delete("to_delete.txt")
    assert not workspace.exists("to_delete.txt")


@pytest.mark.asyncio
async def test_delete_missing_raises(workspace):
    with pytest.raises(FileNotFoundInWorkspaceError):
        await workspace.delete("nope.txt")


# ── Edit ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_text(workspace):
    await workspace.write_text("code.py", "def foo():\n    pass\n")
    n = await workspace.edit_text("code.py", "    pass", "    return 42")
    assert n == 1
    content = await workspace.read_text("code.py")
    assert "return 42" in content
    assert "pass" not in content


@pytest.mark.asyncio
async def test_edit_text_not_found(workspace):
    await workspace.write_text("code.py", "hello world")
    from tracera.errors import WorkspaceError
    with pytest.raises(WorkspaceError, match="not found"):
        await workspace.edit_text("code.py", "MISSING TEXT", "replacement")


# ── List directory ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_directory(workspace, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "README.md").write_text("y")

    entries = await workspace.list_directory(".", max_depth=2)
    names = [e.relative.name for e in entries]
    assert "README.md" in names
    assert "src" in names
    assert "main.py" in names


# ── Grep ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grep_finds_pattern(workspace, tmp_path):
    (tmp_path / "app.py").write_text("def authenticate(user):\n    pass\n")
    (tmp_path / "other.py").write_text("x = 1\n")

    results = await workspace.grep("authenticate")
    assert len(results) == 1
    assert results[0]["file"] == "app.py"
    assert results[0]["line"] == 1


@pytest.mark.asyncio
async def test_grep_no_results(workspace, tmp_path):
    (tmp_path / "empty.py").write_text("hello")
    results = await workspace.grep("NOTFOUND")
    assert results == []


@pytest.mark.asyncio
async def test_grep_case_insensitive(workspace, tmp_path):
    (tmp_path / "test.py").write_text("AUTH_TOKEN = 'abc'\n")
    results = await workspace.grep("auth_token", case_insensitive=True)
    assert len(results) == 1
