"""
Tests for the git tool (Phase 3 → agent) and incremental-index deletion
cleanup (Phase 24).
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from tracera.workspace.sandbox import WorkspaceSandbox


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=tester", "-c", "user.email=tester@example.com", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
    )


def test_git_tool_status_and_diff(tmp_path: Path):
    """GitTool reports status and diff for the workspace repo."""
    from tracera.tools.git_tool import GitTool

    _git(tmp_path, "init")
    (tmp_path / "main.py").write_text("print('hello')\n")
    _git(tmp_path, "add", "main.py")
    _git(tmp_path, "commit", "-m", "initial")
    (tmp_path / "main.py").write_text("print('hello world')\n")

    tool = GitTool(WorkspaceSandbox(tmp_path))

    async def _run():
        r = await tool.execute(operation="status")
        assert r.success, r.error
        assert "Branch:" in r.output
        assert "main.py" in r.output  # modified file listed

        r2 = await tool.execute(operation="diff")
        assert r2.success
        assert "hello world" in r2.output

        r3 = await tool.execute(operation="log", max_count=5)
        assert r3.success
        assert "initial" in r3.output

        r4 = await tool.execute(operation="branch")
        assert r4.success

    asyncio.run(_run())


def test_git_tool_not_a_repo(tmp_path: Path):
    """GitTool fails gracefully outside a git repository."""
    from tracera.tools.git_tool import GitTool

    tool = GitTool(WorkspaceSandbox(tmp_path))

    async def _run():
        r = await tool.execute(operation="status")
        assert not r.success
        assert "git repository" in r.error.lower()

    asyncio.run(_run())


def test_incremental_deletion_cleans_bm25(tmp_path: Path):
    """Deleting a file removes its chunks from BM25 (Phase 24)."""

    class FakeEmbedder:
        dimension = 8

        def embed_batch(self, texts):
            return [[0.1] * 8 for _ in texts]

    class FakeVectorStore:
        def __init__(self):
            self.deleted: list[str] = []

        def upsert_chunks(self, chunks, embeddings):
            pass

        def delete_by_file(self, file_path):
            self.deleted.append(file_path)

    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.incremental import IncrementalIndexer
    from tracera.graph.symbol_graph import SymbolGraph

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "keep.py").write_text("def keep_fn():\n    return 1\n")
    (ws / "drop.py").write_text("def drop_fn():\n    return 2\n")

    bm25 = BM25Index()
    indexer = IncrementalIndexer(
        workspace_root=ws,
        bm25_index=bm25,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        index_dir=tmp_path / "idx",
        symbol_graph=SymbolGraph(),
    )
    indexer.run()
    before = bm25.doc_count
    assert before >= 2

    # Delete drop.py and re-run
    (ws / "drop.py").unlink()
    indexer.run()

    assert bm25.doc_count < before
    # The dropped file's chunks are gone from BM25 (search its unique token)
    assert bm25.search("drop", k=10) == []
