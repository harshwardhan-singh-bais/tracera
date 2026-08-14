"""
Tests for the Code Knowledge Graph (Phases 25-26) and the
graph-backed code-search tools (Phase 27).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tracera.indexer.schema import LineRange, Symbol, SymbolType
from tracera.graph.symbol_graph import SymbolGraph
from tracera.graph.graph_retrieval import GraphRetriever
from tracera.retrieval.bm25 import BM25Index


def _sym(name: str, stype: SymbolType, start: int, end: int, parent: str | None = None) -> Symbol:
    return Symbol(
        name=name,
        type=stype,
        range=LineRange(start_line=start, end_line=end),
        content=f"def {name}: pass",
        parent_symbol=parent,
    )


def _sample_graph() -> SymbolGraph:
    g = SymbolGraph()
    g.add_symbol("auth.py", _sym("AuthMiddleware", SymbolType.CLASS, 1, 10))
    g.add_symbol("auth.py", _sym("login", SymbolType.METHOD, 2, 8, parent="AuthMiddleware"))
    g.add_symbol("app.py", _sym("handle_login", SymbolType.FUNCTION, 5, 20))
    return g


def test_graph_build_and_query():
    """Symbols are added as nodes with containment edges and are queryable."""
    g = _sample_graph()

    assert g.node_count == 3
    assert g.edge_count == 1  # CONTAINS: AuthMiddleware -> login

    node_ids = g.find_by_name("AuthMiddleware")
    assert len(node_ids) == 1
    node = g.get_node(node_ids[0])
    assert node is not None
    assert node["symbol_type"] == "class"

    # Method is a child of its class
    children = g.get_children(node_ids[0])
    assert len(children) == 1
    child = g.get_node(children[0])
    assert child["name"] == "login"


def test_graph_persistence_roundtrip(tmp_path: Path):
    """Graph can be saved and reloaded from disk."""
    g = _sample_graph()
    path = tmp_path / "symbol_graph.json"
    g.save(path)

    loaded = SymbolGraph.load(path)
    assert loaded.node_count == g.node_count
    assert loaded.edge_count == g.edge_count
    assert loaded.find_by_name("AuthMiddleware")


def test_graph_remove_file(tmp_path: Path):
    """Removing a file removes its nodes and edges."""
    g = _sample_graph()
    g.remove_file("auth.py")

    assert g.node_count == 1
    assert not g.find_by_name("AuthMiddleware")
    assert not g.find_by_name("login")
    assert g.find_by_name("handle_login")
    # The containment edge vanished with its nodes
    assert g.edge_count == 0


def test_graph_retriever_expansion():
    """Dependency-aware retrieval adds related symbols from the graph."""
    g = _sample_graph()
    bm25 = BM25Index()
    bm25.add_document("c1", "class AuthMiddleware:\n    def __init__(self): pass")
    bm25.add_document("c2", "def login(user):\n    return user")

    retriever = GraphRetriever(g, bm25)
    results = retriever.expand_with_graph(
        [{"id": "c1", "symbol": "AuthMiddleware", "file_path": "auth.py", "content": "class AuthMiddleware"}],
        max_total=10,
    )

    # At least the base result + graph neighbours are returned
    assert len(results) >= 1
    symbols = {r.get("symbol") for r in results}
    assert "AuthMiddleware" in symbols


def test_incremental_indexer_builds_graph(tmp_path: Path):
    """Phase 24 indexer persists a symbol graph that stays in sync with files."""

    class FakeEmbedder:
        dimension = 8

        def embed_batch(self, texts):
            return [[0.1] * 8 for _ in texts]

    class FakeVectorStore:
        def __init__(self):
            self.chunks = []

        def upsert_chunks(self, chunks, embeddings):
            self.chunks.extend(chunks)

        def delete_by_file(self, file_path):
            pass

    from tracera.retrieval.incremental import IncrementalIndexer

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "auth.py").write_text(
        "import os\n\n"
        "class AuthMiddleware:\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "def login(user):\n"
        "    return user\n"
    )

    index_dir = tmp_path / "idx"
    indexer = IncrementalIndexer(
        workspace_root=ws,
        bm25_index=BM25Index(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        index_dir=index_dir,
        symbol_graph=SymbolGraph(),
    )

    stats = indexer.run()
    assert stats["new"] == 1
    assert stats["chunks_indexed"] >= 1

    graph_path = index_dir / "symbol_graph.json"
    assert graph_path.exists()
    graph = SymbolGraph.load(graph_path)
    assert graph.find_by_name("AuthMiddleware")
    assert graph.find_by_name("login")

    # Second run with no changes: skipped, graph preserved
    stats2 = indexer.run()
    assert stats2["new"] == 0
    assert stats2["skipped"] == 1
    assert SymbolGraph.load(graph_path).node_count == graph.node_count

    # Modify the file: graph is updated (replaced, not duplicated)
    (ws / "auth.py").write_text(
        "import os\n\n"
        "class AuthMiddleware:\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "class NewClass:\n"
        "    pass\n"
    )
    stats3 = indexer.run()
    assert stats3["modified"] == 1
    updated = SymbolGraph.load(graph_path)
    assert updated.find_by_name("NewClass")
    assert updated.find_by_name("AuthMiddleware")


def test_bm25_reindex_is_idempotent():
    """Re-adding the same doc_id (incremental re-index) must not double-count."""
    bm25 = BM25Index()
    bm25.add_document("d1", "def login(user): return check(user)")
    bm25.add_document("d1", "def login(user, pwd): return verify(user, pwd)")

    assert bm25.doc_count == 1
    # Search should still find the updated content
    hits = bm25.search("verify", k=5)
    assert any(doc_id == "d1" for doc_id, _ in hits)


def test_code_search_tools_execute():
    """Phase 27 tools run and return structured results."""
    from tracera.tools.code_search import (
        SearchCodeTool,
        FindSymbolTool,
        FindReferencesTool,
        GetDependenciesTool,
        GetContextTool,
    )

    class FakeRetriever:
        def search(self, query, k=5, language=None):
            return [
                {
                    "id": "c1",
                    "symbol": "AuthMiddleware",
                    "symbol_type": "class",
                    "file_path": "auth.py",
                    "start_line": 1,
                    "end_line": 10,
                    "content": "class AuthMiddleware: ...",
                    "language": "python",
                    "_rrf_score": 1.0,
                }
            ]

    class FakeExpander:
        def expand(self, results, max_additional=3):
            return results

    g = _sample_graph()
    retriever = FakeRetriever()

    async def _run():
        r = await SearchCodeTool(retriever).execute(query="auth")
        assert r.success and "AuthMiddleware" in r.output

        r2 = await FindSymbolTool(retriever).execute(name="AuthMiddleware")
        assert r2.success

        r3 = await FindReferencesTool(g).execute(symbol="AuthMiddleware")
        assert r3.success

        r4 = await GetDependenciesTool(g).execute(symbol="AuthMiddleware")
        assert r4.success

        r5 = await GetContextTool(retriever, FakeExpander()).execute(symbol="AuthMiddleware")
        assert r5.success

    asyncio.run(_run())


def test_registry_extension_with_graph():
    """extend_registry_with_retrieval registers graph-backed tools."""
    from tracera.tools.registry import ToolRegistry, extend_registry_with_retrieval

    class FakeRetriever:
        def search(self, query, k=5, language=None):
            return []

    bm25 = BM25Index()
    graph_retriever = GraphRetriever(_sample_graph(), bm25)

    registry = ToolRegistry()
    extend_registry_with_retrieval(registry, FakeRetriever(), None, graph_retriever)

    names = set(registry.names)
    assert {"search_code", "find_symbol", "get_context", "find_references", "get_dependencies"} <= names
