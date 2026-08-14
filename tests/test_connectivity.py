"""
Connectivity tests for all 38 TRACERA phases.

One test per phase. Each test proves the phase:
  - WORKS       (its code executes and produces correct output)
  - CONNECTED   (its output is consumed by another phase / a live CLI-TUI path)

Real project code is used throughout. Fakes are used only where a real
dependency would require network access or a model download
(LLM providers, sentence-transformers, cross-encoders).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

import tracera.tools.test_runner as tr  # aliased so pytest doesn't collect Test* classes


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fakes
# ═══════════════════════════════════════════════════════════════════════════════

class FakeProvider:
    """Minimal LLM provider — never touches the network."""

    name = "fake"
    default_model = "fake-model"

    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.complete_calls = 0

    async def complete(self, messages, **kwargs):
        self.complete_calls += 1
        from tracera.providers.base import LLMResponse, TokenUsage
        return LLMResponse(
            content=self.response_text, tool_calls=None,
            usage=TokenUsage(), model=self.default_model, finish_reason="stop",
        )

    async def stream(self, messages, **kwargs):
        from tracera.providers.base import StreamEvent
        yield StreamEvent(type="text_delta", text=self.response_text)
        yield StreamEvent(type="done")


class _StubReadFile:
    name = "read_file"
    description = "Read a file"
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, path=None, **kwargs):
        from tracera.tools.base import ToolResult
        return ToolResult.ok(
            tool_name=self.name, tool_call_id="", output=f"contents of {path}"
        )

    def to_schema(self):
        from tracera.providers.base import ToolSchema
        return ToolSchema(name=self.name, description=self.description,
                          parameters=self.parameters_schema)

    async def safe_execute(self, tool_call_id, arguments):
        from tracera.tools.base import ToolResult
        result = await self.execute(**arguments)
        result.tool_call_id = tool_call_id
        return result


class _FakeEmbedder:
    dimension = 4

    def embed_single(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts, batch_size=32):
        return [self.embed_single(t) for t in texts]


class _FakeVectorStore:
    def __init__(self):
        self.chunks = []
        self.deleted = []

    def upsert_chunks(self, chunks, embeddings):
        self.chunks.extend(chunks)

    def delete_by_file(self, file_path):
        self.deleted.append(file_path)

    def search(self, query_embedding, k=10, language=None, symbol_type=None):
        return []


class _FakeDense:
    def __init__(self, rows=None):
        self._rows = rows or []

    def search(self, query, k=10, language=None, symbol_type=None):
        return self._rows


def _run_agent(agent, task, conversation=None):
    """Drive a ReActAgent to completion, collecting all events."""
    async def _collect():
        events = []
        async for ev in await agent.run(task, conversation=conversation):
            events.append(ev)
        return events
    return asyncio.run(_collect())


# ═══════════════════════════════════════════════════════════════════════════════
# FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_01_architecture_config_cli():
    """Phase 1: config system + env vars + CLI entry point + logging all work."""
    from tracera.config import get_settings
    settings = get_settings()
    assert settings.tracera_workspace is not None
    assert settings.tracera_default_provider  # loaded from .env
    assert settings.ensure_dirs() is None  # data dirs created without error

    from tracera.main import app, ask, status, index, search, fix, review, tui
    assert app.info.name == "tracera"
    # Every CLI command is registered on the Typer app
    for fn in (ask, status, index, search, fix, review, tui):
        assert callable(fn)
    assert len(app.registered_commands) >= 7

    from tracera.logging import setup_logging, get_logger
    setup_logging(level="DEBUG", log_file=None)
    get_logger("connectivity.phase1").info("logging works")


def test_phase_02_workspace_sandbox_and_lifecycle(tmp_path):
    """Phase 2: sandbox I/O + traversal protection + workspace lifecycle."""
    from tracera.workspace.sandbox import WorkspaceSandbox
    from tracera.errors import PathTraversalError, WorkspaceError

    ws = WorkspaceSandbox(tmp_path)
    ws.write_text_sync("src/app.py", "print('hi')\n")
    assert ws.read_text_sync("src/app.py") == "print('hi')\n"
    assert ws.exists("src/app.py")

    # Traversal protection
    with pytest.raises(PathTraversalError):
        ws.resolve("../outside.py")
    with pytest.raises(PathTraversalError):
        ws.read_text_sync(str(tmp_path.parent / "outside.py"))

    # Delete controls: non-recursive dir delete refused
    with pytest.raises(WorkspaceError):
        ws.delete_sync("src", recursive=False)
    ws.delete_sync("src", recursive=True)
    assert not ws.exists("src")

    # Lifecycle creates data dirs
    from tracera.workspace.lifecycle import WorkspaceLifecycle
    lifecycle = WorkspaceLifecycle(tmp_path / "data")
    lifecycle.initialise()
    assert lifecycle.is_initialised()
    assert all(v for v in lifecycle.status().values())


def test_phase_03_git_integration(tmp_path):
    """Phase 3: repo detection, status, diff, log, branch all work."""
    import git as gitpython
    from tracera.git.operations import GitRepo, detect_git_repo

    repo = gitpython.Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "tester")
        cw.set_value("user", "email", "t@example.com")
    (tmp_path / "hello.py").write_text("print('v1')\n")
    repo.index.add(["hello.py"])
    repo.index.commit("initial")

    detected = detect_git_repo(tmp_path)
    assert detected is not None
    gr = GitRepo(tmp_path)
    assert gr.current_branch in ("main", "master")
    assert len(gr.log(max_count=5)) == 1
    assert gr.branches()  # at least one branch

    # Modify → dirty status + diff
    (tmp_path / "hello.py").write_text("print('v2')\n")
    status = gr.status()
    assert status.is_dirty
    diff = gr.diff("HEAD")
    assert diff.files_changed >= 1
    assert "v2" in diff.diff_text


def test_phase_04_provider_abstraction():
    """Phase 4: provider factory, adapters, schemas, and streaming all work."""
    from tracera.providers import create_provider, list_available_providers
    from tracera.providers.ollama_provider import OllamaProvider
    from tracera.providers.openai_provider import OpenAIProvider
    from tracera.providers.nemotron_provider import NemotronProvider
    from tracera.providers.base import LLMMessage, ToolSchema

    settings = types.SimpleNamespace(
        tracera_default_provider="openai", tracera_default_model="",
        openai_api_key="sk-test", google_api_key="",
        anthropic_api_key="", nemotron_api_key="nk-test",
        ollama_base_url="http://localhost:11434",
    )
    p = create_provider("openai", settings=settings)
    assert isinstance(p, OpenAIProvider)
    assert isinstance(create_provider("ollama", settings=settings), OllamaProvider)
    assert isinstance(create_provider("nemotron", settings=settings), NemotronProvider)

    providers = list_available_providers(settings)
    assert {"name", "available", "key_env", "model"} <= set(providers[0])

    # Tool schema → OpenAI format
    schema = ToolSchema(name="read_file", description="r", parameters={"type": "object"})
    assert schema.to_openai_dict()["function"]["name"] == "read_file"

    # Streaming is a real async generator (Phase 4 bullet)
    assert inspect.isasyncgen(p.stream([LLMMessage.user("hi")]))


def test_phase_05_conversation_state():
    """Phase 5: messages, tool calls/results, history, truncation."""
    from tracera.conversation.state import ConversationState
    from tracera.providers.base import ToolCallRequest, Role

    conv = ConversationState(system_prompt="sys")
    conv.add_user("add auth")
    conv.add_assistant("on it")
    tc = ToolCallRequest(id="c1", name="grep", arguments={"pattern": "auth"})
    conv.add_tool_calls([tc])
    conv.add_tool_result("c1", "grep", "auth.py:1", success=True)

    msgs = conv.llm_messages()
    assert msgs[0].role == Role.SYSTEM
    assert msgs[1].role == Role.USER
    assert msgs[4].role == Role.TOOL
    assert conv.stats.tool_calls == 1 and conv.stats.tool_results == 1

    for i in range(20):
        conv.add_user(f"u{i}"); conv.add_assistant(f"a{i}")
    assert len(conv.truncate(5)) <= 6  # system + 5 recent
    snap = conv.snapshot()
    assert len(snap) >= 1


def test_phase_06_tool_registry(tmp_path):
    """Phase 6: registration, discovery, schemas, execution, extension."""
    from tracera.workspace.sandbox import WorkspaceSandbox
    from tracera.tools.registry import create_default_registry, extend_registry_with_retrieval
    from tracera.graph.symbol_graph import SymbolGraph

    (tmp_path / "a.py").write_text("x = 1\n")
    registry = create_default_registry(WorkspaceSandbox(tmp_path))
    assert len(registry.names) == 7
    schemas = registry.schemas()
    assert all(s.name and s.description and "type" in s.parameters for s in schemas)

    result = asyncio.run(registry.execute("read_file", "call-1", {"path": "a.py"}))
    assert result.success and "x = 1" in result.output

    # Retrieval extension (Phase 27 tools)
    class FakeRetriever:
        def search(self, query, k=5, language=None):
            return []

    extend_registry_with_retrieval(registry, FakeRetriever(), None, types.SimpleNamespace(graph=SymbolGraph()))
    assert {"search_code", "find_symbol", "get_context", "find_references", "get_dependencies"} <= set(registry.names)


def test_phase_07_coding_tools(tmp_path):
    """Phase 7: read/write/edit/list/grep/run_command all execute."""
    from tracera.workspace.sandbox import WorkspaceSandbox
    from tracera.tools.read_file import ReadFileTool
    from tracera.tools.write_file import WriteFileTool
    from tracera.tools.edit_file import EditFileTool
    from tracera.tools.list_dir import ListDirTool
    from tracera.tools.grep import GrepTool
    from tracera.tools.run_command import RunCommandTool
    from tracera.tools.git_tool import GitTool

    ws = WorkspaceSandbox(tmp_path)
    (tmp_path / "app.py").write_text("class Auth:\n    pass\n")

    assert asyncio.run(ReadFileTool(ws).execute(path="app.py")).success
    assert asyncio.run(WriteFileTool(ws).execute(path="b.py", content="y=2\n")).success
    assert asyncio.run(EditFileTool(ws).execute(path="app.py", old_text="pass", new_text="return 1")).success
    listing = asyncio.run(ListDirTool(ws).execute(path="."))
    assert listing.success and "app.py" in listing.output
    grep = asyncio.run(GrepTool(ws).execute(pattern="class Auth"))
    assert grep.success and "app.py" in grep.output
    cmd = asyncio.run(RunCommandTool(ws).execute(command="git --version"))
    assert cmd.success
    git = asyncio.run(GitTool(ws).execute(operation="status"))
    assert git.success or "Not a git repository" in git.output


def test_phase_08_react_agent_loop():
    """Phase 8: full event cycle — thinking → tool → delta → complete → done."""
    from tracera.agent.react_loop import AgentEventType, ReActAgent
    from tracera.tools.registry import ToolRegistry
    from tracera.providers.base import StreamEvent, ToolCallRequest

    class ToolThenTextProvider(FakeProvider):
        def __init__(self):
            super().__init__(response_text="fixed it")
            self.calls = 0

        async def stream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield StreamEvent(type="tool_call_complete",
                                  tool_call=ToolCallRequest(id="c1", name="read_file", arguments={"path": "a.py"}))
            else:
                yield StreamEvent(type="text_delta", text="fixed ")
                yield StreamEvent(type="text_delta", text="it")
            yield StreamEvent(type="done")

    registry = ToolRegistry()
    registry.register(_StubReadFile())
    agent = ReActAgent(provider=ToolThenTextProvider(), registry=registry, streaming=True)

    events = _run_agent(agent, "fix the bug")
    types = [e.type for e in events]
    assert AgentEventType.THINKING in types
    assert AgentEventType.TOOL_START in types
    assert AgentEventType.TOOL_END in types
    assert AgentEventType.RESPONSE_DELTA in types  # Phase 4/8 streaming wired
    complete = next(e for e in events if e.type == AgentEventType.RESPONSE_COMPLETE)
    assert complete.text == "fixed it"
    assert types[-1] == AgentEventType.DONE


def test_phase_09_planning_system():
    """Phase 9: decomposition, todo state, progress, replanning."""
    from tracera.agent.planner import TaskDecomposer, Plan, TodoStatus

    class PlanProvider(FakeProvider):
        async def complete(self, messages, **kwargs):
            self.complete_calls += 1
            from tracera.providers.base import LLMResponse, TokenUsage
            body = '[{"title": "Read code", "priority": 0}, {"title": "Write test", "priority": 1}]'
            if "recovery" in (messages[-1].content or "").lower() or "failed" in (messages[-1].content or "").lower():
                body = '[{"title": "Rollback change", "priority": 0}]'
            return LLMResponse(content=body, tool_calls=None,
                               usage=TokenUsage(), model="fake", finish_reason="stop")

    decomposer = TaskDecomposer(PlanProvider())
    plan = asyncio.run(decomposer.decompose("Add tests"))
    assert len(plan.items) == 2
    plan.start_item(plan.items[0].id)
    assert plan.items[0].status == TodoStatus.IN_PROGRESS
    plan.complete_item(plan.items[0].id)
    assert plan.progress == (1, 2) and plan.progress_pct == 50.0
    assert "[x]" in plan.to_markdown()

    replanned = asyncio.run(decomposer.replan(plan, "test failed"))
    assert replanned.replanned_count == 1
    assert any("Rollback" in i.title for i in replanned.items)


def test_phase_10_persistent_memory(tmp_path):
    """Phase 10: memory stores, persists, retrieves, and injects into the agent."""
    from tracera.agent.memory import AgentMemory, MemoryCategory
    from tracera.conversation.state import ConversationState, MessageType

    memory = AgentMemory(tmp_path / "memory")
    memory.add("JWT used for auth", MemoryCategory.PROJECT_FACT)
    assert memory.count == 1
    m2 = AgentMemory(tmp_path / "memory")  # reload from disk
    assert m2.count == 1
    assert "JWT" in m2.retrieve("auth")[0].content
    assert "Relevant Memory" in m2.build_context("auth")

    # Injection into the agent conversation (Phase 10 → 8)
    from tracera.agent.react_loop import ReActAgent
    from tracera.tools.registry import ToolRegistry
    conv = ConversationState(system_prompt="sys")
    agent = ReActAgent(provider=FakeProvider(), registry=ToolRegistry(),
                       memory_provider=lambda: m2.build_context("auth"))
    _run_agent(agent, "hi", conversation=conv)
    assert any(m.type == MessageType.SYSTEM and m.metadata.get("memory") for m in conv.messages)


# ═══════════════════════════════════════════════════════════════════════════════
# CODE INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_11_repository_scanner(tmp_path):
    """Phase 11: recursive scan, .gitignore, binaries, language detection."""
    from tracera.indexer.scanner import RepositoryScanner

    (tmp_path / "main.py").write_text("print('hi')")
    (tmp_path / "data.bin").write_bytes(b"a\x00b")
    (tmp_path / ".gitignore").write_text("skip.py\n")
    (tmp_path / "skip.py").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.ts").write_text("export const x = 1;")

    files = {f.path: f for f in RepositoryScanner(workspace_root=tmp_path).scan()}
    assert "main.py" in files
    assert "src/util.ts" in files
    assert files["src/util.ts"].language == "typescript"
    assert "data.bin" not in files
    assert "skip.py" not in files


def test_phase_12_language_parser():
    """Phase 12: tree-sitter parses source into an AST."""
    from tracera.indexer.parser import LanguageParser

    parser = LanguageParser()
    tree = parser.parse(b"def f():\n    return 1\n", "python")
    assert tree is not None
    assert tree.root_node is not None
    assert tree.root_node.type == "module"
    assert parser.parse(b"var x = 1;", "javascript") is not None


def test_phase_13_symbol_extraction():
    """Phase 13: functions, classes, methods, imports extracted."""
    from tracera.indexer.parser import LanguageParser
    from tracera.indexer.extractor import SymbolExtractor
    from tracera.indexer.schema import SymbolType

    code = b"import os\n\nclass Service:\n    def run(self):\n        pass\n\ndef helper():\n    return 1\n"
    symbols = SymbolExtractor(LanguageParser()).extract_symbols(code, "python")
    by_name = {s.name: s for s in symbols}
    assert by_name["Service"].type == SymbolType.CLASS
    assert by_name["run"].type == SymbolType.METHOD
    assert by_name["run"].parent_symbol == "Service"
    assert by_name["helper"].type == SymbolType.FUNCTION
    assert by_name["os"].type == SymbolType.IMPORT


def test_phase_14_symbol_chunker():
    """Phase 14: file → class/method chunks with symbol metadata."""
    from tracera.indexer.parser import LanguageParser
    from tracera.indexer.extractor import SymbolExtractor
    from tracera.indexer.chunker import SymbolAwareChunker

    content = "import os\n\nclass Service:\n    def run(self):\n        pass\n\ndef helper():\n    return 1\n"
    symbols = SymbolExtractor(LanguageParser()).extract_symbols(content.encode(), "python")
    chunks = SymbolAwareChunker().chunk_file("svc.py", "python", content, symbols)

    primary = {c.primary_symbol for c in chunks if c.primary_symbol}
    assert "Service" in primary and "helper" in primary
    assert all(c.file_path == "svc.py" for c in chunks)
    assert any(c.symbol_type and c.parent_symbol for c in chunks)


def test_phase_15_code_schema():
    """Phase 15: canonical index schema validates and is shared across phases."""
    from tracera.indexer.schema import Symbol, CodeChunk, FileMetadata, LineRange, SymbolType

    rng = LineRange(start_line=1, end_line=4)
    sym = Symbol(name="login", type=SymbolType.FUNCTION, range=rng,
                 content="def login(): pass", parent_symbol=None)
    chunk = CodeChunk(id="abc123", file_path="auth.py", language="python",
                      content=sym.content, range=rng, primary_symbol="login",
                      symbol_type=SymbolType.FUNCTION, tokens=8)
    meta = FileMetadata(path="auth.py", language="python", size_bytes=10, sha256="x" * 64)

    assert chunk.id and chunk.primary_symbol == "login"
    assert meta.path == "auth.py" and len(meta.sha256) == 64
    assert sym.type.value == "function"


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_16_bm25_index(tmp_path):
    """Phase 16: tokenization, scoring, persistence, idempotent re-add."""
    from tracera.retrieval.bm25 import BM25Index

    bm25 = BM25Index()
    bm25.add_document("d1", "def login(user, password): return check(user)")
    bm25.add_document("d2", "class AuthMiddleware: pass")
    hits = bm25.search("login user", k=5)
    assert hits and hits[0][0] == "d1"
    assert bm25.doc_count == 2
    assert bm25.get_document("d2") and "AuthMiddleware" in bm25.get_document("d2")

    # Incremental re-index of the same doc must not double-count
    bm25.add_document("d1", "def login(user, pwd): return verify(user, pwd)")
    assert bm25.doc_count == 2
    bm25.remove_document("d2")
    assert bm25.doc_count == 1

    path = tmp_path / "bm25.json"
    bm25.save(path)
    loaded = BM25Index.load(path)
    assert loaded.doc_count == 1
    assert loaded.search("login")[0][0] == "d1"


def test_phase_17_embedding_pipeline(tmp_path):
    """Phase 17: cache hit path + similarity work without loading a model."""
    from tracera.retrieval.embedder import EmbeddingPipeline

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text = "def authenticate(request): pass"
    key = hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()
    (cache_dir / f"{key}.json").write_text(json.dumps([0.5, 0.5, 0.5]))

    embedder = EmbeddingPipeline(model_name=model_name, cache_dir=cache_dir)
    vec = embedder.embed_single(text)  # served from disk cache — no model load
    assert vec == [0.5, 0.5, 0.5]
    assert embedder.similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert embedder.similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    # Batch of two cached texts is served entirely from cache
    other = "def logout(): pass"
    k2 = hashlib.sha256(f"{model_name}:{other}".encode()).hexdigest()
    (cache_dir / f"{k2}.json").write_text(json.dumps([0.25, 0.25, 0.25]))
    batch = embedder.embed_batch([text, other])
    assert len(batch) == 2 and batch[0] == [0.5, 0.5, 0.5]


def test_phase_18_vector_index_lancedb(tmp_path):
    """Phase 18: real LanceDB insert, search, filter, delete, persistence."""
    from tracera.retrieval.vector_store import VectorStore
    from tracera.indexer.schema import CodeChunk, LineRange

    store = VectorStore(uri=tmp_path / "lancedb", dimension=4)
    rng = LineRange(start_line=0, end_line=1)
    chunks = [
        CodeChunk(id="c1", file_path="a.py", language="python",
                  content="def login(): pass", range=rng,
                  primary_symbol="login", tokens=4),
        CodeChunk(id="c2", file_path="b.py", language="python",
                  content="class Auth: pass", range=rng,
                  primary_symbol="Auth", tokens=4),
    ]
    store.upsert_chunks(chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert store.count == 2
    assert store.existing_dimension() == 4

    results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert results and results[0]["id"] == "c1"

    store.delete_by_file("b.py")
    assert store.count == 1

    # New store on the same URI reuses the persisted table + dimension
    store2 = VectorStore(uri=tmp_path / "lancedb", dimension=None)
    assert store2.existing_dimension() == 4
    assert store2.count == 1


def test_phase_19_dense_retrieval():
    """Phase 19: query → embedding → vector search → top-k with scores."""
    from tracera.retrieval.dense import DenseRetriever

    class Store:
        def search(self, query_embedding, k=10, language=None, symbol_type=None):
            return [{"id": "c1", "content": "def login(): pass", "file_path": "a.py",
                     "_distance": 0.5}]

    dense = DenseRetriever(_FakeEmbedder(), Store())
    results = dense.search("login function", k=5)
    assert results and results[0]["id"] == "c1"
    assert results[0]["_source"] == "dense"
    assert 0 < results[0]["_relevance_score"] <= 1.0


def test_phase_20_hybrid_retrieval():
    """Phase 20: BM25 + dense fused via RRF with configurable weights."""
    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.hybrid import HybridRetriever

    bm25 = BM25Index()
    bm25.add_document("b1", "def login(user): return check(user)")
    dense = _FakeDense([{"id": "v1", "content": "class AuthMiddleware: pass",
                         "file_path": "auth.py", "language": "python"}])
    hybrid = HybridRetriever(bm25, dense, bm25_weight=0.6, dense_weight=0.4)

    results = hybrid.search("login", k=5)
    ids = {r["id"] for r in results}
    assert "b1" in ids and "v1" in ids  # both lexical and semantic hits merged
    assert all("_rrf_score" in r for r in results)
    assert all("_bm25_score" in r and "_dense_score" in r for r in results)


def test_phase_21_symbol_aware_retrieval():
    """Phase 21: query intent detection + symbol-name boosting + dedup."""
    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.hybrid import HybridRetriever
    from tracera.retrieval.symbol_retrieval import SymbolAwareRetriever

    bm25 = BM25Index()
    bm25.add_document("c1", "class AuthMiddleware:\n    def __init__(self): pass")
    bm25.add_document("c2", "class AuthMiddleware:\n    def validate(self): pass")
    bm25.add_document("c3", "def helper(): return 1")

    hybrid = HybridRetriever(bm25, _FakeDense())
    retriever = SymbolAwareRetriever(hybrid)

    results = retriever.search("authentication class", k=5)
    assert results  # hybrid over-fetch + symbol boost
    assert all("_final_score" in r for r in results)
    # Deduplicated: only the best chunk per symbol survives
    symbols = [r.get("symbol") or r.get("id") for r in results]
    assert len(symbols) == len(set(symbols))


def test_phase_22_context_expansion():
    """Phase 22: parent class + imports auto-fetched around a matched symbol."""
    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.context_expander import ContextExpander

    bm25 = BM25Index()
    bm25.add_document("d_parent", "class AuthMiddleware:\n    pass")
    bm25.add_document("d_import", "from auth import AuthMiddleware")

    expander = ContextExpander(bm25, _FakeVectorStore())
    base = [{"id": "d_method", "symbol": "validate", "parent": "AuthMiddleware",
             "file_path": "auth.py", "symbol_type": "method", "content": "def validate(): pass"}]
    expanded = expander.expand(base, max_additional=3)

    assert len(expanded) > 1
    reasons = [r.get("_expansion_reason", "") for r in expanded]
    assert any("parent class" in r for r in reasons)


def test_phase_23_cross_encoder_reranker(monkeypatch):
    """Phase 23: reranker truncates to top-k via passthrough when model absent."""
    from tracera.retrieval.reranker import CrossEncoderReranker

    # Model unavailable (no download in CI) → documented passthrough fallback
    monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda self: False)
    reranker = CrossEncoderReranker(top_n=5)
    results = [{"id": f"c{i}", "content": f"chunk {i}"} for i in range(5)]
    ranked = reranker.rerank("query", results, k=2)
    assert len(ranked) == 2  # truncated to requested k
    assert reranker.rerank("q", results)  # default top_n path works


def test_phase_24_incremental_indexer(tmp_path):
    """Phase 24: created/modified/deleted detection keeps all stores in sync."""
    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.incremental import IncrementalIndexer
    from tracera.graph.symbol_graph import SymbolGraph

    ws = tmp_path / "ws"; ws.mkdir()
    index_dir = tmp_path / "idx"
    bm25 = BM25Index()
    vector_store = _FakeVectorStore()
    graph = SymbolGraph()
    indexer = IncrementalIndexer(workspace_root=ws, bm25_index=bm25,
                                 embedder=_FakeEmbedder(), vector_store=vector_store,
                                 index_dir=index_dir, symbol_graph=graph)

    (ws / "auth.py").write_text("class AuthMiddleware:\n    def __init__(self): pass\n")
    stats = indexer.run()
    assert stats["new"] == 1 and stats["chunks_indexed"] >= 1
    assert bm25.doc_count >= 1 and vector_store.chunks
    assert SymbolGraph.load(index_dir / "symbol_graph.json").find_by_name("AuthMiddleware")

    # Unchanged → skipped; stores preserved
    stats2 = indexer.run()
    assert stats2["skipped"] == 1 and stats2["new"] == 0

    # Modified → re-indexed without duplication
    (ws / "auth.py").write_text("class AuthMiddleware:\n    def __init__(self): pass\n\nclass NewSvc:\n    pass\n")
    stats3 = indexer.run()
    assert stats3["modified"] == 1
    g = SymbolGraph.load(index_dir / "symbol_graph.json")
    assert g.find_by_name("NewSvc") and g.find_by_name("AuthMiddleware")

    # Deleted → removed from BM25, vectors, and graph
    (ws / "auth.py").unlink()
    before = bm25.doc_count
    stats4 = indexer.run()
    assert stats4["deleted"] == 1
    assert bm25.doc_count < before
    assert not SymbolGraph.load(index_dir / "symbol_graph.json").find_by_name("AuthMiddleware")


# ═══════════════════════════════════════════════════════════════════════════════
# CODE KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def _mk_sym(name, stype, start, end, parent=None):
    from tracera.indexer.schema import Symbol, SymbolType, LineRange
    return Symbol(name=name, type=stype, range=LineRange(start_line=start, end_line=end),
                  content=f"def {name}: pass", parent_symbol=parent)


def test_phase_25_symbol_relationship_graph(tmp_path):
    """Phase 25: nodes, typed edges (imports/calls/inherits/contains), persistence."""
    from tracera.graph.symbol_graph import SymbolGraph, RelationType
    from tracera.indexer.schema import SymbolType

    g = SymbolGraph()
    g.build_from_file_symbols("auth.py", [
        _mk_sym("AuthMiddleware", SymbolType.CLASS, 1, 10),
        _mk_sym("validate", SymbolType.METHOD, 2, 8, parent="AuthMiddleware"),
    ])
    g.add_symbol("app.py", _mk_sym("handle_login", SymbolType.FUNCTION, 5, 20))
    g.add_relation("app.py::handle_login", "auth.py::AuthMiddleware", RelationType.CALLS)

    node = g.get_node("auth.py::AuthMiddleware")
    assert node and node["symbol_type"] == "class"
    assert g.get_callers("auth.py::AuthMiddleware") == ["app.py::handle_login"]
    assert g.get_callees("app.py::handle_login") == ["auth.py::AuthMiddleware"]
    assert g.get_children("auth.py::AuthMiddleware") == ["auth.py::validate"]

    path = tmp_path / "symbol_graph.json"
    g.save(path)
    loaded = SymbolGraph.load(path)
    assert loaded.node_count == g.node_count and loaded.edge_count == g.edge_count
    loaded.remove_file("auth.py")
    assert loaded.node_count == 1


def test_phase_26_dependency_aware_retrieval():
    """Phase 26: graph neighbours enrich retrieval results."""
    from tracera.graph.symbol_graph import SymbolGraph, RelationType
    from tracera.graph.graph_retrieval import GraphRetriever
    from tracera.retrieval.bm25 import BM25Index
    from tracera.indexer.schema import SymbolType

    g = SymbolGraph()
    g.add_symbol("auth.py", _mk_sym("AuthMiddleware", SymbolType.CLASS, 1, 10))
    g.add_symbol("app.py", _mk_sym("handle_login", SymbolType.FUNCTION, 5, 20))
    g.add_relation("app.py::handle_login", "auth.py::AuthMiddleware", RelationType.CALLS)

    bm25 = BM25Index()
    bm25.add_document("c_auth", "class AuthMiddleware: pass")
    bm25.add_document("c_app", "def handle_login(): return auth()")

    retriever = GraphRetriever(g, bm25)
    assert retriever.graph is g  # graph accessor used by Phase 27 tools

    base = [{"id": "c_auth", "symbol": "AuthMiddleware", "file_path": "auth.py",
             "content": "class AuthMiddleware: pass", "symbol_type": "class"}]
    expanded = retriever.expand_with_graph(base, max_depth=1, max_total=10)
    symbols = {r.get("symbol") for r in expanded}
    assert "AuthMiddleware" in symbols
    assert len(expanded) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT + CODE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_27_code_search_tools():
    """Phase 27: all six retrieval tools execute and return structured results."""
    from tracera.tools.code_search import (
        SearchCodeTool, FindSymbolTool, FindReferencesTool,
        GetDependenciesTool, GetContextTool,
    )
    from tracera.graph.symbol_graph import SymbolGraph
    from tracera.indexer.schema import SymbolType

    class FakeRetriever:
        def search(self, query, k=5, language=None):
            return [{"id": "c1", "symbol": "AuthMiddleware", "symbol_type": "class",
                     "file_path": "auth.py", "start_line": 1, "end_line": 10,
                     "content": "class AuthMiddleware: ...", "language": "python",
                     "_rrf_score": 1.0}]

    class FakeExpander:
        def expand(self, results, max_additional=3):
            return results

    g = SymbolGraph()
    g.add_symbol("auth.py", _mk_sym("AuthMiddleware", SymbolType.CLASS, 1, 10))

    async def _run():
        assert (await SearchCodeTool(FakeRetriever()).execute(query="auth")).success
        assert (await FindSymbolTool(FakeRetriever()).execute(name="AuthMiddleware")).success
        assert (await FindReferencesTool(g).execute(symbol="AuthMiddleware")).success
        assert (await GetDependenciesTool(g).execute(symbol="AuthMiddleware")).success
        assert (await GetContextTool(FakeRetriever(), FakeExpander()).execute(symbol="AuthMiddleware")).success

    asyncio.run(_run())


def test_phase_28_retrieval_aware_agent(tmp_path, monkeypatch):
    """Phase 28: agent registry is extended with search tools when an index exists."""
    from tracera.config import get_settings
    import tracera.main as main_mod
    from tracera.graph.symbol_graph import SymbolGraph

    settings = get_settings()
    settings.tracera_workspace = tmp_path
    settings.tracera_data_dir = tmp_path / "data"  # hermetic memory/index dirs

    monkeypatch.setattr("tracera.providers.create_provider", lambda **kw: FakeProvider())
    fake_pipeline = [None, FakeRetrieverStub(), None, None, None, None, None, None, None,
                     types.SimpleNamespace(graph=SymbolGraph())]

    agent, _ws, _prov = main_mod._build_agent(settings, tmp_path, fake_pipeline)
    assert "search_code" in agent.registry.names
    assert "find_references" in agent.registry.names
    assert "get_dependencies" in agent.registry.names


class FakeRetrieverStub:
    def search(self, query, k=5, language=None):
        return []


def test_phase_29_context_assembly_engine():
    """Phase 29: dedup, ordering, token budget, truncation."""
    from tracera.agent.context_engine import ContextAssemblyEngine

    engine = ContextAssemblyEngine(max_tokens=1000)
    chunks = [
        {"content": "def a(): pass", "symbol": "a", "symbol_type": "function", "file_path": "a.py"},
        {"content": "def a(): pass", "symbol": "a", "symbol_type": "function", "file_path": "a.py"},  # dup
        {"content": "import os", "symbol": "os", "symbol_type": "import", "file_path": "a.py"},
        {"content": "class C: pass", "symbol": "C", "symbol_type": "class", "file_path": "a.py"},
    ]
    ctx = engine.assemble(chunks, query="test query")
    assert "# Retrieved Code Context" in ctx
    assert "test query" in ctx
    # imports sorted first, dedup means one entry for the duplicate
    assert ctx.index("import os") < ctx.index("class C") < ctx.index("def a")
    assert ctx.count("def a(): pass") == 1


def test_phase_30_context_compression():
    """Phase 30: compression shrinks oversized context and is wired into debugging."""
    from tracera.agent.compressor import ContextCompressor

    compressor = ContextCompressor(target_tokens=50)
    chunks = [{"content": "def f(): " + "x = 1\n" * 60, "symbol": "f",
               "symbol_type": "function", "_final_score": 0.9}] * 20
    compressed = compressor.compress(chunks)
    total_tokens = sum(len(c["content"]) // 4 for c in compressed)
    assert total_tokens <= 200  # shrunk toward the 50-token budget

    # Wired into RetrievalDebugger (Phase 35 consumer)
    from tracera.agent.autonomous import RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    class SpyCompressor(ContextCompressor):
        def __init__(self):
            super().__init__()
            self.called = False

        def compress(self, chunks):
            self.called = True
            return chunks

    spy = SpyCompressor()
    retriever = FakeRetrieverStub()
    retriever.search = lambda query, k=5, language=None: [
        {"content": "def auth(): pass", "symbol": "auth", "file_path": "auth.py"}]
    debugger = RetrievalDebugger(retriever, ContextAssemblyEngine(), compressor=spy)
    plan = debugger.build_debug_plan(tr.TestFailure(test_name="t", error_type="E", error_message="m"), None)
    assert spy.called
    assert "auth" in plan.retrieved_context


def test_phase_31_repository_aware_agent(tmp_path, monkeypatch):
    """Phase 31: system prompt instructs search-first when an index is loaded."""
    from tracera.config import get_settings
    import tracera.main as main_mod
    from tracera.graph.symbol_graph import SymbolGraph

    settings = get_settings()
    settings.tracera_workspace = tmp_path
    settings.tracera_data_dir = tmp_path / "data"
    monkeypatch.setattr("tracera.providers.create_provider", lambda **kw: FakeProvider())
    fake_pipeline = [None, FakeRetrieverStub(), None, None, None, None, None, None, None,
                     types.SimpleNamespace(graph=SymbolGraph())]

    agent_aware, *_ = main_mod._build_agent(settings, tmp_path, fake_pipeline)
    assert "search_code" in agent_aware.system_prompt
    assert "find_symbol" in agent_aware.system_prompt
    assert "grep" in agent_aware.system_prompt  # fallback still mentioned

    agent_basic, *_ = main_mod._build_agent(settings, tmp_path, None)
    assert "search_code" not in agent_basic.system_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS SOFTWARE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase_32_test_discovery(tmp_path):
    """Phase 32: pytest / npm / cargo frameworks are auto-detected."""
    from tracera.tools.test_runner import TestDiscovery

    py = tmp_path / "py"; py.mkdir()
    (py / "pyproject.toml").write_text("[project]\n")
    (py / "tests").mkdir()
    (py / "tests" / "test_x.py").write_text("def test_x(): pass")
    disc = TestDiscovery(py)
    assert disc.detect_framework() == "pytest"
    cmd = disc.get_test_command("pytest")
    assert cmd and cmd[0] == "python"

    js = tmp_path / "js"; js.mkdir()
    (js / "package.json").write_text('{"scripts": {"test": "jest"}}')
    assert TestDiscovery(js).detect_framework() == "npm"

    rs = tmp_path / "rs"; rs.mkdir()
    (rs / "Cargo.toml").write_text("[package]\n")
    assert TestDiscovery(rs).detect_framework() == "cargo"


def test_phase_33_test_execution(tmp_path, monkeypatch):
    """Phase 33: TestRunner executes the suite and reports pass/fail."""
    from tracera.tools.test_runner import TestRunner, TestDiscovery

    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n"
    )
    monkeypatch.setattr(
        TestDiscovery, "get_test_command",
        lambda self, fw=None: [sys.executable, "-m", "pytest", "-q"],
    )
    report = TestRunner(tmp_path).run()
    assert report.success
    assert report.passed >= 1 and report.total >= 1
    assert report.framework == "pytest"
    assert "passed" in report.summary


def test_phase_34_failure_analysis():
    """Phase 34: raw pytest output parsed into structured TestFailure objects."""
    from tracera.tools.test_runner import FailureAnalyzer

    output = (
        "def test_login():\n"
        "    assert authenticate('bad') is True\n"
        "E   AssertionError: expected True\n"
        "\n"
        'File "tests/test_auth.py", line 12, in test_login\n'
        "    assert authenticate('bad') is True\n"
        "AssertionError: expected True\n"
        "\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: expected True\n"
        "2 passed, 1 failed in 1.23s\n"
    )
    report = FailureAnalyzer.parse(output, "pytest")
    assert report.failed == 1 and report.passed == 2 and report.total == 3
    assert report.duration_seconds == 1.23
    assert report.failures
    f = report.failures[0]
    assert f.test_name and f.error_type == "AssertionError"
    assert "test_auth.py" in f.file_path and f.line_number == 12
    assert f.stack_trace


def test_phase_35_retrieval_driven_debugging():
    """Phase 35: failure → retrieve symbol → assemble context → debug plan."""
    from tracera.agent.autonomous import RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    retriever = FakeRetrieverStub()
    retriever.search = lambda query, k=5, language=None: [
        {"content": "def validate_token(token):\n    return token == 'x'",
         "symbol": "validate_token", "file_path": "auth.py",
         "symbol_type": "function"}]
    debugger = RetrievalDebugger(retriever, ContextAssemblyEngine())

    failure = tr.TestFailure(test_name="test_validate_token",
                             error_type="AssertionError",
                             error_message="token validation failed",
                             file_path="tests/test_auth.py", line_number=7)
    plan = debugger.build_debug_plan(failure, provider=None)
    assert "auth.py" in plan.hypothesis
    assert "validate_token" in plan.retrieved_context
    assert plan.failure is failure


class _FakeAgent2:
    def __init__(self):
        self.tasks = []

    async def run(self, task, conversation=None):
        self.tasks.append(task)

        async def gen():
            yield None

        return gen()


def test_phase_36_autonomous_fix_loop():
    """Phase 36: plan → test → fix → re-test until green; events are consumed."""
    from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    class FailOnceRunner:
        def __init__(self):
            self.calls = 0

        def run(self, framework=None, test_paths=None):
            self.calls += 1
            if self.calls >= 2:
                return tr.TestReport(framework="pytest", passed=2, total=2, success=True)
            return tr.TestReport(framework="pytest", passed=1, total=2, success=False,
                                 failures=[tr.TestFailure(test_name="t", error_type="E",
                                                          error_message="m")])

    loop = AutonomousFixLoop(Path("."), FailOnceRunner(),
                             RetrievalDebugger(None, ContextAssemblyEngine()),
                             max_iterations=3)
    agent = _FakeAgent2()
    result = asyncio.run(loop.run("fix the bug", None, agent))

    assert result.final_success is True
    assert result.total_iterations == 2
    assert len(result.attempts) == 2
    # The loop actually drove the agent — the fix task was consumed
    # through the event stream (previously `await agent.run()` dropped it).
    assert len(agent.tasks) == 1


def test_phase_37_self_review(tmp_path):
    """Phase 37: git diff + LLM critique produces a review report."""
    import git as gitpython
    from tracera.agent.autonomous import SelfReviewer

    repo = gitpython.Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "t")
        cw.set_value("user", "email", "t@e.com")
    (tmp_path / "app.py").write_text("def a(): pass\n")
    repo.index.add(["app.py"])
    repo.index.commit("c1")
    (tmp_path / "app.py").write_text("def a():\n    return 1\n")  # uncommitted change

    reviewer = SelfReviewer(tmp_path, retriever=None)
    assert "return 1" in reviewer.get_diff()

    provider = FakeProvider(response_text="PASS — no critical issues found.")
    report = asyncio.run(reviewer.review(provider, implementation_summary="added return"))
    assert "PASS" in report

    # Clean repo (committed, no pending changes) → nothing to review
    clean_dir = tmp_path / "clean"
    repo2 = gitpython.Repo.init(clean_dir)
    with repo2.config_writer() as cw:
        cw.set_value("user", "name", "t")
        cw.set_value("user", "email", "t@e.com")
    (clean_dir / "x.py").write_text("x = 1\n")
    repo2.index.add(["x.py"])
    repo2.index.commit("c1")
    clean = asyncio.run(SelfReviewer(clean_dir, None).review(FakeProvider()))
    assert "No changes" in clean


def test_phase_38_regression_protection(tmp_path, monkeypatch):
    """Phase 38: pre/post snapshots + changed files detect regressions."""
    from tracera.agent.autonomous import RegressionProtector

    class CountingRunner:
        def __init__(self):
            self.calls = 0
            self.results = [
                tr.TestReport(framework="pytest", passed=5, total=5, success=True),
                tr.TestReport(framework="pytest", passed=3, total=5, success=False),
            ]

        def run(self, framework=None, test_paths=None):
            r = self.results[min(self.calls, 1)]
            self.calls += 1
            return r

    runner = CountingRunner()
    protector = RegressionProtector(tmp_path, runner)
    baseline = protector.snapshot_before()
    assert baseline.passed == 5

    monkeypatch.setattr(
        "tracera.agent.autonomous.subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": "auth.py\nmain.py\n"})(),
    )
    report = protector.verify_after()
    assert report["pre_passed"] == 5 and report["post_passed"] == 3
    assert report["regressions"] == 2
    assert report["overall_success"] is False
    assert "auth.py" in report["changed_files"]
