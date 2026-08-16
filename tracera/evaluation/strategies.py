"""
Phases 47-48 — Retrieval strategies.

Every strategy exposes the same interface so benchmarks compare them fairly:

    strategy.name          — e.g. "bm25", "dense", "hybrid+reranker"
    strategy.retrieve(query, k=10) -> list[RetrievalHit]
    strategy.kind          — "grep" | "lexical" | "dense" | "hybrid"

    @dataclass RetrievalHit:
        doc_id: str
        score: float
        file_path: str | None   # None when the strategy can't resolve it
        content: str
        symbol: str | None

Strategies implemented: grep (Phase 48 baseline), BM25, dense, hybrid,
hybrid + cross-encoder reranker (Phase 47).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tracera.logging import get_logger

log = get_logger("evaluation.strategies")


@dataclass
class RetrievalHit:
    """A single retrieval result, normalised across strategies."""

    doc_id: str
    score: float
    file_path: str | None = None
    content: str = ""
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 4),
            "file_path": self.file_path,
            "symbol": self.symbol,
            "content": self.content[:200],
        }


# ── Base strategy ─────────────────────────────────────────────────────────────

class RetrievalStrategy:
    """Common base: measures and records its own latency per retrieval."""

    name: str = "strategy"
    kind: str = "hybrid"

    def __init__(self) -> None:
        self.last_latency_ms: float = 0.0
        self.last_result_bytes: int = 0

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalHit]:
        t0 = time.perf_counter()
        hits = self._retrieve(query, k)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_result_bytes = sum(len(h.content) for h in hits)
        return hits

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


# ── Grep baseline (Phase 48) ──────────────────────────────────────────────────

class GrepStrategy(RetrievalStrategy):
    """Baseline: literal grep over the workspace. Returns file hits."""

    name = "grep"
    kind = "grep"

    #: Directories that must never be walked (vendored deps, build output,
    #: hidden dirs) — keeps the baseline fast and results meaningful.
    _SKIP_DIRS = {
        ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
        ".tracera", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "dist", "build", ".idea", ".vscode", "target", "site-packages",
    }

    _SOURCE_SUFFIXES = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".h", ".cpp", ".rb", ".php", ".swift", ".kt", ".md",
        ".yml", ".yaml", ".json", ".toml", ".sql", ".sh", ".css",
        ".html", ".vue", ".svelte", ".cs", ".scala", ".ex", ".exs",
    }

    def __init__(self, workspace: Path, *, search_fn: Callable | None = None) -> None:
        """
        Args:
            workspace: root directory to grep.
            search_fn: optional async callable(query) -> list[str] of file
                paths (e.g. WorkspaceSandbox.grep). Defaults to a plain
                recursive text search that requires no code index.
        """
        super().__init__()
        self.workspace = Path(workspace)
        self._search_fn = search_fn

    def _search(self, query: str) -> list[str]:
        if self._search_fn is not None:
            try:
                return list(self._search_fn(query))
            except Exception:
                pass  # fall back to plain grep
        # Plain recursive grep over source-ish files (no index required).
        tokens = query.lower().split()
        if not tokens:
            return []
        hits: list[str] = []
        for path in self.workspace.rglob("*"):
            if path.is_dir():
                continue
            parts = path.parts
            # Never walk vendored deps / build output / hidden dirs — the
            # baseline would read tens of thousands of files (e.g. .venv).
            if any(part in self._SKIP_DIRS or part.startswith(".")
                   for part in parts[:-1]):
                continue
            if path.suffix not in self._SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if all(t in text.lower() for t in tokens):
                hits.append(str(path))
        return hits

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        files = self._search(query)[:k]
        return [
            RetrievalHit(
                doc_id=str(f),
                score=1.0,
                file_path=str(f),
                content="",
                symbol=None,
            )
            for f in files
        ]


# ── BM25 (Phase 47) ───────────────────────────────────────────────────────────

def build_doc_resolver(vector_store: Any) -> Callable[[str], str | None]:
    """
    Build a doc_id → file_path resolver from the vector store table.

    The LanceDB table stores id + file_path per chunk, so BM25 hits (which
    only know doc ids) can be mapped to files for ground-truth matching.
    Falls back to returning None (doc-id matching only) when unavailable.
    """
    try:
        table = vector_store._get_or_create_table()  # noqa: SLF001
        arrow = table.to_arrow()
        mapping: dict[str, str] = {}
        ids = arrow.column("id").to_pylist()
        paths = arrow.column("file_path").to_pylist()
        for doc_id, file_path in zip(ids, paths):
            mapping[str(doc_id)] = str(file_path)
        return lambda doc_id: mapping.get(doc_id)
    except Exception as e:
        log.warning("Doc resolver unavailable (%s) — BM25 uses doc-id matching", e)
        return lambda doc_id: None


class BM25Strategy(RetrievalStrategy):
    """Lexical BM25 (Phase 16 index) — exact keyword matching."""

    name = "bm25"
    kind = "lexical"

    def __init__(self, bm25: Any, *, resolve_doc: Callable[[str], str | None] | None = None) -> None:
        super().__init__()
        self._bm25 = bm25
        self._resolve = resolve_doc or (lambda doc_id: None)

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for doc_id, score in self._bm25.search(query, k=k):
            content = self._bm25.get_document(doc_id) or ""
            hits.append(
                RetrievalHit(
                    doc_id=doc_id,
                    score=float(score),
                    file_path=self._resolve(doc_id),
                    content=content,
                )
            )
        return hits


# ── Dense (Phase 47) ──────────────────────────────────────────────────────────

class DenseStrategy(RetrievalStrategy):
    """Dense semantic retrieval over the vector store."""

    name = "dense"
    kind = "dense"

    def __init__(self, dense: Any) -> None:
        super().__init__()
        self._dense = dense

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for row in self._dense.search(query, k=k):
            hits.append(
                RetrievalHit(
                    doc_id=row.get("id", ""),
                    score=float(row.get("_relevance_score", 0.0)),
                    file_path=row.get("file_path"),
                    content=row.get("content", ""),
                    symbol=row.get("symbol"),
                )
            )
        return hits


# ── Hybrid (Phase 47) ─────────────────────────────────────────────────────────

class HybridStrategy(RetrievalStrategy):
    """BM25 + Dense fused with Reciprocal Rank Fusion."""

    name = "hybrid"
    kind = "hybrid"

    def __init__(self, hybrid: Any) -> None:
        super().__init__()
        self._hybrid = hybrid

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for row in self._hybrid.search(query, k=k):
            hits.append(
                RetrievalHit(
                    doc_id=row.get("id", ""),
                    score=float(row.get("_rrf_score", 0.0)),
                    file_path=row.get("file_path"),
                    content=row.get("content", ""),
                    symbol=row.get("symbol"),
                )
            )
        return hits


# ── Hybrid + reranker (Phase 47) ──────────────────────────────────────────────

def cross_encoder_available(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> bool:
    """
    True when the cross-encoder model is already downloaded in the Hugging
    Face cache. Benchmarks must NOT trigger a network download mid-run, so
    the reranked strategy is only built when the model is present.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_name = model_name.replace("/", "--")
        return (Path(HF_HUB_CACHE) / f"models--{cache_name}").is_dir()
    except Exception:
        return False


class RerankedHybridStrategy(RetrievalStrategy):
    """Hybrid candidates re-ranked by a cross-encoder."""

    name = "hybrid+reranker"
    kind = "hybrid"

    def __init__(self, hybrid: Any, reranker: Any, *, candidate_k: int = 20) -> None:
        super().__init__()
        self._hybrid = hybrid
        self._reranker = reranker
        self._candidate_k = candidate_k

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        candidates = self._hybrid.search(query, k=self._candidate_k)
        reranked = self._reranker.rerank(query, candidates, k=k)
        hits: list[RetrievalHit] = []
        for row in reranked:
            hits.append(
                RetrievalHit(
                    doc_id=row.get("id", ""),
                    score=float(row.get("_rerank_score", 0.0)),
                    file_path=row.get("file_path"),
                    content=row.get("content", ""),
                    symbol=row.get("symbol"),
                )
            )
        return hits


# ── Factory ───────────────────────────────────────────────────────────────────

def build_strategies(
    *,
    workspace: Path | None = None,
    bm25: Any = None,
    dense: Any = None,
    hybrid: Any = None,
    reranker: Any = None,
    resolve_doc: Callable[[str], str | None] | None = None,
    include: list[str] | None = None,
) -> dict[str, RetrievalStrategy]:
    """
    Build the strategy dict for a benchmark.

    Only strategies whose dependencies are provided are built, so a
    benchmark can run against a partial pipeline (e.g. BM25 only).

    Args:
        workspace: root for the grep baseline.
        bm25: BM25Index instance.
        dense: DenseRetriever instance.
        hybrid: HybridRetriever instance.
        reranker: CrossEncoderReranker instance.
        resolve_doc: doc_id → file_path resolver for BM25.
        include: subset of strategy names to build (default: all available).
    """
    builders: dict[str, Callable[[], RetrievalStrategy]] = {}
    if workspace is not None:
        builders["grep"] = lambda: GrepStrategy(workspace)
    if bm25 is not None:
        builders["bm25"] = lambda: BM25Strategy(bm25, resolve_doc=resolve_doc)
    if dense is not None:
        builders["dense"] = lambda: DenseStrategy(dense)
    if hybrid is not None:
        builders["hybrid"] = lambda: HybridStrategy(hybrid)
    if hybrid is not None and reranker is not None and cross_encoder_available():
        builders["hybrid+reranker"] = lambda: RerankedHybridStrategy(hybrid, reranker)

    want = set(include) if include else set(builders)
    return {name: builders[name]() for name in want if name in builders}
