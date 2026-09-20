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

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.retrieval.dedupe import dedupe_by_file, overfetch_k

log = get_logger("evaluation.strategies")

#: Function words dropped before matching. Without this, a query like "how does
#: the scanner handle nested gitignore files" matches nearly every file in the
#: repo on "the", and the baseline degrades into a stopword-counting contest.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "does", "did", "how", "what",
    "where", "which", "that", "this", "with", "from", "into", "when", "why",
    "who", "its", "not", "but", "all", "any", "can", "has", "have", "had",
    "they", "them", "then", "than", "over", "under", "each", "more", "most",
    "some", "such", "only", "also", "is", "be", "to", "of", "in", "on", "at",
    "as", "by", "or", "it",
})


def tokenize(query: str) -> list[str]:
    """Split a query into lowercase word tokens."""
    return [t for t in re.split(r"[^a-z0-9_]+", query.lower()) if t]


def _matching_lines(text: str, terms: set[str], *, limit: int = 20) -> str:
    """
    The lines that matched, capped — i.e. what ``grep -n`` would print.

    The baseline is deliberately given *only* its matching lines, not the whole
    file. Handing it whole files would inflate its context size and flatter
    TRACERA; understating our own advantage is the safer error.
    """
    out: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if any(term in low for term in terms):
            out.append(f"{number}:{line}")
            if len(out) >= limit:
                break
    return "\n".join(out)


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

    def __init__(self, *, max_per_file: int | None = None) -> None:
        self.last_latency_ms: float = 0.0
        self.last_result_bytes: int = 0
        #: Cap on chunks returned per file. None disables it, which is how every
        #: baseline ran before this existed. Hybrid's top-5 on this repository was
        #: regularly only two or three distinct files, so the freed slots go to
        #: files that would otherwise have been cut off.
        #:
        #: Measured at k=10 over the 120-query set: cap 2 is recall-neutral
        #: (hybrid recall@5 0.9458 -> 0.9417) for -12% context bytes; cap 1 takes
        #: -31% bytes but costs recall@10 (0.983 -> 0.950), because a file's
        #: symbol can live only on a dropped duplicate chunk. Full numbers and the
        #: precision caveat are in tracera/retrieval/dedupe.py.
        self.max_per_file = max_per_file

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalHit]:
        t0 = time.perf_counter()
        # Over-fetch when deduplicating: requesting exactly k and then dropping
        # duplicates would return fewer than k, the opposite of the intent.
        fetch_k = overfetch_k(k, filtering=bool(self.max_per_file))
        hits = self._retrieve(query, fetch_k)
        hits = dedupe_by_file(
            hits, max_per_file=self.max_per_file, file_path_of=lambda h: h.file_path
        )[:k]
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
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".tracera",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "target",
        "site-packages",
    }

    _SOURCE_SUFFIXES = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".md",
        ".yml",
        ".yaml",
        ".json",
        ".toml",
        ".sql",
        ".sh",
        ".css",
        ".html",
        ".vue",
        ".svelte",
        ".cs",
        ".scala",
        ".ex",
        ".exs",
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
        #: File text, read once and reused across queries. Without this the
        #: baseline re-reads the whole corpus per query (~35s each on a 250-file
        #: repo), which made a 100-query benchmark take over an hour. A real
        #: grep tool gets the same benefit from the OS page cache.
        self._text_cache: dict[Path, str] = {}
        #: Greppable files, discovered once. Re-walking the tree per query made
        #: the baseline quadratic in repo size.
        self._files: list[Path] | None = None

    def _search(self, query: str) -> list[str]:
        """Workspace-relative posix paths of matching files, best match first."""
        return [path for path, _ in self._search_with_content(query)]

    def _source_files(self) -> list[Path]:
        """
        Every greppable source file, discovered once and cached.

        ``rglob("*")`` walks *into* skipped directories before the filter ever
        sees them, so a repo with a ``.venv`` or ``node_modules`` paid the full
        traversal on **every** query — ~15s each here, almost entirely inside
        directories the baseline is not allowed to read. ``os.walk`` prunes in
        place, so those trees are never entered at all.
        """
        if self._files is None:
            found: list[Path] = []
            for root, dirnames, filenames in os.walk(self.workspace):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in self._SKIP_DIRS and not d.startswith(".")
                ]
                for name in filenames:
                    path = Path(root) / name
                    if path.suffix in self._SOURCE_SUFFIXES:
                        found.append(path)
            found.sort()
            self._files = found
        return self._files

    def _search_with_content(self, query: str) -> list[tuple[str, str]]:
        """
        Grep the workspace; return ``(relative_path, matching_text)`` pairs.

        Paths come back **workspace-relative and posix-normalised**, the same
        form every ground-truth entry and every other strategy uses. Returning
        absolute Windows paths made the baseline unmatchable: it scored 0.000 on
        every query while still costing 33s each, which reads as "grep is
        terrible" rather than "the baseline can never match".

        Matching is "any query term", ranked by how many distinct terms a file
        contains. The previous all-terms rule required every word including
        function words, so a natural-language query matched nothing at all.
        """
        if self._search_fn is not None:
            try:
                # An injected search function returns paths only.
                return [(str(p), "") for p in self._search_fn(query)]
            except Exception:
                pass  # fall back to plain grep

        terms = {t for t in tokenize(query) if len(t) >= 3 and t not in _STOPWORDS}
        if not terms:
            return []

        scored: list[tuple[int, str, str]] = []
        for path in self._source_files():
            text = self._text_cache.get(path)
            if text is None:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                self._text_cache[path] = text

            lowered = text.lower()
            matched = {t for t in terms if t in lowered}
            if not matched:
                continue
            try:
                rel = path.relative_to(self.workspace).as_posix()
            except ValueError:  # pragma: no cover - workspace is always a parent
                rel = path.as_posix()
            scored.append((len(matched), rel, _matching_lines(text, matched)))

        # Most distinct query terms first; ties broken by path for determinism.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(rel, snippet) for _, rel, snippet in scored]

    def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        found = self._search_with_content(query)[:k]
        return [
            RetrievalHit(
                doc_id=rel,
                score=1.0,
                file_path=rel,
                content=snippet,
                symbol=None,
            )
            for rel, snippet in found
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

    def __init__(
        self, bm25: Any, *, resolve_doc: Callable[[str], str | None] | None = None
    ) -> None:
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
    max_per_file: int | None = None,
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
        max_per_file: cap on chunks returned per file (None disables it).
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
    strategies = {name: builders[name]() for name in want if name in builders}
    # Applied uniformly after construction so an A/B on this policy compares
    # the policy and not the strategy set: every arm gets the same cap.
    for strategy in strategies.values():
        strategy.max_per_file = max_per_file
    return strategies
