"""
Phase 24 — Incremental / Live Indexing.

Detects which files have changed (created/modified/deleted) since the last
index run using SHA256 hashes, and only re-indexes those files.
This avoids rebuilding the entire repository index on every run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tracera.indexer.schema import CodeChunk, FileMetadata
from tracera.indexer.scanner import RepositoryScanner
from tracera.indexer.parser import LanguageParser
from tracera.indexer.extractor import SymbolExtractor
from tracera.indexer.chunker import SymbolAwareChunker
from tracera.graph.symbol_graph import SymbolGraph
from tracera.retrieval.bm25 import BM25Index
from tracera.retrieval.embedder import EmbeddingPipeline
from tracera.retrieval.vector_store import VectorStore
from tracera.logging import get_logger

log = get_logger("retrieval.incremental")

_MANIFEST_FILENAME = "index_manifest.json"
_GRAPH_FILENAME = "symbol_graph.json"
_FILE_CHUNKS_FILENAME = "file_chunks.json"


class IncrementalIndexer:
    """
    Manages the full indexing pipeline with change detection.

    Stores a manifest file (.tracera/index/index_manifest.json) that maps
    each file path to its last-known SHA256 hash.
    On each run, only changed/new files are re-indexed.
    """

    def __init__(
        self,
        workspace_root: Path,
        bm25_index: BM25Index,
        embedder: EmbeddingPipeline,
        vector_store: VectorStore,
        index_dir: Path,
        max_file_size: int = 2 * 1024 * 1024,
        symbol_graph: SymbolGraph | None = None,
    ) -> None:
        self._workspace = workspace_root
        self._bm25 = bm25_index
        self._embedder = embedder
        self._vector_store = vector_store
        self._index_dir = index_dir
        self._max_file_size = max_file_size
        self._manifest_path = index_dir / _MANIFEST_FILENAME
        self._graph_path = index_dir / _GRAPH_FILENAME
        self._file_chunks_path = index_dir / _FILE_CHUNKS_FILENAME

        # file_path → [chunk_ids] — lets us remove deleted files from BM25
        self._file_chunks: dict[str, list[str]] = {}
        if self._file_chunks_path.exists():
            try:
                self._file_chunks = json.loads(self._file_chunks_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Failed to load file→chunk map (%s) — starting empty", e)

        # Phase 25: symbol relationship graph (persisted alongside the index)
        if symbol_graph is not None:
            self._graph = symbol_graph
        elif self._graph_path.exists():
            try:
                self._graph = SymbolGraph.load(self._graph_path)
                log.info(
                    "Loaded symbol graph: %d nodes, %d edges",
                    self._graph.node_count, self._graph.edge_count,
                )
            except Exception as e:
                log.warning("Failed to load symbol graph (%s) — starting fresh", e)
                self._graph = SymbolGraph()
        else:
            self._graph = SymbolGraph()

        # Pipeline components
        self._scanner = RepositoryScanner(workspace_root, max_file_size=max_file_size)
        self._parser = LanguageParser()
        self._extractor = SymbolExtractor(self._parser)
        self._chunker = SymbolAwareChunker()

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, str]:
        """Load {file_path: sha256} manifest from disk."""
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        """Save complete index snapshot metadata to disk."""
        import subprocess
        import datetime
        # Get git SHA
        git_sha = ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self._workspace), capture_output=True, text=True, timeout=10
            )
            git_sha = result.stdout.strip()
        except Exception:
            pass
        # Get parser versions
        parser_versions = {lang: "0.1.0" for lang in self._parser.languages()}
        # Full snapshot metadata
        full_manifest = {
            "snapshot": {
                "repository": str(self._workspace),
                "git_sha": git_sha,
                "generation_timestamp": datetime.datetime.utcnow().isoformat(),
                "files_indexed": len(manifest),
                "files_excluded": 0,
                "parser_versions": parser_versions,
                "index_version": "1.0.0"
            },
            "files": manifest
        }
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")

    # ── Change detection ──────────────────────────────────────────────────────

    def _detect_changes(
        self, current_files: list[FileMetadata], manifest: dict[str, str]
    ) -> tuple[list[FileMetadata], list[FileMetadata], list[str]]:
        """
        Compare current file state against manifest.

        Returns:
            (new_files, modified_files, deleted_file_paths)
        """
        current_map = {f.path: f for f in current_files}
        new_files = []
        modified_files = []

        for fmeta in current_files:
            old_hash = manifest.get(fmeta.path)
            if old_hash is None:
                new_files.append(fmeta)
            elif old_hash != fmeta.sha256:
                modified_files.append(fmeta)

        deleted = [path for path in manifest if path not in current_map]

        return new_files, modified_files, deleted

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_file(self, fmeta: FileMetadata) -> list[CodeChunk]:
        """Parse, extract symbols, and chunk a single file."""
        if not fmeta.language:
            return []  # Skip undetected languages

        try:
            file_path = self._workspace / fmeta.path
            content = file_path.read_text(encoding="utf-8", errors="replace")
            code_bytes = content.encode("utf-8")

            symbols = self._extractor.extract_symbols(code_bytes, fmeta.language)

            # Phase 25: keep the symbol relationship graph in sync with the index.
            # Re-adding a modified file first drops its old nodes, then re-adds them.
            self._graph.remove_file(fmeta.path)
            self._graph.build_from_file_symbols(fmeta.path, symbols)

            chunks = self._chunker.chunk_file(fmeta.path, fmeta.language, content, symbols)
            return chunks
        except Exception as e:
            log.warning("Failed to index %s: %s", fmeta.path, e)
            return []

    def _embed_and_store(self, chunks: list[CodeChunk]) -> None:
        """Embed chunks and upsert into BM25 + vector store."""
        if not chunks:
            return

        # BM25 — add docs
        for chunk in chunks:
            self._bm25.add_document(chunk.id, chunk.content)

        # Dense — batch embed
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        self._vector_store.upsert_chunks(chunks, embeddings)

    def _remove_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a deleted file from BM25 + vectors."""
        for chunk_id in self._file_chunks.pop(file_path, []):
            self._bm25.remove_document(chunk_id)
        self._vector_store.delete_by_file(file_path)
        self._file_chunks.pop(file_path, None)
        log.info("Removed deleted file from index: %s", file_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, full_rebuild: bool = False) -> dict:
        """
        Run the incremental indexing pipeline.

        Args:
            full_rebuild: If True, re-index everything regardless of changes.

        Returns:
            Stats dict with counts of new/modified/deleted/skipped files.
        """
        log.info("Starting incremental indexing (full_rebuild=%s)", full_rebuild)
        manifest = {} if full_rebuild else self._load_manifest()
        if full_rebuild:
            self._graph = SymbolGraph()
            self._file_chunks = {}

        # Scan workspace
        current_files = list(self._scanner.scan())
        log.info("Found %d source files in workspace", len(current_files))

        new_files, modified_files, deleted_paths = self._detect_changes(current_files, manifest)

        stats = {
            "total_scanned": len(current_files),
            "new": len(new_files),
            "modified": len(modified_files),
            "deleted": len(deleted_paths),
            "skipped": len(current_files) - len(new_files) - len(modified_files),
            "chunks_indexed": 0,
        }

        # Process deletions
        for path in deleted_paths:
            self._remove_file(path)
            self._graph.remove_file(path)
            manifest.pop(path, None)

        # Process new + modified files
        to_index = new_files + modified_files
        log.info("Indexing %d files (%d new, %d modified)", len(to_index), len(new_files), len(modified_files))

        for fmeta in to_index:
            # Drop stale chunk ids for modified files before re-adding
            for old_chunk_id in self._file_chunks.pop(fmeta.path, []):
                self._bm25.remove_document(old_chunk_id)

            chunks = self._index_file(fmeta)
            self._embed_and_store(chunks)
            self._file_chunks[fmeta.path] = [c.id for c in chunks]
            manifest[fmeta.path] = fmeta.sha256
            stats["chunks_indexed"] += len(chunks)

        # Persist manifest, BM25 index, symbol graph, and file→chunk map
        self._save_manifest(manifest)
        bm25_path = self._index_dir / "bm25.json"
        self._bm25.save(bm25_path)
        self._graph.save(self._graph_path)
        self._file_chunks_path.write_text(
            json.dumps(self._file_chunks), encoding="utf-8"
        )

        log.info(
            "Indexing complete: %d new, %d modified, %d deleted, %d chunks indexed",
            stats["new"], stats["modified"], stats["deleted"], stats["chunks_indexed"],
        )
        return stats

    # ── Freshness checks ────────────────────────────────────────────────────────
    def check_freshness(self) -> dict[str, dict[str, str]]:
        """Return freshness state for every file in the index."""
        import subprocess
        from datetime import datetime
        current_files = {f.path: f for f in self._scanner.scan()}
        old_manifest = self._load_manifest()
        old_files = old_manifest.get("files", {})
        
        # Get git status to find uncommitted changes
        git_modified = set()
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self._workspace), capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        git_modified.add(str(self._workspace / parts[1]))
        except Exception:
            pass

        freshness = {}
        for path, old_sha in old_files.items():
            if path not in current_files:
                freshness[path] = {"state": "missing", "reason": "File not found on filesystem"}
            else:
                current_sha = current_files[path].sha256
                if current_sha != old_sha:
                    if path in git_modified:
                        freshness[path] = {"state": "edited_uncommitted", "reason": "File modified since last commit"}
                    else:
                        freshness[path] = {"state": "stale_index", "reason": "Filesystem SHA doesn't match index SHA"}
                else:
                    freshness[path] = {"state": "fresh", "reason": "Index matches filesystem"}
        return freshness