"""
Phase 18 — Vector Index (LanceDB).

Persists CodeChunk embeddings into a local LanceDB table and exposes
similarity search with optional metadata filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tracera.indexer.schema import CodeChunk, SymbolType
from tracera.logging import get_logger

log = get_logger("retrieval.vector_store")

_TABLE_NAME = "code_chunks"


class VectorStore:
    """
    LanceDB-backed vector store for CodeChunk embeddings.

    Schema:
        id          — chunk ID
        content     — raw code text
        file_path   — relative file path
        language    — detected language
        symbol      — primary symbol name (optional)
        symbol_type — class | function | method | ...
        parent      — parent symbol name (optional)
        start_line  — start line in the file
        end_line    — end line in the file
        vector      — embedding (float32[])
    """

    def __init__(self, uri: str | Path, dimension: int | None = None) -> None:
        self._uri = str(uri)
        self._dimension = dimension
        self._db: Any = None
        self._table: Any = None

    # ── Lazy connect ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._db is not None:
            return
        try:
            import lancedb
        except ImportError:
            raise RuntimeError("lancedb not installed. Run: uv add lancedb")
        log.debug("Connecting to LanceDB at %s", self._uri)
        self._db = lancedb.connect(self._uri)

    def existing_dimension(self) -> int | None:
        """
        Return the vector dimension of an already-created table, or None.
        Lets the pipeline reuse the stored dimension instead of assuming 384.
        """
        try:
            self._connect()
            if _TABLE_NAME not in self._db.list_tables().tables:
                return None
            table = self._db.open_table(_TABLE_NAME)
            field = table.schema.field("vector")
            size = getattr(field.type, "list_size", None)
            if size and size > 0:
                return int(size)
            return None
        except Exception:
            return None

    def _get_or_create_table(self) -> Any:
        if self._table is not None:
            return self._table
        self._connect()
        if _TABLE_NAME in self._db.list_tables().tables:
            self._table = self._db.open_table(_TABLE_NAME)
            # Adopt the real dimension from the existing table
            if self._dimension is None:
                dim = self.existing_dimension()
                if dim:
                    self._dimension = dim
        else:
            if self._dimension is None:
                raise RuntimeError(
                    "VectorStore dimension is required to create a new table."
                )
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("language", pa.string()),
                pa.field("symbol", pa.string()),
                pa.field("symbol_type", pa.string()),
                pa.field("parent", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
            ])
            self._table = self._db.create_table(_TABLE_NAME, schema=schema)
            log.info("Created LanceDB table: %s (dim=%d)", _TABLE_NAME, self._dimension)
        return self._table

    # NOTE: LanceDB deprecated table_names() in favor of list_tables().
    # All call sites now use list_tables().

    # ── Insert ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Upsert code chunks with their embeddings into the vector store."""
        if not chunks:
            return
        table = self._get_or_create_table()

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append({
                "id": chunk.id,
                "content": chunk.content,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "symbol": chunk.primary_symbol or "",
                "symbol_type": chunk.symbol_type.value if chunk.symbol_type else "",
                "parent": chunk.parent_symbol or "",
                "start_line": chunk.range.start_line,
                "end_line": chunk.range.end_line,
                "vector": embedding,
            })

        # Merge (overwrite) by deleting existing IDs first
        ids = [r["id"] for r in rows]
        id_list = ", ".join(f"'{i}'" for i in ids)
        try:
            table.delete(f"id IN ({id_list})")
        except Exception:
            pass  # Table may be empty

        table.add(rows)
        log.debug("Upserted %d chunks into LanceDB", len(rows))

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        language: str | None = None,
        symbol_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for nearest neighbours.

        Returns list of row dicts sorted by similarity (descending).
        """
        table = self._get_or_create_table()

        query = table.search(query_embedding).limit(k)
        if language:
            query = query.where(f"language = '{language}'", prefilter=True)
        if symbol_type:
            query = query.where(f"symbol_type = '{symbol_type}'", prefilter=True)

        results = query.to_list()
        return results

    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a specific file (for incremental re-indexing)."""
        table = self._get_or_create_table()
        table.delete(f"file_path = '{file_path}'")
        log.debug("Deleted chunks for file: %s", file_path)

    @property
    def count(self) -> int:
        """Number of vectors stored."""
        try:
            table = self._get_or_create_table()
            return table.count_rows()
        except Exception:
            return 0
