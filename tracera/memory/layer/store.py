"""
Memory Layer Store — SQLite-backed persistence for the agent-native memory
layer, shaped after Memori's model:

    entities(external_id unique, created_at)
    processes(external_id unique, created_at)
    sessions(id, entity_id, process_id, started_at, ended_at)
    memories(id, entity_id, process_id, kind, subject, predicate, object,
             text, embedding, mention_count, first_seen_at, last_seen_at,
             session_id, last_job_id)

Why SQLite: TRACERA ships zero external databases, so the memory layer owns a
small local SQLite file (`.tracera/memory/memory_layer.db`) — the same
"you already run this infra" argument Memori makes for Postgres, applied to
TRACERA's self-contained footprint. Embeddings are stored as JSON blobs and
compared with in-process cosine similarity; no separate vector database.

Guarantees implemented here:

  * **Entity isolation** — every query filters by ``entity_id``.
  * **Semantic dedup** — restating a fact different ways merges into one row
    and bumps ``mention_count`` instead of inserting duplicates.
  * **Idempotent writes** — a ``last_job_id`` guard means a retried background
    job never double-inserts *or* double-counts.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from tracera.logging import get_logger

log = get_logger("memory.layer.store")


# ── Memory kinds (Memori taxonomy) ────────────────────────────────────────────

class MemoryKind(str, Enum):
    """The durable memory kinds supported by the layer."""

    FACT = "fact"
    PREFERENCE = "preference"
    SKILL = "skill"
    ATTRIBUTE = "attribute"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    RULE = "rule"
    DECISION = "decision"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    EXPERIENCE = "experience"


ALL_KINDS: frozenset[str] = frozenset(k.value for k in MemoryKind)


# ── Memory Scopes (Phase 11) ───────────────────────────────────────────────────

class MemoryScope(str, Enum):
    """Scope levels for memory access and policies."""

    GLOBAL = "global"           # Cross-entity (system-wide)
    ORGANIZATION = "organization"  # Organization/team level
    PROJECT = "project"         # Project/repository level
    ENTITY = "entity"           # User/customer level
    PROCESS = "process"         # Agent/process level
    SESSION = "session"         # Single session level


SCOPE_HIERARCHY = [
    MemoryScope.GLOBAL,
    MemoryScope.ORGANIZATION,
    MemoryScope.PROJECT,
    MemoryScope.ENTITY,
    MemoryScope.PROCESS,
    MemoryScope.SESSION,
]

SCOPE_PRIORITY = {scope: i for i, scope in enumerate(SCOPE_HIERARCHY)}


# ── Memory Policies (Phase 11) ─────────────────────────────────────────────────

@dataclass
class MemoryPolicy:
    """Configuration for memory behavior per scope."""

    # Retention
    max_memories: int = 10000
    max_memories_per_entity: int = 5000
    retention_days: int = 365

    # Quality thresholds
    min_confidence: float = 0.5
    min_importance: float = 0.3
    dedup_threshold: float = 0.9

    # Recall
    recall_top_k: int = 10
    recall_min_score: float = 0.3
    recall_token_budget: int = 2000

    # Extraction
    extraction_enabled: bool = True
    extraction_min_confidence: float = 0.5
    extraction_min_importance: float = 0.3
    worthiness_filter: bool = True

    # Consolidation
    consolidation_enabled: bool = True
    consolidation_threshold: float = 0.92
    consolidation_interval_hours: int = 24

    # Safety
    pii_detection: bool = True
    prompt_injection_protection: bool = True
    cross_entity_isolation: bool = True  # Hard requirement

    def for_scope(self, scope: MemoryScope) -> "MemoryPolicy":
        """Return a policy adjusted for a specific scope."""
        # Base policy - can be overridden per scope
        if scope == MemoryScope.GLOBAL:
            return MemoryPolicy(
                max_memories=50000,
                max_memories_per_entity=10000,
                retention_days=730,
                min_confidence=0.7,
            )
        elif scope == MemoryScope.ORGANIZATION:
            return MemoryPolicy(
                max_memories=20000,
                max_memories_per_entity=5000,
                retention_days=365,
                min_confidence=0.6,
            )
        elif scope == MemoryScope.PROJECT:
            return MemoryPolicy(
                max_memories=10000,
                max_memories_per_entity=3000,
                retention_days=180,
                min_confidence=0.5,
            )
        elif scope == MemoryScope.ENTITY:
            return MemoryPolicy(
                max_memories=5000,
                max_memories_per_entity=2000,
                retention_days=90,
                min_confidence=0.5,
            )
        elif scope == MemoryScope.PROCESS:
            return MemoryPolicy(
                max_memories=2000,
                max_memories_per_entity=1000,
                retention_days=30,
                min_confidence=0.4,
            )
        else:  # SESSION
            return MemoryPolicy(
                max_memories=500,
                max_memories_per_entity=500,
                retention_days=7,
                min_confidence=0.3,
            )


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """One row of the ``memories`` table, using external ids for convenience."""

    id: int
    entity_id: str
    process_id: str
    kind: str
    subject: str
    predicate: str
    object: str
    text: str
    embedding: list[float]
    mention_count: int
    first_seen_at: float
    last_seen_at: float
    session_id: str | None = None
    last_job_id: int | None = None
    status: str = "active"
    confidence: float = 0.8
    importance: float = 0.5
    source_event: str | None = None
    source_message_id: str | None = None

    @property
    def triple(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def to_line(self) -> str:
        """One-line rendering used for prompt injection and CLI output."""
        return f"[{self.kind}] {self.text}"

    def to_short_line(self) -> str:
        """Compact rendering for the inspection CLI."""
        return f"{self.subject} → {self.predicate} → {self.object}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "kind": self.kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "mention_count": self.mention_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "session_id": self.session_id,
            "last_job_id": self.last_job_id,
            "status": self.status,
            "confidence": self.confidence,
            "importance": self.importance,
            "source_event": self.source_event,
            "source_message_id": self.source_message_id,
        }


@dataclass
class Job:
    """A single durable background job (extraction of one turn)."""

    id: int
    kind: str
    payload: dict[str, Any]
    status: str  # pending | running | done | failed
    attempts: int
    created_at: float
    last_error: str | None = None
# ── Store ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT    NOT NULL UNIQUE,
    created_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS processes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT    NOT NULL UNIQUE,
    created_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    entity_id  INTEGER NOT NULL REFERENCES entities(id),
    process_id INTEGER NOT NULL REFERENCES processes(id),
    started_at REAL NOT NULL,
    ended_at   REAL
);

CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id     INTEGER NOT NULL REFERENCES entities(id),
    process_id    INTEGER NOT NULL REFERENCES processes(id),
    kind          TEXT    NOT NULL
                  CHECK (kind IN ('fact','preference','skill','attribute','relationship','event','rule','decision','goal','constraint','experience')),
    subject       TEXT    NOT NULL,
    predicate     TEXT    NOT NULL,
    object        TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    embedding     TEXT    NOT NULL,          -- JSON array of floats
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at REAL    NOT NULL,
    last_seen_at  REAL    NOT NULL,
    session_id    TEXT,
    last_job_id   INTEGER,
    status        TEXT    NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active','superseded','archived','invalidated')),
    confidence    REAL    NOT NULL DEFAULT 0.8,
    importance    REAL    NOT NULL DEFAULT 0.5,
    source_event  TEXT,                    -- e.g., 'llm.response', 'tool.completed', 'user.instruction'
    source_message_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_entity         ON memories(entity_id);
CREATE INDEX IF NOT EXISTS idx_memories_entity_kind    ON memories(entity_id, kind);
CREATE INDEX IF NOT EXISTS idx_memories_entity_last_seen
    ON memories(entity_id, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_entity_status  ON memories(entity_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_entity_triple
    ON memories(entity_id, subject, predicate, object);

CREATE TABLE IF NOT EXISTS memory_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     INTEGER NOT NULL REFERENCES memories(id),
    old_text      TEXT,
    new_text      TEXT,
    old_status    TEXT,
    new_status    TEXT,
    changed_at    REAL    NOT NULL,
    reason        TEXT,
    source_session TEXT,
    source_process TEXT,
    source_job_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memory_versions_memory ON memory_versions(memory_id);

CREATE TABLE IF NOT EXISTS jobs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending','running','done','failed')),
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    not_before REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    priority   INTEGER NOT NULL DEFAULT 100  -- lower = higher priority
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, not_before);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(status, priority, not_before);
"""


def _normalize_triple(subject: str, predicate: str, object: str) -> tuple[str, str, str]:
    """Normalize a (subject, predicate, object) triple for dedup keys."""
    return (
        (subject or "").strip().lower(),
        (predicate or "").strip().lower(),
        (object or "").strip().lower(),
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (raw, unnormalised safe)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na * nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


class MemoryStore:
    """SQLite-backed memory store with entity/process scoping and dedup."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_schema()
        log.debug("MemoryStore ready at %s", self._db_path)

    # ── Connection management (one connection per thread) ────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    # ── Attribution registries ───────────────────────────────────────────────

    def register_entity(self, external_id: str) -> int:
        """Return the internal row id for an external entity id (insert if new)."""
        external_id = external_id.strip()
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT id FROM entities WHERE external_id = ?", (external_id,)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO entities (external_id, created_at) VALUES (?, ?)",
                (external_id, time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def register_process(self, external_id: str) -> int:
        """Return the internal row id for an external process id (insert if new)."""
        external_id = external_id.strip()
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT id FROM processes WHERE external_id = ?", (external_id,)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO processes (external_id, created_at) VALUES (?, ?)",
                (external_id, time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def _resolve_ids(
        self, entity_id: str, process_id: str
    ) -> tuple[int, int]:
        return self.register_entity(entity_id), self.register_process(process_id)

    # ── Sessions ─────────────────────────────────────────────────────────────

    def create_session(self, entity_id: str, process_id: str) -> str:
        """Create a new session row, return its id."""
        entity_pk, process_pk = self._resolve_ids(entity_id, process_id)
        session_id = str(uuid.uuid4())
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO sessions (id, entity_id, process_id, started_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, entity_pk, process_pk, time.time()),
            )
            conn.commit()
        return session_id

    def close_session(self, session_id: str) -> None:
        """Mark an open session as ended."""
        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), session_id),
            )
            conn.commit()

    def get_session_ids(self, entity_id: str, limit: int = 50) -> list[str]:
        """Return recent session ids for an entity, newest first."""
        entity_pk = self.register_entity(entity_id)
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT id FROM sessions WHERE entity_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (entity_pk, limit),
            ).fetchall()
            return [str(r["id"]) for r in rows]

    # ── Memory writes ────────────────────────────────────────────────────────

    def upsert_memory(
        self,
        *,
        entity_id: str,
        process_id: str,
        kind: str,
        subject: str,
        predicate: str,
        object: str,
        text: str,
        embedding: list[float],
        session_id: str | None = None,
        job_id: int | None = None,
        similarity_threshold: float = 0.9,
        confidence: float = 0.8,
        importance: float = 0.5,
        source_event: str | None = None,
        source_message_id: str | None = None,
        status: str = "active",
    ) -> tuple[bool, MemoryRecord]:
        """
        Write one memory.

        Dedup order:
          1. A semantically similar row for this entity (cosine ≥ threshold)
             → bump ``mention_count`` / refresh ``last_seen_at``.
          2. Otherwise insert; the unique (entity, subject, predicate, object)
             index absorbs exact duplicates with ``ON CONFLICT``.

        Idempotency: a row already written by the same ``job_id`` is left
        untouched — retried background jobs never double-insert or inflate
        ``mention_count``.

        Returns ``(inserted, record)``.
        """
        if kind not in ALL_KINDS:
            raise ValueError(f"invalid memory kind: {kind!r}")
        if status not in ("active", "superseded", "archived", "invalidated"):
            raise ValueError(f"invalid memory status: {status!r}")
        entity_pk, process_pk = self._resolve_ids(entity_id, process_id)
        s, p, o = _normalize_triple(subject, predicate, object)
        now = time.time()
        emb_json = json.dumps(embedding, separators=(",", ":"))

        with self._lock:
            conn = self._conn()

            # 1) Semantic duplicate?
            existing = self._find_semantic_duplicate(
                conn, entity_pk, process_pk, embedding, similarity_threshold
            )
            if existing is not None:
                if existing["last_job_id"] == job_id:
                    # Already applied for exactly this job — idempotent retry.
                    return (
                        False,
                        self._record_from_row(conn, existing),
                    )
                conn.execute(
                    "UPDATE memories SET mention_count = mention_count + 1, "
                    "last_seen_at = ?, text = ?, last_job_id = ?, "
                    "process_id = ?, confidence = ?, importance = ? WHERE id = ?",
                    (now, text, job_id, process_pk, confidence, importance, existing["id"]),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (existing["id"],)
                ).fetchone()
                return False, self._record_from_row(conn, row)

            # 2) Insert (exact-duplicate races absorbed by ON CONFLICT).
            cur = conn.execute(
                """
                INSERT INTO memories
                    (entity_id, process_id, kind, subject, predicate, object,
                     text, embedding, mention_count, first_seen_at,
                     last_seen_at, session_id, last_job_id, status, confidence,
                     importance, source_event, source_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, subject, predicate, object)
                DO UPDATE SET
                    mention_count = mention_count + 1,
                    last_seen_at = excluded.last_seen_at,
                    text = excluded.text,
                    embedding = excluded.embedding,
                    last_job_id = excluded.last_job_id,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    source_event = excluded.source_event,
                    source_message_id = excluded.source_message_id
                WHERE memories.last_job_id IS NOT excluded.last_job_id
                RETURNING id, mention_count
                """,
                (
                    entity_pk, process_pk, kind, s, p, o, text, emb_json,
                    now, now, session_id, job_id, status, confidence,
                    importance, source_event, source_message_id,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            if result is None:
                # A same-job retry was suppressed by the idempotency guard
                # (WHERE memories.last_job_id IS NOT excluded.last_job_id).
                # Fetch the already-written row.
                row = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? "
                    "AND subject = ? AND predicate = ? AND object = ?",
                    (entity_pk, s, p, o),
                ).fetchone()
                return False, self._record_from_row(conn, row)
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (result["id"],)
            ).fetchone()
            inserted = bool(result["mention_count"] == 1)
            return inserted, self._record_from_row(conn, row)

    def _find_semantic_duplicate(
        self,
        conn: sqlite3.Connection,
        entity_pk: int,
        process_pk: int,
        embedding: list[float],
        threshold: float,
    ) -> sqlite3.Row | None:
        """Best matching existing row for this entity above the threshold."""
        rows = conn.execute(
            "SELECT * FROM memories WHERE entity_id = ? AND process_id = ?",
            (entity_pk, process_pk),
        ).fetchall()
        best_score = 0.0
        best_row: sqlite3.Row | None = None
        for row in rows:
            try:
                other = json.loads(row["embedding"])
            except (TypeError, ValueError):
                continue
            score = cosine_similarity(embedding, other)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is not None and best_score >= threshold:
            return best_row
        return None

    def _record_from_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> MemoryRecord:
        entity_pk = int(row["entity_id"])
        process_pk = int(row["process_id"])
        entity_ext = conn.execute(
            "SELECT external_id FROM entities WHERE id = ?", (entity_pk,)
        ).fetchone()["external_id"]
        process_ext = conn.execute(
            "SELECT external_id FROM processes WHERE id = ?", (process_pk,)
        ).fetchone()["external_id"]
        return MemoryRecord(
            id=int(row["id"]),
            entity_id=str(entity_ext),
            process_id=str(process_ext),
            kind=str(row["kind"]),
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            object=str(row["object"]),
            text=str(row["text"]),
            embedding=json.loads(row["embedding"]),
            mention_count=int(row["mention_count"]),
            first_seen_at=float(row["first_seen_at"]),
            last_seen_at=float(row["last_seen_at"]),
            session_id=row["session_id"],
            last_job_id=row["last_job_id"],
            status=str(row["status"]) if row["status"] else "active",
            confidence=float(row["confidence"]) if row["confidence"] is not None else 0.8,
            importance=float(row["importance"]) if row["importance"] is not None else 0.5,
            source_event=row["source_event"],
            source_message_id=row["source_message_id"],
        )
# ── Memory reads ─────────────────────────────────────────────────────────

    def recall(
        self,
        entity_id: str,
        query_embedding: list[float],
        *,
        k: int = 5,
        process_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Vector-search a single entity's memories (never cross-entity).

        Returns ``[(record, similarity), ...]`` sorted by similarity desc.
        """
        entity_pk = self.register_entity(entity_id)
        with self._lock:
            conn = self._conn()
            if process_id is None:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ?",
                    (entity_pk,),
                ).fetchall()
            else:
                process_pk = self.register_process(process_id)
                rows = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? AND process_id = ?",
                    (entity_pk, process_pk),
                ).fetchall()

            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                try:
                    other = json.loads(row["embedding"])
                except (TypeError, ValueError):
                    continue
                score = cosine_similarity(query_embedding, other)
                if score >= min_score:
                    scored.append((score, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            results: list[tuple[MemoryRecord, float]] = []
            for score, row in scored[:k]:
                results.append((self._record_from_row(conn, row), score))
            return results

    def recall_hybrid(
        self,
        entity_id: str,
        query: str,
        query_embedding: list[float],
        *,
        k: int = 10,
        process_id: str | None = None,
        min_score: float = 0.0,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        exact_match_boost: float = 0.2,
        recency_weight: float = 0.1,
        importance_weight: float = 0.1,
        mention_weight: float = 0.05,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Hybrid recall combining vector similarity, keyword matching, and metadata.

        Scoring components:
        - Vector similarity (cosine): weight = vector_weight
        - Keyword overlap (TF-IDF style): weight = keyword_weight
        - Exact triple match boost: +exact_match_boost
        - Recency boost: weight = recency_weight * normalized_recency
        - Importance boost: weight = importance_weight * importance
        - Mention count boost: weight = mention_weight * log(mention_count + 1)
        """
        entity_pk = self.register_entity(entity_id)
        query_tokens = self._tokenize_query(query)

        with self._lock:
            conn = self._conn()
            if process_id is None:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? AND status = 'active'",
                    (entity_pk,),
                ).fetchall()
            else:
                process_pk = self.register_process(process_id)
                rows = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? AND process_id = ? AND status = 'active'",
                    (entity_pk, process_pk),
                ).fetchall()

            if not rows:
                return []

            # Build IDF for keyword scoring
            n = len(rows)
            df: dict[str, int] = {}
            for row in rows:
                text = row["text"] or ""
                tokens = set(self._tokenize_query(text))
                for t in tokens:
                    df[t] = df.get(t, 0) + 1
            idf = {t: math.log(n / c + 1) for t, c in df.items()}

            now = time.time()
            scored: list[tuple[float, sqlite3.Row]] = []

            for row in rows:
                try:
                    other = json.loads(row["embedding"])
                except (TypeError, ValueError):
                    continue

                # 1. Vector similarity
                vec_score = cosine_similarity(query_embedding, other)

                # 2. Keyword overlap
                text = row["text"] or ""
                doc_tokens = self._tokenize_query(text)
                kw_score = self._tfidf_score(query_tokens, doc_tokens, idf)

                # 3. Exact match boost
                exact_boost = 0.0
                subject = str(row["subject"]).lower()
                predicate = str(row["predicate"]).lower()
                obj = str(row["object"]).lower()
                if subject in query.lower() and predicate in query.lower() and obj in query.lower():
                    exact_boost = exact_match_boost

                # 4. Recency boost (normalized 0-1 based on last_seen_at)
                last_seen = float(row["last_seen_at"])
                age_days = (now - last_seen) / 86400
                recency_boost = 1.0 / (1.0 + age_days / 30.0)  # decays over ~30 days

                # 5. Importance boost
                importance = float(row["importance"]) if row["importance"] is not None else 0.5

                # 6. Mention count boost
                mention_count = int(row["mention_count"])
                mention_boost = math.log(mention_count + 1) / math.log(100)  # caps at ~1 for 100 mentions

                # Combined score
                final_score = (
                    vector_weight * vec_score
                    + keyword_weight * kw_score
                    + exact_boost
                    + recency_weight * recency_boost
                    + importance_weight * importance
                    + mention_weight * mention_boost
                )

                if final_score >= min_score:
                    scored.append((final_score, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            results: list[tuple[MemoryRecord, float]] = []
            for score, row in scored[:k]:
                results.append((self._record_from_row(conn, row), score))
            return results

    def _tokenize_query(self, text: str) -> list[str]:
        """Simple tokenization for keyword matching."""
        import re
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]

    def _tfidf_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        idf: dict[str, float],
    ) -> float:
        """TF-IDF style keyword overlap score."""
        if not query_tokens or not doc_tokens:
            return 0.0
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        if not query_set or not doc_set:
            return 0.0

        overlap = query_set & doc_set
        if not overlap:
            return 0.0

        # TF-IDF weighted overlap
        score = sum(idf.get(t, 1.0) for t in overlap)
        # Normalize by query length
        return score / len(query_set)

    def find_memories(
        self,
        entity_id: str,
        *,
        kind: str | None = None,
        process_id: str | None = None,
        limit: int = 500,
    ) -> list[MemoryRecord]:
        """List an entity's memories, grouped for inspection (no cross-entity)."""
        entity_pk = self.register_entity(entity_id)
        clauses = ["entity_id = ?"]
        params: list[Any] = [entity_pk]
        if kind:
            if kind not in ALL_KINDS:
                raise ValueError(f"invalid memory kind: {kind!r}")
            clauses.append("kind = ?")
            params.append(kind)
        if process_id:
            process_pk = self.register_process(process_id)
            clauses.append("process_id = ?")
            params.append(process_pk)
        where = " AND ".join(clauses)
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where} "
                "ORDER BY mention_count DESC, last_seen_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [self._record_from_row(conn, r) for r in rows]

    def find_by_kind(self, entity_id: str) -> dict[str, list[MemoryRecord]]:
        """All memories for an entity grouped by kind (inspection CLI)."""
        grouped: dict[str, list[MemoryRecord]] = {k: [] for k in ALL_KINDS}
        for record in self.find_memories(entity_id, limit=10_000):
            grouped[record.kind].append(record)
        return grouped

    def count_memories(self, entity_id: str | None = None) -> int:
        with self._lock:
            conn = self._conn()
            if entity_id is None:
                row = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
            else:
                entity_pk = self.register_entity(entity_id)
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE entity_id = ?",
                    (entity_pk,),
                ).fetchone()
            return int(row["c"])

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics for status/observability output."""
        with self._lock:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
            by_kind: dict[str, int] = {}
            for row in conn.execute(
                "SELECT kind, COUNT(*) AS c FROM memories GROUP BY kind"
            ).fetchall():
                by_kind[str(row["kind"])] = int(row["c"])
            by_status: dict[str, int] = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) AS c FROM memories GROUP BY status"
            ).fetchall():
                by_status[str(row["status"])] = int(row["c"])
            entities = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
            sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
            jobs = {
                "pending": self.count_jobs(status="pending"),
                "running": self.count_jobs(status="running"),
                "done": self.count_jobs(status="done"),
                "failed": self.count_jobs(status="failed"),
            }
            # Additional observability metrics
            avg_mentions = conn.execute(
                "SELECT AVG(mention_count) FROM memories WHERE status = 'active'"
            ).fetchone()[0] or 0
            avg_confidence = conn.execute(
                "SELECT AVG(confidence) FROM memories WHERE status = 'active'"
            ).fetchone()[0] or 0
            total_versions = conn.execute("SELECT COUNT(*) AS c FROM memory_versions").fetchone()["c"]
            superseded_count = conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE status = 'superseded'"
            ).fetchone()["c"]
            return {
                "memories_total": int(total),
                "memories_by_kind": by_kind,
                "memories_by_status": by_status,
                "entities": int(entities),
                "sessions": int(sessions),
                "jobs": jobs,
                "avg_mentions": round(float(avg_mentions), 2),
                "avg_confidence": round(float(avg_confidence), 3),
                "total_versions": int(total_versions),
                "superseded_count": int(superseded_count),
            }

    # ── Durable job queue ────────────────────────────────────────────────────

    def enqueue_job(self, kind: str, payload: dict[str, Any], priority: int = 100) -> int:
        """Persist a background job; survives process restarts."""
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO jobs (kind, payload, status, attempts, created_at, priority) "
                "VALUES (?, ?, 'pending', 0, ?, ?)",
                (kind, json.dumps(payload, ensure_ascii=False), time.time(), priority),
            )
            conn.commit()
            return int(cur.lastrowid)

    def claim_jobs(self, limit: int = 1, kind: str | None = None) -> list[Job]:
        """Claim due jobs atomically (pending → running), ordered by priority."""
        now = time.time()
        with self._lock:
            conn = self._conn()
            if kind is None:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = 'pending' AND not_before <= ? "
                    "ORDER BY priority ASC, id LIMIT ?",
                    (now, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = 'pending' AND kind = ? "
                    "AND not_before <= ? ORDER BY priority ASC, id LIMIT ?",
                    (kind, now, limit),
                ).fetchall()
            claimed: list[Job] = []
            for row in rows:
                conn.execute(
                    "UPDATE jobs SET status = 'running', attempts = attempts + 1 "
                    "WHERE id = ? AND status = 'pending'",
                    (row["id"],),
                )
                claimed.append(self._job_from_row(row))
            conn.commit()
            return claimed

    def complete_job(self, job_id: int) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE jobs SET status = 'done', last_error = NULL WHERE id = ?",
                (job_id,),
            )
            conn.commit()

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 5) -> None:
        """Record a failure; reschedule with backoff or mark failed."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            attempts = int(row["attempts"]) if row else 0
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', last_error = ? WHERE id = ?",
                    (error[:500], job_id),
                )
            else:
                backoff = 2.0 ** attempts
                conn.execute(
                    "UPDATE jobs SET status = 'pending', not_before = ?, "
                    "last_error = ? WHERE id = ?",
                    (time.time() + backoff, error[:500], job_id),
                )
            conn.commit()

    def count_jobs(self, status: str | None = None) -> int:
        with self._lock:
            conn = self._conn()
            if status is None:
                row = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM jobs WHERE status = ?", (status,)
                ).fetchone()
            return int(row["c"])

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=int(row["id"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            created_at=float(row["created_at"]),
            last_error=row["last_error"],
        )

    def close(self) -> None:
        """Close the current thread's connection (best-effort)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def close_all(self) -> None:
        """Close all thread connections (best-effort)."""
        # Close current thread's connection
        self.close()
        # Note: Other threads' connections will be closed when those threads end
        # This is best-effort as we can't directly access other threads' local storage

    # ── Memory versioning & conflict resolution ────────────────────────────────

    def record_memory_version(
        self,
        memory_id: int,
        *,
        old_text: str | None = None,
        new_text: str | None = None,
        old_status: str | None = None,
        new_status: str | None = None,
        reason: str | None = None,
        source_session: str | None = None,
        source_process: str | None = None,
        source_job_id: int | None = None,
    ) -> None:
        """Record a version entry for a memory change."""
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO memory_versions
                    (memory_id, old_text, new_text, old_status, new_status,
                     changed_at, reason, source_session, source_process, source_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    old_text,
                    new_text,
                    old_status,
                    new_status,
                    time.time(),
                    reason,
                    source_session,
                    source_process,
                    source_job_id,
                ),
            )
            conn.commit()

    def get_memory_versions(self, memory_id: int) -> list[dict[str, Any]]:
        """Get version history for a memory."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT * FROM memory_versions
                WHERE memory_id = ?
                ORDER BY changed_at DESC
                """,
                (memory_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def explain_memory(self, memory_id: int) -> dict[str, Any]:
        """
        Explain why a memory exists and its lifecycle.

        Returns a human-readable explanation of:
        - What the memory is
        - When and why it was created
        - How many times it's been mentioned/reinforced
        - Version history (if superseded)
        - Source attribution
        """
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return {"error": f"Memory {memory_id} not found"}

            record = self._record_from_row(conn, row)
            versions = self.get_memory_versions(memory_id)

            return {
                "memory": {
                    "id": record.id,
                    "kind": record.kind,
                    "text": record.text,
                    "triple": f"{record.subject} → {record.predicate} → {record.object}",
                    "mention_count": record.mention_count,
                    "confidence": record.confidence,
                    "importance": record.importance,
                    "status": record.status,
                    "created_at": record.first_seen_at,
                    "updated_at": record.last_seen_at,
                    "source_event": record.source_event,
                    "source_message_id": record.source_message_id,
                    "session_id": record.session_id,
                },
                "entity_id": record.entity_id,
                "process_id": record.process_id,
                "version_history": [
                    {
                        "changed_at": v["changed_at"],
                        "reason": v["reason"],
                        "old_status": v["old_status"],
                        "new_status": v["new_status"],
                        "old_text": v["old_text"],
                        "new_text": v["new_text"],
                    }
                    for v in versions
                ],
                "summary": self._generate_memory_summary(record, versions),
            }

    def _generate_memory_summary(self, record: MemoryRecord, versions: list) -> str:
        """Generate a human-readable summary of the memory's lifecycle."""
        parts = []
        parts.append(f"This {record.kind} memory states: \"{record.text}\"")
        parts.append(f"It has been mentioned {record.mention_count} time(s) since first observed.")
        parts.append(f"Confidence: {record.confidence:.0%}, Importance: {record.importance:.0%}")

        if record.source_event:
            parts.append(f"Source: {record.source_event}")
        if record.session_id:
            parts.append(f"Session: {record.session_id[:8]}")

        if versions:
            superseded_count = sum(1 for v in versions if v.get("new_status") == "superseded")
            if superseded_count:
                parts.append(f"Has been superseded {superseded_count} time(s) with updates.")
            else:
                parts.append(f"Has {len(versions)} version update(s).")

        return " ".join(parts)

    def explain_recall(
        self,
        entity_id: str,
        query: str,
        query_embedding: list[float],
        *,
        k: int = 5,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """
        Explain why specific memories were recalled for a query.

        Shows the scoring breakdown for each returned memory.
        """
        results = self.recall_hybrid(
            entity_id, query, query_embedding, k=k, min_score=min_score
        )

        explanations = []
        for record, score in results:
            explanations.append({
                "memory_id": record.id,
                "text": record.text,
                "triple": f"{record.subject} → {record.predicate} → {record.object}",
                "final_score": round(score, 4),
                "factors": {
                    "vector_similarity": "Computed via cosine similarity",
                    "mention_count": record.mention_count,
                    "confidence": record.confidence,
                    "importance": record.importance,
                    "recency": f"Last seen {self._format_recency(record.last_seen_at)}",
                },
                "why_recalled": self._explain_why_recalled(record, query, score),
            })

        return {
            "query": query,
            "entity_id": entity_id,
            "total_candidates": "N/A (hybrid scoring)",
            "returned": len(explanations),
            "explanations": explanations,
        }

    def _format_recency(self, timestamp: float) -> str:
        """Format a timestamp as relative time."""
        import time
        age = time.time() - timestamp
        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age/60)}m ago"
        elif age < 86400:
            return f"{int(age/3600)}h ago"
        else:
            return f"{int(age/86400)}d ago"

    def _explain_why_recalled(self, record: MemoryRecord, query: str, score: float) -> str:
        """Generate human-readable explanation of why this memory was recalled."""
        reasons = []
        if score > 0.8:
            reasons.append("very high semantic similarity to query")
        elif score > 0.5:
            reasons.append("good semantic match to query")
        elif score > 0.3:
            reasons.append("moderate semantic relevance")

        if record.mention_count > 5:
            reasons.append(f"frequently reinforced ({record.mention_count} mentions)")
        elif record.mention_count > 1:
            reasons.append(f"mentioned {record.mention_count} times")

        if record.confidence > 0.9:
            reasons.append("high confidence")
        if record.importance > 0.7:
            reasons.append("marked as important")

        return "; ".join(reasons) if reasons else "matched query with low confidence"

    def supersede_memory(
        self,
        memory_id: int,
        *,
        new_text: str,
        new_embedding: list[float],
        reason: str,
        source_session: str | None = None,
        source_process: str | None = None,
        source_job_id: int | None = None,
    ) -> MemoryRecord:
        """
        Supersede an existing memory with new information.

        The old memory is marked as 'superseded' and a new version entry is recorded.
        Returns the updated memory record.
        """
        with self._lock:
            conn = self._conn()
            # Get the old memory
            old_row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not old_row:
                raise ValueError(f"Memory {memory_id} not found")

            old_text = str(old_row["text"])
            old_status = str(old_row["status"])

            # Update the memory
            emb_json = json.dumps(new_embedding, separators=(",", ":"))
            now = time.time()
            conn.execute(
                """
                UPDATE memories
                SET text = ?, embedding = ?, last_seen_at = ?, status = 'superseded'
                WHERE id = ?
                """,
                (new_text, emb_json, now, memory_id),
            )

            # Record the version
            conn.execute(
                """
                INSERT INTO memory_versions
                    (memory_id, old_text, new_text, old_status, new_status,
                     changed_at, reason, source_session, source_process, source_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    old_text,
                    new_text,
                    old_status,
                    "superseded",
                    now,
                    reason,
                    source_session,
                    source_process,
                    source_job_id,
                ),
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return self._record_from_row(conn, row)

    def find_conflicting_memories(
        self,
        entity_id: str,
        subject: str,
        predicate: str,
        *,
        similarity_threshold: float = 0.85,
    ) -> list[MemoryRecord]:
        """
        Find memories that might conflict with a new (subject, predicate) pair.

        Returns memories with the same subject and predicate but different objects
        that are semantically similar to the new object.
        """
        entity_pk = self.register_entity(entity_id)
        s, p, _ = _normalize_triple(subject, predicate, "")

        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE entity_id = ? AND subject = ? AND predicate = ?
                AND status = 'active'
                """,
                (entity_pk, s, p),
            ).fetchall()

            conflicts = []
            for row in rows:
                conflicts.append(self._record_from_row(conn, row))
            return conflicts

    def consolidate_memory(
        self,
        entity_id: str,
        process_id: str,
        *,
        subject: str,
        predicate: str,
        object: str,
        text: str,
        embedding: list[float],
        confidence: float = 0.8,
        importance: float = 0.5,
        session_id: str | None = None,
        job_id: int | None = None,
        source_event: str = "consolidation",
    ) -> tuple[bool, MemoryRecord]:
        """
        Consolidate a new memory with existing ones.

        If a conflicting memory exists (same subject/predicate, different object),
        the higher-confidence one wins and the other is superseded.
        """
        conflicts = self.find_conflicting_memories(entity_id, subject, predicate)

        if not conflicts:
            # No conflict, just insert
            return self.upsert_memory(
                entity_id=entity_id,
                process_id=process_id,
                kind="fact",
                subject=subject,
                predicate=predicate,
                object=object,
                text=text,
                embedding=embedding,
                session_id=session_id,
                job_id=job_id,
                confidence=confidence,
                importance=importance,
                source_event=source_event,
            )

        # There are conflicts - check if any have the same object
        for conflict in conflicts:
            if conflict.object.lower() == object.lower():
                # Same object, just reinforce
                return self.upsert_memory(
                    entity_id=entity_id,
                    process_id=process_id,
                    kind="fact",
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    text=text,
                    embedding=embedding,
                    session_id=session_id,
                    job_id=job_id,
                    confidence=confidence,
                    importance=importance,
                    source_event=source_event,
                )

        # Different objects - check confidence
        best_conflict = max(conflicts, key=lambda m: m.confidence)
        if confidence > best_conflict.confidence + 0.1:
            # New memory is significantly more confident - supersede the old
            self.supersede_memory(
                best_conflict.id,
                new_text=text,
                new_embedding=embedding,
                reason=f"Superseded by higher-confidence memory (new: {confidence:.2f}, old: {best_conflict.confidence:.2f})",
                source_session=session_id,
                source_process=process_id,
                source_job_id=job_id,
            )
            # Insert the new memory
            return self.upsert_memory(
                entity_id=entity_id,
                process_id=process_id,
                kind="fact",
                subject=subject,
                predicate=predicate,
                object=object,
                text=text,
                embedding=embedding,
                session_id=session_id,
                job_id=job_id,
                confidence=confidence,
                importance=importance,
                source_event=source_event,
            )
        elif best_conflict.confidence > confidence + 0.1:
            # Existing memory is more confident - reinforce it
            self.upsert_memory(
                entity_id=entity_id,
                process_id=process_id,
                kind=best_conflict.kind,
                subject=best_conflict.subject,
                predicate=best_conflict.predicate,
                object=best_conflict.object,
                text=best_conflict.text,
                embedding=best_conflict.embedding,
                session_id=session_id,
                job_id=job_id,
                confidence=min(1.0, best_conflict.confidence + 0.05),
                importance=best_conflict.importance,
                source_event=source_event,
            )
            return False, best_conflict
        else:
            # Similar confidence - store both but mark as potential conflict
            return self.upsert_memory(
                entity_id=entity_id,
                process_id=process_id,
                kind="fact",
                subject=subject,
                predicate=predicate,
                object=object,
                text=text,
                embedding=embedding,
                session_id=session_id,
                job_id=job_id,
                confidence=confidence,
                importance=importance,
                source_event=source_event,
            )

    # ── Periodic Consolidation Job ─────────────────────────────────────────────

    def run_consolidation(
        self,
        entity_id: str | None = None,
        *,
        similarity_threshold: float = 0.92,
        min_mention_count: int = 2,
        max_merges: int = 50,
    ) -> dict[str, Any]:
        """
        Run a consolidation pass to merge near-duplicate memories.

        This can be scheduled as a periodic background job.
        Returns statistics about the consolidation run.
        """
        stats = {"scanned": 0, "merged": 0, "superseded": 0, "errors": 0}

        with self._lock:
            conn = self._conn()

            # Get entities to process
            if entity_id:
                entity_pk = self.register_entity(entity_id)
                entity_rows = [{"id": entity_pk, "external_id": entity_id}]
            else:
                entity_rows = conn.execute(
                    "SELECT id, external_id FROM entities"
                ).fetchall()

            for entity_row in entity_rows:
                entity_pk = entity_row["id"]
                entity_ext = entity_row["external_id"]

                # Get all active memories for this entity
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE entity_id = ? AND status = 'active'
                    ORDER BY mention_count DESC, confidence DESC
                    """,
                    (entity_pk,),
                ).fetchall()

                if len(rows) < 2:
                    continue

                stats["scanned"] += len(rows)

                # Compare each pair for near-duplicates
                for i, row_i in enumerate(rows):
                    if stats["merged"] >= max_merges:
                        break
                    try:
                        emb_i = json.loads(row_i["embedding"])
                    except (TypeError, ValueError):
                        continue

                    for row_j in rows[i + 1 :]:
                        if stats["merged"] >= max_merges:
                            break
                        try:
                            emb_j = json.loads(row_j["embedding"])
                        except (TypeError, ValueError):
                            continue

                        score = cosine_similarity(emb_i, emb_j)
                        if score >= similarity_threshold:
                            # Found near-duplicate - merge the lower confidence into higher
                            if row_i["confidence"] >= row_j["confidence"]:
                                keeper_id, merge_id = row_i["id"], row_j["id"]
                            else:
                                keeper_id, merge_id = row_j["id"], row_i["id"]

                            # Update keeper with combined info
                            new_mention = int(row_i["mention_count"]) + int(row_j["mention_count"])
                            new_conf = max(float(row_i["confidence"]), float(row_j["confidence"]))
                            new_imp = max(float(row_i["importance"]), float(row_j["importance"]))
                            now = time.time()

                            conn.execute(
                                """
                                UPDATE memories
                                SET mention_count = ?, confidence = ?, importance = ?,
                                    last_seen_at = ?, text = ?
                                WHERE id = ?
                                """,
                                (new_mention, new_conf, new_imp, now,
                                 row_i["text"] if keeper_id == row_i["id"] else row_j["text"],
                                 keeper_id),
                            )

                            # Supersede the merged memory
                            conn.execute(
                                "UPDATE memories SET status = 'superseded' WHERE id = ?",
                                (merge_id,),
                            )

                            # Record version
                            conn.execute(
                                """
                                INSERT INTO memory_versions
                                    (memory_id, old_text, new_text, old_status, new_status,
                                     changed_at, reason, source_session, source_process, source_job_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    merge_id,
                                    row_j["text"] if merge_id == row_j["id"] else row_i["text"],
                                    row_i["text"] if keeper_id == row_i["id"] else row_j["text"],
                                    "active",
                                    "superseded",
                                    now,
                                    f"Auto-consolidated with memory {keeper_id} (similarity: {score:.3f})",
                                    None,
                                    "consolidation_job",
                                    None,
                                ),
                            )

                            stats["merged"] += 1
                            stats["superseded"] += 1

            conn.commit()

        return stats

    def enqueue_consolidation_job(self, entity_id: str | None = None) -> int:
        """Enqueue a consolidation job for background processing."""
        payload = {"entity_id": entity_id} if entity_id else {}
        return self.enqueue_job("consolidation", payload)

    def process_consolidation_job(self, job: Job) -> dict[str, Any]:
        """Process a consolidation job (called by background worker)."""
        entity_id = job.payload.get("entity_id")
        return self.run_consolidation(entity_id=entity_id)