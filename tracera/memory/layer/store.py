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
import re
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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

    GLOBAL = "global"  # Cross-entity (system-wide)
    ORGANIZATION = "organization"  # Organization/team level
    PROJECT = "project"  # Project/repository level
    ENTITY = "entity"  # User/customer level
    PROCESS = "process"  # Agent/process level
    SESSION = "session"  # Single session level


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

    def for_scope(self, scope: MemoryScope) -> MemoryPolicy:
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
    # ── bi-temporal validity (schema v2) ──
    valid_at: float | None = None
    invalid_at: float | None = None
    expired_at: float | None = None
    superseded_by: int | None = None
    # ── entity resolution (schema v2) ──
    subject_key: str | None = None
    object_key: str | None = None
    # ── usage telemetry (schema v2) ──
    recall_count: int = 0
    last_recalled_at: float | None = None
    useful_count: int = 0
    harmful_count: int = 0
    decay_score: float | None = None
    # ── provenance / hierarchy (schema v2) ──
    scope: str = "entity"
    source_kind: str | None = None
    summary_of: str | None = None

    @property
    def triple(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    @property
    def is_current(self) -> bool:
        """True when the fact is believed *and* not yet invalidated."""
        return self.status == "active" and self.invalid_at is None

    def valid_window(self) -> str:
        """Human-readable validity window, e.g. ``2026-01-04 → present``."""
        def _fmt(ts: float | None) -> str:
            if ts is None:
                return "present"
            return time.strftime("%Y-%m-%d", time.localtime(ts))

        return f"{_fmt(self.valid_at)} → {_fmt(self.invalid_at)}"

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
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
            "superseded_by": self.superseded_by,
            "subject_key": self.subject_key,
            "object_key": self.object_key,
            "recall_count": self.recall_count,
            "last_recalled_at": self.last_recalled_at,
            "useful_count": self.useful_count,
            "harmful_count": self.harmful_count,
            "decay_score": self.decay_score,
            "scope": self.scope,
            "source_kind": self.source_kind,
            "summary_of": self.summary_of,
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

-- ═══ Schema v2: bi-temporal facts, entity graph, feedback, fast vectors ═══

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Canonical entities: resolves "the user" / "Acer" / "my dev" to one node so
-- facts about the same thing stop fragmenting across spellings.
CREATE TABLE IF NOT EXISTS canonical_entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id     INTEGER NOT NULL REFERENCES entities(id),
    canonical     TEXT    NOT NULL,
    kind          TEXT    NOT NULL DEFAULT 'concept',
    aliases       TEXT    NOT NULL DEFAULT '[]',
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at REAL    NOT NULL,
    last_seen_at  REAL    NOT NULL,
    UNIQUE(entity_id, canonical)
);

CREATE INDEX IF NOT EXISTS idx_canonical_entity
    ON canonical_entities(entity_id, canonical);

-- The memory graph: one edge per (src, predicate, dst) with its own validity
-- window, so graph recall respects the same temporal rules as flat recall.
CREATE TABLE IF NOT EXISTS entity_edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id),
    src        TEXT    NOT NULL,
    predicate  TEXT    NOT NULL,
    dst        TEXT    NOT NULL,
    memory_id  INTEGER REFERENCES memories(id),
    weight     REAL    NOT NULL DEFAULT 1.0,
    valid_at   REAL,
    invalid_at REAL,
    created_at REAL    NOT NULL,
    UNIQUE(entity_id, src, predicate, dst)
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(entity_id, src);
CREATE INDEX IF NOT EXISTS idx_entity_edges_dst ON entity_edges(entity_id, dst);

-- Explicit relevance feedback: the only signal that can correct a ranking the
-- embedding model got wrong. Feeds ``importance`` and recall scoring.
CREATE TABLE IF NOT EXISTS memory_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  INTEGER NOT NULL REFERENCES memories(id),
    signal     TEXT    NOT NULL CHECK (signal IN ('useful','harmful','irrelevant')),
    weight     REAL    NOT NULL DEFAULT 1.0,
    query      TEXT,
    created_at REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);
"""


#: Columns added in schema v2 (``ALTER TABLE`` for pre-existing databases).
_V2_MEMORY_COLUMNS: list[tuple[str, str]] = [
    # ── bi-temporal validity (Zep/Graphiti-style) ──
    ("valid_at", "REAL"),  # when the fact became true (event time)
    ("invalid_at", "REAL"),  # when it stopped being true (NULL = still true)
    ("expired_at", "REAL"),  # when the system stopped believing it (txn time)
    ("superseded_by", "INTEGER"),  # successor memory id, when superseded
    # ── entity resolution ──
    ("subject_key", "TEXT"),
    ("object_key", "TEXT"),
    # ── fast vector path ──
    ("embedding_f32", "BLOB"),  # float32 little-endian; preferred over `embedding`
    ("embedding_dim", "INTEGER"),
    # ── usage telemetry (feeds decay + ranking) ──
    ("recall_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_recalled_at", "REAL"),
    ("useful_count", "INTEGER NOT NULL DEFAULT 0"),
    ("harmful_count", "INTEGER NOT NULL DEFAULT 0"),
    ("decay_score", "REAL"),
    # ── provenance / hierarchy ──
    ("scope", "TEXT NOT NULL DEFAULT 'entity'"),
    ("source_kind", "TEXT"),
    ("summary_of", "TEXT"),  # session id when this row is a session summary
]

_V2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memories_temporal
    ON memories(entity_id, invalid_at);
CREATE INDEX IF NOT EXISTS idx_memories_subject_key
    ON memories(entity_id, subject_key);
CREATE INDEX IF NOT EXISTS idx_memories_active
    ON memories(entity_id, status, invalid_at);
"""

#: Current schema version written to ``schema_meta`` and ``PRAGMA user_version``.
SCHEMA_VERSION = 2


def _normalize_triple(subject: str, predicate: str, object: str) -> tuple[str, str, str]:
    """Normalize a (subject, predicate, object) triple for dedup keys."""
    return (
        (subject or "").strip().lower(),
        (predicate or "").strip().lower(),
        (object or "").strip().lower(),
    )


#: Predicates that can only hold **one** value at a time.
#:
#: A user has one employer, one timezone, one preferred editor — a second,
#: different value is a *replacement*, not an addition.
#:
#: Matching is **token-wise** (the predicate is split on ``_``/``-``/space), not
#: substring, so ``analysis_result`` is not mistaken for ``is_*``.
#:
#: The tokens below name *slots* (``language``, ``editor``, ``database``). A slot
#: only makes a predicate single-valued when the predicate actually reads as a
#: slot reference — which is why :data:`MULTI_VALUED_PREDICATE_TOKENS` is checked
#: first and vetoes the match.
FUNCTIONAL_PREDICATE_TOKENS: frozenset[str] = frozenset(
    {
        "employer", "works", "workplace", "job", "occupation", "role", "position",
        "company", "organisation", "organization", "title",
        "name", "username", "nickname", "email", "phone", "address", "city",
        "country", "timezone", "locale", "location", "birthday", "birthdate",
        "age", "gender", "pronouns",
        "language", "editor", "ide", "shell", "terminal", "os",
        "database", "db", "datastore", "framework", "hosting", "cloud",
        "model", "plan", "tier", "subscription", "version", "status", "state",
        "manager", "favorite", "favourite", "preferred", "prefers", "default",
        "primary", "current", "setting", "config",
    }
)

#: Whole-predicate matches for short forms that would be ambiguous as tokens.
FUNCTIONAL_PREDICATE_EXACT: frozenset[str] = frozenset(
    {
        "employer", "works_at", "works_for", "job_title", "job_role",
        "favorite_color", "favourite_color", "preferred_editor", "db_choice",
        "ci", "test_runner", "package_manager", "operating_system",
        "belongs_to", "reports_to", "main_language", "primary_language",
    }
)

#: Predicates that look functional via their ``*_preference`` / ``favorite_*`` shape.
FUNCTIONAL_PREDICATE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(favorite|favourite|preferred|default|current|primary|main)_"),
    re.compile(r"^prefers?_"),
    re.compile(r"_(preference|choice|setting|config)$"),
)

#: Verbs that make a predicate **multi-valued no matter what noun follows**.
#:
#: ``likes_language``, ``uses_framework``, ``knows_person`` all end in a noun that
#: appears in :data:`FUNCTIONAL_PREDICATE_TOKENS`, so a naive token intersection
#: would treat them as single-valued and silently retire "User likes Rust" when
#: the user says "User likes Go". Checking the verb first is what keeps the
#: detector on the conservative side of its own contract.
MULTI_VALUED_PREDICATE_TOKENS: frozenset[str] = frozenset(
    {
        "likes", "like", "liked", "loves", "love", "loved", "enjoys", "enjoy",
        "uses", "use", "used", "using", "knows", "know", "knew",
        "imports", "import", "imported", "depends", "requires", "needs",
        "calls", "call", "invokes", "reads", "writes", "handles",
        "contains", "includes", "supports", "targets", "mentions",
        "references", "owns", "speaks", "studied", "learning", "maintains",
        "authors", "visited",
    }
)

#: Whole-phrase multi-valued predicates — the verb form that token-splitting
#: cannot recover (``works_with`` splits to ``{works, with}``, and ``works``
#: alone is functional because of ``works_at``).
MULTI_VALUED_PREDICATE_EXACT: frozenset[str] = frozenset(
    {
        "works_with", "worked_on", "works_on", "contributes_to",
        "collaborates_with", "friends_with", "member_of", "subscribes_to",
        "interested_in", "familiar_with", "learning_about", "visited_place",
        "has_skill", "has_experience", "speaks_language", "known_for",
    }
)

_PREDICATE_SPLIT = re.compile(r"[^a-z0-9]+")


def _is_functional_predicate(predicate: str) -> bool:
    """
    True when a predicate is single-valued, so a new value *replaces* the old.

    Conservative by design: a false negative merely keeps two facts side by
    side (the pre-v1 behaviour), while a false positive would silently erase a
    belief the user actually holds. Hence the multi-valued verb veto runs
    before the slot-noun heuristic.
    """
    p = (predicate or "").strip().lower()
    if not p:
        return False
    if p in MULTI_VALUED_PREDICATE_EXACT:
        return False
    if p in FUNCTIONAL_PREDICATE_EXACT:
        return True
    if any(rx.search(p) for rx in FUNCTIONAL_PREDICATE_REGEXES):
        return True
    tokens = {t for t in _PREDICATE_SPLIT.split(p) if t}
    if tokens & MULTI_VALUED_PREDICATE_TOKENS:
        return False
    return bool(tokens & FUNCTIONAL_PREDICATE_TOKENS)


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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Embedding (de)serialisation ────────────────────────────────────────────────
# Stored as raw little-endian float32 BLOBs (4 bytes/dim) instead of JSON
# (~19 bytes/dim). 4.5x smaller on disk and a memcpy instead of a parse.

def pack_embedding(vector: list[float]) -> bytes:
    """Serialise a float vector to a compact float32 blob."""
    import array

    arr = array.array("f", vector)
    if sys.byteorder == "big":  # keep the on-disk format little-endian
        arr.byteswap()
    return arr.tobytes()


def unpack_embedding(blob: bytes, dim: int | None = None) -> list[float]:
    """Inverse of :func:`pack_embedding`."""
    import array

    arr = array.array("f")
    arr.frombytes(blob)
    if sys.byteorder == "big":
        arr.byteswap()
    if dim is not None and len(arr) > dim:
        arr = arr[:dim]
    return list(arr)


def embedding_from_row(row: sqlite3.Row) -> list[float] | None:
    """
    Read a row's embedding, preferring the v2 float32 blob and falling back to
    the legacy JSON text column. Returns None when the row has neither.
    """
    try:
        blob = row["embedding_f32"]
    except (IndexError, KeyError):
        blob = None
    if isinstance(blob, (bytes, bytearray, memoryview)) and len(blob) >= 4:
        try:
            return unpack_embedding(bytes(blob))
        except (ValueError, TypeError):
            pass
    raw = row["embedding"]
    if isinstance(raw, (bytes, bytearray, memoryview)) and len(raw) >= 4:
        try:
            return unpack_embedding(bytes(raw))
        except (ValueError, TypeError):
            pass
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (TypeError, ValueError):
            return None
    return None


def canonical_key(name: str) -> str:
    """
    Normalise an entity/attribute name into a resolution key.

    ``"The User"``, ``"the_user"`` and ``"the-user"`` all collapse to ``"user"``,
    and ``"AuthMiddleware"`` / ``"auth_middleware"`` / ``"auth-middleware"`` all
    collapse to ``"auth_middleware"`` — so facts about the same thing stop
    fragmenting across spellings and code conventions.
    """
    text = (name or "").strip()
    # Drop leading articles/possessives that carry no identity.
    text = re.sub(r"(?i)^(the|a|an|my|our|your|their|his|her)\s+", "", text)
    text = re.sub(r"['’]s\b", "", text)
    # camelCase / PascalCase / ACRONYMWord → snake_case, so code identifiers and
    # prose names land on the same key.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = text.lower()
    # Unify separators, then squash remaining punctuation.
    text = re.sub(r"[\s\-_/]+", "_", text)
    text = re.sub(r"[^a-z0-9_.]", "", text)
    return text.strip("_.")


#: Subject keys that always mean "the human using this tool".
_SELF_SUBJECTS = {"user", "me", "i", "myself", "the_user", "user_self"}


def is_self_subject(name: str) -> bool:
    return canonical_key(name) in _SELF_SUBJECTS


class VectorIndex:
    """
    In-process embedding matrix for one entity, backed by numpy.

    Replaces the old per-row ``json.loads`` + Python cosine loop: recall becomes
    a single matrix-vector product (microseconds for thousands of memories)
    instead of N parses and N interpreter-level dot products.

    The index is maintained **incrementally** — a new memory appends one row
    rather than forcing a full rebuild, which keeps bulk ingest linear instead
    of quadratic. :class:`MemoryStore` marks an index dirty only when a row's
    embedding actually changes underneath it.
    """

    __slots__ = ("ids", "_matrix", "_norms", "generation", "_positions")

    def __init__(self, generation: int = 0) -> None:
        self.ids: list[int] = []
        self._matrix: Any = None
        self._norms: Any = None
        self._positions: dict[int, int] = {}
        self.generation = generation

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def matrix(self) -> Any:
        return self._matrix

    def build(self, pairs: list[tuple[int, list[float]]], generation: int) -> None:
        import numpy as np

        self.ids = [pid for pid, _ in pairs]
        self._positions = {pid: i for i, pid in enumerate(self.ids)}
        self.generation = generation
        if not pairs:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._norms = np.zeros((0,), dtype=np.float32)
            return
        dim = max(len(vec) for _, vec in pairs)
        matrix = np.zeros((len(pairs), dim), dtype=np.float32)
        for i, (_, vec) in enumerate(pairs):
            if vec:
                matrix[i, : len(vec)] = vec
        self._matrix = matrix
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        self._norms = norms

    def append(self, memory_id: int, vector: list[float]) -> None:
        """
        Add one memory to the index without rebuilding it.

        Grows the matrix by doubling its capacity so a bulk insert is amortised
        O(1) per row instead of O(n) per row.
        """
        import numpy as np

        if not vector:
            return
        if memory_id in self._positions:
            self.update(memory_id, vector)
            return
        if self._matrix is None or self._matrix.shape[0] == 0:
            self.build([(memory_id, vector)], self.generation)
            return

        dim = max(self._matrix.shape[1], len(vector))
        if dim != self._matrix.shape[1]:
            grown = np.zeros((self._matrix.shape[0], dim), dtype=np.float32)
            grown[:, : self._matrix.shape[1]] = self._matrix
            self._matrix = grown
            self._norms = np.linalg.norm(self._matrix, axis=1)
            self._norms[self._norms == 0.0] = 1.0

        capacity = self._matrix.shape[0]
        used = len(self.ids)
        if used >= capacity:
            new_capacity = max(8, capacity * 2)
            grown = np.zeros((new_capacity, dim), dtype=np.float32)
            grown[:used] = self._matrix[:used]
            self._matrix = grown
            norms = np.zeros((new_capacity,), dtype=np.float32)
            norms[:used] = self._norms[:used]
            self._norms = norms

        row = np.zeros((dim,), dtype=np.float32)
        row[: len(vector)] = vector
        self._matrix[used] = row
        norm = float(np.linalg.norm(row)) or 1.0
        self._norms[used] = norm
        self._positions[memory_id] = used
        self.ids.append(memory_id)

    def update(self, memory_id: int, vector: list[float]) -> None:
        """Replace one memory's vector in place."""
        import numpy as np

        position = self._positions.get(memory_id)
        if position is None or self._matrix is None:
            self.append(memory_id, vector)
            return
        dim = self._matrix.shape[1]
        row = np.zeros((dim,), dtype=np.float32)
        if len(vector) >= dim:
            row[:] = vector[:dim]
        else:
            row[: len(vector)] = vector
        self._matrix[position] = row
        self._norms[position] = float(np.linalg.norm(row)) or 1.0

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Return ``[(memory_id, cosine), ...]`` for the top-k most similar rows."""
        import numpy as np

        count = len(self.ids)
        if self._matrix is None or count == 0 or not query:
            return []
        dim = self._matrix.shape[1]
        q = np.zeros((dim,), dtype=np.float32)
        if len(query) >= dim:
            q[:] = query[:dim]
        else:
            q[: len(query)] = query
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        sims = (self._matrix[:count] @ q) / (self._norms[:count] * qn)
        k = min(k, count)
        if k <= 0:
            return []
        if k < count:
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
        else:
            top = np.argsort(-sims)
        return [(self.ids[int(i)], float(sims[int(i)])) for i in top]


class MemoryStore:
    """SQLite-backed memory store with entity/process scoping and dedup."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        # Vector-index cache: {entity_pk: (generation, VectorIndex)}
        self._vector_index: dict[int, tuple[int, VectorIndex]] = {}
        # Bumped on every write so stale matrices are rebuilt lazily.
        self._generation = 0
        # Scoring components from the most recent hybrid recall (debug surface).
        self._last_breakdown: dict[int, dict[str, float]] = {}
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
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """
        Bring a pre-v2 database up to the current schema, in place.

        Idempotent: every step is guarded by an existence check, so it is safe
        to run on a fresh database, a v1 database, and every startup after.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        added: list[str] = []
        for name, decl in _V2_MEMORY_COLUMNS:
            if name in existing:
                continue
            # NOT NULL columns need a default to satisfy SQLite's ALTER TABLE.
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
            added.append(name)
        conn.executescript(_V2_INDEXES)

        # Backfill validity for pre-v2 rows: a row was believed from the moment
        # it was first seen. Without this every legacy memory would look
        # "always invalid" to an as-of query.
        conn.execute(
            "UPDATE memories SET valid_at = first_seen_at WHERE valid_at IS NULL"
        )
        # Backfill entity resolution keys from the raw triple.
        rows = conn.execute(
            "SELECT id, subject, object FROM memories WHERE subject_key IS NULL"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE memories SET subject_key = ?, object_key = ? WHERE id = ?",
                (canonical_key(row["subject"]), canonical_key(row["object"]), row["id"]),
            )

        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        current = int(version["value"]) if version else 1
        if added or current < SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if added:
                log.info(
                    "Memory schema migrated v%d → v%d (added: %s)",
                    current,
                    SCHEMA_VERSION,
                    ", ".join(added),
                )

    # ── Vector index ─────────────────────────────────────────────────────────

    def _index_append(self, entity_pk: int, memory_id: int, vector: list[float]) -> None:
        """Add a newly inserted row to the cached matrix (no full rebuild)."""
        cached = self._vector_index.get(entity_pk)
        if cached is None:
            return
        index = cached[1]
        index.append(memory_id, vector)
        self._vector_index[entity_pk] = (self._generation, index)

    def _index_update(self, entity_pk: int, memory_id: int, vector: list[float]) -> None:
        """Replace a row's vector in the cached matrix."""
        cached = self._vector_index.get(entity_pk)
        if cached is None:
            return
        cached[1].update(memory_id, vector)
        self._vector_index[entity_pk] = (self._generation, cached[1])

    def _index_mark_dirty(self, entity_pk: int) -> None:
        """Force a rebuild of one entity's matrix on next use."""
        self._vector_index.pop(entity_pk, None)

    def _bump_generation(self) -> None:
        """
        Advance the store generation.

        The cached matrices are maintained incrementally, so this only matters
        for callers that replaced rows wholesale; per-entity invalidation goes
        through :meth:`_index_mark_dirty`.
        """
        self._generation += 1

    def _get_index(self, conn: sqlite3.Connection, entity_pk: int) -> VectorIndex:
        """
        Return the (cached) embedding matrix for one entity, building it once
        and then keeping it current incrementally.
        """
        cached = self._vector_index.get(entity_pk)
        if cached is not None:
            return cached[1]

        rows = conn.execute(
            "SELECT id, embedding, embedding_f32 FROM memories WHERE entity_id = ?",
            (entity_pk,),
        ).fetchall()
        pairs: list[tuple[int, list[float]]] = []
        for row in rows:
            vec = embedding_from_row(row)
            if vec:
                pairs.append((int(row["id"]), vec))
        index = VectorIndex()
        index.build(pairs, self._generation)
        self._vector_index[entity_pk] = (self._generation, index)
        return index

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

    def _resolve_ids(self, entity_id: str, process_id: str) -> tuple[int, int]:
        return self.register_entity(entity_id), self.register_process(process_id)

    # ── Sessions ─────────────────────────────────────────────────────────────

    def create_session(self, entity_id: str, process_id: str) -> str:
        """Create a new session row, return its id."""
        entity_pk, process_pk = self._resolve_ids(entity_id, process_id)
        session_id = str(uuid.uuid4())
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO sessions (id, entity_id, process_id, started_at) VALUES (?, ?, ?, ?)",
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
                "SELECT id FROM sessions WHERE entity_id = ? ORDER BY started_at DESC LIMIT ?",
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
        valid_at: float | None = None,
        source_kind: str | None = None,
        scope: str = "entity",
        invalidate_contradictions: bool = True,
    ) -> tuple[bool, MemoryRecord]:
        """
        Write one memory.

        Dedup order:
          1. A semantically similar row for this **entity** (cosine ≥ threshold)
             → bump ``mention_count`` / refresh ``last_seen_at``.
          2. Otherwise insert; the unique (entity, subject, predicate, object)
             index absorbs exact duplicates with ``ON CONFLICT``.

        After a successful insert, a *functional* predicate (single-valued, e.g.
        ``employer``) whose existing value disagrees with the new one has the old
        row **invalidated** rather than deleted: ``invalid_at`` is stamped,
        ``superseded_by`` points at the successor, and a version row records why.
        The old fact stays queryable through :meth:`recall_as_of`.

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
        subj_key = canonical_key(subject)
        obj_key = canonical_key(object)
        now = time.time()
        valid_from = valid_at if valid_at is not None else now
        emb_json = json.dumps(embedding, separators=(",", ":"))
        emb_blob = pack_embedding(embedding) if embedding else None

        with self._lock:
            conn = self._conn()

            # 1) Semantic duplicate? (entity-scoped — a fact restated by a
            #    different agent process is still the same fact about the user)
            existing = self._find_semantic_duplicate(
                conn, entity_pk, embedding, similarity_threshold
            )
            if existing is not None:
                if existing["last_job_id"] == job_id:
                    # Already applied for exactly this job — idempotent retry.
                    return False, self._record_from_row(conn, existing)
                conn.execute(
                    "UPDATE memories SET mention_count = mention_count + 1, "
                    "last_seen_at = ?, text = ?, last_job_id = ?, "
                    "process_id = ?, confidence = ?, importance = ? WHERE id = ?",
                    (now, text, job_id, process_pk, confidence, importance, existing["id"]),
                )
                # Graph nodes are canonicalised from the *original* names, not
                # the lowercased dedup triple — lowercasing first would erase
                # the camelCase boundary that "AuthMiddleware" → "auth_middleware"
                # depends on.
                self._sync_entity_graph(
                    conn, entity_pk, existing["id"], subject, predicate, object, now, valid_from
                )
                conn.commit()
                # The existing row keeps its embedding, so the index is already
                # correct — no rebuild needed on the reinforcement path.
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (existing["id"],)
                ).fetchone()
                return False, self._record_from_row(conn, row)

            # 2) Insert (exact-duplicate races absorbed by ON CONFLICT).
            cur = conn.execute(
                """
                INSERT INTO memories
                    (entity_id, process_id, kind, subject, predicate, object,
                     text, embedding, embedding_f32, embedding_dim, mention_count,
                     first_seen_at, last_seen_at, session_id, last_job_id, status,
                     confidence, importance, source_event, source_message_id,
                     valid_at, subject_key, object_key, source_kind, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, subject, predicate, object)
                DO UPDATE SET
                    mention_count = mention_count + 1,
                    last_seen_at = excluded.last_seen_at,
                    text = excluded.text,
                    embedding = excluded.embedding,
                    embedding_f32 = excluded.embedding_f32,
                    embedding_dim = excluded.embedding_dim,
                    last_job_id = excluded.last_job_id,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    source_event = excluded.source_event,
                    source_message_id = excluded.source_message_id
                WHERE memories.last_job_id IS NOT excluded.last_job_id
                RETURNING id, mention_count
                """,
                (
                    entity_pk,
                    process_pk,
                    kind,
                    s,
                    p,
                    o,
                    text,
                    emb_json,
                    emb_blob,
                    len(embedding),
                    now,
                    now,
                    session_id,
                    job_id,
                    status,
                    confidence,
                    importance,
                    source_event,
                    source_message_id,
                    valid_from,
                    subj_key,
                    obj_key,
                    source_kind,
                    scope,
                ),
            )
            result = cur.fetchone()
            if result is None:
                # A same-job retry was suppressed by the idempotency guard
                # (WHERE memories.last_job_id IS NOT excluded.last_job_id).
                row = conn.execute(
                    "SELECT * FROM memories WHERE entity_id = ? "
                    "AND subject = ? AND predicate = ? AND object = ?",
                    (entity_pk, s, p, o),
                ).fetchone()
                conn.commit()
                return False, self._record_from_row(conn, row)

            memory_id = int(result["id"])
            inserted = bool(result["mention_count"] == 1)

            # 3) Bi-temporal contradiction resolution.
            if invalidate_contradictions and inserted and status == "active":
                self._invalidate_contradictions(
                    conn,
                    entity_pk=entity_pk,
                    memory_id=memory_id,
                    subject_key=subj_key,
                    predicate=p,
                    object_key=obj_key,
                    confidence=confidence,
                    valid_from=valid_from,
                    session_id=session_id,
                    process_id=process_id,
                    job_id=job_id,
                )

            self._sync_entity_graph(
                conn, entity_pk, memory_id, subject, predicate, object, now, valid_from
            )
            conn.commit()
            # Keep the cached matrix current in O(1) instead of rebuilding it.
            if inserted:
                self._index_append(entity_pk, memory_id, embedding)
            else:
                self._index_update(entity_pk, memory_id, embedding)
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return inserted, self._record_from_row(conn, row)

    def _find_semantic_duplicate(
        self,
        conn: sqlite3.Connection,
        entity_pk: int,
        embedding: list[float],
        threshold: float,
        *,
        exclude_id: int | None = None,
    ) -> sqlite3.Row | None:
        """
        Best matching existing row for this entity above the threshold.

        Uses the cached embedding matrix (one matmul) instead of parsing every
        row's JSON embedding in Python. Candidates are then filtered to
        currently-believed rows before the best one is chosen.
        """
        if not embedding:
            return None
        index = self._get_index(conn, entity_pk)
        if index.size == 0:
            return None

        # Over-fetch so that inactive rows filtered out below can't starve the
        # result when many superseded duplicates exist.
        candidates = index.search(embedding, k=min(max(8, index.size), 64))
        if not candidates:
            return None
        best_id, best_score = candidates[0]
        if best_score < threshold:
            return None

        for memory_id, score in candidates:
            if score < threshold:
                break
            if exclude_id is not None and memory_id == exclude_id:
                continue
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None or row["status"] != "active":
                continue
            if row["invalid_at"] is not None:
                continue
            return row
        return None

    # ── Bi-temporal contradiction resolution ─────────────────────────────────

    def _invalidate_contradictions(
        self,
        conn: sqlite3.Connection,
        *,
        entity_pk: int,
        memory_id: int,
        subject_key: str,
        predicate: str,
        object_key: str,
        confidence: float,
        valid_from: float,
        session_id: str | None,
        process_id: str,
        job_id: int | None,
    ) -> list[int]:
        """
        Retire older, disagreeing values for a functional predicate.

        Only *functional* predicates are touched (a user has one employer, not
        five), and only when the new fact is at least as confident as the old
        one — so a low-confidence extraction can never erase a strong belief.

        Returns the ids of the rows that were invalidated.
        """
        if not _is_functional_predicate(predicate):
            return []
        rows = conn.execute(
            """
            SELECT id, object_key, object, text, confidence, status, invalid_at
            FROM memories
            WHERE entity_id = ? AND subject_key = ? AND predicate = ?
              AND id != ? AND status = 'active' AND invalid_at IS NULL
            """,
            (entity_pk, subject_key, predicate, memory_id),
        ).fetchall()
        invalidated: list[int] = []
        now = time.time()
        for row in rows:
            if (row["object_key"] or canonical_key(row["object"])) == object_key:
                continue
            if confidence + 1e-9 < float(row["confidence"] or 0.0):
                continue
            conn.execute(
                "UPDATE memories SET status = 'superseded', invalid_at = ?, "
                "expired_at = ?, superseded_by = ? WHERE id = ?",
                (valid_from, now, memory_id, row["id"]),
            )
            conn.execute(
                """
                INSERT INTO memory_versions
                    (memory_id, old_text, new_text, old_status, new_status,
                     changed_at, reason, source_session, source_process, source_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["text"],
                    None,
                    row["status"],
                    "superseded",
                    now,
                    f"invalidated: {predicate!r} superseded by memory {memory_id} "
                    f"(object changed)",
                    session_id,
                    process_id,
                    job_id,
                ),
            )
            invalidated.append(int(row["id"]))
        if invalidated:
            log.debug(
                "Invalidated %d contradicting memory(ies) for %s.%s",
                len(invalidated),
                subject_key,
                predicate,
            )
        return invalidated

    # ── Entity graph ─────────────────────────────────────────────────────────

    def _sync_entity_graph(
        self,
        conn: sqlite3.Connection,
        entity_pk: int,
        memory_id: int,
        subject: str,
        predicate: str,
        object: str,
        now: float,
        valid_from: float,
    ) -> None:
        """
        Keep the memory graph in step with the flat store.

        Every memory becomes one edge ``(subject) -[predicate]-> (object)`` using
        canonical node names, so ``"AuthMiddleware calls UserService"`` and
        ``"auth_middleware -> user_service"`` land on the same two nodes.
        """
        src = canonical_key(subject)
        dst = canonical_key(object)
        if not src or not dst or src == dst:
            return
        conn.execute(
            """
            INSERT INTO entity_edges
                (entity_id, src, predicate, dst, memory_id, weight, valid_at, created_at)
            VALUES (?, ?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT(entity_id, src, predicate, dst) DO UPDATE SET
                memory_id = excluded.memory_id,
                weight = entity_edges.weight + 1.0,
                invalid_at = NULL
            """,
            (entity_pk, src, predicate, dst, memory_id, valid_from, now),
        )
        for name in (src, dst):
            kind = "self" if is_self_subject(name) else "concept"
            conn.execute(
                """
                INSERT INTO canonical_entities
                    (entity_id, canonical, kind, mention_count, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(entity_id, canonical) DO UPDATE SET
                    mention_count = canonical_entities.mention_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (entity_pk, name, kind, now, now),
            )

    def _record_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> MemoryRecord:
        entity_pk = int(row["entity_id"])
        process_pk = int(row["process_id"])
        entity_ext = conn.execute(
            "SELECT external_id FROM entities WHERE id = ?", (entity_pk,)
        ).fetchone()["external_id"]
        process_ext = conn.execute(
            "SELECT external_id FROM processes WHERE id = ?", (process_pk,)
        ).fetchone()["external_id"]

        def opt(name: str, default: Any = None) -> Any:
            """Read a v2 column defensively (old rows / old schemas)."""
            try:
                return row[name]
            except (IndexError, KeyError):
                return default

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
            valid_at=_as_float(opt("valid_at")),
            invalid_at=_as_float(opt("invalid_at")),
            expired_at=_as_float(opt("expired_at")),
            superseded_by=_as_int(opt("superseded_by")),
            subject_key=opt("subject_key"),
            object_key=opt("object_key"),
            recall_count=int(opt("recall_count", 0) or 0),
            last_recalled_at=_as_float(opt("last_recalled_at")),
            useful_count=int(opt("useful_count", 0) or 0),
            harmful_count=int(opt("harmful_count", 0) or 0),
            decay_score=_as_float(opt("decay_score")),
            scope=str(opt("scope", "entity") or "entity"),
            source_kind=opt("source_kind"),
            summary_of=opt("summary_of"),
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
        as_of: float | None = None,
        kinds: list[str] | None = None,
        record_hits: bool = True,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Vector-search a single entity's memories (never cross-entity).

        Runs against the cached embedding matrix — one matmul — instead of
        parsing every row's JSON embedding in Python.

        ``as_of`` performs a point-in-time query: it returns what was *believed
        and true* at that timestamp, which is how superseded facts stay
        answerable ("what was the user's employer in March?").

        Returns ``[(record, similarity), ...]`` sorted by similarity desc.
        """
        entity_pk = self.register_entity(entity_id)
        with self._lock:
            conn = self._conn()
            index = self._get_index(conn, entity_pk)
            if index.size == 0:
                return []
            hits = index.search(query_embedding, k=min(max(k * 4, k), index.size))
            if not hits:
                return []
            ids = [mid for mid, _ in hits]
            rows = self._fetch_rows(conn, ids, process_id, as_of, kinds)
            results: list[tuple[MemoryRecord, float]] = []
            for memory_id, score in hits:
                row = rows.get(memory_id)
                if row is None or score < min_score:
                    continue
                results.append((self._record_from_row(conn, row), score))
                if len(results) >= k:
                    break
            if record_hits and results:
                self._record_recall_hits(conn, [r.id for r, _ in results])
                conn.commit()
            return results

    def _fetch_rows(
        self,
        conn: sqlite3.Connection,
        memory_ids: list[int],
        process_id: str | None,
        as_of: float | None,
        kinds: list[str] | None,
    ) -> dict[int, sqlite3.Row]:
        """Load candidate rows once, applying the active/temporal/kind filters."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        clauses = [f"id IN ({placeholders})"]
        params: list[Any] = list(memory_ids)
        if process_id is not None:
            clauses.append("process_id = ?")
            params.append(self.register_process(process_id))
        if kinds:
            bad = [k for k in kinds if k not in ALL_KINDS]
            if bad:
                raise ValueError(f"invalid memory kind(s): {bad!r}")
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if as_of is None:
            clauses.append("status = 'active'")
            clauses.append("invalid_at IS NULL")
        else:
            # Point-in-time: the fact must have become true by `as_of` and not
            # yet have been invalidated at that moment.
            clauses.append("(valid_at IS NULL OR valid_at <= ?)")
            params.append(as_of)
            clauses.append("(invalid_at IS NULL OR invalid_at > ?)")
            params.append(as_of)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}", tuple(params)
        ).fetchall()
        return {int(r["id"]): r for r in rows}

    def _record_recall_hits(self, conn: sqlite3.Connection, memory_ids: list[int]) -> None:
        """Bump usage counters for recalled memories (feeds decay + ranking)."""
        now = time.time()
        conn.executemany(
            "UPDATE memories SET recall_count = recall_count + 1, "
            "last_recalled_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )

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
        as_of: float | None = None,
        kinds: list[str] | None = None,
        include_invalidated: bool = False,
        record_hits: bool = True,
        graph_expansion: bool = False,
        graph_hops: int = 1,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Hybrid recall combining vector similarity, keyword matching, and metadata.

        Scoring is **normalised** so ``min_score`` is a real relevance floor:

            relevance = (w_vec·vector + w_kw·keyword) / (w_vec + w_kw)   # 0..1
            quality   = (w_rec·recency + w_imp·importance + w_men·mentions)
                        / (w_rec + w_imp + w_men)                       # 0..1
            score     = relevance × (1 - q·quality) + exact_boost       # 0..1

        The previous additive form summed unnormalised terms, so every memory
        collected a ~0.16 "metadata floor" and a completely unrelated row could
        clear a 0.3 threshold. Here an irrelevant memory scores ≈0 regardless of
        how important or recent it is, and ``min_score=0.3`` means "at least 30%
        relevant".

        ``graph_expansion`` additionally pulls in one-hop neighbour facts of the
        top hits from the memory graph, which recovers facts phrased around an
        entity the query never names.
        """
        entity_pk = self.register_entity(entity_id)
        query_tokens = self._tokenize_query(query)

        with self._lock:
            conn = self._conn()
            index = self._get_index(conn, entity_pk)
            if index.size == 0:
                return []

            # Over-fetch so temporal/kind filters and graph expansion have room.
            pool = min(index.size, max(k * 8, 64))
            vector_hits = index.search(query_embedding, k=pool)
            if not vector_hits:
                return []
            vector_scores = dict(vector_hits)
            if graph_expansion:
                for neighbour_id in self._graph_neighbours(
                    conn, entity_pk, [mid for mid, _ in vector_hits][: max(3, k // 2)],
                    hops=graph_hops, as_of=as_of,
                ):
                    vector_scores.setdefault(neighbour_id, 0.0)

            rows = self._fetch_rows(
                conn, list(vector_scores), process_id, as_of, kinds
            )
            if not rows and not include_invalidated:
                return []
            if include_invalidated:
                missing = [mid for mid in vector_scores if mid not in rows]
                if missing:
                    placeholders = ",".join("?" * len(missing))
                    for row in conn.execute(
                        f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(missing)
                    ).fetchall():
                        rows[int(row["id"])] = row
            if not rows:
                return []

            # Build IDF for keyword scoring
            n = len(rows)
            df: dict[str, int] = {}
            tokenised: dict[int, list[str]] = {}
            for memory_id, row in rows.items():
                tokens = self._tokenize_query(row["text"] or "")
                tokenised[memory_id] = tokens
                for t in set(tokens):
                    df[t] = df.get(t, 0) + 1
            idf = {t: math.log(n / c + 1) for t, c in df.items()}

            now = time.time()
            q_lower = query.lower()
            scored: list[tuple[float, int]] = []
            breakdown: dict[int, dict[str, float]] = {}

            w_rel = (vector_weight + keyword_weight) or 1.0
            w_qual = (recency_weight + importance_weight + mention_weight) or 1.0

            for memory_id, row in rows.items():
                vec_score = vector_scores.get(memory_id, 0.0)
                kw_score = self._tfidf_score(query_tokens, tokenised[memory_id], idf)

                relevance = (vector_weight * vec_score + keyword_weight * kw_score) / w_rel
                relevance = max(0.0, min(1.0, relevance))

                subject = str(row["subject"]).lower()
                predicate = str(row["predicate"]).lower()
                obj = str(row["object"]).lower()
                exact = (
                    exact_match_boost
                    if (subject and subject in q_lower)
                    and (predicate and predicate in q_lower)
                    and (obj and obj in q_lower)
                    else 0.0
                )

                last_seen = float(row["last_seen_at"] or 0.0)
                age_days = max(0.0, (now - last_seen) / 86400.0)
                recency = 1.0 / (1.0 + age_days / 30.0)

                importance = float(row["importance"] or 0.5)
                mentions = int(row["mention_count"] or 1)
                mention_score = math.log(mentions + 1) / math.log(100)
                mention_score = max(0.0, min(1.0, mention_score))

                quality = (
                    recency_weight * recency
                    + importance_weight * importance
                    + mention_weight * mention_score
                ) / w_qual

                # Feedback nudges quality, bounded so it can never fabricate
                # relevance for an unrelated memory.
                useful = int(row["useful_count"] or 0) if "useful_count" in row.keys() else 0
                harmful = int(row["harmful_count"] or 0) if "harmful_count" in row.keys() else 0
                if useful or harmful:
                    quality = max(0.0, min(1.0, quality + 0.15 * (useful - harmful) / (useful + harmful + 1)))

                final_score = relevance * (1.0 - 0.25 * (1.0 - quality)) + exact
                final_score = max(0.0, min(1.0, final_score))

                if final_score >= min_score:
                    scored.append((final_score, memory_id))
                    breakdown[memory_id] = {
                        "vector": round(vec_score, 4),
                        "keyword": round(kw_score, 4),
                        "relevance": round(relevance, 4),
                        "recency": round(recency, 4),
                        "importance": round(importance, 4),
                        "mentions": mention_score,
                        "quality": round(quality, 4),
                        "exact_boost": exact,
                    }

            scored.sort(key=lambda x: x[0], reverse=True)
            results: list[tuple[MemoryRecord, float]] = []
            for score, memory_id in scored[:k]:
                results.append((self._record_from_row(conn, rows[memory_id]), score))
            self._last_breakdown = breakdown
            if record_hits and results:
                self._record_recall_hits(conn, [r.id for r, _ in results])
                conn.commit()
            return results

    #: Component breakdown from the most recent hybrid recall (debug surface).
    def last_score_breakdown(self) -> dict[int, dict[str, float]]:
        """Per-memory scoring components from the most recent hybrid recall."""
        return self._last_breakdown

    def _graph_neighbours(
        self,
        conn: sqlite3.Connection,
        entity_pk: int,
        memory_ids: list[int],
        *,
        hops: int = 1,
        as_of: float | None = None,
        limit: int = 32,
    ) -> list[int]:
        """
        Walk the memory graph outward from the given memories.

        Each seed memory contributes its subject/object nodes; one hop of
        ``entity_edges`` from those nodes yields neighbour memories. This is what
        lets "what do you know about the auth service?" surface a fact stored as
        "AuthMiddleware calls UserService".
        """
        if not memory_ids or hops < 1:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        seeds = conn.execute(
            f"SELECT subject_key, object_key FROM memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        nodes: set[str] = set()
        for row in seeds:
            for key in (row["subject_key"], row["object_key"]):
                if key:
                    nodes.add(key)
        if not nodes:
            return []

        visited = set(nodes)
        frontier = set(nodes)
        neighbour_memories: list[int] = []
        for _ in range(hops):
            if not frontier:
                break
            ph = ",".join("?" * len(frontier))
            params: list[Any] = [entity_pk, *frontier, *frontier]
            temporal = ""
            if as_of is not None:
                temporal = (
                    " AND (valid_at IS NULL OR valid_at <= ?)"
                    " AND (invalid_at IS NULL OR invalid_at > ?)"
                )
                params.extend([as_of, as_of])
            rows = conn.execute(
                f"""
                SELECT src, dst, memory_id FROM entity_edges
                WHERE entity_id = ? AND (src IN ({ph}) OR dst IN ({ph})){temporal}
                """,
                tuple(params),
            ).fetchall()
            next_frontier: set[str] = set()
            for row in rows:
                for key in (row["src"], row["dst"]):
                    if key and key not in visited:
                        visited.add(key)
                        next_frontier.add(key)
                if row["memory_id"] is not None:
                    neighbour_memories.append(int(row["memory_id"]))
            frontier = next_frontier
        return neighbour_memories[:limit]

    def recall_as_of(
        self,
        entity_id: str,
        query_embedding: list[float],
        as_of: float,
        *,
        k: int = 5,
        process_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Point-in-time recall: what was believed and true at ``as_of``."""
        return self.recall(
            entity_id,
            query_embedding,
            k=k,
            process_id=process_id,
            min_score=min_score,
            as_of=as_of,
            record_hits=False,
        )

    def timeline(
        self,
        entity_id: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """
        Every version of a fact in chronological order, including invalidated ones.

        Answers "how did this belief change over time" — the query Zep/Graphiti
        is known for. Pass ``subject``/``predicate`` to scope to one relation.
        """
        entity_pk = self.register_entity(entity_id)
        clauses = ["entity_id = ?"]
        params: list[Any] = [entity_pk]
        if subject:
            clauses.append("subject_key = ?")
            params.append(canonical_key(subject))
        if predicate:
            clauses.append("predicate = ?")
            params.append((predicate or "").strip().lower())
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                "ORDER BY COALESCE(valid_at, first_seen_at) ASC, id ASC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [self._record_from_row(conn, r) for r in rows]

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
        status: str | None = None,
        as_of: float | None = None,
        current_only: bool = False,
        order: str = "mentions",
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """
        List an entity's memories (never cross-entity).

        ``current_only`` restricts to facts that are still believed *and* still
        true — the right default for display. ``as_of`` gives a point-in-time
        view; ``status`` filters to one lifecycle state.
        """
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
        if status:
            clauses.append("status = ?")
            params.append(status)
        if current_only:
            clauses.append("status = 'active'")
            clauses.append("invalid_at IS NULL")
        if as_of is not None:
            clauses.append("(valid_at IS NULL OR valid_at <= ?)")
            params.append(as_of)
            clauses.append("(invalid_at IS NULL OR invalid_at > ?)")
            params.append(as_of)
        order_sql = {
            "mentions": "mention_count DESC, last_seen_at DESC",
            "recent": "COALESCE(valid_at, first_seen_at) DESC, id DESC",
            "importance": "importance DESC, mention_count DESC",
        }.get(order, "mention_count DESC, last_seen_at DESC")
        where = " AND ".join(clauses)
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return [self._record_from_row(conn, r) for r in rows]

    def get_memory(self, memory_id: int) -> MemoryRecord | None:
        """Fetch a single memory by id (None when missing)."""
        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record_from_row(conn, row) if row else None

    # ── Deletion, export, import ─────────────────────────────────────────────

    def delete_memory(self, memory_id: int, *, hard: bool = False, reason: str = "user_request") -> bool:
        """
        Remove a memory.

        Soft delete (default) stamps ``status='invalidated'`` and ``expired_at``
        so the row leaves recall immediately but stays in the audit trail and is
        still visible to :meth:`recall_as_of`. ``hard=True`` physically deletes
        the row plus its versions, feedback and graph edges — the GDPR-style
        "forget me" path.
        """
        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return False
            now = time.time()
            if hard:
                conn.execute("DELETE FROM memory_feedback WHERE memory_id = ?", (memory_id,))
                conn.execute("DELETE FROM memory_versions WHERE memory_id = ?", (memory_id,))
                conn.execute("UPDATE entity_edges SET memory_id = NULL WHERE memory_id = ?", (memory_id,))
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                conn.execute(
                    "UPDATE memories SET status = 'invalidated', expired_at = ?, "
                    "invalid_at = COALESCE(invalid_at, ?) WHERE id = ?",
                    (now, now, memory_id),
                )
                conn.execute(
                    """
                    INSERT INTO memory_versions
                        (memory_id, old_text, new_text, old_status, new_status,
                         changed_at, reason)
                    VALUES (?, ?, NULL, ?, 'invalidated', ?, ?)
                    """,
                    (memory_id, row["text"], row["status"], now, reason),
                )
            conn.commit()
            # A soft delete only changes status (the row is filtered on read);
            # a hard delete removes the row, so drop its vector from the cache.
            if hard:
                self._index_mark_dirty(int(row["entity_id"]))
            return True

    def forget(self, entity_id: str, query: str, query_embedding: list[float], *, threshold: float = 0.9, hard: bool = False) -> list[int]:
        """Delete every memory matching a query above ``threshold``. Returns ids."""
        hits = self.recall(entity_id, query_embedding, k=50, min_score=threshold, record_hits=False)
        removed = [r.id for r, _ in hits if self.delete_memory(r.id, hard=hard)]
        return removed

    def export_entity(self, entity_id: str, *, include_invalidated: bool = True) -> dict[str, Any]:
        """
        Export one entity's memories + graph as a portable JSON structure.

        Embeddings are omitted (they are model-specific and regenerated on
        import), which keeps exports diffable and small.
        """
        entity_pk = self.register_entity(entity_id)
        with self._lock:
            conn = self._conn()
            clauses = ["entity_id = ?"]
            if not include_invalidated:
                clauses.append("status = 'active' AND invalid_at IS NULL")
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY id", (entity_pk,)
            ).fetchall()
            memories = []
            for row in rows:
                data = self._record_from_row(conn, row).as_dict()
                data.pop("embedding", None)
                memories.append(data)
            edges = [
                {
                    "src": r["src"],
                    "predicate": r["predicate"],
                    "dst": r["dst"],
                    "weight": r["weight"],
                    "valid_at": r["valid_at"],
                    "invalid_at": r["invalid_at"],
                }
                for r in conn.execute(
                    "SELECT * FROM entity_edges WHERE entity_id = ?", (entity_pk,)
                ).fetchall()
            ]
            return {
                "format": "tracera-memory/v2",
                "exported_at": time.time(),
                "entity_id": entity_id,
                "memories": memories,
                "edges": edges,
            }

    def import_entity(self, payload: dict[str, Any], *, embed_fn: Any = None) -> dict[str, int]:
        """
        Import an :meth:`export_entity` payload.

        Memories are re-embedded with the caller's embedder (``embed_fn``) so a
        store moved between embedding models still works. Idempotent: existing
        (subject, predicate, object) triples are reinforced, not duplicated.
        """
        if payload.get("format") not in (None, "tracera-memory/v2"):
            raise ValueError(f"unsupported memory export format: {payload.get('format')!r}")
        entity_id = payload.get("entity_id") or "default"
        stats = {"imported": 0, "reinforced": 0, "skipped": 0}
        for mem in payload.get("memories", []):
            text = mem.get("text") or ""
            if not text:
                stats["skipped"] += 1
                continue
            embedding = embed_fn(text) if embed_fn else []
            inserted, _ = self.upsert_memory(
                entity_id=entity_id,
                process_id=mem.get("process_id") or "import",
                kind=mem.get("kind") or "fact",
                subject=mem.get("subject") or "user",
                predicate=mem.get("predicate") or "imported",
                object=mem.get("object") or text[:80],
                text=text,
                embedding=embedding,
                confidence=float(mem.get("confidence") or 0.8),
                importance=float(mem.get("importance") or 0.5),
                source_event="import",
                source_kind="import",
                valid_at=mem.get("valid_at"),
                invalidate_contradictions=False,
            )
            stats["imported" if inserted else "reinforced"] += 1
            # Preserve the original lifecycle state (e.g. superseded history).
            if mem.get("status") and mem["status"] != "active":
                with self._lock:
                    conn = self._conn()
                    conn.execute(
                        "UPDATE memories SET status = ?, invalid_at = ?, superseded_by = ? "
                        "WHERE entity_id = ? AND subject = ? AND predicate = ? AND object = ?",
                        (
                            mem["status"],
                            mem.get("invalid_at"),
                            mem.get("superseded_by"),
                            self.register_entity(entity_id),
                            (mem.get("subject") or "user").strip().lower(),
                            (mem.get("predicate") or "imported").strip().lower(),
                            (mem.get("object") or text[:80]).strip().lower(),
                        ),
                    )
                    conn.commit()
        # Embeddings are unchanged by the lifecycle restore above; the upserts
        # already appended their vectors to the cached matrix.
        return stats

    # ── Feedback ─────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        memory_id: int,
        signal: str,
        *,
        weight: float = 1.0,
        query: str | None = None,
    ) -> dict[str, Any]:
        """
        Record whether a recalled memory actually helped.

        This is the only signal that can correct a ranking the embedding model
        got wrong. ``useful`` nudges ``importance`` up and the recall score up;
        ``harmful`` does the reverse and can push a memory below the recall
        floor without deleting it.
        """
        if signal not in ("useful", "harmful", "irrelevant"):
            raise ValueError(f"invalid feedback signal: {signal!r}")
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT importance, useful_count, harmful_count FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"memory {memory_id} not found")
            conn.execute(
                "INSERT INTO memory_feedback (memory_id, signal, weight, query, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, signal, weight, query, time.time()),
            )
            if signal == "useful":
                conn.execute(
                    "UPDATE memories SET useful_count = useful_count + 1, "
                    "importance = MIN(1.0, importance + 0.05 * ?) WHERE id = ?",
                    (weight, memory_id),
                )
            elif signal in ("harmful", "irrelevant"):
                conn.execute(
                    "UPDATE memories SET harmful_count = harmful_count + 1, "
                    "importance = MAX(0.0, importance - 0.1 * ?) WHERE id = ?",
                    (weight, memory_id),
                )
            conn.commit()
            updated = conn.execute(
                "SELECT importance, useful_count, harmful_count FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            return {
                "memory_id": memory_id,
                "signal": signal,
                "importance": float(updated["importance"]),
                "useful_count": int(updated["useful_count"]),
                "harmful_count": int(updated["harmful_count"]),
            }

    # ── Decay, retention and garbage collection ──────────────────────────────

    def apply_decay(self, *, half_life_days: float = 90.0) -> int:
        """
        Recompute ``decay_score`` for every memory.

        An exponential decay on time-since-last-seen, damped by how often the
        memory was actually useful and by whether it was ever recalled. This is
        what makes forgetting principled instead of "whatever the user deletes" —
        the gap every memory server in the market currently leaves open.
        """
        lam = math.log(2) / max(1.0, half_life_days)
        now = time.time()
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT id, last_seen_at, mention_count, importance, useful_count, "
                "harmful_count, recall_count FROM memories WHERE status = 'active'"
            ).fetchall()
            updates = []
            for row in rows:
                age_days = max(0.0, (now - float(row["last_seen_at"] or now)) / 86400.0)
                recency = math.exp(-lam * age_days)
                reinforcement = math.log1p(int(row["mention_count"] or 1)) / math.log1p(20)
                feedback = (int(row["useful_count"] or 0) - int(row["harmful_count"] or 0)) / (
                    int(row["useful_count"] or 0) + int(row["harmful_count"] or 0) + 1
                )
                score = recency * (0.5 + 0.3 * reinforcement + 0.2 * float(row["importance"] or 0.5))
                score = max(0.0, min(1.0, score + 0.1 * feedback))
                updates.append((round(score, 4), int(row["id"])))
            conn.executemany("UPDATE memories SET decay_score = ? WHERE id = ?", updates)
            conn.commit()
        return len(updates)

    def run_gc(
        self,
        *,
        retention_days: int | None = 365,
        decay_floor: float = 0.05,
        min_age_days: int = 30,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Archive stale memories: expired by retention, or decayed past the floor.

        Archiving (not deleting) keeps the audit trail intact and stays
        reversible — ``status='archived'`` rows leave recall but can be restored.
        A memory that was ever marked ``useful`` is never collected.
        """
        now = time.time()
        cutoff = now - retention_days * 86400 if retention_days else None
        age_cutoff = now - min_age_days * 86400
        self.apply_decay()
        with self._lock:
            conn = self._conn()
            clauses = ["status = 'active'", "useful_count = 0", "last_seen_at < ?"]
            params: list[Any] = [age_cutoff]
            if cutoff is not None:
                clauses.append("last_seen_at < ?")
                params.append(cutoff)
            clauses.append("COALESCE(decay_score, 1.0) < ?")
            params.append(decay_floor)
            rows = conn.execute(
                f"SELECT id, text FROM memories WHERE {' AND '.join(clauses)}", tuple(params)
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if ids and not dry_run:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE memories SET status = 'archived', expired_at = ? WHERE id IN ({placeholders})",
                    (now, *ids),
                )
                conn.commit()
            return {
                "candidates": len(ids),
                "archived": 0 if dry_run else len(ids),
                "dry_run": dry_run,
                "ids": ids[:50],
            }

    # ── Memory graph queries ─────────────────────────────────────────────────

    def entity_graph(self, entity_id: str, *, node: str | None = None, limit: int = 100) -> dict[str, Any]:
        """
        Inspect the memory graph: top nodes, or one node's edges.

        Unlike a raw triple dump this returns *canonical* nodes, so every
        spelling of the same entity is already merged into one node.
        """
        entity_pk = self.register_entity(entity_id)
        with self._lock:
            conn = self._conn()
            if node:
                key = canonical_key(node)
                edges = conn.execute(
                    "SELECT * FROM entity_edges WHERE entity_id = ? AND (src = ? OR dst = ?) "
                    "ORDER BY weight DESC LIMIT ?",
                    (entity_pk, key, key, limit),
                ).fetchall()
                return {
                    "node": key,
                    "outgoing": [
                        {"dst": r["dst"], "predicate": r["predicate"], "weight": r["weight"]}
                        for r in edges if r["src"] == key
                    ],
                    "incoming": [
                        {"src": r["src"], "predicate": r["predicate"], "weight": r["weight"]}
                        for r in edges if r["dst"] == key
                    ],
                }
            nodes = conn.execute(
                "SELECT canonical, kind, mention_count FROM canonical_entities "
                "WHERE entity_id = ? ORDER BY mention_count DESC LIMIT ?",
                (entity_pk, limit),
            ).fetchall()
            edge_count = conn.execute(
                "SELECT COUNT(*) AS c FROM entity_edges WHERE entity_id = ?", (entity_pk,)
            ).fetchone()["c"]
            return {
                "nodes": [
                    {"canonical": r["canonical"], "kind": r["kind"], "mentions": r["mention_count"]}
                    for r in nodes
                ],
                "edge_count": int(edge_count),
            }

    def resolve_entity(self, entity_id: str, name: str) -> str | None:
        """Resolve a free-text name to its canonical node, if it exists."""
        entity_pk = self.register_entity(entity_id)
        key = canonical_key(name)
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT canonical FROM canonical_entities WHERE entity_id = ? AND canonical = ?",
                (entity_pk, key),
            ).fetchone()
            return row["canonical"] if row else None

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
            avg_mentions = (
                conn.execute(
                    "SELECT AVG(mention_count) FROM memories WHERE status = 'active'"
                ).fetchone()[0]
                or 0
            )
            avg_confidence = (
                conn.execute(
                    "SELECT AVG(confidence) FROM memories WHERE status = 'active'"
                ).fetchone()[0]
                or 0
            )
            total_versions = conn.execute("SELECT COUNT(*) AS c FROM memory_versions").fetchone()[
                "c"
            ]
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
            row = conn.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            attempts = int(row["attempts"]) if row else 0
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', last_error = ? WHERE id = ?",
                    (error[:500], job_id),
                )
            else:
                backoff = 2.0**attempts
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
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
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
        parts.append(f'This {record.kind} memory states: "{record.text}"')
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
        results = self.recall_hybrid(entity_id, query, query_embedding, k=k, min_score=min_score)

        explanations = []
        for record, score in results:
            explanations.append(
                {
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
                }
            )

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
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        else:
            return f"{int(age / 86400)}d ago"

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
        Supersede an existing memory, **in place**.

        The row keeps its identity but takes the new text and is marked
        ``superseded``. Kept for backwards compatibility with the original
        contract; the temporal columns are stamped so point-in-time queries
        exclude it correctly.

        Prefer :meth:`replace_memory` when you want the previous value to remain
        queryable — this method overwrites it.
        """
        with self._lock:
            conn = self._conn()
            # Get the old memory
            old_row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not old_row:
                raise ValueError(f"Memory {memory_id} not found")

            old_text = str(old_row["text"])
            old_status = str(old_row["status"])

            # Update the memory
            emb_json = json.dumps(new_embedding, separators=(",", ":"))
            emb_blob = pack_embedding(new_embedding) if new_embedding else None
            now = time.time()
            conn.execute(
                """
                UPDATE memories
                SET text = ?, embedding = ?, embedding_f32 = ?, embedding_dim = ?,
                    last_seen_at = ?, status = 'superseded',
                    invalid_at = COALESCE(invalid_at, ?), expired_at = ?
                WHERE id = ?
                """,
                (
                    new_text,
                    emb_json,
                    emb_blob,
                    len(new_embedding),
                    now,
                    now,
                    now,
                    memory_id,
                ),
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
            self._index_update(int(old_row["entity_id"]), memory_id, new_embedding)

            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record_from_row(conn, row)

    def replace_memory(
        self,
        memory_id: int,
        *,
        new_text: str,
        new_embedding: list[float],
        reason: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        kind: str | None = None,
        confidence: float = 0.8,
        importance: float | None = None,
        session_id: str | None = None,
        process_id: str | None = None,
        job_id: int | None = None,
    ) -> tuple[MemoryRecord, MemoryRecord]:
        """
        Retire a memory and insert its successor, **preserving both versions**.

        This is the history-preserving form of supersession: the old row keeps
        its original text and gets ``invalid_at`` + ``superseded_by``, while a new
        active row carries the new value. ``recall_as_of(t)`` can then answer
        "what did we believe in March?" long after the fact changed.

        Returns ``(old_record, new_record)``.
        """
        with self._lock:
            conn = self._conn()
            old_row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not old_row:
                raise ValueError(f"Memory {memory_id} not found")

            now = time.time()
            new_subject = subject or str(old_row["subject"])
            new_predicate = predicate or str(old_row["predicate"])
            new_object = object or str(old_row["object"])
            new_kind = kind or str(old_row["kind"])
            new_importance = (
                importance if importance is not None
                else float(old_row["importance"] or 0.5)
            )
            entity_pk = int(old_row["entity_id"])
            entity_ext = conn.execute(
                "SELECT external_id FROM entities WHERE id = ?", (entity_pk,)
            ).fetchone()["external_id"]
            process_ext = process_id or conn.execute(
                "SELECT external_id FROM processes WHERE id = ?", (int(old_row["process_id"]),)
            ).fetchone()["external_id"]

            # 1) Insert the successor first so the old row can point at it.
            _inserted, successor = self.upsert_memory(
                entity_id=str(entity_ext),
                process_id=str(process_ext),
                kind=new_kind,
                subject=new_subject,
                predicate=new_predicate,
                object=new_object,
                text=new_text,
                embedding=new_embedding,
                session_id=session_id,
                job_id=job_id,
                similarity_threshold=1.1,  # never merge the successor into the old row
                confidence=confidence,
                importance=new_importance,
                source_event="memory.replaced",
                source_kind="reconciliation",
                invalidate_contradictions=False,
            )

            # 2) Retire the old row, keeping its text for the audit trail.
            conn.execute(
                "UPDATE memories SET status = 'superseded', invalid_at = ?, "
                "expired_at = ?, superseded_by = ? WHERE id = ?",
                (now, now, successor.id, memory_id),
            )
            conn.execute(
                """
                INSERT INTO memory_versions
                    (memory_id, old_text, new_text, old_status, new_status,
                     changed_at, reason, source_session, source_process, source_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    str(old_row["text"]),
                    new_text,
                    str(old_row["status"]),
                    "superseded",
                    now,
                    reason,
                    session_id,
                    process_ext,
                    job_id,
                ),
            )
            conn.commit()
            # The successor was inserted through upsert_memory (which appended
            # it to the cached matrix); the retired row keeps its embedding.

            old_final = self._record_from_row(
                conn, conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            )
            return old_final, successor

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

    def preview_consolidation(
        self,
        entity_id: str | None = None,
        *,
        similarity_threshold: float = 0.92,
        max_candidates: int = 50,
    ) -> dict[str, Any]:
        """
        Dry-run of :meth:`run_consolidation` — find near-duplicate memory pairs
        WITHOUT modifying anything, so the CLI can show what *would* be merged.
        """
        candidates: list[dict[str, Any]] = []
        scanned = 0

        with self._lock:
            conn = self._conn()

            if entity_id:
                entity_pk = self.register_entity(entity_id)
                entity_rows = [{"id": entity_pk, "external_id": entity_id}]
            else:
                entity_rows = conn.execute("SELECT id, external_id FROM entities").fetchall()

            for entity_row in entity_rows:
                entity_pk = entity_row["id"]
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
                scanned += len(rows)

                for i, row_i in enumerate(rows):
                    if len(candidates) >= max_candidates:
                        break
                    try:
                        emb_i = json.loads(row_i["embedding"])
                    except (TypeError, ValueError):
                        continue
                    for row_j in rows[i + 1 :]:
                        if len(candidates) >= max_candidates:
                            break
                        try:
                            emb_j = json.loads(row_j["embedding"])
                        except (TypeError, ValueError):
                            continue
                        score = cosine_similarity(emb_i, emb_j)
                        if score >= similarity_threshold:
                            candidates.append(
                                {
                                    "entity": entity_row["external_id"],
                                    "keeper_text": (
                                        row_i["text"]
                                        if row_i["confidence"] >= row_j["confidence"]
                                        else row_j["text"]
                                    )[:80],
                                    "merge_text": (
                                        row_j["text"]
                                        if row_i["confidence"] >= row_j["confidence"]
                                        else row_i["text"]
                                    )[:80],
                                    "similarity": round(score, 3),
                                }
                            )

        return {"scanned": scanned, "candidates": candidates}

    def run_consolidation(
        self,
        entity_id: str | None = None,
        *,
        similarity_threshold: float = 0.92,
        min_mention_count: int = 2,
        max_merges: int = 50,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run a consolidation pass to merge near-duplicate memories.

        Only currently-believed rows participate: merging into a superseded row
        would resurrect a retired fact. ``dry_run`` reports what *would* merge
        without touching anything.

        This can be scheduled as a periodic background job.
        Returns statistics about the consolidation run.
        """
        stats = {"scanned": 0, "merged": 0, "superseded": 0, "errors": 0, "dry_run": dry_run}

        with self._lock:
            conn = self._conn()

            # Get entities to process
            if entity_id:
                entity_pk = self.register_entity(entity_id)
                entity_rows = [{"id": entity_pk, "external_id": entity_id}]
            else:
                entity_rows = conn.execute("SELECT id, external_id FROM entities").fetchall()

            for entity_row in entity_rows:
                entity_pk = entity_row["id"]
                entity_ext = entity_row["external_id"]

                # Get all active memories for this entity
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE entity_id = ? AND status = 'active' AND invalid_at IS NULL
                    ORDER BY mention_count DESC, confidence DESC
                    """,
                    (entity_pk,),
                ).fetchall()

                if len(rows) < 2:
                    continue

                stats["scanned"] += len(rows)

                # Similarity for every pair, computed once as a matmul over the
                # entity's embeddings. The previous nested loop re-parsed both
                # rows' JSON embeddings for every (i, j) pair — O(n²) parses.
                ids, matrix = self._entity_matrix(conn, entity_pk, [int(r["id"]) for r in rows])
                if matrix is None or len(ids) < 2:
                    continue
                position = {memory_id: idx for idx, memory_id in enumerate(ids)}
                row_by_id = {int(r["id"]): r for r in rows}

                merged_ids: set[int] = set()
                for i, row_i in enumerate(rows):
                    if stats["merged"] >= max_merges:
                        break
                    idx_i = position.get(int(row_i["id"]))
                    if idx_i is None or int(row_i["id"]) in merged_ids:
                        continue
                    sims = matrix @ matrix[idx_i]
                    for idx_j in range(idx_i + 1, len(ids)):
                        if stats["merged"] >= max_merges:
                            break
                        if int(ids[idx_j]) in merged_ids:
                            continue
                        score = float(sims[idx_j])
                        if score < similarity_threshold:
                            continue

                        row_j = row_by_id[int(ids[idx_j])]
                        # Found near-duplicate - merge the lower confidence into higher
                        if row_i["confidence"] >= row_j["confidence"]:
                            keeper_id, merge_id = row_i["id"], row_j["id"]
                        else:
                            keeper_id, merge_id = row_j["id"], row_i["id"]

                        new_mention = int(row_i["mention_count"]) + int(row_j["mention_count"])
                        new_conf = max(float(row_i["confidence"]), float(row_j["confidence"]))
                        new_imp = max(float(row_i["importance"]), float(row_j["importance"]))
                        now = time.time()
                        merged_ids.add(int(merge_id))
                        stats["merged"] += 1
                        stats["superseded"] += 1
                        if dry_run:
                            continue

                        # Update keeper with combined info
                        conn.execute(
                            """
                            UPDATE memories
                            SET mention_count = ?, confidence = ?, importance = ?,
                                last_seen_at = ?, text = ?
                            WHERE id = ?
                            """,
                            (
                                new_mention,
                                new_conf,
                                new_imp,
                                now,
                                row_i["text"] if keeper_id == row_i["id"] else row_j["text"],
                                keeper_id,
                            ),
                        )

                        # Supersede the merged memory, stamping its validity end.
                        conn.execute(
                            "UPDATE memories SET status = 'superseded', invalid_at = ?, "
                            "expired_at = ?, superseded_by = ? WHERE id = ?",
                            (now, now, keeper_id, merge_id),
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

            if not dry_run:
                conn.commit()

        return stats

    def _entity_matrix(
        self,
        conn: sqlite3.Connection,
        entity_pk: int,
        memory_ids: list[int],
    ) -> tuple[list[int], Any]:
        """Normalised embedding matrix for a subset of one entity's memories."""
        import numpy as np

        placeholders = ",".join("?" * len(memory_ids))
        rows = conn.execute(
            f"SELECT id, embedding, embedding_f32 FROM memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        ids: list[int] = []
        vectors: list[list[float]] = []
        for row in rows:
            vec = embedding_from_row(row)
            if vec:
                ids.append(int(row["id"]))
                vectors.append(vec)
        if not vectors:
            return ids, None
        dim = max(len(v) for v in vectors)
        matrix = np.zeros((len(vectors), dim), dtype=np.float32)
        for i, vec in enumerate(vectors):
            matrix[i, : len(vec)] = vec
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
        return ids, matrix

    def enqueue_consolidation_job(self, entity_id: str | None = None) -> int:
        """Enqueue a consolidation job for background processing."""
        payload = {"entity_id": entity_id} if entity_id else {}
        return self.enqueue_job("consolidation", payload)

    def process_consolidation_job(self, job: Job) -> dict[str, Any]:
        """Process a consolidation job (called by background worker)."""
        entity_id = job.payload.get("entity_id")
        return self.run_consolidation(entity_id=entity_id)
