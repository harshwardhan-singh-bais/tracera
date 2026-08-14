"""
Phase 16 — BM25 Lexical Index.

Provides tokenization, an inverted index, and BM25F scoring for
keyword-based code retrieval. Works entirely offline with no API calls.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from tracera.logging import get_logger

log = get_logger("retrieval.bm25")

# BM25 hyper-parameters (standard defaults)
_K1 = 1.5   # term frequency saturation
_B = 0.75   # length normalisation factor


def _tokenize(text: str) -> list[str]:
    """
    Simple code-aware tokenizer.
    Splits on whitespace, punctuation, and camelCase/snake_case boundaries.
    Lowercases and removes stop-tokens.
    """
    # Split camelCase: e.g. "getUserById" -> "get User By Id"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Replace non-alphanumeric with space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = text.lower().split()
    # Filter very short tokens (1-char) — they carry no information
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """
    An in-memory BM25 index over CodeChunk objects (or any string-keyed docs).

    Usage:
        index = BM25Index()
        index.add_document("doc1", "def login(user, password): ...")
        index.add_document("doc2", "class AuthMiddleware: ...")
        results = index.search("login function", k=5)
    """

    def __init__(self) -> None:
        # doc_id → raw text content
        self._docs: dict[str, str] = {}
        # doc_id → list of tokens
        self._tokenized: dict[str, list[str]] = {}
        # token → {doc_id: count}
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        # doc_id → token count (doc length)
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._doc_count: int = 0

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a single document to the index.

        Idempotent: re-adding an existing doc_id replaces its content
        (used by incremental re-indexing of modified files) without
        double-counting documents.
        """
        existed = doc_id in self._docs
        if existed:
            # Drop the old postings for this doc_id before re-indexing it
            for token in set(self._tokenized.get(doc_id, [])):
                self._inverted[token].pop(doc_id, None)
                if not self._inverted[token]:
                    del self._inverted[token]

        tokens = _tokenize(text)
        self._docs[doc_id] = text
        self._tokenized[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)
        if not existed:
            self._doc_count += 1

        # Update inverted index with term frequencies
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for term, count in tf.items():
            self._inverted[term][doc_id] = count

        # Recalculate average doc length
        if self._doc_count > 0:
            self._avg_doc_len = sum(self._doc_lengths.values()) / self._doc_count

    def add_documents(self, docs: list[tuple[str, str]]) -> None:
        """Bulk-add (doc_id, text) pairs."""
        for doc_id, text in docs:
            self.add_document(doc_id, text)
        log.debug("BM25 index: %d documents indexed", self._doc_count)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index (for incremental re-indexing)."""
        if doc_id not in self._docs:
            return
        tokens = self._tokenized.pop(doc_id, [])
        self._docs.pop(doc_id, None)
        self._doc_lengths.pop(doc_id, None)
        self._doc_count = max(0, self._doc_count - 1)

        for token in set(tokens):
            self._inverted[token].pop(doc_id, None)
            if not self._inverted[token]:
                del self._inverted[token]

        if self._doc_count > 0:
            self._avg_doc_len = sum(self._doc_lengths.values()) / self._doc_count
        else:
            self._avg_doc_len = 0.0

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, query_tokens: list[str], doc_id: str) -> float:
        """Compute BM25 score for a single document."""
        dl = self._doc_lengths.get(doc_id, 0)
        score = 0.0
        for term in query_tokens:
            if term not in self._inverted:
                continue
            tf = self._inverted[term].get(doc_id, 0)
            if tf == 0:
                continue
            df = len(self._inverted[term])  # document frequency
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (_K1 + 1)) / (
                tf + _K1 * (1 - _B + _B * dl / max(self._avg_doc_len, 1))
            )
            score += idf * tf_norm
        return score

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Search the index.

        Returns:
            List of (doc_id, score) sorted by descending BM25 score.
        """
        if not self._docs:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Only score documents that contain at least one query term
        candidate_doc_ids: set[str] = set()
        for term in query_tokens:
            candidate_doc_ids.update(self._inverted.get(term, {}).keys())

        scored = [
            (doc_id, self._score(query_tokens, doc_id))
            for doc_id in candidate_doc_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_document(self, doc_id: str) -> str | None:
        return self._docs.get(doc_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize index to disk as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "docs": self._docs,
            "tokenized": self._tokenized,
            "inverted": {k: dict(v) for k, v in self._inverted.items()},
            "doc_lengths": self._doc_lengths,
            "avg_doc_len": self._avg_doc_len,
            "doc_count": self._doc_count,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        log.debug("BM25 index saved: %s (%d docs)", path, self._doc_count)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load index from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls()
        idx._docs = data["docs"]
        idx._tokenized = data["tokenized"]
        idx._inverted = defaultdict(dict, {k: dict(v) for k, v in data["inverted"].items()})
        idx._doc_lengths = data["doc_lengths"]
        idx._avg_doc_len = data["avg_doc_len"]
        idx._doc_count = data["doc_count"]
        log.debug("BM25 index loaded: %s (%d docs)", path, idx._doc_count)
        return idx

    @property
    def doc_count(self) -> int:
        return self._doc_count
