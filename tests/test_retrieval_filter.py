"""
Retrieval language filtering — the tests that fail if `--lang` starts leaking.

Two related defects lived here:

1. ``BM25Index.add_document`` accepted a ``metadata`` argument and **silently
   discarded it**, so BM25 candidates carried no language.
2. Hybrid retrieval fused those unfiltered BM25 candidates with the filtered
   dense results, so ``--lang python`` could return rows from other languages —
   and a BM25-only hit was rebuilt as a partial record with an empty
   ``file_path``, i.e. a result you could not locate.

Both sides are now filtered, and BM25-only hits are rebuilt from metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from tracera.retrieval.bm25 import BM25Index
from tracera.retrieval.hybrid import HybridRetriever


class _FakeDense:
    """Dense side that records the filter it was given and returns nothing."""

    def __init__(self) -> None:
        self.seen_language: object | None = "unset"

    def search(self, query: str, k: int = 10, language=None) -> list[dict]:
        self.seen_language = language
        return []


def _index_with_docs() -> BM25Index:
    idx = BM25Index()
    idx.add_document("py1", "def login(user, password): pass",
                     metadata={"file_path": "auth.py", "language": "python"})
    idx.add_document("ts1", "function login(user: string) {}",
                     metadata={"file_path": "auth.ts", "language": "typescript"})
    idx.add_document("tsx1", "export function Login() { return <div/> }",
                     metadata={"file_path": "Login.tsx", "language": "tsx"})
    return idx


# ── BM25 metadata ────────────────────────────────────────────────────────────


def test_bm25_stores_metadata():
    """The `metadata` argument must not be dropped on the floor."""
    idx = _index_with_docs()
    assert idx.get_metadata("py1") == {"file_path": "auth.py", "language": "python"}


def test_bm25_metadata_survives_a_round_trip(tmp_path: Path):
    path = tmp_path / "bm25.json"
    _index_with_docs().save(path)
    loaded = BM25Index.load(path)
    assert loaded.get_metadata("tsx1") == {
        "file_path": "Login.tsx",
        "language": "tsx",
    }


def test_bm25_load_tolerates_an_index_without_metadata(tmp_path: Path):
    """
    Indexes written before metadata existed must still load.

    They report *unknown* (None) rather than an empty dict, so callers can tell
    "no metadata" apart from "metadata with no language".
    """
    path = tmp_path / "bm25.json"
    _index_with_docs().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["metadata"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = BM25Index.load(path)
    assert loaded.doc_count == 3
    assert loaded.get_metadata("py1") is None


def test_bm25_remove_drops_metadata():
    idx = _index_with_docs()
    idx.remove_document("py1")
    assert idx.get_metadata("py1") is None


# ── Hybrid language filtering ────────────────────────────────────────────────


def test_language_filter_removes_bm25_candidates_from_other_languages():
    """A BM25-only hit must not bypass the language filter."""
    dense = _FakeDense()
    hybrid = HybridRetriever(_index_with_docs(), dense)
    results = hybrid.search("login", k=10, language="python")

    assert [r["id"] for r in results] == ["py1"], [r["id"] for r in results]
    assert results[0]["language"] == "python"


def test_language_filter_accepts_an_alias_group():
    """'typescript' expands to tsx, and both sides honour it."""
    hybrid = HybridRetriever(_index_with_docs(), _FakeDense())
    results = hybrid.search("login", k=10, language=["typescript", "tsx"])
    assert {r["id"] for r in results} == {"ts1", "tsx1"}


def test_unknown_language_is_dropped_under_a_filter(tmp_path: Path):
    """
    An index with no metadata cannot prove a candidate's language, so under a
    filter those candidates are excluded rather than guessed at.
    """
    path = tmp_path / "bm25.json"
    _index_with_docs().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["metadata"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = BM25Index.load(path)

    hybrid = HybridRetriever(legacy, _FakeDense())
    assert hybrid.search("login", k=10, language="python") == []


def test_bm25_only_hit_carries_a_real_file_path():
    """
    A BM25-only result must be locatable.

    It used to be rebuilt as ``{"file_path": "", "language": ""}`` — a chunk of
    text with no way to find it.
    """
    hybrid = HybridRetriever(_index_with_docs(), _FakeDense())
    results = hybrid.search("login", k=10)

    by_id = {r["id"]: r for r in results}
    assert by_id["py1"]["file_path"] == "auth.py"
    assert by_id["py1"]["language"] == "python"
    assert by_id["tsx1"]["file_path"] == "Login.tsx"


def test_without_a_filter_nothing_is_excluded():
    """No filter means no filtering — the BM25 side still contributes."""
    hybrid = HybridRetriever(_index_with_docs(), _FakeDense())
    assert {r["id"] for r in hybrid.search("login", k=10)} == {"py1", "ts1", "tsx1"}


def test_the_filter_is_passed_to_the_dense_side_unchanged():
    dense = _FakeDense()
    HybridRetriever(_index_with_docs(), dense).search("login", language=["typescript", "tsx"])
    assert dense.seen_language == ["typescript", "tsx"]
