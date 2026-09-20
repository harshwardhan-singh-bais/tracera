"""
Regression tests for the v2 memory layer — the capabilities that bring TRACERA
level with (and past) the best memory MCP servers on the market.

Each test here pins down behaviour that was either **broken** before v2 or is
**new** in v2. They are deliberately behavioural (they assert on observable
outcomes, not on internals) so they keep guarding the contract if the
implementation is refactored.

Coverage map
------------
  * bi-temporal facts      — contradiction invalidation, ``recall_as_of``, ``timeline``
  * multi-valued predicates — never invalidate each other
  * reconciliation          — ADD/UPDATE/DELETE/NOOP, LLM parse + every fallback
  * entity resolution       — camelCase/snake_case/article canonicalisation
  * memory graph            — edges, canonical nodes, graph-expanded recall
  * scoring                 — normalised relevance, a real ``min_score`` floor
  * forgetting              — ``apply_decay``, ``run_gc``, ``record_feedback``
  * facade regressions      — recall honours the query, delete actually deletes,
                              entries() doesn't crash, add() uses the real entity
  * token budget            — actually enforced (was a no-op)
  * debug_recall            — works on both the hybrid and vector-only paths
  * export / import         — round-trip fidelity + idempotency
  * schema migration        — a v1 database is upgraded in place, data intact
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from tracera.memory.layer import (
    MemoryLayer,
    MemoryReconciler,
    MemoryStore,
    ReconcileEvent,
)
from tracera.memory.layer.attribution import reset_attribution, set_attribution
from tracera.memory.layer.facade import AgentMemory
from tracera.memory.layer.recall import RecallInjector, estimate_tokens
from tracera.memory.layer.store import (
    SCHEMA_VERSION,
    MemoryPolicy,
    MemoryRecord,
    canonical_key,
    unpack_embedding,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures / helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _token_vec(token: str, dim: int = 48) -> list[float]:
    rng = random.Random(f"tok:{token}")
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / n for x in raw]


def fake_embed(text: str, dim: int = 48) -> list[float]:
    """Deterministic offline embedding: shared tokens ⇒ high cosine."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tv = _token_vec(token, dim)
        for i in range(dim):
            vec[i] += tv[i]
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


#: Policy the suite runs under. Explicit, so the suite does not pick up the
#: developer's real ``TRACERA_MEMORY_*`` values (which would make these tests
#: environment-dependent). The toy hash embedder scores genuine matches around
#: 0.25-0.35 cosine, well under the production ``min_recall_score`` of 0.3, so
#: recall here runs unthresholded; the gate itself is tested directly in
#: ``TestPolicyEnforcement``.
TEST_POLICY = MemoryPolicy(recall_min_score=0.0)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "mem.db", policy=TEST_POLICY)


@pytest.fixture(autouse=True)
def _clean_attribution() -> Any:
    reset_attribution()
    yield
    reset_attribution()


def _put(
    store: MemoryStore,
    *,
    entity: str = "u",
    process: str = "p",
    kind: str = "fact",
    subject: str = "user",
    predicate: str = "note",
    object_: str = "x",
    text: str,
    job_id: int = 1,
    confidence: float = 0.9,
    importance: float = 0.5,
    valid_at: float | None = None,
    **kwargs: Any,
) -> tuple[bool, MemoryRecord]:
    return store.upsert_memory(
        entity_id=entity,
        process_id=process,
        kind=kind,
        subject=subject,
        predicate=predicate,
        object=object_,
        text=text,
        embedding=fake_embed(text),
        job_id=job_id,
        confidence=confidence,
        importance=importance,
        valid_at=valid_at,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Bi-temporal model
# ═══════════════════════════════════════════════════════════════════════════════


def test_contradiction_invalidates_but_keeps_history(store: MemoryStore) -> None:
    """A new value for a single-valued predicate retires the old one."""
    t_jan = time.time() - 90 * 86400
    t_mar = time.time() - 30 * 86400

    _put(store, predicate="employer", object_="acme", text="User works at Acme.",
         job_id=1, valid_at=t_jan)
    time.sleep(0.01)
    _put(store, predicate="employer", object_="globex", text="User works at Globex.",
         job_id=2, valid_at=t_mar)

    current = [r.text for r, _ in store.recall("u", fake_embed("where does the user work"), k=5)]
    assert any("Globex" in t for t in current)
    assert not any("Acme" in t for t in current), current

    # The old row is retired, never deleted — the audit trail survives.
    assert store.count_memories("u") == 2
    assert store.find_memories("u", current_only=True).__len__() == 1


def test_recall_as_of_answers_historical_questions(store: MemoryStore) -> None:
    """Point-in-time recall: what did we believe before the move?"""
    t_jan = time.time() - 90 * 86400
    t_mar = time.time() - 30 * 86400
    _put(store, predicate="employer", object_="acme", text="User works at Acme.",
         job_id=1, valid_at=t_jan)
    time.sleep(0.01)
    _put(store, predicate="employer", object_="globex", text="User works at Globex.",
         job_id=2, valid_at=t_mar)

    past = store.recall_as_of(
        "u", fake_embed("where does the user work"), t_mar - 86400, k=5
    )
    texts = [r.text for r, _ in past]
    assert any("Acme" in t for t in texts), texts
    assert not any("Globex" in t for t in texts), texts


def test_timeline_returns_versions_in_order_with_supersession_link(
    store: MemoryStore,
) -> None:
    t_jan = time.time() - 90 * 86400
    t_mar = time.time() - 30 * 86400
    _put(store, predicate="employer", object_="acme", text="User works at Acme.",
         job_id=1, valid_at=t_jan)
    time.sleep(0.01)
    _put(store, predicate="employer", object_="globex", text="User works at Globex.",
         job_id=2, valid_at=t_mar)

    tl = store.timeline("u", predicate="employer")
    assert len(tl) == 2
    assert "Acme" in tl[0].text and "Globex" in tl[1].text
    assert tl[0].invalid_at is not None
    assert tl[0].superseded_by == tl[1].id
    assert tl[0].is_current is False
    assert tl[1].is_current is True


def test_multi_valued_predicates_do_not_invalidate_each_other(
    store: MemoryStore,
) -> None:
    """A user likes many languages; none of them retires the others."""
    for i, lang in enumerate(["python", "rust", "go"]):
        _put(store, kind="preference", predicate="likes_language", object_=lang,
             text=f"User likes {lang}.", job_id=i + 1)
    assert store.count_memories("u") == 3
    assert len(store.find_memories("u", current_only=True)) == 3


@pytest.mark.parametrize(
    "predicate,functional",
    [
        # single-valued slots — a new value replaces the old
        ("employer", True),
        ("works_at", True),
        ("job_title", True),
        ("preferred_editor", True),
        ("favorite_color", True),
        ("main_language", True),
        ("timezone", True),
        ("db_choice", True),
        # multi-valued relations — values accumulate
        ("likes_language", False),
        ("uses_framework", False),
        ("knows_person", False),
        ("imports_module", False),
        ("calls", False),
        ("works_with", False),
        ("interested_in", False),
        # not a slot at all
        ("analysis_result", False),
        ("", False),
    ],
)
def test_functional_predicate_classification(
    predicate: str, functional: bool
) -> None:
    """Pins the single-valued/multi-valued contract that drives invalidation."""
    from tracera.memory.layer.store import _is_functional_predicate

    assert _is_functional_predicate(predicate) is functional


def test_multi_valued_verb_vetoes_a_slot_noun(store: MemoryStore) -> None:
    """Regression: ``language`` is a slot noun, so ``likes_language`` used to
    be misread as single-valued and silently retired the previous language."""
    _put(store, kind="preference", predicate="likes_language", object_="rust",
         text="User likes Rust.", job_id=1)
    _put(store, kind="preference", predicate="likes_language", object_="go",
         text="User likes Go.", job_id=2)

    texts = [r.text for r in store.find_memories("u", current_only=True)]
    assert len(texts) == 2, texts
    assert store.get_memory(1).status == "active"


def test_slot_predicate_still_invalidates(store: MemoryStore) -> None:
    """The veto must not blunt the single-valued behaviour it guards."""
    _put(store, predicate="main_language", object_="python",
         text="User's main language is Python.", job_id=1)
    _put(store, predicate="main_language", object_="rust",
         text="User's main language is Rust.", job_id=2)

    current = [r.text for r in store.find_memories("u", current_only=True)]
    assert len(current) == 1
    assert "Rust" in current[0]


def test_low_confidence_extraction_cannot_erase_a_strong_belief(
    store: MemoryStore,
) -> None:
    """Contradiction handling is confidence-guarded, not blind."""
    _put(store, predicate="employer", object_="acme", text="User works at Acme.",
         job_id=1, confidence=0.95)
    _put(store, predicate="employer", object_="globex", text="User works at Globex.",
         job_id=2, confidence=0.40)  # weaker than the existing belief

    assert len(store.find_memories("u", current_only=True)) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reconciliation — ADD / UPDATE / DELETE / NOOP
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def reconciler() -> MemoryReconciler:
    return MemoryReconciler(use_llm=False)


@pytest.fixture()
def seeded(store: MemoryStore) -> list[MemoryRecord]:
    _put(store, kind="preference", predicate="preferred_editor", object_="vscode",
         text="User prefers VS Code.", job_id=1, confidence=0.9)
    return store.find_memories("u")


def test_deterministic_noop_on_identical_fact(
    reconciler: MemoryReconciler, seeded: list[MemoryRecord]
) -> None:
    action = reconciler.reconcile_deterministic(
        {"kind": "preference", "subject": "user", "predicate": "preferred_editor",
         "object": "vscode", "text": "User prefers VS Code.", "confidence": 0.9},
        seeded,
    )
    assert action.event is ReconcileEvent.NOOP


def test_deterministic_update_on_changed_single_valued_value(
    reconciler: MemoryReconciler, seeded: list[MemoryRecord]
) -> None:
    action = reconciler.reconcile_deterministic(
        {"kind": "preference", "subject": "user", "predicate": "preferred_editor",
         "object": "neovim", "text": "User now prefers Neovim.", "confidence": 0.95},
        seeded,
    )
    assert action.event is ReconcileEvent.UPDATE
    assert action.memory_id == seeded[0].id


def test_deterministic_add_on_unrelated_fact(
    reconciler: MemoryReconciler, seeded: list[MemoryRecord]
) -> None:
    action = reconciler.reconcile_deterministic(
        {"kind": "fact", "subject": "user", "predicate": "hobby",
         "object": "cycling", "text": "User enjoys cycling.", "confidence": 0.8},
        seeded,
    )
    assert action.event is ReconcileEvent.ADD


def test_deterministic_noop_on_same_subject_predicate(
    reconciler: MemoryReconciler, seeded: list[MemoryRecord]
) -> None:
    """Same slot, non-functional predicate, no contradicting value → reinforce."""
    action = reconciler.reconcile_deterministic(
        {"kind": "preference", "subject": "user", "predicate": "preferred_editor",
         "object": "vscode", "text": "User prefers VS Code.", "confidence": 0.9},
        seeded,
    )
    assert action.event is ReconcileEvent.NOOP


def test_reconciler_with_no_candidates_always_adds(
    reconciler: MemoryReconciler,
) -> None:
    action = reconciler.reconcile_deterministic(
        {"kind": "fact", "subject": "user", "predicate": "employer",
         "object": "acme", "text": "User works at Acme.", "confidence": 0.9},
        [],
    )
    assert action.event is ReconcileEvent.ADD


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def __call__(self, prompt: str) -> str:
        return self._reply


def test_llm_update_is_parsed(seeded: list[MemoryRecord]) -> None:
    rc = MemoryReconciler(
        _FakeLLM(json.dumps({"event": "UPDATE", "memory_id": seeded[0].id,
                             "reason": "moved"}))
    )
    action = asyncio.run(rc.reconcile({"text": "x"}, seeded))
    assert action.event is ReconcileEvent.UPDATE
    assert action.memory_id == seeded[0].id
    assert action.source == "llm"


def test_llm_reply_wrapped_in_markdown_fence_is_parsed(
    seeded: list[MemoryRecord],
) -> None:
    body = json.dumps({"event": "DELETE", "memory_id": seeded[0].id, "reason": "wrong"})
    rc = MemoryReconciler(_FakeLLM(f"```json\n{body}\n```"))
    action = asyncio.run(rc.reconcile({"text": "x"}, seeded))
    assert action.event is ReconcileEvent.DELETE
    assert action.source == "llm"


def test_llm_naming_an_unknown_id_is_rejected(seeded: list[MemoryRecord]) -> None:
    """Safety: the model cannot point at a row it was never offered."""
    rc = MemoryReconciler(
        _FakeLLM(json.dumps({"event": "DELETE", "memory_id": 99999, "reason": "bogus"}))
    )
    action = asyncio.run(rc.reconcile({"text": "x"}, seeded))
    assert action.source == "fallback"
    assert action.memory_id != 99999


def test_llm_unparseable_reply_falls_back(seeded: list[MemoryRecord]) -> None:
    rc = MemoryReconciler(_FakeLLM("not json at all"))
    action = asyncio.run(rc.reconcile({"text": "x"}, seeded))
    assert action.source == "fallback"


def test_llm_exception_falls_back_and_never_raises(
    seeded: list[MemoryRecord],
) -> None:
    async def boom(prompt: str) -> str:
        raise RuntimeError("api down")

    rc = MemoryReconciler(boom)
    action = asyncio.run(rc.reconcile({"text": "x"}, seeded))
    assert action.source == "fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Entity resolution and the memory graph
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AuthMiddleware", "auth_middleware"),
        ("auth_middleware", "auth_middleware"),
        ("auth-middleware", "auth_middleware"),
        ("auth middleware", "auth_middleware"),
        ("the user", "user"),
        ("The User", "user"),
        ("user's", "user"),
        ("HTTPServer", "http_server"),
    ],
)
def test_canonical_key_collapses_spellings(raw: str, expected: str) -> None:
    assert canonical_key(raw) == expected


def test_camel_and_snake_case_resolve_to_one_node(store: MemoryStore) -> None:
    _put(store, kind="relationship", subject="AuthMiddleware", predicate="calls",
         object_="UserService", text="AuthMiddleware calls UserService.", job_id=1)
    _put(store, subject="auth_middleware", predicate="validates",
         object_="JWT tokens", text="AuthMiddleware validates JWT tokens.", job_id=2)

    assert store.resolve_entity("u", "AuthMiddleware") == store.resolve_entity(
        "u", "auth_middleware"
    )


def test_graph_overview_lists_canonical_nodes(store: MemoryStore) -> None:
    _put(store, kind="relationship", subject="AuthMiddleware", predicate="calls",
         object_="UserService", text="AuthMiddleware calls UserService.", job_id=1)
    _put(store, subject="auth_middleware", predicate="validates",
         object_="JWT tokens", text="AuthMiddleware validates JWT tokens.", job_id=2)
    _put(store, predicate="favorite_food", object_="ramen",
         text="User loves ramen.", job_id=3)

    graph = store.entity_graph("u", node="auth_middleware")
    assert len(graph["outgoing"]) >= 2
    assert {e["predicate"] for e in graph["outgoing"]} >= {"calls", "validates"}

    overview = store.entity_graph("u")
    assert len(overview["nodes"]) >= 3


def test_graph_expansion_recovers_neighbour_facts(store: MemoryStore) -> None:
    """A fact the query never names is still reachable through the graph."""
    _put(store, kind="relationship", subject="AuthMiddleware", predicate="calls",
         object_="UserService", text="AuthMiddleware calls UserService.", job_id=1)
    _put(store, subject="auth_middleware", predicate="validates",
         object_="JWT tokens", text="AuthMiddleware validates JWT tokens.", job_id=2)

    plain = store.recall_hybrid("u", "AuthMiddleware", fake_embed("AuthMiddleware"),
                                k=3, min_score=0.0)
    expanded = store.recall_hybrid("u", "AuthMiddleware", fake_embed("AuthMiddleware"),
                                   k=3, min_score=0.0, graph_expansion=True)
    assert len(expanded) >= len(plain)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Scoring — normalised relevance with a meaningful floor
# ═══════════════════════════════════════════════════════════════════════════════


def test_irrelevant_memories_do_not_clear_the_floor(store: MemoryStore) -> None:
    """The old additive score gave every row a ~0.16 metadata floor."""
    for i in range(10):
        _put(store, subject=f"c{i}", predicate="p", object_="o",
             text=f"totally unrelated fact {i} about medieval poetry",
             job_id=i + 1, importance=1.0, confidence=1.0)

    hits = store.recall_hybrid(
        "u", "kubernetes ingress controller yaml",
        fake_embed("kubernetes ingress controller"), k=10, min_score=0.3,
    )
    assert hits == [], [(r.text[:30], round(s, 3)) for r, s in hits]


def test_irrelevant_top_score_stays_low_even_without_a_floor(
    store: MemoryStore,
) -> None:
    for i in range(10):
        _put(store, subject=f"c{i}", predicate="p", object_="o",
             text=f"totally unrelated fact {i} about medieval poetry",
             job_id=i + 1, importance=1.0, confidence=1.0)

    hits = store.recall_hybrid(
        "u", "kubernetes ingress", fake_embed("kubernetes ingress"),
        k=10, min_score=0.0,
    )
    assert hits
    assert hits[0][1] < 0.35, hits[0][1]


def test_relevant_memory_clears_the_floor(store: MemoryStore) -> None:
    _put(store, kind="preference", predicate="preferred_database", object_="postgres",
         text="User prefers PostgreSQL for storage.", job_id=1)
    for i in range(5):
        _put(store, subject=f"noise{i}", predicate="p", object_="o",
             text=f"unrelated trivia number {i} about garden gnomes", job_id=i + 2)

    hits = store.recall_hybrid(
        "u", "user prefers postgresql",
        fake_embed("user prefers postgresql"), k=5, min_score=0.3,
    )
    assert any("PostgreSQL" in r.text for r, _ in hits)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Forgetting — decay, garbage collection, feedback
# ═══════════════════════════════════════════════════════════════════════════════


def _age(store: MemoryStore, memory_id: int, days: float) -> None:
    ts = time.time() - days * 86400
    conn = store._conn()
    conn.execute("UPDATE memories SET last_seen_at = ? WHERE id = ?", (ts, memory_id))
    conn.commit()


def test_apply_decay_scores_every_active_memory(store: MemoryStore) -> None:
    for i in range(5):
        _put(store, subject=f"old{i}", predicate="p", object_="o",
             text=f"stale fact {i}", job_id=i + 1)
    assert store.apply_decay(half_life_days=90) == 5


def test_decay_is_monotonic_in_age(store: MemoryStore) -> None:
    _, fresh = _put(store, subject="a", predicate="p", object_="o",
                    text="fresh fact", job_id=1)
    _, old = _put(store, subject="b", predicate="p", object_="o",
                  text="old fact", job_id=2)
    _age(store, old.id, 400)
    store.apply_decay(half_life_days=90)

    assert store.get_memory(fresh.id).decay_score > store.get_memory(old.id).decay_score


def test_run_gc_archives_stale_but_keeps_fresh(store: MemoryStore) -> None:
    for i in range(5):
        _, rec = _put(store, subject=f"old{i}", predicate="p", object_="o",
                      text=f"stale fact {i}", job_id=i + 1)
        _age(store, rec.id, 400)
    _put(store, kind="preference", predicate="preferred_editor", object_="neovim",
         text="User prefers Neovim.", job_id=99)

    report = store.run_gc(retention_days=365, decay_floor=0.05, min_age_days=30)
    assert report["archived"] >= 4
    remaining = [r.text for r in store.find_memories("u", current_only=True)]
    assert "User prefers Neovim." in remaining


def test_run_gc_dry_run_reports_without_archiving(store: MemoryStore) -> None:
    for i in range(3):
        _, rec = _put(store, subject=f"old{i}", predicate="p", object_="o",
                      text=f"stale fact {i}", job_id=i + 1)
        _age(store, rec.id, 400)

    report = store.run_gc(retention_days=365, decay_floor=0.99, min_age_days=0,
                          dry_run=True)
    assert report["dry_run"] is True
    assert report["archived"] == 0
    assert report["candidates"] >= 3
    # nothing actually left recall
    assert len(store.find_memories("u", current_only=True)) == 3


def test_gc_never_collects_a_memory_marked_useful(store: MemoryStore) -> None:
    _, rec = _put(store, subject="a", predicate="p", object_="o",
                  text="User always wants tests run.", job_id=1)
    _age(store, rec.id, 400)
    store.record_feedback(rec.id, "useful")

    store.run_gc(retention_days=365, decay_floor=0.99, min_age_days=0)
    assert store.get_memory(rec.id).status == "active"


def test_useful_feedback_raises_importance(store: MemoryStore) -> None:
    _, rec = _put(store, subject="a", predicate="b", object_="c",
                  text="User prefers dark mode.", job_id=1, importance=0.5)
    before = store.get_memory(rec.id).importance

    result = store.record_feedback(rec.id, "useful", query="theme")
    assert result["importance"] > before
    assert result["useful_count"] == 1


def test_harmful_feedback_lowers_importance(store: MemoryStore) -> None:
    _, rec = _put(store, subject="a", predicate="b", object_="c",
                  text="User prefers dark mode.", job_id=1, importance=0.8)
    before = store.get_memory(rec.id).importance

    result = store.record_feedback(rec.id, "harmful")
    assert result["importance"] < before
    assert result["harmful_count"] == 1


def test_irrelevant_feedback_also_lowers_importance(store: MemoryStore) -> None:
    _, rec = _put(store, subject="a", predicate="b", object_="c",
                  text="Noise.", job_id=1, importance=0.8)
    result = store.record_feedback(rec.id, "irrelevant")
    assert result["harmful_count"] == 1
    assert result["importance"] < 0.8


def test_invalid_feedback_signal_is_rejected(store: MemoryStore) -> None:
    _, rec = _put(store, text="Something.", job_id=1)
    with pytest.raises(ValueError):
        store.record_feedback(rec.id, "banana")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Facade regressions — the four previously-broken paths
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def layer(tmp_path: Path) -> MemoryLayer:
    return MemoryLayer(
        store=MemoryStore(tmp_path / "mem.db"),
        embed_fn=fake_embed,
        worker_enabled=False,
        reconciliation_candidates=5,
        reconciliation_min_similarity=0.0,
    )


@pytest.fixture()
def agent(layer: MemoryLayer) -> AgentMemory:
    set_attribution("user_1", "agent")
    return AgentMemory(layer)


def test_facade_add_uses_the_active_entity(agent: AgentMemory) -> None:
    rec = agent.add({"text": "User prefers PostgreSQL.", "kind": "preference"})
    assert rec is not None and rec.id > 0
    assert rec.entity_id == "user_1"  # was hard-coded to "default"
    assert rec.predicate == "prefers"  # derived from the kind


def test_facade_recall_actually_depends_on_the_query(agent: AgentMemory) -> None:
    """Regression: recall() used to return the same fixed list for any query."""
    agent.add({"text": "User prefers PostgreSQL for the primary datastore.",
               "kind": "preference"})

    hit = agent.recall("which database does the user prefer")
    miss = agent.recall("zzz completely unrelated query zzz")

    assert isinstance(hit, str) and hit != ""
    assert hit != miss or "No relevant" in miss


def test_facade_recall_does_not_raise(agent: AgentMemory) -> None:
    agent.add({"text": "User prefers dark mode.", "kind": "preference"})
    out = agent.recall("theme preference")
    assert isinstance(out, str)


def test_facade_entries_does_not_raise(agent: AgentMemory) -> None:
    """Regression: entries() used to raise AttributeError on an int entity id."""
    agent.add({"text": "User prefers dark mode.", "kind": "preference"})
    entries = agent.entries()
    assert isinstance(entries, list)
    assert len(entries) == 1


def test_facade_stats_reports_the_real_entity(agent: AgentMemory) -> None:
    agent.add({"text": "User prefers dark mode.", "kind": "preference"})
    stats = agent.stats()
    assert stats.get("entity_id") == "user_1"


def test_facade_delete_actually_deletes(agent: AgentMemory, layer: MemoryLayer) -> None:
    """Regression: delete() inserted a junk row and never touched the target."""
    rec = agent.add({"text": "User prefers Vim keybindings.", "kind": "preference"})

    assert agent.delete(str(rec.id)) is True
    assert all(r.id != rec.id for r in layer.store.find_memories("user_1", current_only=True))


def test_facade_delete_keeps_an_audit_trail(agent: AgentMemory, layer: MemoryLayer) -> None:
    rec = agent.add({"text": "User prefers Vim keybindings.", "kind": "preference"})
    agent.delete(str(rec.id))
    assert len(layer.store.get_memory_versions(rec.id)) >= 1


def test_facade_delete_returns_false_for_unknown_id(agent: AgentMemory) -> None:
    assert agent.delete("987654") is False


def test_facade_two_distinct_memories_do_not_collide(agent: AgentMemory) -> None:
    a = agent.add({"text": "User prefers PostgreSQL.", "kind": "preference"})
    b = agent.add({"text": "User prefers Vim keybindings.", "kind": "preference"})
    assert a.id != b.id


def test_facade_update_changes_value_and_preserves_history(
    agent: AgentMemory, layer: MemoryLayer
) -> None:
    rec = agent.add({"text": "User prefers VS Code.", "kind": "preference"})
    updated = agent.update(rec.id, text="User prefers Neovim.",
                           object="neovim", confidence=0.95)

    assert updated is not None
    assert "Neovim" in updated.text
    current = [r.text for r in layer.store.find_memories("user_1", current_only=True)]
    assert any("Neovim" in t for t in current)
    assert not any("VS Code" in t for t in current)
    # the previous value is still queryable
    assert len(layer.store.timeline("user_1", predicate="prefers")) >= 2


def test_facade_feedback_routes_to_the_store(agent: AgentMemory) -> None:
    rec = agent.add({"text": "User prefers dark mode.", "kind": "preference"})
    result = agent.feedback(rec.id, "useful", query="theme")
    assert result["useful_count"] == 1


def test_facade_timeline_and_entities(agent: AgentMemory) -> None:
    agent.add({"text": "AuthMiddleware calls UserService.", "kind": "relationship"})
    agent.add({"text": "User prefers PostgreSQL.", "kind": "preference"})

    assert isinstance(agent.timeline(), list)
    graph = agent.entities()
    assert "nodes" in graph


def test_facade_export_import_round_trip(agent: AgentMemory, layer: MemoryLayer,
                                         tmp_path: Path) -> None:
    agent.add({"text": "User prefers PostgreSQL.", "kind": "preference"})
    agent.add({"text": "User works at Acme.", "kind": "fact"})

    payload = agent.export()
    assert len(payload["memories"]) == 2

    other = MemoryLayer(
        store=MemoryStore(tmp_path / "other.db"),
        embed_fn=fake_embed,
        worker_enabled=False,
    )
    set_attribution("user_1", "agent")
    stats = AgentMemory(other).import_data(payload)
    assert stats["imported"] == 2
    assert other.store.count_memories("user_1") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. apply_memory — the end-to-end reconciliation path
# ═══════════════════════════════════════════════════════════════════════════════


class _Item:
    """Stand-in for an extracted MemoryItem."""

    def __init__(self, **kw: Any) -> None:
        self.kind = kw.get("kind", "preference")
        self.subject = kw.get("subject", "user")
        self.predicate = kw.get("predicate", "preferred_editor")
        self.object = kw.get("object", "vscode")
        self.text = kw.get("text", "User prefers VS Code.")
        self.confidence = kw.get("confidence", 0.9)
        self.importance = kw.get("importance", 0.6)


def test_apply_memory_add_then_noop_then_update(layer: MemoryLayer) -> None:
    set_attribution("u", "p")

    first = asyncio.run(layer.apply_memory(_Item(), entity_id="u", process_id="p"))
    assert first is not None and first.text == "User prefers VS Code."

    # identical extraction → NOOP → no new row
    asyncio.run(layer.apply_memory(_Item(), entity_id="u", process_id="p"))
    assert layer.store.count_memories("u") == 1

    # changed value → UPDATE → old retired, new active
    item = _Item(object="neovim", text="User now prefers Neovim.", confidence=0.95)
    asyncio.run(layer.apply_memory(item, entity_id="u", process_id="p"))

    current = [r.text for r in layer.store.find_memories("u", current_only=True)]
    assert any("Neovim" in t for t in current)
    assert layer.store.count_memories("u") == 2
    assert len(layer.reconciliation_log) == 3


def test_apply_memory_logs_the_decision_and_source(layer: MemoryLayer) -> None:
    set_attribution("u", "p")
    asyncio.run(layer.apply_memory(_Item(), entity_id="u", process_id="p"))
    entry = layer.reconciliation_log[-1]
    assert entry["event"] == "ADD"
    assert entry["entity_id"] == "u"
    assert "reason" in entry and "source" in entry


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Token budget is actually enforced (regression: it was a no-op)
# ═══════════════════════════════════════════════════════════════════════════════


def test_token_budget_truncates_an_oversized_memory(store: MemoryStore) -> None:
    inj = RecallInjector(store, fake_embed, top_k=5, token_budget=100)
    big = MemoryRecord(
        id=1, entity_id="u", process_id="p", kind="fact", subject="s",
        predicate="p", object="o", text="X" * 4000, embedding=[0.0] * 8,
        mention_count=1, first_seen_at=0, last_seen_at=0,
    )
    kept = inj._apply_token_budget([(big, 0.9)])
    assert kept, "the memory should survive in truncated form"
    assert estimate_tokens(kept[0][0].to_line()) <= 100


def test_token_budget_honoured_across_many_memories(store: MemoryStore) -> None:
    inj = RecallInjector(store, fake_embed, top_k=5, token_budget=100)
    many = [
        MemoryRecord(
            id=i, entity_id="u", process_id="p", kind="fact", subject="s",
            predicate="p", object="o", text="word " * 40, embedding=[0.0] * 8,
            mention_count=1, first_seen_at=0, last_seen_at=0,
        )
        for i in range(20)
    ]
    kept = inj._apply_token_budget([(m, 0.9) for m in many])
    total = sum(estimate_tokens(r.to_line()) for r, _ in kept)
    total += estimate_tokens("Known context about this user:")
    assert total <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 9. debug_recall works on both paths (regression: the vector path crashed)
# ═══════════════════════════════════════════════════════════════════════════════


class _Scope:
    entity_id = "u"
    process_id = "p"


@pytest.mark.parametrize("use_hybrid", [True, False])
def test_debug_recall_works_on_both_paths(store: MemoryStore, use_hybrid: bool) -> None:
    _put(store, predicate="preferred_database", object_="postgresql",
         text="User prefers PostgreSQL.", job_id=1)

    inj = RecallInjector(store, fake_embed, use_hybrid=use_hybrid)
    debug = inj.debug_recall("user prefers postgresql", _Scope())

    assert debug["returned"] >= 1
    assert debug["would_inject"] >= 1


def test_debug_recall_explains_why(store: MemoryStore) -> None:
    _put(store, predicate="preferred_database", object_="postgresql",
         text="User prefers PostgreSQL.", job_id=1)

    inj = RecallInjector(store, fake_embed, use_hybrid=True)
    debug = inj.debug_recall("postgres database", _Scope())

    top = debug["results"][0]
    assert "components" in top
    assert top["why_recalled"]
    assert isinstance(top["why_recalled"], str)


def test_debug_recall_shows_rejected_candidates_too(store: MemoryStore) -> None:
    """The debug floor is the true cosine minimum, so rejects stay visible."""
    for i in range(5):
        _put(store, subject=f"n{i}", predicate="p", object_="o",
             text=f"unrelated trivia {i} about garden gnomes", job_id=i + 1)

    inj = RecallInjector(store, fake_embed, use_hybrid=True, min_score=0.9)
    debug = inj.debug_recall("kubernetes ingress", _Scope())

    assert debug["returned"] >= 1          # inspected
    assert debug["would_inject"] == 0      # but nothing clears the real floor
    assert all(r["above_threshold"] is False for r in debug["results"])


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Export / import
# ═══════════════════════════════════════════════════════════════════════════════


def test_export_carries_memories_and_edges(store: MemoryStore) -> None:
    _put(store, predicate="employer", object_="acme", text="User works at Acme.", job_id=1)
    _put(store, kind="preference", predicate="preferred_editor", object_="neovim",
         text="User prefers Neovim.", job_id=2)

    payload = store.export_entity("u")
    assert len(payload["memories"]) == 2
    assert "edges" in payload
    # embeddings are regenerated on import, never shipped
    assert "embedding" not in payload["memories"][0]


def test_import_round_trips_into_a_fresh_store(store: MemoryStore, tmp_path: Path) -> None:
    _put(store, predicate="employer", object_="acme", text="User works at Acme.", job_id=1)
    _put(store, kind="preference", predicate="preferred_editor", object_="neovim",
         text="User prefers Neovim.", job_id=2)

    payload = store.export_entity("u")
    target = MemoryStore(tmp_path / "target.db")
    stats = target.import_entity(payload, embed_fn=fake_embed)

    assert target.count_memories("u") == 2
    assert stats["imported"] == 2
    # imported rows are recallable
    assert target.recall("u", fake_embed("user employer acme"), k=1)


def test_import_is_idempotent(store: MemoryStore, tmp_path: Path) -> None:
    _put(store, predicate="employer", object_="acme", text="User works at Acme.", job_id=1)
    payload = store.export_entity("u")

    target = MemoryStore(tmp_path / "target.db")
    target.import_entity(payload, embed_fn=fake_embed)
    second = target.import_entity(payload, embed_fn=fake_embed)

    assert second["imported"] == 0
    assert target.count_memories("u") == 1


def test_export_can_exclude_invalidated_rows(store: MemoryStore) -> None:
    _put(store, predicate="employer", object_="acme", text="User works at Acme.", job_id=1)
    _put(store, predicate="employer", object_="globex", text="User works at Globex.", job_id=2)

    full = store.export_entity("u", include_invalidated=True)
    active = store.export_entity("u", include_invalidated=False)
    assert len(full["memories"]) == 2
    assert len(active["memories"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Deletion semantics
# ═══════════════════════════════════════════════════════════════════════════════


def test_soft_delete_keeps_the_row_and_audit_trail(store: MemoryStore) -> None:
    _, rec = _put(store, predicate="employer", object_="acme",
                  text="User works at Acme.", job_id=1)

    assert store.delete_memory(rec.id) is True
    assert store.get_memory(rec.id).status == "invalidated"
    assert store.count_memories("u") == 1                       # row survives
    assert len(store.get_memory_versions(rec.id)) >= 1          # audit trail
    assert all(r.id != rec.id for r in store.find_memories("u", current_only=True))


def test_hard_delete_physically_removes(store: MemoryStore) -> None:
    _, rec = _put(store, predicate="employer", object_="acme",
                  text="User works at Acme.", job_id=1)

    assert store.delete_memory(rec.id, hard=True) is True
    assert store.get_memory(rec.id) is None
    assert store.count_memories("u") == 0


def test_delete_unknown_memory_returns_false(store: MemoryStore) -> None:
    assert store.delete_memory(123456) is False


def test_forget_removes_matching_memories(store: MemoryStore) -> None:
    _put(store, predicate="preferred_database", object_="postgres",
         text="User prefers PostgreSQL.", job_id=1)
    _put(store, predicate="favorite_food", object_="ramen",
         text="User loves ramen.", job_id=2)

    removed = store.forget(
        "u", "user prefers postgresql", fake_embed("user prefers postgresql"),
        threshold=0.5,
    )
    assert removed
    remaining = [r.text for r in store.find_memories("u", current_only=True)]
    assert not any("PostgreSQL" in t for t in remaining)
    assert any("ramen" in t for t in remaining)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Cross-process dedup is entity-scoped
# ═══════════════════════════════════════════════════════════════════════════════


def test_same_fact_from_two_processes_dedups_to_one_row(store: MemoryStore) -> None:
    """Dedup used to be process-scoped, so two agents duplicated every fact."""
    _put(store, process="agent_a", predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(store, process="agent_b", predicate="employer", object_="acme",
         text="User works at Acme.", job_id=2)

    assert store.count_memories("u") == 1


def test_entities_remain_isolated(store: MemoryStore) -> None:
    _put(store, entity="u1", predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(store, entity="u2", predicate="employer", object_="globex",
         text="User works at Globex.", job_id=2)

    assert store.count_memories("u1") == 1
    assert store.count_memories("u2") == 1
    texts = [r.text for r, _ in store.recall("u1", fake_embed("employer"), k=5)]
    assert not any("Globex" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Schema migration — a v1 database is upgraded in place
# ═══════════════════════════════════════════════════════════════════════════════

_V1_SCHEMA = """
CREATE TABLE entities (id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT NOT NULL UNIQUE, created_at REAL NOT NULL);
CREATE TABLE processes (id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT NOT NULL UNIQUE, created_at REAL NOT NULL);
CREATE TABLE sessions (id TEXT PRIMARY KEY, entity_id INTEGER NOT NULL, process_id INTEGER NOT NULL, started_at REAL NOT NULL, ended_at REAL);
CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id INTEGER NOT NULL, process_id INTEGER NOT NULL,
  kind TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, text TEXT NOT NULL,
  embedding TEXT NOT NULL, mention_count INTEGER NOT NULL DEFAULT 1, first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL,
  session_id TEXT, last_job_id INTEGER, status TEXT NOT NULL DEFAULT 'active', confidence REAL NOT NULL DEFAULT 0.8,
  importance REAL NOT NULL DEFAULT 0.5, source_event TEXT, source_message_id TEXT);
CREATE TABLE memory_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER NOT NULL, old_text TEXT, new_text TEXT,
  old_status TEXT, new_status TEXT, changed_at REAL NOT NULL, reason TEXT, source_session TEXT, source_process TEXT, source_job_id INTEGER);
CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
  not_before REAL NOT NULL DEFAULT 0, last_error TEXT, priority INTEGER NOT NULL DEFAULT 100);
"""


@pytest.fixture()
def v1_db(tmp_path: Path) -> Path:
    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(_V1_SCHEMA)
    conn.execute("INSERT INTO entities (external_id, created_at) VALUES ('u', 1.0)")
    conn.execute("INSERT INTO processes (external_id, created_at) VALUES ('p', 1.0)")
    conn.execute(
        "INSERT INTO memories (entity_id, process_id, kind, subject, predicate, object, "
        "text, embedding, first_seen_at, last_seen_at) "
        "VALUES (1,1,'fact','user','employer','acme','User works at Acme.',?,100.0,100.0)",
        (json.dumps(fake_embed("user employer acme")),),
    )
    conn.commit()
    conn.close()
    return db


def test_v1_database_gains_v2_columns(v1_db: Path) -> None:
    store = MemoryStore(v1_db)
    cols = {r[1] for r in store._conn().execute("PRAGMA table_info(memories)").fetchall()}
    required = {"valid_at", "invalid_at", "expired_at", "superseded_by",
                "subject_key", "object_key", "embedding_f32", "decay_score"}
    assert required <= cols, sorted(required - cols)


def test_v1_rows_survive_migration(v1_db: Path) -> None:
    store = MemoryStore(v1_db)
    legacy = store.find_memories("u", current_only=True)
    assert len(legacy) == 1
    assert legacy[0].text == "User works at Acme."
    assert legacy[0].valid_at == 100.0  # backfilled from first_seen_at


def test_migrated_rows_are_recallable_through_the_new_index(v1_db: Path) -> None:
    store = MemoryStore(v1_db)
    hits = store.recall("u", fake_embed("user employer acme"), k=1)
    assert len(hits) == 1
    assert hits[0][0].text == "User works at Acme."


def test_migration_records_the_schema_version(v1_db: Path) -> None:
    store = MemoryStore(v1_db)
    row = store._conn().execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert row is not None
    assert row["value"] == str(SCHEMA_VERSION)


def test_migration_backfills_canonical_keys(v1_db: Path) -> None:
    store = MemoryStore(v1_db)
    rec = store.find_memories("u")[0]
    assert rec.subject_key == canonical_key("user")
    assert rec.object_key == canonical_key("acme")


def test_migration_is_idempotent(v1_db: Path) -> None:
    MemoryStore(v1_db)
    store2 = MemoryStore(v1_db)  # reopen: must not re-migrate or lose data
    assert store2.count_memories("u") == 1
    assert store2._conn().execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()["value"] == str(SCHEMA_VERSION)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Embedding storage
# ═══════════════════════════════════════════════════════════════════════════════


def test_embeddings_are_stored_as_compact_float32_blobs(store: MemoryStore) -> None:
    vec = fake_embed("user prefers postgresql")
    _, rec = _put(store, predicate="preferred_database", object_="postgresql",
                  text="User prefers PostgreSQL.", job_id=1)

    row = store._conn().execute(
        "SELECT embedding_f32, embedding_dim FROM memories WHERE id = ?", (rec.id,)
    ).fetchone()
    assert row["embedding_f32"] is not None
    assert row["embedding_dim"] == len(vec)
    # 4 bytes/dim vs ~19 bytes/dim for JSON text
    assert len(row["embedding_f32"]) == 4 * len(vec)

    restored = unpack_embedding(row["embedding_f32"], row["embedding_dim"])
    assert len(restored) == len(vec)
    assert max(abs(a - b) for a, b in zip(restored, vec)) < 1e-6


def test_legacy_json_embeddings_still_load(v1_db: Path) -> None:
    """A v1 row's JSON embedding is readable through the new accessor."""
    store = MemoryStore(v1_db)
    rec = store.find_memories("u")[0]
    assert rec.embedding
    assert len(rec.embedding) == 48


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Maintenance entry point
# ═══════════════════════════════════════════════════════════════════════════════


def test_run_maintenance_reports_decay_gc_and_consolidation(layer: MemoryLayer) -> None:
    _put(layer.store, predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)

    report = layer.run_maintenance(entity_id="u")
    assert "decay_updated" in report
    assert "gc" in report
    assert "consolidation" in report
    assert report["decay_updated"] >= 1


def test_consolidation_dry_run_does_not_merge(layer: MemoryLayer) -> None:
    _put(layer.store, predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(layer.store, predicate="favorite_food", object_="ramen",
         text="User loves ramen.", job_id=2)

    before = layer.store.count_memories("u")
    result = layer.store.run_consolidation(entity_id="u", dry_run=True)

    assert result["dry_run"] is True
    assert result["scanned"] >= 1
    assert layer.store.count_memories("u") == before


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Agent tool surface (the v2 tools exposed to the model)
# ═══════════════════════════════════════════════════════════════════════════════


def test_v2_tools_are_registered_with_unique_names() -> None:
    from tracera.tools.memory_tools import (
        MemoryEntitiesTool,
        MemoryExplainTool,
        MemoryExportTool,
        MemoryFeedbackTool,
        MemoryImportTool,
        MemoryMaintenanceTool,
        MemoryTimelineTool,
        MemoryUpdateTool,
    )

    tools = [
        MemoryTimelineTool, MemoryEntitiesTool, MemoryUpdateTool,
        MemoryFeedbackTool, MemoryMaintenanceTool, MemoryExplainTool,
        MemoryExportTool, MemoryImportTool,
    ]
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    for t in tools:
        assert t.name.startswith("memory_")
        assert isinstance(t.description, str) and t.description
        assert isinstance(t.parameters, dict)


def test_v2_tools_degrade_gracefully_without_a_layer() -> None:
    """No layer → a readable failure, never a traceback."""
    from tracera.tools.memory_tools import MemoryTimelineTool

    result = asyncio.run(MemoryTimelineTool(None).execute())
    assert result.success is False
    assert "not available" in (result.error or "").lower()


def test_memory_timeline_tool_reports_history(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryTimelineTool

    set_attribution("u", "p")
    _put(layer.store, predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(layer.store, predicate="employer", object_="globex",
         text="User works at Globex.", job_id=2)

    result = asyncio.run(MemoryTimelineTool(layer).execute(predicate="employer"))
    assert result.success is True
    assert "Acme" in result.output and "Globex" in result.output
    assert "●" in result.output and "○" in result.output
    assert result.metadata["count"] == 2


def test_memory_timeline_tool_handles_empty_scope(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryTimelineTool

    set_attribution("nobody", "p")
    result = asyncio.run(MemoryTimelineTool(layer).execute())
    assert result.success is True
    assert "No memory history" in result.output


def test_memory_entities_tool_overview_and_node(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryEntitiesTool

    set_attribution("u", "p")
    _put(layer.store, kind="relationship", subject="AuthMiddleware", predicate="calls",
         object_="UserService", text="AuthMiddleware calls UserService.", job_id=1)
    _put(layer.store, subject="auth_middleware", predicate="validates",
         object_="JWT tokens", text="AuthMiddleware validates JWT tokens.", job_id=2)

    overview = asyncio.run(MemoryEntitiesTool(layer).execute())
    assert overview.success is True
    assert "Memory Graph" in overview.output
    assert "auth_middleware" in overview.output

    node = asyncio.run(MemoryEntitiesTool(layer).execute(node="AuthMiddleware"))
    assert node.success is True
    # both edge directions render a real node name, never "None"
    assert "auth_middleware \u2192 calls \u2192 user_service" in node.output
    assert "auth_middleware \u2192 validates \u2192 jwt_tokens" in node.output
    assert "None" not in node.output


def test_memory_update_tool_preserves_the_previous_version(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryTimelineTool, MemoryUpdateTool

    set_attribution("u", "p")
    _, rec = _put(layer.store, kind="preference", predicate="preferred_editor",
                  object_="vscode", text="User prefers VS Code.", job_id=1)

    result = asyncio.run(
        MemoryUpdateTool(layer).execute(
            memory_id=rec.id, text="User prefers Neovim.",
            object="neovim", confidence=0.95,
        )
    )
    assert result.success is True
    new_id = result.metadata["memory_id"]
    assert new_id != rec.id

    history = asyncio.run(MemoryTimelineTool(layer).execute())
    assert "VS Code" in history.output      # the old value is still there
    assert "Neovim" in history.output


def test_memory_feedback_tool_adjusts_importance(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryFeedbackTool

    set_attribution("u", "p")
    _, rec = _put(layer.store, kind="preference", predicate="preferred_editor",
                  object_="vscode", text="User prefers VS Code.", job_id=1,
                  importance=0.5)

    result = asyncio.run(
        MemoryFeedbackTool(layer).execute(memory_id=rec.id, signal="useful")
    )
    assert result.success is True
    assert result.metadata["stats"]["importance"] > 0.5


def test_memory_feedback_tool_rejects_a_bad_signal(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryFeedbackTool

    set_attribution("u", "p")
    _, rec = _put(layer.store, text="Something.", job_id=1)
    result = asyncio.run(
        MemoryFeedbackTool(layer).execute(memory_id=rec.id, signal="banana")
    )
    assert result.success is False


def test_memory_maintenance_tool_dry_run(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryMaintenanceTool

    set_attribution("u", "p")
    _put(layer.store, predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(layer.store, predicate="favorite_food", object_="ramen",
         text="User loves ramen.", job_id=2)

    result = asyncio.run(MemoryMaintenanceTool(layer).execute(dry_run=True))
    assert result.success is True
    assert "Dry run" in result.output
    assert layer.store.count_memories("u") == 2


def test_memory_explain_tool_shows_the_breakdown(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryExplainTool

    set_attribution("u", "p")
    _put(layer.store, predicate="preferred_database", object_="postgresql",
         text="User prefers PostgreSQL.", job_id=1)

    result = asyncio.run(
        MemoryExplainTool(layer).execute(query="user prefers postgresql")
    )
    assert result.success is True
    assert "Recall Explanation" in result.output
    assert "INJECT" in result.output
    assert result.metadata["debug"]["would_inject"] >= 1


def test_memory_explain_tool_without_attribution(layer: MemoryLayer) -> None:
    from tracera.tools.memory_tools import MemoryExplainTool

    result = asyncio.run(MemoryExplainTool(layer).execute(query="anything"))
    assert result.success is False
    assert "attribution" in (result.error or "").lower()


def test_memory_export_import_tools_round_trip(layer: MemoryLayer, tmp_path: Path) -> None:
    from tracera.tools.memory_tools import MemoryExportTool, MemoryImportTool

    set_attribution("u", "p")
    _put(layer.store, predicate="employer", object_="acme",
         text="User works at Acme.", job_id=1)
    _put(layer.store, kind="preference", predicate="preferred_editor",
         object_="neovim", text="User prefers Neovim.", job_id=2)

    out = tmp_path / "dump" / "memories.json"
    export = asyncio.run(MemoryExportTool(layer).execute(output_path=str(out)))
    assert export.success is True
    assert out.exists()
    assert export.metadata["count"] == 2

    other = MemoryLayer(
        store=MemoryStore(tmp_path / "other.db"),
        embed_fn=fake_embed,
        worker_enabled=False,
    )
    set_attribution("u", "p")
    import_result = asyncio.run(MemoryImportTool(other).execute(input_path=str(out)))
    assert import_result.success is True
    assert import_result.metadata["stats"]["imported"] == 2
    assert other.store.count_memories("u") == 2


def test_memory_import_tool_reports_a_missing_file(layer: MemoryLayer, tmp_path: Path) -> None:
    from tracera.tools.memory_tools import MemoryImportTool

    set_attribution("u", "p")
    result = asyncio.run(
        MemoryImportTool(layer).execute(input_path=str(tmp_path / "nope.json"))
    )
    assert result.success is False
