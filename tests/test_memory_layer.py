"""
Tests for the TRACERA Memory Layer (Memori-style agent-native memory).

Covers the guardrails that matter most:

  * attribution enforcement — no attribution → no memory read/write
  * recall injection — top-k memories land in the outbound system prompt
  * extraction — structured JSON parsing + classification of a real turn
  * dedup — restating the same fact three times → one row, mention_count=3
  * entity isolation — one entity never sees another entity's memories
  * hot-path latency — the wrapped call returns to the caller before
    extraction completes (extraction genuinely runs after the response)
  * idempotent writes under retry
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tracera.memory.layer import (
    MemoryExtractor,
    MemoryLayer,
    MemoryStore,
)
from tracera.memory.layer.attribution import (
    AttributionError,
    current_attribution,
    reset_attribution,
    set_attribution,
    set_session_id,
)
from tracera.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    StreamEvent,
    TokenUsage,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test doubles
# ═══════════════════════════════════════════════════════════════════════════════

def _token_vec(token: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-random unit-ish vector per token (cached)."""
    rng = random.Random(f"tok:{token}")
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / n for x in raw]


def fake_embed(text: str, dim: int = 32) -> list[float]:
    """Deterministic token-bag embedding (offline):

    * identical text ⇒ identical vector
    * similar wording (shared tokens) ⇒ high cosine similarity
    * unrelated content ⇒ near-zero similarity
    """
    import re

    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tv = _token_vec(token, dim)
        for i in range(dim):
            vec[i] += tv[i]
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


EXTRACTION_JSON = json.dumps([
    {
        "kind": "preference",
        "subject": "user",
        "predicate": "favorite_color",
        "object": "blue",
        "text": "User's favorite color is blue.",
        "confidence": 0.95,
        "importance": 0.7,
    },
    {
        "kind": "fact",
        "subject": "user",
        "predicate": "preferred_language",
        "object": "python",
        "text": "User works primarily in Python.",
        "confidence": 0.9,
        "importance": 0.6,
    },
])


class FakeProvider(LLMProvider):
    """A scripted provider: extraction prompts get `extraction` content,
    anything else gets `normal`. Records the system prompts it receives."""

    def __init__(
        self,
        *,
        normal: str = "hello there",
        extraction: str | None = EXTRACTION_JSON,
        delay: float = 0.0,
    ) -> None:
        self._normal = normal
        self._extraction = extraction
        self._delay = delay
        self.systems: list[str | None] = []
        self.user_texts: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.systems.append(system)
        texts = [m.content or "" for m in messages if m.role.value == "user"]
        self.user_texts.extend(texts)
        joined = "\n".join(texts)
        if self._extraction is not None and "You extract structured memory" in joined:
            content = self._extraction
        else:
            content = self._normal
        return LLMResponse(
            content=content,
            tool_calls=None,
            usage=TokenUsage(total_tokens=len(content)),
            model=model or self.default_model,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> Any:
        self.systems.append(system)
        yield StreamEvent(type="text_delta", text="hel")
        yield StreamEvent(type="text_delta", text="lo")
        yield StreamEvent(type="done")


def make_layer(db_path: Path, **kwargs: Any) -> MemoryLayer:
    """Build a fully-wired layer around a fake provider + fake embedder."""
    store = MemoryStore(db_path / "mem.db")
    layer = MemoryLayer(
        store=store,
        embed_fn=fake_embed,
        top_k=kwargs.pop("top_k", 5),
        dedup_threshold=kwargs.pop("dedup_threshold", 0.9),
        enabled_processes=kwargs.pop("enabled_processes", None),
        worker_enabled=False,
        recall_grouped=False,
        **kwargs,
    )
    provider = FakeProvider()
    layer.register(provider)
    return layer


@pytest.fixture(autouse=True)
def _clean_attribution() -> Any:
    """Every test starts with no attribution scope set."""
    reset_attribution()
    set_session_id(None)
    yield
    reset_attribution()
    set_session_id(None)
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Attribution enforcement
# ═══════════════════════════════════════════════════════════════════════════════

async def test_no_attribution_creates_no_memory(tmp_path):
    """An LLM call without attribution must not create memory."""
    layer = make_layer(tmp_path)
    provider = layer.wrapped_provider

    await provider.complete([LLMMessage.user("hello")])

    assert layer.store.count_memories() == 0
    assert layer.store.count_jobs() == 0
    # but the LLM call still went through untouched
    assert len(layer.wrapped_provider.inner.systems) == 1


async def test_no_attribution_logs_warning(tmp_path, caplog):
    layer = make_layer(tmp_path)
    provider = layer.wrapped_provider
    with caplog.at_level(logging.WARNING, logger="tracera.memory.layer"):
        await provider.complete([LLMMessage.user("hello")])
    assert any("no attribution" in r.message.lower() for r in caplog.records)


def test_attribution_requires_both_ids():
    with pytest.raises(AttributionError):
        set_attribution("user_1", "")  # empty process
    with pytest.raises(AttributionError):
        set_attribution("", "agent")
    scope = set_attribution("user_1", "agent")
    assert current_attribution() == scope
    assert scope.entity_id == "user_1"
    assert scope.process_id == "agent"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Recall injects top-k memories into the outbound prompt
# ═══════════════════════════════════════════════════════════════════════════════

async def test_recall_injects_top_k_into_system_prompt(tmp_path):
    layer = make_layer(tmp_path, top_k=5)
    store = layer.store
    # Three relevant color memories + two clearly unrelated ones.
    seeds = [
        ("User's favorite color is blue.", "favorite color blue", "fact"),
        ("User's favorite color is green.", "favorite color green", "fact"),
        ("User's favorite color is red.", "favorite color red", "fact"),
        ("User writes Python daily.", "python programming", "skill"),
        ("User's team ships on Fridays.", "friday release", "fact"),
    ]
    for i, (text, emb_text, kind) in enumerate(seeds):
        store.upsert_memory(
            entity_id="user_1", process_id="agent", kind=kind,
            subject="user", predicate=f"k{i}", object=emb_text.split()[-1],
            text=text, embedding=fake_embed(emb_text), job_id=100 + i,
        )

    layer.attribution("user_1", "agent")
    provider = layer.wrapped_provider
    await provider.complete(
        [LLMMessage.user("what is my favorite color?")],
        system="You are a helpful assistant.",
    )

    system = provider.inner.systems[-1]
    assert system is not None
    assert "Known context about this user:" in system
    # the three color memories are relevant → injected
    for text, _, _ in seeds[:3]:
        assert text in system
    # the unrelated memories are filtered out by the relevance floor
    assert "Python daily" not in system
    assert "Fridays" not in system
    # all three color memories should be present (order may vary by embedding similarity)
    color_lines = [
        line for line in system.splitlines()
        if line.startswith("- [")
    ]
    color_texts = [line for line in color_lines if "favorite color" in line]
    assert len(color_texts) == 3
    for text, _, _ in seeds[:3]:
        assert text in system


async def test_recall_no_match_leaves_prompt_unchanged(tmp_path):
    layer = make_layer(tmp_path)
    # An entity with no memories gets an untouched prompt.
    layer.attribution("user_1", "agent")
    provider = layer.wrapped_provider
    system = "You are a helpful assistant."
    await provider.complete([LLMMessage.user("how do I fix a type error?")], system=system)
    assert provider.inner.systems[-1] == system  # untouched
# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extraction classifies a sample turn
# ═══════════════════════════════════════════════════════════════════════════════

async def _fake_call_llm(prompt: str) -> str:
    return EXTRACTION_JSON


async def test_extraction_classifies_turn():
    extractor = MemoryExtractor(_fake_call_llm)
    items = await extractor.extract_turn(
        "My favorite color is blue and I mainly code in Python",
        "Noted — blue and Python!",
    )
    assert len(items) == 2
    by_predicate = {i.predicate: i for i in items}
    assert by_predicate["favorite_color"].kind == "preference"
    assert by_predicate["favorite_color"].object == "blue"
    assert by_predicate["preferred_language"].kind == "fact"
    assert by_predicate["preferred_language"].object == "python"


async def test_extraction_handles_empty_and_json_fences():
    async def empty(prompt: str) -> str:
        return "[]"

    async def fenced(prompt: str) -> str:
        return "```json\n" + EXTRACTION_JSON + "\n```"

    assert await MemoryExtractor(empty).extract_turn("hi", "yo") == []
    items = await MemoryExtractor(fenced).extract_turn("a", "b")
    assert len(items) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dedup: same fact restated N times → one row with mention_count = N
# ═══════════════════════════════════════════════════════════════════════════════

async def test_dedup_same_fact_three_times_one_row_mention_three(tmp_path):
    layer = make_layer(tmp_path)
    store = layer.store
    payload = {
        "user_message": "My favorite color is blue.",
        "assistant_message": "Great, remembering that.",
        "entity_id": "user_1",
        "process_id": "agent",
        "session_id": "s-1",
    }
    for _ in range(3):  # three distinct turns / jobs
        job_id = store.enqueue_job("extract_turn", payload)
        job = store.claim_jobs(limit=1)[0]
        assert job.id == job_id
        await layer._execute_job(job)

    records = store.find_memories("user_1")
    # Two distinct triples were extracted — each deduped into ONE row.
    assert len(records) == 2
    by_predicate = {r.predicate: r for r in records}
    assert by_predicate["favorite_color"].mention_count == 3
    assert by_predicate["preferred_language"].mention_count == 3


async def test_write_idempotent_under_retry(tmp_path):
    """Re-running the SAME background job must not double-count."""
    layer = make_layer(tmp_path)
    store = layer.store
    payload = {
        "user_message": "My favorite color is blue.",
        "assistant_message": "OK!",
        "entity_id": "user_1",
        "process_id": "agent",
        "session_id": "s-1",
    }
    job_id = store.enqueue_job("extract_turn", payload)
    job = store.claim_jobs(limit=1)[0]
    await layer._execute_job(job)   # first run → insert
    await layer._execute_job(job)   # retry of the same job → no-op

    records = store.find_memories("user_1")
    assert len(records) == 2
    # both triples written at mention_count=1 — retry did not inflate them
    assert all(r.mention_count == 1 for r in records)
# ═══════════════════════════════════════════════════════════════════════════════
# 5. Entity isolation
# ═══════════════════════════════════════════════════════════════════════════════

async def test_recall_never_crosses_entities(tmp_path):
    layer = make_layer(tmp_path)
    store = layer.store
    store.upsert_memory(
        entity_id="user_1", process_id="agent", kind="fact",
        subject="user", predicate="favorite_color", object="blue",
        text="User's favorite color is blue.",
        embedding=fake_embed("favorite color blue"), job_id=1,
    )
    store.upsert_memory(
        entity_id="user_2", process_id="agent", kind="fact",
        subject="user", predicate="favorite_color", object="green",
        text="User's favorite color is green.",
        embedding=fake_embed("favorite color green"), job_id=2,
    )

    # Direct store-level isolation
    hits = store.recall("user_2", fake_embed("what color does the user like"), k=5)
    assert hits and hits[0][0].entity_id == "user_2"
    assert "green" in hits[0][0].object
    assert all(r.entity_id == "user_2" for r, _ in hits)

    # Wrapper-level isolation: user_2's call must never receive user_1's memories
    layer.attribution("user_2", "agent")
    provider = layer.wrapped_provider
    await provider.complete([LLMMessage.user("favorite color")], system="sys")
    system = provider.inner.systems[-1] or ""
    assert "user_1" not in system
    assert "blue" not in system            # user_1's value is absent
    assert "Known context about this user:" in system
    assert "green" in system


async def test_write_scoped_to_entity(tmp_path):
    """Extraction writes for entity A never land under entity B."""
    layer = make_layer(tmp_path)
    store = layer.store
    payload = {
        "user_message": "My favorite color is blue.",
        "assistant_message": "OK",
        "entity_id": "user_1",
        "process_id": "agent",
        "session_id": "s-1",
    }
    job_id = store.enqueue_job("extract_turn", payload)
    job = store.claim_jobs(limit=1)[0]
    await layer._execute_job(job)

    assert store.count_memories("user_1") == 2  # two extracted triples
    assert store.count_memories("user_2") == 0
# ═══════════════════════════════════════════════════════════════════════════════
# 6. Hot-path latency / async extraction
# ═══════════════════════════════════════════════════════════════════════════════

async def test_wrapped_call_returns_before_extraction(tmp_path):
    """Response path must not wait for extraction: <50ms overhead, and memory
    appears only after the queued job runs."""
    layer = make_layer(tmp_path)
    store = layer.store
    # A non-trivial corpus to make the recall scan realistic.
    # Use distinct enough embeddings to avoid semantic dedup.
    for i in range(100):
        store.upsert_memory(
            entity_id="user_1", process_id="agent", kind="fact",
            subject="user", predicate=f"seed{i}", object=str(i),
            text=f"Seed memory number {i}.",
            embedding=fake_embed(f"distinct seed memory number {i} unique"), job_id=i,
        )

    layer.attribution("user_1", "agent")
    provider = layer.wrapped_provider

    start = time.perf_counter()
    await provider.complete([LLMMessage.user("what is seed 7?")])
    elapsed = time.perf_counter() - start

    assert elapsed < 0.050, f"hot path took {elapsed * 1000:.2f} ms"
    # extraction was only QUEUED — not run — while the caller already returned
    assert store.count_jobs(status="pending") == 1
    initial_count = store.count_memories("user_1")

    # Now run the queued job and verify the write lands afterwards.
    job = store.claim_jobs(limit=1)[0]
    await layer._execute_job(job)
    assert store.count_memories("user_1") == initial_count + 2  # two extracted memories


async def test_background_worker_persists_extraction(tmp_path):
    """The durable worker thread consumes the job and writes memory."""
    layer = make_layer(tmp_path)
    layer.start()
    try:
        layer.attribution("user_1", "agent")
        provider = layer.wrapped_provider
        await provider.complete([LLMMessage.user("my favorite color is blue")])
        for _ in range(100):  # poll until the worker writes
            if layer.store.count_memories("user_1") >= 2:
                break
            await asyncio.sleep(0.05)
        records = layer.store.find_memories("user_1")
        assert len(records) == 2
        assert {r.predicate for r in records} == {"favorite_color", "preferred_language"}
    finally:
        layer.stop()
# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-process policy & sessions
# ═══════════════════════════════════════════════════════════════════════════════

async def test_process_allowlist(tmp_path):
    layer = make_layer(tmp_path, enabled_processes=["support_agent"])
    store = layer.store
    provider = layer.wrapped_provider

    layer.attribution("user_1", "other_process")
    await provider.complete([LLMMessage.user("hello")])
    assert store.count_jobs() == 0  # disabled process → passthrough, no memory

    layer.attribution("user_1", "support_agent")
    await provider.complete([LLMMessage.user("hello")])
    assert store.count_jobs() == 1  # allowed process → memory active


async def test_session_lifecycle(tmp_path):
    layer = make_layer(tmp_path)
    store = layer.store
    layer.attribution("user_1", "agent")
    session_id = layer.new_session()
    assert session_id in store.get_session_ids("user_1")

    provider = layer.wrapped_provider
    await provider.complete([LLMMessage.user("my favorite color is blue")])
    payload = store.claim_jobs(limit=10)[-1].payload
    assert payload["session_id"] == session_id  # turn tagged with session

    store.close_session(session_id)
    layer.set_session(session_id)  # resumable


async def test_streaming_path_enqueues_turn(tmp_path):
    layer = make_layer(tmp_path)
    store = layer.store
    layer.attribution("user_1", "agent")
    provider = layer.wrapped_provider

    events = []
    async for event in provider.stream([LLMMessage.user("tell me a secret")]):
        events.append(event.type)
    assert events == ["text_delta", "text_delta", "done"]
    assert store.count_jobs() == 1  # turn captured after stream completes