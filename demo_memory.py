#!/usr/bin/env python3
"""
TRACERA Memory System - End-to-End Demo & Benchmark

This script demonstrates the complete memory system workflow:
1. Agent starts with attribution
2. User interacts with agent
3. Memories are automatically recalled and injected
4. Background extraction creates structured memories
5. Knowledge graph builds relationships
6. Consolidation merges duplicates
7. Supersession handles conflicts
8. Debug/explanation tools show internals

Run: python demo_memory.py
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

# Core memory components
from tracera.memory.layer.store import MemoryStore
from tracera.memory.layer.facade import MemoryLayer
from tracera.memory.layer.recall import RecallInjector
from tracera.memory.layer.extract import MemoryExtractor, is_memory_worthy
from tracera.memory.layer.queue import BackgroundWorker
from tracera.memory.layer.events import EventPipeline, EventType, MemoryEvent, build_default_pipeline
from tracera.memory.triples import Triple, TripleStore
from tracera.memory.layer.attribution import set_attribution, current_attribution
from tracera.providers.base import (
    LLMMessage, LLMProvider, LLMResponse, StreamEvent, TokenUsage, Role
)

# Demo uses fake provider/embedder for reproducibility
import math


def fake_embed(text: str, dim: int = 32) -> list[float]:
    """Deterministic embedding for demo."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = hash(token) % dim
        vec[h] += 1.0
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class FakeProvider(LLMProvider):
    """Scripted provider for demo."""
    def __init__(self):
        self._call_count = 0
        self.systems = []
        self._extraction_responses = [
            # First turn: user states preference
            json.dumps([{
                "kind": "preference",
                "subject": "user",
                "predicate": "favorite_color",
                "object": "blue",
                "text": "User's favorite color is blue.",
                "confidence": 0.95,
                "importance": 0.7,
            }]),
            # Second turn: user states fact
            json.dumps([{
                "kind": "fact",
                "subject": "user",
                "predicate": "uses_database",
                "object": "postgresql",
                "text": "User uses PostgreSQL as primary database.",
                "confidence": 0.9,
                "importance": 0.8,
            }]),
            # Third turn: user states rule
            json.dumps([{
                "kind": "rule",
                "subject": "user",
                "predicate": "requires",
                "object": "tests_before_commit",
                "text": "Always run tests before committing.",
                "confidence": 0.85,
                "importance": 0.9,
            }]),
        ]
        self._normal_responses = [
            "Noted! I'll remember that.",
            "Got it - PostgreSQL it is.",
            "Understood. Tests before commits.",
        ]

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.2,
        max_tokens=8192,
        tools=None,
        system=None,
    ):
        self.systems.append(system)
        self._call_count += 1

        # Check if this is an extraction call
        user_text = "".join(m.content or "" for m in messages if m.role == Role.USER)
        if "extract structured memory" in user_text.lower():
            resp = self._extraction_responses.pop(0) if self._extraction_responses else "[]"
            return LLMResponse(
                content=resp,
                tool_calls=None,
                usage=TokenUsage(total_tokens=100),
                model="fake",
                finish_reason="stop"
            )

        resp = self._normal_responses.pop(0) if self._normal_responses else "OK"
        return LLMResponse(
            content=resp,
            tool_calls=None,
            usage=TokenUsage(total_tokens=50),
            model="fake",
            finish_reason="stop"
        )

    async def stream(
        self,
        messages,
        *,
        model=None,
        temperature=0.2,
        max_tokens=8192,
        tools=None,
        system=None,
    ):
        self.systems.append(system)
        yield StreamEvent(type="text_delta", text="OK")
        yield StreamEvent(type="done")


async def run_demo():
    print("=" * 70)
    print("TRACERA Persistent Agent Memory - End-to-End Demo")
    print("=" * 70)

    # Use a fixed temp directory instead of TemporaryDirectory to avoid Windows file locking issues
    tmpdir = Path(".demo_tmp") / uuid.uuid4().hex[:8]
    tmpdir.mkdir(parents=True, exist_ok=True)
    db_path = tmpdir / "demo_memory.db"

    # Initialize
    print("\n[Initializing memory system...]")
    store = MemoryStore(db_path)
    embedder = lambda t: fake_embed(t)

    layer = MemoryLayer(
        store=store,
        embed_fn=embedder,
        top_k=5,
        dedup_threshold=0.9,
        min_recall_score=0.3,
        extraction_model=None,
        worker_enabled=True,
        min_extraction_confidence=0.5,
        min_extraction_importance=0.3,
        enable_worthiness_filter=True,
        enable_safety=True,
        recall_use_hybrid=True,
        recall_token_budget=2000,
        recall_grouped=True,
    )

    provider = FakeProvider()
    wrapped = layer.register(provider)
    layer.start()
    print(f"   [OK] Memory layer started (worker running: {layer._worker.running})")

    # Set Attribution
    print("\n[Setting attribution: entity=user_123, process=coding_agent]")
    layer.attribution("user_123", "coding_agent")

    # Turn 1: User states preference
    print("\n" + "-" * 50)
    print("Turn 1: User states preference")
    print("-" * 50)

    user_msg = "My favorite color is blue and I want you to remember that."
    print(f"   User: {user_msg}")

    resp = await wrapped.complete(
        [LLMMessage.user(user_msg)],
        system="You are a helpful assistant."
    )

    print(f"   Agent: {resp.content}")
    print(f"   [Memory extraction queued (background)]")

    # Turn 2: User states fact
    print("\nTurn 2: User states database preference")
    print("-" * 50)

    user_msg = "I use PostgreSQL as my primary database."
    print(f"   User: {user_msg}")

    resp = await wrapped.complete(
        [LLMMessage.user(user_msg)],
        system="You are a helpful assistant."
    )

    print(f"   Agent: {resp.content}")

    # Turn 3: User states rule
    print("\nTurn 3: User states a rule")
    print("-" * 50)

    user_msg = "Always run tests before committing code."
    print(f"   User: {user_msg}")

    resp = await wrapped.complete(
        [LLMMessage.user(user_msg)],
        system="You are a helpful assistant."
    )

    print(f"   Agent: {resp.content}")

    # Wait for background extraction
    print("\n[Waiting for background extraction to complete...]")
    for _ in range(20):
        if store.count_jobs(status="pending") == 0:
            break
        await asyncio.sleep(0.1)

    # Recall Demonstration
    print("\nRecall Demonstration: 'What is my favorite color?'")
    print("-" * 50)

    resp = await wrapped.complete(
        [LLMMessage.user("What is my favorite color?")],
        system="You are a helpful assistant."
    )

    print(f"   Agent: {resp.content}")
    print(f"   [System prompt included memory injection:]")
    sys_prompt = provider.systems[-1]
    if sys_prompt and "Known context about this user:" in sys_prompt:
        for line in sys_prompt.split("\n"):
            if line.strip().startswith("- ["):
                print(f"      {line.strip()}")

    # Knowledge Graph Construction
    print("\nKnowledge Graph Construction")
    print("-" * 50)

    triple_store = TripleStore()
    triple_store.add_triple(Triple(
        subject="user", predicate="prefers", object="blue",
        confidence=0.9, source="conversation"
    ))
    triple_store.add_triple(Triple(
        subject="user", predicate="uses_database", object="postgresql",
        confidence=0.85, source="conversation"
    ))
    triple_store.add_triple(Triple(
        subject="postgresql", predicate="is_a", object="database",
        confidence=0.99, source="code_analysis"
    ))

    print(f"   Triples: {triple_store.triple_count}")
    print(f"   Nodes: {triple_store.node_count}")

    expanded = triple_store.expand_query_with_graph("postgresql", max_hops=2)
    print(f"   Expanded from 'postgresql': {len(expanded)} related triples")
    for t in expanded[:5]:
        print(f"      {t.subject} -> {t.predicate} -> {t.object} (conf: {t.confidence:.2f})")

    # Memory Consolidation
    print("\nMemory Consolidation")
    print("-" * 50)

    store.upsert_memory(
        entity_id="user_123", process_id="coding_agent", kind="fact",
        subject="user", predicate="favorite_color", object="blue",
        text="User's favorite color is blue (repeated).",
        embedding=fake_embed("user favorite color blue"),
        confidence=0.8, importance=0.5, job_id=100,
    )

    stats = store.run_consolidation(entity_id="user_123", similarity_threshold=0.9)
    print(f"   Scanned: {stats['scanned']}, Merged: {stats['merged']}, Superseded: {stats['superseded']}")

    # Supersession & Version History
    print("\nSupersession & Version History")
    print("-" * 50)

    _, rec = store.upsert_memory(
        entity_id="user_123", process_id="coding_agent", kind="preference",
        subject="user", predicate="favorite_color", object="blue",
        text="User's favorite color is blue.",
        embedding=fake_embed("user favorite color blue"),
        confidence=0.9, importance=0.7, job_id=1,
    )

    store.supersede_memory(
        rec.id,
        new_text="User's favorite color changed to green.",
        new_embedding=fake_embed("user favorite color green"),
        reason="User updated preference",
        source_session="sess-1",
        source_process="coding_agent",
    )

    versions = store.get_memory_versions(rec.id)
    print(f"   Version history entries: {len(versions)}")
    for v in versions:
        print(f"      {time.ctime(v['changed_at'])}: {v['reason']}")
        print(f"         Old: {v['old_text'][:50]}...")
        print(f"         New: {v['new_text'][:50]}...")

    explanation = store.explain_memory(rec.id)
    print(f"   Summary: {explanation['summary']}")

    # Debug & Explanation
    print("\nDebug Recall & Explanation")
    print("-" * 50)

    debug = layer._recaller.debug_recall(
        "what database do I use",
        current_attribution()
    )
    print(f"   Query: {debug['query']}")
    print(f"   Candidates: {debug['total_candidates']}, Returned: {debug['returned']}")
    print(f"   Timing: {debug['timing_ms']:.1f}ms")
    for r in debug['results'][:3]:
        print(f"      [{r['kind']}] {r['text'][:50]}... (score: {r['final_score']:.3f})")

    explain = store.explain_recall("user_123", "what database do I use", fake_embed("what database do I use"))
    print(f"\n   Recall explanation:")
    for exp in explain['explanations'][:2]:
        print(f"      Memory {exp['memory_id']}: {exp['why_recalled']}")

    # Statistics
    print("\nSystem Statistics")
    print("-" * 50)

    stats = store.stats()
    print(f"   Total memories: {stats['memories_total']}")
    print(f"   By kind: {stats['memories_by_kind']}")
    print(f"   By status: {stats['memories_by_status']}")
    print(f"   Entities: {stats['entities']}, Sessions: {stats['sessions']}")
    print(f"   Avg mentions: {stats['avg_mentions']}, Avg confidence: {stats['avg_confidence']}")
    print(f"   Versions: {stats['total_versions']}, Superseded: {stats['superseded_count']}")

    worker_stats = layer._worker.get_stats()
    print(f"   Worker: {worker_stats['jobs_processed']} processed, {worker_stats['jobs_failed']} failed")
    print(f"   Avg latency: {worker_stats['avg_latency_ms']:.1f}ms")

    # Cleanup
    print("\nCleanup")
    print("-" * 50)
    layer.stop()
    store.close_all()
    print("   [OK] Worker stopped and connections closed")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    # Clean up temp directory
    shutil.rmtree(tmpdir, ignore_errors=True)


async def run_benchmark():
    """Run performance benchmarks."""
    print("\n" + "=" * 70)
    print("TRACERA Memory System - Benchmark")
    print("=" * 70)

    # Use a fixed temp directory instead of TemporaryDirectory to avoid Windows file locking issues
    tmpdir = Path(".demo_tmp") / ("bench_" + uuid.uuid4().hex[:8])
    tmpdir.mkdir(parents=True, exist_ok=True)
    db_path = tmpdir / "bench_memory.db"
    store = MemoryStore(db_path)

    # Pre-populate with memories
    print("\nPopulating with 1000 memories...")
    start = time.perf_counter()
    for i in range(1000):
        store.upsert_memory(
            entity_id="user_bench", process_id="bench_agent", kind="fact",
            subject="user", predicate=f"fact_{i}", object=f"value_{i}",
            text=f"Benchmark memory {i} with some content.",
            embedding=fake_embed(f"benchmark memory {i}"),
            confidence=0.8, importance=0.5, job_id=i,
        )
    elapsed = time.perf_counter() - start
    print(f"   Inserted 1000 memories in {elapsed*1000:.1f}ms ({1000/elapsed:.0f} ops/sec)")

    # Benchmark recall
    print("\nBenchmarking recall (hybrid)...")
    query = "what is fact 500 about"
    query_emb = fake_embed(query)

    # Warm up
    store.recall_hybrid("user_bench", query, query_emb, k=10)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        store.recall_hybrid("user_bench", query, query_emb, k=10)
    elapsed = time.perf_counter() - start
    print(f"   {iterations} hybrid recalls in {elapsed*1000:.1f}ms ({iterations/elapsed:.0f} ops/sec)")

    # Benchmark vector-only recall
    print("\nBenchmarking vector-only recall...")
    start = time.perf_counter()
    for _ in range(iterations):
        store.recall("user_bench", query_emb, k=10)
    elapsed = time.perf_counter() - start
    print(f"   {iterations} vector recalls in {elapsed*1000:.1f}ms ({iterations/elapsed:.0f} ops/sec)")

    # Benchmark consolidation
    print("\nBenchmarking consolidation...")
    start = time.perf_counter()
    stats = store.run_consolidation(entity_id="user_bench", similarity_threshold=0.92)
    elapsed = time.perf_counter() - start
    print(f"   Consolidation in {elapsed*1000:.1f}ms (scanned: {stats['scanned']}, merged: {stats['merged']})")

    # Benchmark worker throughput
    print("\nBenchmarking worker throughput...")
    from tracera.memory.layer.queue import BackgroundWorker

    def dummy_handler(job):
        pass  # No-op

    worker = BackgroundWorker(store, dummy_handler, batch_size=50, max_attempts=2)
    worker.start()

    # Enqueue 500 jobs
    for i in range(500):
        store.enqueue_job("bench_job", {"id": i}, priority=100)

    start = time.perf_counter()
    while store.count_jobs(status="pending") > 0:
        await asyncio.sleep(0.01)
    elapsed = time.perf_counter() - start

    worker.stop()
    store.close_all()
    processed = worker.stats.jobs_processed
    print(f"   Processed {processed} jobs in {elapsed:.2f}s ({processed/elapsed:.0f} jobs/sec)")
    print(f"   Avg latency: {worker.stats.avg_latency_ms():.1f}ms")

    # Clean up temp directory
    shutil.rmtree(tmpdir, ignore_errors=True)


async def main():
    await run_demo()
    await run_benchmark()

    print("\n" + "=" * 70)
    print("All demos and benchmarks completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())