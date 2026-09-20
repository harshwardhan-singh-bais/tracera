"""
Tests for Phases 45-50 — evaluation dataset, metrics, strategies,
retrieval benchmark, agent benchmark, ablation framework.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracera.evaluation.ablation import (
    AblationConfig,
    AblationFramework,
    default_ablation_configs,
)
from tracera.evaluation.agent_benchmark import AgentBenchmark
from tracera.evaluation.dataset import EvalQuery, EvaluationDataset, example_dataset
from tracera.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tracera.evaluation.retrieval_benchmark import RetrievalBenchmark
from tracera.evaluation.strategies import (
    BM25Strategy,
    RetrievalHit,
    build_strategies,
)

# ════════════════════════════════════════════════════════════════════════════
# Phase 45 — dataset
# ════════════════════════════════════════════════════════════════════════════


def test_dataset_add_and_persist(tmp_path):
    ds = EvaluationDataset("test")
    ds.add_query("Where is JWT auth?", files=["auth/middleware.py"], symbols=["validate_token"])
    assert len(ds) == 1

    path = ds.save(tmp_path / "ds.json")
    loaded = EvaluationDataset.load(path)
    assert loaded.name == "test"
    assert loaded[0].query == "Where is JWT auth?"
    assert "auth/middleware.py" in loaded[0].ground_truth


def test_example_dataset_has_queries():
    ds = example_dataset()
    assert len(ds) >= 3
    assert all(q.query for q in ds)


def test_ground_truth_case_insensitive():
    q = EvalQuery(query="q", files=["Auth/Middleware.py"])
    assert "auth/middleware.py" in q.ground_truth


# ════════════════════════════════════════════════════════════════════════════
# Phase 46 — metrics
# ════════════════════════════════════════════════════════════════════════════


def _hits(*paths):
    return [RetrievalHit(doc_id=f"d{i}", score=1.0, file_path=p) for i, p in enumerate(paths)]


def test_recall_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["b.py", "c.py", "z.py"]
    assert recall_at_k(hits, gt, k=1) == 0.0
    assert recall_at_k(hits, gt, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(hits, gt, k=3) == pytest.approx(2 / 3)


def test_precision_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["a.py"]
    assert precision_at_k(hits, gt, k=2) == pytest.approx(0.5)


def test_reciprocal_rank():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["b.py"]
    assert reciprocal_rank(hits, gt) == pytest.approx(0.5)
    assert reciprocal_rank(hits, ["zzz.py"]) == 0.0


def test_ndcg_at_k():
    hits = _hits("a.py", "b.py", "c.py")
    gt = ["a.py", "c.py"]
    # Relevant at ranks 1 and 3: DCG = 1 + 1/log2(3) ≈ 1.631
    # Ideal: 1 + 1/log2(2) = 2
    assert ndcg_at_k(hits, gt, k=3) == pytest.approx(1.631 / 2.0, abs=0.01)


def test_mrr_mean():
    rankings = [
        (_hits("a.py", "b.py"), ["b.py"]),  # RR 0.5
        (_hits("a.py"), ["a.py"]),  # RR 1.0
        (_hits("a.py"), ["zzz.py"]),  # RR 0.0
    ]
    assert mean_reciprocal_rank(rankings) == pytest.approx(0.5)


# ════════════════════════════════════════════════════════════════════════════
# Phases 47-48 — strategies + benchmark
# ════════════════════════════════════════════════════════════════════════════


def _make_bm25():
    from tracera.retrieval.bm25 import BM25Index

    bm25 = BM25Index()
    bm25.add_document("d1", "def validate_token(): jwt middleware", {"file_path": "auth/token.py"})
    bm25.add_document(
        "d2", "class AuthMiddleware: handles authentication", {"file_path": "auth/middleware.py"}
    )
    bm25.add_document("d3", "def retry_db_call(): database retries", {"file_path": "db/retry.py"})
    return bm25


def test_bm25_strategy_returns_hits():
    bm25 = _make_bm25()
    strategy = BM25Strategy(bm25)
    hits = strategy.retrieve("authentication middleware", k=5)
    assert hits
    assert hits[0].doc_id in {"d1", "d2"}
    assert strategy.last_latency_ms >= 0.0


def test_grep_strategy_baseline(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def authenticate(): pass\nJWT token validation with authentication flow here"
    )
    (tmp_path / "other.txt").write_text("noise")
    strategy = build_strategies(workspace=tmp_path)["grep"]
    hits = strategy.retrieve("JWT authentication", k=5)
    assert hits
    assert any("auth.py" in (h.file_path or "") for h in hits)


def test_retrieval_benchmark_full_run(tmp_path):
    bm25 = _make_bm25()
    dataset = EvaluationDataset(
        "t",
        [
            EvalQuery(query="authentication middleware", docs=["d1", "d2"]),
            EvalQuery(query="database retry", docs=["d3"]),
        ],
    )
    strategies = build_strategies(workspace=tmp_path, bm25=bm25)
    report = RetrievalBenchmark(dataset, strategies).run()

    assert "grep" in report.strategies
    assert "bm25" in report.strategies
    bm25_scores = report.strategies["bm25"]
    assert 0.0 <= bm25_scores.recall_5 <= 1.0
    assert bm25_scores.mrr >= 0.0
    md = report.to_markdown()
    assert "R@5" in md or "recall" in md.lower()


# ════════════════════════════════════════════════════════════════════════════
# Phase 49 — agent benchmark
# ════════════════════════════════════════════════════════════════════════════


async def _fake_runner(task: str) -> dict:
    return {
        "success": True,
        "output": "done",
        "iterations": 3,
        "tool_calls": 5,
        "tool_names": {"search_code", "read_file", "edit_file"},
        "tokens_in": 1000,
        "tokens_out": 500,
    }


async def test_agent_benchmark_measures_all_metrics():
    bench = AgentBenchmark(_fake_runner, tasks=["task one", "task two"])
    report = await bench.run()

    assert report.n == 2
    assert report.success_rate == 1.0
    assert report.mean_iterations == 3.0
    assert report.mean_tool_calls == 5.0
    assert report.mean_retrieval_calls == 1.0  # only search_code counts
    assert report.mean_tokens == 1500.0
    assert report.mean_latency_ms >= 0.0
    assert report.mean_cost_usd > 0.0
    assert "Success rate" in report.to_markdown()


async def test_agent_benchmark_failure_and_test_verification():
    async def flaky_runner(task: str) -> dict:
        if "bad" in task:
            return {"success": False, "error": "boom"}
        return {"success": True, "output": "ok"}

    bench = AgentBenchmark(
        flaky_runner,
        tasks=["good task", "bad task"],
        verify_tests=lambda: True,
    )
    report = await bench.run()
    assert report.success_rate == 0.5
    assert report.tests_passed_rate == 1.0


# ════════════════════════════════════════════════════════════════════════════
# Phase 50 — ablation framework
# ════════════════════════════════════════════════════════════════════════════


def test_default_ablation_configs_cover_roadmap():
    configs = default_ablation_configs()
    labels = [c.label for c in configs]
    assert "Agent" in labels
    assert "Agent + BM25" in labels
    assert "Agent + Dense" in labels
    assert "Agent + Hybrid" in labels
    assert "Agent + Hybrid + Reranker" in labels
    assert "Agent + Hybrid + Graph" in labels


async def test_ablation_framework_runs_arms():
    async def build_agent(config: AblationConfig):
        async def runner(task: str) -> dict:
            return {"success": True, "output": "ok", "tool_names": set()}

        return runner

    framework = AblationFramework(["task"], build_agent, configs=default_ablation_configs())
    report = await framework.run()
    assert len(report.arms) == 6
    assert report.best_arm() is not None
    md = report.to_markdown()
    assert "Configuration" in md


# ════════════════════════════════════════════════════════════════════════════
# Regression: the benchmark must be able to measure something
# ════════════════════════════════════════════════════════════════════════════


def test_recall_cannot_exceed_one_when_chunks_repeat_a_file() -> None:
    """
    Recall counts *ground-truth items* hit, not hits.

    Retrieval normally returns several chunks from the same file. Counting hits
    let one ground-truth file contribute more than once, and the benchmark
    reported recall@10 of 1.45 — a value recall cannot take.
    """
    hits = [
        RetrievalHit(doc_id=f"c{i}", score=1.0, file_path="a/b.py")
        for i in range(5)
    ]
    # Two ground-truth items; only one of them (the file) is present.
    assert recall_at_k(hits, ["a/b.py", "SomeSymbol"], k=10) == 0.5

    # Even with a single item and many hits, recall is capped at 1.0.
    assert recall_at_k(hits, ["a/b.py"], k=10) == 1.0


def test_grep_baseline_reports_relative_paths_and_content(tmp_path) -> None:
    """
    The grep baseline must be matchable, and must report the context it costs.

    It returned absolute Windows paths while ground truth is workspace-relative,
    so it scored 0.000 on every query despite costing ~33s each — which reads as
    "grep is terrible" rather than "the baseline can never match". It also
    reported empty content, so its context size was structurally 0 bytes and the
    token comparison could not be made at all.
    """
    from tracera.evaluation.strategies import GrepStrategy

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "scanner.py").write_text(
        "def scan():\n    pass\n# nested gitignore handling\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text("unrelated = True\n", encoding="utf-8")

    hits = GrepStrategy(tmp_path)._retrieve("nested gitignore handling", 5)

    assert hits, "grep returned nothing for a query whose terms are present"
    top = hits[0]
    assert not Path(top.file_path).is_absolute(), top.file_path
    assert top.file_path == "pkg/scanner.py", top.file_path
    assert top.content, "empty content means the baseline reports 0 context bytes"


def test_grep_baseline_ignores_function_words(tmp_path) -> None:
    """
    A natural-language query must not match every file on the word "the".

    The baseline previously required *all* query tokens to appear, so a query
    containing function words matched nothing; relaxing that naively would have
    matched everything instead.
    """
    from tracera.evaluation.strategies import GrepStrategy

    (tmp_path / "hit.py").write_text("nested = 1\n", encoding="utf-8")
    (tmp_path / "miss.py").write_text("nothing = 1\n", encoding="utf-8")

    hits = GrepStrategy(tmp_path)._retrieve("how does the nested work", 5)

    assert [h.file_path for h in hits] == ["hit.py"]


def test_grep_baseline_never_enters_skipped_directories(tmp_path) -> None:
    """
    The skip-list must prune the walk, not just filter its results.

    ``rglob("*")`` descended into ``.venv`` / ``node_modules`` and discarded the
    entries afterwards, so any repo carrying one paid a full traversal of tens of
    thousands of files on *every* query (~15s each here — the whole reason a
    120-query run had to be abandoned). Vendored trees must never be entered.
    """
    from tracera.evaluation.strategies import GrepStrategy

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_text("nested = 1\n", encoding="utf-8")
    for skipped in ("node_modules", ".venv", ".next", "__pycache__"):
        vendored = tmp_path / skipped
        vendored.mkdir()
        (vendored / "vendored.py").write_text("nested = 1\n", encoding="utf-8")

    strategy = GrepStrategy(tmp_path)
    found = strategy._source_files()

    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["pkg/real.py"]
    # Discovered once, not re-walked per query.
    assert strategy._source_files() is found
    assert [h.file_path for h in strategy._retrieve("nested", 5)] == ["pkg/real.py"]


def test_ablation_arms_can_actually_differ() -> None:
    """
    The ablation baseline must be built *without* the retrieval tools.

    ``tool_filter`` was computed in ``eval_ablation`` and then never passed to
    ``_build_agent`` — and the retrieval pipeline was never passed either. Every
    arm therefore ran the identical agent with **no retrieval tools at all**: the
    study compared six copies of itself, and nothing in its output reveals that.
    An ablation whose arms cannot differ is not an ablation, it is a table.

    This is a source-level check because the defect is a dropped argument, which
    no runtime assertion in the benchmark can observe.
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "tracera" / "main.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)

    builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_agent"
    )
    assert "tool_filter" in [a.arg for a in builder.args.args], (
        "_build_agent cannot accept a tool filter, so no arm can differ"
    )

    def calls_in(func_name: str) -> list[ast.Call]:
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == func_name
        )
        return [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_agent"
        ]

    for func_name in ("eval_agent", "eval_ablation"):
        calls = calls_in(func_name)
        assert calls, f"{func_name} no longer builds an agent"
        for call in calls:
            assert len(call.args) >= 3, (
                f"{func_name} calls _build_agent at line {call.lineno} with no "
                "retrieval pipeline — the agent under test has no retrieval tools"
            )

    forwarded = [
        kw
        for call in calls_in("eval_ablation")
        for kw in call.keywords
        if kw.arg == "tool_filter"
    ]
    assert forwarded, (
        "eval_ablation never forwards tool_filter, so every arm gets the same "
        "tool set and the study measures nothing"
    )
    # Forwarding a literal None is the same as not forwarding it at all, and a
    # presence-only check cannot tell the two apart.
    assert not any(
        isinstance(kw.value, ast.Constant) and kw.value.value is None
        for kw in forwarded
    ), (
        "eval_ablation forwards `tool_filter=None`, which is indistinguishable "
        "from omitting it — the ablation arms cannot differ"
    )


def test_agent_task_runner_populates_every_field_the_benchmark_reads() -> None:
    """
    A field the benchmark reads but the runner never writes reports a zero.

    That zero is indistinguishable from a real measurement of zero: the token
    columns were empty for every task in every arm, so "no difference between
    arms" was *unobservable* rather than true, and the one number the agent
    benchmark exists to produce could never appear. `tool_calls` was the same,
    and `success` was set to True by any completion at all — including one where
    the model answered from its own priors without reading a single file.
    """
    import asyncio

    from tracera.agent.react_loop import AgentEvent, AgentEventType
    from tracera.main import _run_agent_task

    class FakeAgent:
        async def run(self, task: str):
            # `agent.run` is a coroutine returning an async iterator, not an
            # async generator function — the runner awaits it, then iterates.
            async def events():
                yield AgentEvent(
                    type=AgentEventType.TOOL_END, iteration=1, tool_name="read_file"
                )
                yield AgentEvent(
                    type=AgentEventType.RESPONSE_COMPLETE,
                    iteration=1,
                    text="done",
                    metadata={
                        "iterations": 2,
                        "tool_calls": 1,
                        "tokens_in": 120,
                        "tokens_out": 30,
                    },
                )

            return events()

    outcome = asyncio.run(_run_agent_task(FakeAgent(), "t"))

    assert outcome["success"] is True
    assert outcome["iterations"] == 2
    assert outcome["tool_calls"] == 1, "the benchmark's tool-call column would read 0"
    assert outcome["tokens_in"] == 120, "the benchmark's token column would read 0"
    assert outcome["tokens_out"] == 30
    assert outcome["tool_names"] == {"read_file"}

    class ErrorAgent:
        async def run(self, task: str):
            async def events():
                yield AgentEvent(type=AgentEventType.ERROR, iteration=1, text="boom")

            return events()

    failed = asyncio.run(_run_agent_task(ErrorAgent(), "t"))
    assert failed["success"] is False, "an errored task was counted as a success"
    assert failed["error"] == "boom"


def test_failed_run_still_reports_the_tokens_it_burned() -> None:
    """
    A run that fails after several LLM calls cost real money.

    Usage used to be published only on the success completion, so the error and
    max-iterations exits reported nothing at all. That zero is not a measurement,
    it is a *missing* one — and averaging it in makes whichever arm fails most
    look cheapest, which is precisely the comparison a benchmark exists to make
    honestly.
    """
    import asyncio

    from tracera.agent.react_loop import AgentEvent, AgentEventType
    from tracera.main import _run_agent_task

    class FailsAfterSpending:
        async def run(self, task: str):
            async def events():
                yield AgentEvent(
                    type=AgentEventType.DONE,
                    iteration=3,
                    metadata={
                        "terminated_by_error": True,
                        "iterations": 3,
                        "tool_calls": 2,
                        "tokens_in": 4200,
                        "tokens_out": 55,
                        "usage_reported": True,
                    },
                )
                yield AgentEvent(type=AgentEventType.ERROR, iteration=3, text="rate limited")

            return events()

    outcome = asyncio.run(_run_agent_task(FailsAfterSpending(), "t"))

    assert outcome["tokens_in"] == 4200, "the tokens a failed run burned went unreported"
    assert outcome["tokens_out"] == 55
    assert outcome["tool_calls"] == 2
    assert outcome["success"] is False


def test_a_run_that_never_completes_is_not_a_success() -> None:
    """
    ``success`` must start False.

    The benchmark reads ``outcome.get("success", True)``, so a runner that never
    sets the key turns an unfinished run into a success — at zero tokens, which
    makes it the best-looking result in the report.
    """
    import asyncio

    from tracera.agent.react_loop import AgentEvent, AgentEventType
    from tracera.main import _run_agent_task

    class Silent:
        async def run(self, task: str):
            async def events():
                yield AgentEvent(type=AgentEventType.THINKING, iteration=1, text="...")

            return events()

    outcome = asyncio.run(_run_agent_task(Silent(), "t"))

    assert outcome["success"] is False, "a run that produced nothing counted as a success"


def test_benchmark_flags_tasks_whose_usage_was_never_reported() -> None:
    """
    A zero nobody measured must not be averaged in silently.

    The report has to state how many tasks were actually measured, or the token
    mean reads as though every task contributed to it.
    """
    import asyncio

    from tracera.evaluation.agent_benchmark import AgentBenchmark

    async def runner(task: str) -> dict:
        if task == "measured":
            return {
                "output": "x",
                "success": True,
                "tokens_in": 1000,
                "tokens_out": 100,
                "usage_reported": True,
            }
        # No usage_reported key: the benchmark must infer "not measured" from the
        # numbers rather than assume the zero is real.
        return {"output": "y", "success": True, "tokens_in": 0, "tokens_out": 0}

    report = asyncio.run(AgentBenchmark(runner, tasks=["measured", "silent"]).run())

    assert report.tasks_with_usage == 1
    assert report.to_dict()["tasks_with_usage"] == 1
    assert report.results[1].usage_reported is False, "an unmeasured zero was taken as measured"

    markdown = report.to_markdown()
    assert "measured on 1/2 tasks" in markdown
    assert "1 of 2 task(s) reported no token usage" in markdown


def test_a_hung_task_is_abandoned_instead_of_hanging_the_run() -> None:
    """
    A provider call that never returns must not silence the whole benchmark.

    A real `eval agent` run was killed at 9m45s with its log frozen after a
    successful HTTP 200 and no report written at all. The budget abandons that
    task and lets the run finish with something to read.
    """
    import asyncio

    from tracera.evaluation.agent_benchmark import AgentBenchmark

    async def never_returns(task: str) -> dict:
        await asyncio.sleep(3600)
        return {"success": True}

    report = asyncio.run(
        AgentBenchmark(never_returns, tasks=["stuck"], task_timeout_s=0.05).run()
    )

    result = report.results[0]
    assert result.timed_out is True, "a hung task was not marked as timed out"
    assert result.success is False, "a timed-out task counted as a success"
    assert "timed out" in (result.error or "")
    assert report.tasks_timed_out == 1
    assert report.to_dict()["tasks_timed_out"] == 1
    assert "1 of 1 task(s) timed out" in report.to_markdown()


def test_a_runner_that_ignores_cancellation_cannot_stall_the_guard() -> None:
    """
    The cancelled task is deliberately not awaited.

    ``asyncio.wait_for`` waits for the cancellation to complete, so a runner that
    catches ``CancelledError`` and carries on would hang inside the very guard
    meant to prevent hanging. This asserts the benchmark returns on its own
    budget regardless of what the abandoned task does.

    The runner below ignores cancellation for 0.3s — long enough to be caught,
    short enough that ``asyncio.run``'s shutdown does not stall the test suite.
    """
    import asyncio
    import time

    from tracera.evaluation.agent_benchmark import AgentBenchmark

    async def swallows_cancellation(task: str) -> dict:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)
        return {"success": True}

    async def scenario() -> tuple[object, float]:
        bench = AgentBenchmark(swallows_cancellation, tasks=["stuck"], task_timeout_s=0.05)
        started = time.perf_counter()
        report = await bench.run()
        return report, time.perf_counter() - started

    report, elapsed = asyncio.run(scenario())

    assert report.tasks_timed_out == 1
    assert elapsed < 0.25, (
        f"the guard waited {elapsed:.2f}s for a runner that ignored cancellation"
    )


def test_a_budget_of_zero_disables_the_timeout() -> None:
    """0 means "no limit", so a slow-but-working task is never cut off."""
    import asyncio

    from tracera.evaluation.agent_benchmark import AgentBenchmark

    async def quick(task: str) -> dict:
        await asyncio.sleep(0.01)
        return {"success": True, "tokens_in": 5, "tokens_out": 1, "usage_reported": True}

    report = asyncio.run(AgentBenchmark(quick, tasks=["t"], task_timeout_s=0).run())

    assert report.results[0].success is True
    assert report.results[0].timed_out is False
    assert report.tasks_timed_out == 0


def test_dedupe_by_file_keeps_the_cap_and_the_order() -> None:
    """
    The cap is per file, rank order survives, and unresolvable paths are kept.

    Grouping the unresolved ones into one bucket would cap them together — a
    dedup helper silently discarding results it had no basis to rank.
    """
    from tracera.retrieval.dedupe import dedupe_by_file

    items = [{"file_path": p} for p in ("a.py", "a.py", "a.py", "b.py", None, None)]

    def path_of(item):
        return item["file_path"]

    assert dedupe_by_file(items, max_per_file=2, file_path_of=path_of) == [
        {"file_path": "a.py"},
        {"file_path": "a.py"},
        {"file_path": "b.py"},
        {"file_path": None},
        {"file_path": None},
    ]
    assert len(dedupe_by_file(items, max_per_file=1, file_path_of=path_of)) == 4
    assert dedupe_by_file(items, max_per_file=None, file_path_of=path_of) == items
    assert dedupe_by_file(items, max_per_file=0, file_path_of=path_of) == items


def _clustered_strategy(*, max_per_file=None):
    """A strategy whose hits are ten chunks each from three files."""

    from tracera.evaluation.strategies import RetrievalHit, RetrievalStrategy

    class Pool(RetrievalStrategy):
        name = "pool"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.asked_for = 0

        def _retrieve(self, query: str, k: int) -> list[RetrievalHit]:
            self.asked_for = k
            pool: list[RetrievalHit] = []
            for letter in ("a", "b", "c"):
                for i in range(10):
                    pool.append(
                        RetrievalHit(
                            doc_id=f"{letter}{len(pool)}",
                            score=1.0 - len(pool) / 100.0,
                            file_path=f"src/{letter}.py",
                            # Deliberately uneven: a file's later chunks are
                            # its larger bodies. Uniform sizes would make
                            # dedup's byte effect an identity and the guard
                            # below vacuous.
                            content="x" * (100 + 40 * i),
                        )
                    )
            return pool[:k]

    return Pool(max_per_file=max_per_file)


def test_per_file_dedup_diversifies_without_shrinking_the_window() -> None:
    """
    Dedup must diversify *and* still fill k slots.

    Those are opposite failures: dedup that does not dedup, and dedup that
    returns fewer than k because it dropped duplicates from a window that was
    only ever k wide. The second is why the strategy over-fetches, and
    ``len(hits) == k`` is what tells them apart — a coverage-only assertion would
    pass on a shrunken window.
    """
    plain = _clustered_strategy()
    baseline = plain.retrieve("q", k=5)

    assert len(baseline) == 5
    assert {h.file_path for h in baseline} == {"src/a.py"}, "fixture is not clustered"
    assert plain.asked_for == 5, "with no cap there is nothing to over-fetch for"

    capped = _clustered_strategy(max_per_file=2)
    hits = capped.retrieve("q", k=5)

    assert len(hits) == 5, "dedup shrank the window instead of diversifying it"
    assert len({h.file_path for h in hits}) == 3, "dedup did not diversify"
    assert capped.asked_for > 5, "the cap must widen the request before it filters"


def test_dedup_cuts_context_bytes_at_a_fixed_k() -> None:
    """
    Dedup returns a *smaller* window at unchanged k — the claim, pinned.

    This is not a truism, and it was originally asserted backwards. Because a
    file's later chunks are its larger bodies, keeping only each file's
    top-ranked chunk and refilling the window from other files lowers the byte
    total without widening k. Measured at k=10 over 120 queries: -31% hybrid,
    -43% BM25, -35% dense.

    The fixture's chunk sizes are deliberately uneven. With uniform sizes dedup
    would swap equal-sized chunks and the byte count could not move, so the
    assertion would hold no matter what the code did.
    """
    plain = _clustered_strategy()
    capped = _clustered_strategy(max_per_file=2)

    wide = plain.retrieve("q", k=5)
    narrow = capped.retrieve("q", k=5)

    assert len(wide) == len(narrow) == 5, "the window must stay k wide"
    assert len({h.file_path for h in narrow}) == 3, "dedup did not diversify"
    assert capped.last_result_bytes < plain.last_result_bytes, (
        f"deduped {capped.last_result_bytes}B vs plain {plain.last_result_bytes}B"
    )
