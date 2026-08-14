"""Tests for Phases 35/36/38 — autonomous debugging, fix loop, regression protection."""

import pytest

# Aliased so pytest doesn't try to collect these dataclasses as test classes
from tracera.tools.test_runner import TestFailure as Failure, TestReport as Report


# ── Phase 35: RetrievalDebugger ───────────────────────────────────────────────

class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.searched = 0

    def search(self, query, k=8):
        self.searched += 1
        return self._chunks


class _FakeCompressor:
    def __init__(self):
        self.called = False
        self.input_chunks = None

    def compress(self, chunks):
        self.called = True
        self.input_chunks = chunks
        return chunks


def test_retrieval_debugger_uses_compressor():
    from tracera.agent.autonomous import RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    chunks = [{"content": "def auth(): pass", "symbol": "auth", "file_path": "auth.py"}]
    retriever = _FakeRetriever(chunks)
    compressor = _FakeCompressor()
    debugger = RetrievalDebugger(retriever, ContextAssemblyEngine(), compressor=compressor)

    failure = Failure(
        test_name="test_auth", error_type="AssertionError",
        error_message="auth failed", file_path="auth.py", line_number=3,
    )
    plan = debugger.build_debug_plan(failure, provider=None)

    assert retriever.searched == 1
    assert compressor.called
    assert compressor.input_chunks == chunks
    assert "auth.py" in plan.hypothesis
    assert "auth" in plan.retrieved_context


def test_retrieval_debugger_tolerates_missing_retriever():
    from tracera.agent.autonomous import RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    debugger = RetrievalDebugger(None, ContextAssemblyEngine())
    failure = Failure(test_name="t", error_type="E", error_message="m")
    plan = debugger.build_debug_plan(failure, provider=None)
    assert plan.retrieved_context == ""


# ── Phase 36: AutonomousFixLoop consumes the agent event stream ───────────────

class _PassingTestRunner:
    def run(self, framework=None, test_paths=None):
        return Report(framework="pytest", passed=3, total=3, success=True)


class _FakeAgent:
    """Fake ReActAgent whose run() is an async generator, like the real one."""

    def __init__(self):
        self.tasks_run = []

    async def run(self, task, conversation=None):
        self.tasks_run.append(task)

        async def _events():
            yield None  # pretend we drive the loop
            yield None

        return _events()


@pytest.mark.asyncio
async def test_fix_loop_stops_on_pass(tmp_path):
    from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    runner = _PassingTestRunner()
    agent = _FakeAgent()
    debugger = RetrievalDebugger(None, ContextAssemblyEngine())

    loop = AutonomousFixLoop(tmp_path, runner, debugger, max_iterations=3)
    result = await loop.run("fix the auth bug", provider=None, agent=agent)

    assert result.final_success is True
    assert result.total_iterations == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].success is True
    # Tests passed on the first attempt → no fix task needed
    assert len(agent.tasks_run) == 0


@pytest.mark.asyncio
async def test_fix_loop_drives_agent_on_failure(tmp_path):
    from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    class _FailingRunner:
        def __init__(self):
            self.calls = 0

        def run(self, framework=None, test_paths=None):
            self.calls += 1
            if self.calls >= 2:  # pass after the first fix attempt
                return Report(framework="pytest", passed=3, total=3, success=True)
            return Report(
                framework="pytest", passed=2, total=3, success=False,
                failures=[Failure(test_name="t", error_type="E", error_message="m")],
            )

    runner = _FailingRunner()
    agent = _FakeAgent()
    debugger = RetrievalDebugger(None, ContextAssemblyEngine())

    loop = AutonomousFixLoop(tmp_path, runner, debugger, max_iterations=3)
    result = await loop.run("fix the bug", provider=None, agent=agent)

    assert result.final_success is True
    assert result.total_iterations == 2
    assert len(agent.tasks_run) == 1  # agent driven once to fix the failure


# ── Phase 9 → 36: fix loop plans and replans after failures ───────────────────

class _ReplanDecomposer:
    def __init__(self):
        self.decompose_calls = 0
        self.replan_calls = 0

    async def decompose(self, task):
        from tracera.agent.planner import Plan
        self.decompose_calls += 1
        plan = Plan(task)
        plan.add_item("Step 1")
        return plan

    async def replan(self, plan, reason):
        self.replan_calls += 1
        plan.add_item(f"[Recovery] {reason}")
        return plan


@pytest.mark.asyncio
async def test_fix_loop_replans_on_failure(tmp_path):
    from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger
    from tracera.agent.context_engine import ContextAssemblyEngine

    class _FailThenPassRunner:
        def __init__(self):
            self.calls = 0

        def run(self, framework=None, test_paths=None):
            self.calls += 1
            if self.calls >= 2:
                return Report(framework="pytest", passed=2, total=2, success=True)
            return Report(
                framework="pytest", passed=1, total=2, success=False,
                failures=[Failure(test_name="t", error_type="E", error_message="m")],
            )

    decomposer = _ReplanDecomposer()
    debugger = RetrievalDebugger(None, ContextAssemblyEngine())
    loop = AutonomousFixLoop(
        tmp_path, _FailThenPassRunner(), debugger,
        max_iterations=3, decomposer=decomposer,
    )

    result = await loop.run("fix the bug", provider=None, agent=_FakeAgent())

    assert result.final_success is True
    assert decomposer.decompose_calls == 1
    assert decomposer.replan_calls == 1  # replanned after the first failed attempt


# ── Phase 38: RegressionProtector ─────────────────────────────────────────────

class _CountingRunner:
    def __init__(self):
        self.calls = 0
        self.results = [
            Report(framework="pytest", passed=5, total=5, success=True),
            Report(framework="pytest", passed=4, total=5, success=False),
        ]

    def run(self, framework=None, test_paths=None):
        r = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return r


def test_regression_protector_detects_regressions(tmp_path, monkeypatch):
    from tracera.agent.autonomous import RegressionProtector

    runner = _CountingRunner()
    protector = RegressionProtector(tmp_path, runner)

    baseline = protector.snapshot_before()
    assert baseline.passed == 5

    monkeypatch.setattr(
        "tracera.agent.autonomous.subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": "auth.py\nmain.py\n"})(),
    )
    report = protector.verify_after()

    assert report["pre_passed"] == 5
    assert report["post_passed"] == 4
    assert report["regressions"] == 1
    assert report["overall_success"] is False
    assert "auth.py" in report["changed_files"]
