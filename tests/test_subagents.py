"""
Tests for Phases 42-44 — sub-agent framework, task delegation,
and result aggregation.
"""

from __future__ import annotations

import pytest

from tracera.agent.orchestrator import (
    AgentResult,
    AgentStatus,
    ResultAggregator,
    SharedTaskState,
    TaskOrchestrator,
    assign_role,
)
from tracera.agent.subagents import (
    ROLE_TOOL_SETS,
    SubAgentRole,
    build_sub_agent_fleet,
    filter_registry,
    role_spec,
)
from tracera.tools.registry import ToolRegistry


# ════════════════════════════════════════════════════════════════════════════
# Phase 42 — sub-agent framework
# ════════════════════════════════════════════════════════════════════════════


def _dummy_tool(name: str):
    from tracera.tools.base import Tool

    class _T(Tool):
        @property
        def name(self) -> str:
            return name

        @property
        def description(self) -> str:
            return f"tool {name}"

        @property
        def parameters_schema(self) -> dict:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            from tracera.tools.base import ToolResult
            return ToolResult.ok(name, "", "ok")

    return _T()


def test_role_specs_cover_all_roles():
    for role in SubAgentRole:
        spec = role_spec(role)
        assert spec.role == role
        assert spec.label
        assert spec.description


def test_filter_registry_keeps_only_allowed_tools():
    registry = ToolRegistry()
    for name in ("read_file", "grep", "write_file", "edit_file", "run_command"):
        registry.register(_dummy_tool(name))

    filtered = filter_registry(registry, ROLE_TOOL_SETS[SubAgentRole.RESEARCHER])
    names = set(filtered.names)
    assert "read_file" in names
    assert "grep" in names
    assert "write_file" not in names
    assert "run_command" not in names


def test_fleet_builds_all_five_roles():
    class _FakeProvider:
        name = "fake"
        default_model = "fake-model"

    registry = ToolRegistry()
    registry.register(_dummy_tool("read_file"))
    registry.register(_dummy_tool("run_command"))

    fleet = build_sub_agent_fleet(_FakeProvider(), registry)
    assert set(fleet.keys()) == set(SubAgentRole)
    for role, agent in fleet.items():
        assert agent.label == role_spec(role).label
        assert set(agent.registry.names) <= ROLE_TOOL_SETS[role]


# ════════════════════════════════════════════════════════════════════════════
# Phase 43 — task delegation
# ════════════════════════════════════════════════════════════════════════════


def test_assign_role_heuristic():
    assert assign_role("where is authentication handled") == SubAgentRole.RESEARCHER
    assert assign_role("run the test suite") == SubAgentRole.TESTER
    assert assign_role("implement the login endpoint") == SubAgentRole.CODER
    assert assign_role("review the recent diff") == SubAgentRole.REVIEWER
    assert assign_role("why is the test failing") == SubAgentRole.DEBUGGER


def test_shared_task_state_accumulates():
    state = SharedTaskState()
    state.append("files.touched", "auth.py")
    state.append("symbols.found", "validate_token")
    assert state.get("files.touched") == ["auth.py"]
    snapshot = state.snapshot()
    assert snapshot["symbols.found"] == ["validate_token"]
    # snapshot must not share mutation
    state.append("files.touched", "db.py")
    assert snapshot["files.touched"] == ["auth.py"]


def test_shared_task_state_merge():
    a = SharedTaskState()
    a.append("files.touched", "a.py")
    b = SharedTaskState()
    b.append("files.touched", "b.py")
    a.merge(b)
    assert set(a.get("files.touched")) == {"a.py", "b.py"}


# ════════════════════════════════════════════════════════════════════════════
# Phase 44 — result aggregation
# ════════════════════════════════════════════════════════════════════════════


def _result(role: SubAgentRole, status: AgentStatus, error: str | None = None) -> AgentResult:
    return AgentResult(
        role=role,
        task="step",
        status=status,
        output="output" if status == AgentStatus.SUCCESS else "",
        error=error,
        iterations=2,
        tool_calls=3,
        tokens=100,
        latency_ms=50.0,
    )


def test_aggregator_collects_and_scores():
    from tracera.agent.orchestrator import DelegationPlan, DelegationStep

    plan = DelegationPlan(main_task="task", steps=[DelegationStep("step")])
    results = [
        _result(SubAgentRole.RESEARCHER, AgentStatus.SUCCESS),
        _result(SubAgentRole.TESTER, AgentStatus.FAILED, "tests failed"),
    ]
    state = SharedTaskState()
    report = ResultAggregator().aggregate("task", plan, results, state)

    assert report.success_rate == 0.5
    assert report.total_tool_calls == 6
    assert report.total_tokens == 200
    assert report.total_latency_ms == 100.0
    # Coder not present → no coder-vs-tester conflict, but tester failure
    # is recorded as a conflict.
    assert any(c["role"] == "tester" for c in report.conflicts)


def test_aggregator_detects_coder_tester_conflict():
    from tracera.agent.orchestrator import DelegationPlan, DelegationStep

    plan = DelegationPlan(main_task="task", steps=[DelegationStep("step")])
    results = [
        _result(SubAgentRole.CODER, AgentStatus.SUCCESS),
        _result(SubAgentRole.TESTER, AgentStatus.FAILED, "1 failed"),
    ]
    report = ResultAggregator().aggregate("task", plan, results, SharedTaskState())
    assert any(c["role"] == "coder-vs-tester" for c in report.conflicts)


def test_report_markdown():
    from tracera.agent.orchestrator import DelegationPlan, DelegationStep

    plan = DelegationPlan(main_task="task", steps=[DelegationStep("step")])
    results = [_result(SubAgentRole.RESEARCHER, AgentStatus.SUCCESS)]
    report = ResultAggregator().aggregate("task", plan, results, SharedTaskState())
    md = report.to_markdown()
    assert "Researcher" in md
    assert "Delegation report" in md


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator end-to-end with a fake sub-agent fleet
# ════════════════════════════════════════════════════════════════════════════


class _FakeSpecializedAgent:
    """Minimal stand-in for SpecializedAgent that streams canned events."""

    def __init__(self, role: SubAgentRole, output: str = "done") -> None:
        self.role = role
        self._output = output
        self.label = role.value.title()

    def run(self, task: str, **kwargs):
        from tracera.agent.react_loop import AgentEvent, AgentEventType

        async def _gen():
            yield AgentEvent(type=AgentEventType.THINKING, iteration=0, text="thinking")
            yield AgentEvent(type=AgentEventType.RESPONSE_COMPLETE, text=self._output)
            yield AgentEvent(type=AgentEventType.DONE)

        return _gen()


class _FakeDecomposer:
    """Returns a fixed two-step plan (no LLM needed)."""

    def __init__(self, titles: list[str]) -> None:
        self.titles = titles

    async def decompose(self, task: str):
        from tracera.agent.planner import Plan
        plan = Plan(task)
        for title in self.titles:
            plan.add_item(title=title)
        return plan


async def test_orchestrator_delegates_and_aggregates():
    fleet = {
        SubAgentRole.RESEARCHER: _FakeSpecializedAgent(SubAgentRole.RESEARCHER, "found it"),
        SubAgentRole.CODER: _FakeSpecializedAgent(SubAgentRole.CODER, "implemented"),
    }
    orch = TaskOrchestrator(
        fleet,
        decomposer=_FakeDecomposer([
            "explain the login flow",
            "implement the login page",
        ]),
    )

    events = []
    async for event in orch.delegate("implement the login page and explain the flow"):
        events.append(event["type"])
        if event["type"] == "report":
            report = event["report"]
            assert report.success_rate == 1.0
            assert len(report.results) == 2
            roles = {r.role for r in report.results}
            assert roles == {SubAgentRole.RESEARCHER, SubAgentRole.CODER}

    assert "plan_ready" in events
    assert "agent_start" in events
    assert "agent_end" in events
    assert "report" in events


async def test_orchestrator_execute_returns_report():
    fleet = {
        SubAgentRole.RESEARCHER: _FakeSpecializedAgent(SubAgentRole.RESEARCHER),
    }
    orch = TaskOrchestrator(fleet)
    report = await orch.execute("where is authentication handled")
    assert report.main_task == "where is authentication handled"
    assert report.results[0].role == SubAgentRole.RESEARCHER
    assert report.results[0].status == AgentStatus.SUCCESS
