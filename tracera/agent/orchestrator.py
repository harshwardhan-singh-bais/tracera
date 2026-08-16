"""
Phases 43-44 — Task delegation & result aggregation.

The orchestrator coordinates specialized sub-agents for a single task:

    Main agent task
        ↓
    Planner (decompose into steps)
        ↓
    assign each step → sub-agent role (researcher / coder / tester / ...)
        ↓
    run sub-agents (optionally in parallel), collecting AgentResults
        ↓
    ResultAggregator: merge findings, resolve conflicts, maintain shared
    task state, and produce the final report passed back to the main agent

Shared task state is a simple key-value store every sub-agent can read and
update, so later agents see what earlier ones decided (files touched,
symbols located, constraints agreed on).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from tracera.agent.subagents import (
    ROLE_LABELS,
    SpecializedAgent,
    SubAgentRole,
)
from tracera.logging import get_logger

log = get_logger("agent.orchestrator")


# ── Shared task state ─────────────────────────────────────────────────────────

class SharedTaskState:
    """
    Mutable, thread-unsafe shared context for a delegation run.

    Keys are lower-case dotted namespaces, e.g. ``files.touched``,
    ``symbols.found``, ``decisions``, ``constraints``. Sub-agents receive a
    snapshot as context and write their findings back afterwards.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            "constraints": [],
            "files.touched": [],
            "symbols.found": [],
            "findings": [],
            "decisions": [],
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def append(self, key: str, value: Any) -> None:
        self._data.setdefault(key, []).append(value)

    def extend(self, key: str, values: list[Any]) -> None:
        self._data.setdefault(key, []).extend(values)

    def snapshot(self) -> dict[str, Any]:
        """Deep-ish copy for handing to an agent without shared mutation."""
        return json.loads(json.dumps(self._data, default=str))

    def merge(self, other: "SharedTaskState") -> None:
        """Merge another state's lists into this one (used to combine agents)."""
        for key, value in other._data.items():
            if isinstance(value, list):
                self.extend(key, value)
            elif key not in self._data:
                self._data[key] = value

    def __repr__(self) -> str:
        return f"<SharedTaskState keys={list(self._data.keys())}>"


def _state_context(state: SharedTaskState) -> str:
    """Render the shared state as a compact text block for an agent prompt."""
    lines = ["Shared task state:"]
    for key, value in state._data.items():
        if isinstance(value, list) and value:
            items = value if len(value) <= 8 else value[:8] + [f"... ({len(value)} total)"]
            lines.append(f"- {key}: {', '.join(str(i) for i in items)}")
    return "\n".join(lines)


# ── Delegation plan ───────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    """Outcome of one sub-agent execution."""

    role: SubAgentRole
    task: str
    status: AgentStatus = AgentStatus.PENDING
    output: str = ""
    findings: list[str] = field(default_factory=list)
    error: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def label(self) -> str:
        return ROLE_LABELS[self.role]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "label": self.label,
            "task": self.task,
            "status": self.status.value,
            "output": self.output,
            "findings": self.findings,
            "error": self.error,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class DelegationStep:
    """A single step of the delegation plan: task text + role assignment."""

    task: str
    role: SubAgentRole | None = None  # None = main agent handles it
    status: AgentStatus = AgentStatus.PENDING


@dataclass
class DelegationPlan:
    """The plan produced by the planner: ordered steps with role assignments."""

    main_task: str
    steps: list[DelegationStep] = field(default_factory=list)

    @property
    def pending(self) -> list[DelegationStep]:
        return [s for s in self.steps if s.status == AgentStatus.PENDING]

    def to_markdown(self) -> str:
        lines = [f"## Delegation plan: {self.main_task}\n"]
        for i, step in enumerate(self.steps, 1):
            role = ROLE_LABELS[step.role] if step.role else "Main agent"
            lines.append(f"{i}. **[{role}]** {step.task}")
        return "\n".join(lines)


# ── Role assignment ───────────────────────────────────────────────────────────

_ROLE_KEYWORDS: dict[SubAgentRole, tuple[str, ...]] = {
    SubAgentRole.RESEARCHER: (
        "understand", "explain", "where", "how does", "what does", "locate",
        "find", "search", "research", "investigate the code", "architecture",
        "overview", "summarize", "summarise",
    ),
    SubAgentRole.TESTER: (
        "test", "run tests", "pytest", "unit test", "regression",
    ),
    SubAgentRole.REVIEWER: (
        "review", "check", "verify", "audit", "code quality", "lint",
    ),
    SubAgentRole.DEBUGGER: (
        "debug", "fix the bug", "why is", "failing", "failure", "crash",
        "traceback", "error", "exception",
    ),
    SubAgentRole.CODER: (
        "implement", "write", "create", "add", "refactor", "change",
        "update", "edit", "build", "make", "fix", "modify",
    ),
}

#: Roles the orchestrator can assign on its own (no LLM needed).
#: Tie-break order matters: implementation tasks (Coder) and debugging
#: (Debugger) win over generic questions; Debugger beats Tester for
#: "why is the test failing"-style tasks.
_ASSIGNABLE = (
    SubAgentRole.CODER,
    SubAgentRole.DEBUGGER,
    SubAgentRole.TESTER,
    SubAgentRole.REVIEWER,
    SubAgentRole.RESEARCHER,
)


def assign_role(task: str) -> SubAgentRole | None:
    """
    Heuristic role assignment: keyword-score the task against each role and
    return the best match (ties broken by ``_ASSIGNABLE`` order).

    Returns None when no keyword matches (the main agent keeps the step).
    Deterministic and free — used when no LLM is available or configured.
    """
    lowered = task.lower()
    best_role: SubAgentRole | None = None
    best_score = 0
    for role in _ASSIGNABLE:
        score = sum(1 for kw in _ROLE_KEYWORDS[role] if kw in lowered)
        if score > best_score:
            best_role, best_score = role, score
    return best_role


# ── Orchestrator ──────────────────────────────────────────────────────────────

class TaskOrchestrator:
    """
    Delegates a task to specialized sub-agents and aggregates their results.

    Flow:
        async with orchestrator.delegate(task) → iterates events:

            plan_ready → delegation plan (steps + role assignments)
            agent_start → a sub-agent begins (role, step)
            agent_end   → a sub-agent finished (AgentResult)
            report      → final aggregated report

    Usage:

        orchestrator = TaskOrchestrator(fleet, decomposer=None)
        async for event in await orchestrator.delegate(task):
            ...
    """

    def __init__(
        self,
        fleet: dict[SubAgentRole, SpecializedAgent],
        *,
        decomposer: Any | None = None,
        parallel: bool = False,
    ) -> None:
        """
        Args:
            fleet: role → SpecializedAgent, from build_sub_agent_fleet().
            decomposer: optional TaskDecomposer for LLM-based step generation.
            parallel: run independent steps concurrently (default sequential).
        """
        self.fleet = fleet
        self.decomposer = decomposer
        self.parallel = parallel
        self.state = SharedTaskState()

    # ── Planning ─────────────────────────────────────────────────────────────

    async def _build_plan(self, task: str) -> DelegationPlan:
        """Decompose the task; fall back to a single delegated step."""
        plan = DelegationPlan(main_task=task)
        if self.decomposer is not None:
            try:
                llm_plan = await self.decomposer.decompose(task)
                for item in llm_plan.items:
                    plan.steps.append(DelegationStep(task=item.title))
                if plan.steps:
                    return plan
            except Exception as e:
                log.warning("Delegation decomposition failed: %s", e)
        plan.steps.append(DelegationStep(task=task))
        return plan

    def _assign(self, plan: DelegationPlan) -> None:
        """Assign a role to each step (heuristic; LLM assignment optional)."""
        for step in plan.steps:
            if step.role is None:
                step.role = assign_role(step.task)

    # ── Delegation ───────────────────────────────────────────────────────────

    async def delegate(self, task: str) -> AsyncIterator[dict[str, Any]]:
        """
        Run the full delegation pipeline, yielding event dicts:

        - {"type": "plan_ready", "plan": DelegationPlan}
        - {"type": "agent_start", "step": DelegationStep}
        - {"type": "agent_end", "step": DelegationStep, "result": AgentResult}
        - {"type": "report", "report": OrchestrationReport}
        """
        plan = await self._build_plan(task)
        self._assign(plan)
        yield {"type": "plan_ready", "plan": plan}

        results: list[AgentResult] = []

        if self.parallel:
            # Build all (step, result) futures, then await in order.
            futures = [
                self._run_step(step, plan) for step in plan.steps
                if step.role is not None and step.role in self.fleet
            ]
            ordered = await asyncio.gather(*futures)
            for step, result in ordered:
                step.status = result.status
                yield {"type": "agent_end", "step": step, "result": result}
                results.append(result)
        else:
            for step in plan.steps:
                if step.role is None or step.role not in self.fleet:
                    step.status = AgentStatus.SKIPPED
                    yield {"type": "agent_skipped", "step": step}
                    continue
                yield {"type": "agent_start", "step": step}
                result = await self._run_agent(step)
                step.status = result.status
                yield {"type": "agent_end", "step": step, "result": result}
                results.append(result)

        report = ResultAggregator().aggregate(task, plan, results, self.state)
        yield {"type": "report", "report": report}

    async def _run_step(
        self, step: DelegationStep, plan: DelegationPlan
    ) -> tuple[DelegationStep, AgentResult]:
        """Parallel-friendly wrapper around _run_agent."""
        return step, await self._run_agent(step)

    async def _run_agent(self, step: DelegationStep) -> AgentResult:
        """Execute one step with its assigned sub-agent, collecting metrics."""
        agent = self.fleet[step.role]  # type: ignore[index]
        result = AgentResult(role=step.role, task=step.task, status=AgentStatus.RUNNING)
        result.started_at = time.time()

        prompt = f"{step.task}\n\n{_state_context(self.state)}"
        try:
            stream = agent.run(prompt)
            # ReActAgent.run() returns an async generator; some wrappers
            # return an awaitable that must be awaited first. Normalise both.
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                if event.type.value == "response_complete":
                    result.output = event.text or ""
                elif event.type.value == "tool_end":
                    result.tool_calls += 1
                elif event.type.value == "thinking":
                    result.iterations = event.iteration + 1
                elif event.type.value == "error":
                    result.error = event.text
        except Exception as e:
            log.exception("Sub-agent %s failed", result.label)
            result.error = str(e)

        result.status = AgentStatus.SUCCESS if not result.error else AgentStatus.FAILED
        result.finished_at = time.time()
        result.latency_ms = (result.finished_at - result.started_at) * 1000

        # Persist findings into the shared state for downstream agents.
        if result.output:
            self.state.append(f"findings.{result.role.value}", result.output[:500])
        return result

    async def execute(self, task: str) -> "OrchestrationReport":
        """Collect all events and return the final report."""
        report: OrchestrationReport | None = None
        async for event in self.delegate(task):
            if event["type"] == "report":
                report = event["report"]
        assert report is not None, "delegation produced no report"
        return report


# ── Aggregation ───────────────────────────────────────────────────────────────

@dataclass
class OrchestrationReport:
    """The final aggregated output of a delegation run."""

    main_task: str
    plan: DelegationPlan
    results: list[AgentResult]
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    shared_state: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0

    @property
    def success_rate(self) -> float:
        done = [r for r in self.results if r.status == AgentStatus.SUCCESS]
        return (len(done) / len(self.results)) if self.results else 0.0

    def to_markdown(self) -> str:
        lines = [f"## Delegation report: {self.main_task}\n"]
        for r in self.results:
            icon = {"success": "✓", "failed": "✗", "skipped": "⊘"}.get(r.status.value, "○")
            lines.append(f"- {icon} **[{r.label}]** {r.task[:80]}")
            if r.output:
                lines.append(f"  > {r.output.strip()[:200]}")
            if r.error:
                lines.append(f"  > ⚠ {r.error[:200]}")
        if self.conflicts:
            lines.append("\n**Conflicts:**")
            for c in self.conflicts:
                lines.append(f"- {c['message']}")
        if self.summary:
            lines.append(f"\n**Summary:** {self.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_task": self.main_task,
            "plan": [s.task for s in self.plan.steps],
            "results": [r.to_dict() for r in self.results],
            "conflicts": self.conflicts,
            "summary": self.summary,
            "shared_state": self.shared_state,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
        }


class ResultAggregator:
    """
    Phase 44 — collects sub-agent results, resolves conflicts, and produces
    the report passed back to the main agent.
    """

    def aggregate(
        self,
        main_task: str,
        plan: DelegationPlan,
        results: list[AgentResult],
        state: SharedTaskState,
    ) -> OrchestrationReport:
        conflicts: list[dict[str, Any]] = []
        for r in results:
            if r.status == AgentStatus.FAILED and r.error:
                conflicts.append(
                    {"message": f"{r.label} failed: {r.error[:200]}", "role": r.role.value}
                )

        # Conflict resolution: if the Tester says tests fail but the Coder
        # claims success, flag the discrepancy for the main agent.
        coder = next((r for r in results if r.role == SubAgentRole.CODER), None)
        tester = next((r for r in results if r.role == SubAgentRole.TESTER), None)
        if coder and tester and tester.status == AgentStatus.FAILED:
            conflicts.append(
                {
                    "message": (
                        f"Coder reported success but the test run failed — "
                        f"main agent should re-verify: {tester.error[:160]}"
                    ),
                    "role": "coder-vs-tester",
                }
            )

        summary = self._summarize(results)
        report = OrchestrationReport(
            main_task=main_task,
            plan=plan,
            results=results,
            conflicts=conflicts,
            summary=summary,
            shared_state=state.snapshot(),
        )
        for r in results:
            report.total_latency_ms += r.latency_ms
            report.total_tokens += r.tokens
            report.total_tool_calls += r.tool_calls
        return report

    @staticmethod
    def _summarize(results: list[AgentResult]) -> str:
        if not results:
            return "No sub-agents were dispatched."
        ok = [r for r in results if r.status == AgentStatus.SUCCESS]
        failed = [r for r in results if r.status == AgentStatus.FAILED]
        parts = [f"{len(ok)}/{len(results)} sub-agents succeeded."]
        if failed:
            names = ", ".join(r.label for r in failed)
            parts.append(f"Failed: {names}.")
        return " ".join(parts)
