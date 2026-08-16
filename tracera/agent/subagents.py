"""
Phase 42 — Sub-agent framework.

Specialized agents built on the same ReAct loop, but each with a
role-specific system prompt and a filtered tool subset:

    Researcher — explores the codebase, gathers context, answers questions
    Coder      — implements changes (read/write/edit + git)
    Tester     — runs the test suite, diagnoses failures
    Reviewer   — reviews diffs / code for correctness (read-only + git)
    Debugger   — investigates failures, inspects state, runs commands

Each sub-agent is a plain :class:`~tracera.agent.react_loop.ReActAgent`
whose registry only contains the tools its role is allowed to use, so it
streams the same events the main agent does and can be driven by the same
orchestrator / TUI code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from tracera.logging import get_logger
from tracera.tools.registry import ToolRegistry

log = get_logger("agent.subagents")


class SubAgentRole(str, Enum):
    """The five specialized agent roles."""

    RESEARCHER = "researcher"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"


ROLE_LABELS = {
    SubAgentRole.RESEARCHER: "Researcher",
    SubAgentRole.CODER: "Coder",
    SubAgentRole.TESTER: "Tester",
    SubAgentRole.REVIEWER: "Reviewer",
    SubAgentRole.DEBUGGER: "Debugger",
}


# ── Role tool subsets ─────────────────────────────────────────────────────────

#: Read-only tool names shared by every role that only observes the codebase.
_READ_TOOLS = {
    "read_file",
    "list_dir",
    "grep",
    "search_code",
    "find_symbol",
    "find_definition",
    "get_context",
    "find_references",
    "get_dependencies",
}

_WRITE_TOOLS = {"write_file", "edit_file", "delete_file"}

#: Tool names per role. Roles may overlap; the filter is applied against the
#: tools that actually exist in the registry, so missing tools are ignored.
ROLE_TOOL_SETS: dict[SubAgentRole, set[str]] = {
    SubAgentRole.RESEARCHER: _READ_TOOLS,
    SubAgentRole.CODER: _READ_TOOLS | _WRITE_TOOLS | {"git", "run_command"},
    SubAgentRole.TESTER: _READ_TOOLS | {"git", "run_command"},
    SubAgentRole.REVIEWER: _READ_TOOLS | {"git"},
    SubAgentRole.DEBUGGER: _READ_TOOLS | _WRITE_TOOLS | {"git", "run_command"},
}


# ── Role system prompts ───────────────────────────────────────────────────────

ROLE_SYSTEM_PROMPTS: dict[SubAgentRole, str] = {
    SubAgentRole.RESEARCHER: (
        "You are TRACERA's Researcher sub-agent. Your job is to explore the "
        "codebase and answer questions about it precisely and concisely.\n"
        "- Always use search_code / find_symbol / get_context to locate the "
        "relevant code before answering.\n"
        "- Cite the file paths and symbols you base your answer on.\n"
        "- You are read-only: never modify files. If a change is needed, "
        "report exactly what should change and where."
    ),
    SubAgentRole.CODER: (
        "You are TRACERA's Coder sub-agent. Your job is to implement changes "
        "in the codebase.\n"
        "- Read the relevant code first; never edit blind.\n"
        "- Prefer edit_file over write_file for targeted changes.\n"
        "- After editing, run the tests or a lint/build command to verify."
    ),
    SubAgentRole.TESTER: (
        "You are TRACERA's Tester sub-agent. Your job is to run tests and "
        "produce a clear report of what passes and what fails.\n"
        "- Use run_command to execute the test suite (pytest, npm test, ...).\n"
        "- Summarize pass/fail counts and list failing tests with their "
        "error messages and locations.\n"
        "- You are read-only for source files; report problems rather than "
        "fixing them yourself."
    ),
    SubAgentRole.REVIEWER: (
        "You are TRACERA's Reviewer sub-agent. Your job is to review code or "
        "diffs for correctness, security, and style.\n"
        "- Inspect the actual code with read_file / grep before judging.\n"
        "- Check for bugs, edge cases, security issues, and broken tests.\n"
        "- You are read-only: report findings, never modify files."
    ),
    SubAgentRole.DEBUGGER: (
        "You are TRACERA's Debugger sub-agent. Your job is to diagnose "
        "failures and find root causes.\n"
        "- Reproduce the failure first (run the failing test or command).\n"
        "- Inspect the code path with read_file / get_context.\n"
        "- State the root cause and the minimal fix; apply fixes only after "
        "confirming the diagnosis."
    ),
}


# ── Specialized agent ─────────────────────────────────────────────────────────

@dataclass
class SubAgentSpec:
    """Description of a sub-agent for delegation decisions."""

    role: SubAgentRole
    label: str
    description: str


def role_spec(role: SubAgentRole) -> SubAgentSpec:
    return SubAgentSpec(
        role=role,
        label=ROLE_LABELS[role],
        description=ROLE_SYSTEM_PROMPTS[role].splitlines()[0].strip("."),
    )


class SpecializedAgent:
    """
    A role-scoped agent: a ReActAgent restricted to the role's tool subset.

    ``run()`` streams the same AgentEvents as the main agent, so the TUI and
    the orchestrator handle sub-agents exactly like the main loop.
    """

    def __init__(
        self,
        role: SubAgentRole,
        agent: Any,
    ) -> None:
        self.role = role
        self.agent = agent  # ReActAgent

    @property
    def label(self) -> str:
        return ROLE_LABELS[self.role]

    @property
    def name(self) -> str:
        return self.agent.name if hasattr(self.agent, "name") else self.role.value

    @property
    def registry(self) -> ToolRegistry:
        return self.agent.registry

    def run(self, task: str, **kwargs: Any) -> AsyncIterator[Any]:
        return self.agent.run(task, **kwargs)

    async def ask(self, task: str, **kwargs: Any) -> str:
        return await self.agent.ask(task, **kwargs)

    def __repr__(self) -> str:
        return f"<SpecializedAgent role={self.role.value} tools={len(self.registry)}>"


def filter_registry(
    registry: ToolRegistry,
    allowed: set[str],
) -> ToolRegistry:
    """
    Return a new ToolRegistry containing only the tools whose names are in
    *allowed*. Missing tools are silently ignored so a role never breaks when
    e.g. the retrieval tools are absent (no code index).
    """
    filtered = ToolRegistry()
    for tool in registry.tools:
        if tool.name in allowed:
            filtered.register(tool)
    return filtered


def build_sub_agent(
    role: SubAgentRole,
    provider: Any,
    registry: ToolRegistry,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    max_iterations: int = 50,
    max_tool_calls: int = 200,
    context_budget_tokens: int = 12_000,
    memory_provider=None,
    memory_writer=None,
    streaming: bool = True,
) -> SpecializedAgent:
    """
    Build a :class:`SpecializedAgent` for *role*.

    The role's system prompt is prepended with an identity line naming the
    role, and the registry is filtered to the role's tool subset.
    """
    from tracera.agent.react_loop import ReActAgent

    filtered = filter_registry(registry, ROLE_TOOL_SETS[role])
    system_prompt = (
        f"You are the {ROLE_LABELS[role]} sub-agent of the TRACERA system.\n"
        f"{ROLE_SYSTEM_PROMPTS[role]}"
    )

    agent = ReActAgent(
        provider=provider,
        registry=filtered,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        memory_provider=memory_provider,
        memory_writer=memory_writer,
        streaming=streaming,
        context_budget_tokens=context_budget_tokens,
    )
    log.info("Built %s sub-agent (%d tools)", ROLE_LABELS[role], len(filtered))
    return SpecializedAgent(role, agent)


def build_sub_agent_fleet(
    provider: Any,
    registry: ToolRegistry,
    *,
    roles: list[SubAgentRole] | None = None,
    **kwargs: Any,
) -> dict[SubAgentRole, SpecializedAgent]:
    """
    Build one SpecializedAgent per role in *roles* (default: all five).
    Returns a dict keyed by role.
    """
    fleet: dict[SubAgentRole, SpecializedAgent] = {}
    for role in roles or list(SubAgentRole):
        fleet[role] = build_sub_agent(role, provider, registry, **kwargs)
    return fleet
