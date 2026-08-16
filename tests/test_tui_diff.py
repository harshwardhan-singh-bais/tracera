"""Tests for the v3 TUI layer: diff summaries and normalized phase events."""

import pytest

from tracera.agent.react_loop import AGENT_PHASES, AgentEventType, ReActAgent
from tracera.providers.base import (
    LLMResponse,
    StreamEvent,
    TokenUsage,
    ToolCallRequest,
)
from tracera.tools.base import Tool, ToolResult
from tracera.tools.registry import ToolRegistry
from tracera.tui.diffutil import (
    DIFFABLE_TOOLS,
    MAX_DIFF_BYTES,
    compute_diff,
    is_image,
)


# ── diffutil ──────────────────────────────────────────────────────────────────

class TestComputeDiff:
    def test_insert_only(self):
        lines, added, removed = compute_diff("a\nb\n", "a\nb\nc\n", "f.py")
        assert added == 1 and removed == 0
        assert any(k == "add" and v == "c" for k, v in lines)
        assert lines[0][0] == "hunk" and "f.py" in lines[0][1]

    def test_replace_counts_both_sides(self):
        lines, added, removed = compute_diff(
            "def foo():\n    return 1\n",
            "def foo():\n    return 42\n",
            "f.py",
        )
        assert added == 1 and removed == 1
        kinds = {k for k, _ in lines}
        assert {"add", "del", "ctx"}.issubset(kinds)

    def test_no_change_returns_empty(self):
        lines, added, removed = compute_diff("same\n", "same\n", "f.py")
        assert lines == [] and added == 0 and removed == 0

    def test_context_trimmed_leading(self):
        big = "".join(f"line {i}\n" for i in range(50))
        lines, _, _ = compute_diff(big, big + "tail\n", "f.py")
        ctx = [t for k, t in lines if k == "ctx"]
        assert len(ctx) == 3  # only the 3 lines before the insertion
        assert not any(k == "ellipsis" for k, _ in lines)

    def test_context_trimmed_middle_with_ellipsis(self):
        big = "".join(f"line {i}\n" for i in range(50))
        middle = big.replace("line 25\n", "line 25a\n")
        lines, _, _ = compute_diff(big, middle, "f.py")
        assert any(k == "ellipsis" for k, _ in lines)


class TestHelpers:
    def test_is_image(self):
        assert is_image("a.png") and is_image("B.JPG") and is_image("x.webp")
        assert not is_image("a.py") and not is_image("a.txt")

    def test_diffable_tools(self):
        assert DIFFABLE_TOOLS == ("edit_file", "write_file")
        assert MAX_DIFF_BYTES > 0


# ── Phase events from the agent loop ──────────────────────────────────────────

class _StubTool(Tool):
    name = "read_file"
    description = "Read a file"
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, path=None, **kwargs):
        return ToolResult.ok(tool_name=self.name, tool_call_id="", output=f"contents of {path}")


class _ToolCallingProvider:
    """Provider that asks for a tool call, then finishes with text."""

    name = "fake-phase"
    default_model = "fake-model"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="t1", name="read_file", arguments={"path": "a.py"})
                ],
                usage=TokenUsage(),
                model=self.default_model,
                finish_reason="tool_calls",
            )
        return LLMResponse(
            content="All done.",
            tool_calls=None,
            usage=TokenUsage(),
            model=self.default_model,
            finish_reason="stop",
        )

    async def stream(self, messages, **kwargs):
        yield StreamEvent(type="done")


def _make_agent() -> ReActAgent:
    registry = ToolRegistry()
    registry.register(_StubTool())
    return ReActAgent(provider=_ToolCallingProvider(), registry=registry, streaming=False)


async def _collect_phases(agent: ReActAgent) -> list[str]:
    phases: list[str] = []
    async for event in await agent.run("do the task"):
        if event.type == AgentEventType.PHASE_UPDATE:
            assert event.phase in AGENT_PHASES
            phases.append(event.phase)
    return phases


class TestPhaseEvents:
    async def test_emits_normalized_phase_sequence(self):
        phases = await _collect_phases(_make_agent())
        # planning → think → run (tool) → think → generate (final text)
        assert phases == ["planning", "thinking", "running", "thinking", "generating"]

    async def test_single_turn_no_tools(self):
        class _TextOnly(_ToolCallingProvider):
            async def complete(self, messages, **kwargs):
                return LLMResponse(
                    content="Hi.",
                    tool_calls=None,
                    usage=TokenUsage(),
                    model=self.default_model,
                    finish_reason="stop",
                )

        agent = ReActAgent(
            provider=_TextOnly(), registry=ToolRegistry(), streaming=False
        )
        phases = await _collect_phases(agent)
        assert phases == ["planning", "thinking", "generating"]

    async def test_phase_field_defaults_to_none(self):
        from tracera.agent.react_loop import AgentEvent

        ev = AgentEvent(type=AgentEventType.THINKING)
        assert ev.phase is None
