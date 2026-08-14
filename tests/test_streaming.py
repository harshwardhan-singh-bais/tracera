"""Tests for Phase 4/8 streaming — RESPONSE_DELTA emission and complete() fallback."""

import pytest

from tracera.providers.base import LLMResponse, StreamEvent, TokenUsage, ToolCallRequest
from tracera.tools.base import Tool, ToolResult
from tracera.tools.registry import ToolRegistry


class _StreamingProvider:
    """Provider with real streaming support (like the OpenAI adapters)."""

    name = "fake-stream"
    default_model = "fake-model"

    def __init__(self, deltas=None, complete_text="complete-text"):
        self.deltas = deltas or ["Hello ", "world"]
        self.complete_text = complete_text
        self.complete_calls = 0

    async def complete(self, messages, **kwargs):
        self.complete_calls += 1
        return LLMResponse(
            content=self.complete_text, tool_calls=None,
            usage=TokenUsage(), model=self.default_model, finish_reason="stop",
        )

    async def stream(self, messages, **kwargs):
        for d in self.deltas:
            yield StreamEvent(type="text_delta", text=d)
        yield StreamEvent(type="done")


class _NoStreamProvider(_StreamingProvider):
    """Provider whose stream yields nothing usable (base-class behaviour)."""

    async def stream(self, messages, **kwargs):
        yield StreamEvent(type="done")


class _StubReadFile(Tool):
    """Registered stub so tool-call streaming can be exercised."""

    name = "read_file"
    description = "Read a file"
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, path=None, **kwargs):
        return ToolResult.ok(tool_name=self.name, tool_call_id="", output=f"contents of {path}")


def _registry_with_read_file() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_StubReadFile())
    return registry


@pytest.mark.asyncio
async def test_agent_emits_response_deltas():
    from tracera.agent.react_loop import AgentEventType, ReActAgent

    provider = _StreamingProvider()
    agent = ReActAgent(provider=provider, registry=ToolRegistry(), streaming=True)

    deltas = []
    final = None
    async for ev in await agent.run("hello"):
        if ev.type == AgentEventType.RESPONSE_DELTA:
            deltas.append(ev.text)
        elif ev.type == AgentEventType.RESPONSE_COMPLETE:
            final = ev.text

    assert deltas == ["Hello ", "world"]
    assert final == "Hello world"
    assert provider.complete_calls == 0  # stream was used, not complete()


@pytest.mark.asyncio
async def test_agent_falls_back_to_complete_without_streaming():
    from tracera.agent.react_loop import AgentEventType, ReActAgent

    provider = _NoStreamProvider()
    agent = ReActAgent(provider=provider, registry=ToolRegistry(), streaming=True)

    final = None
    async for ev in await agent.run("hi"):
        if ev.type == AgentEventType.RESPONSE_COMPLETE:
            final = ev.text

    assert final == "complete-text"
    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_agent_no_streaming_mode_uses_complete():
    from tracera.agent.react_loop import AgentEventType, ReActAgent

    provider = _StreamingProvider()
    agent = ReActAgent(provider=provider, registry=ToolRegistry(), streaming=False)

    deltas = 0
    final = None
    async for ev in await agent.run("hi"):
        if ev.type == AgentEventType.RESPONSE_DELTA:
            deltas += 1
        elif ev.type == AgentEventType.RESPONSE_COMPLETE:
            final = ev.text

    assert deltas == 0
    assert final == "complete-text"
    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_streaming_tool_calls_are_collected():
    from tracera.agent.react_loop import AgentEventType, ReActAgent

    class _ToolStreamProvider(_StreamingProvider):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0

        async def stream(self, messages, **kwargs):
            self.stream_calls += 1
            if self.stream_calls == 1:
                # First response: request a tool call
                yield StreamEvent(
                    type="tool_call_complete",
                    tool_call=ToolCallRequest(
                        id="call-1", name="read_file", arguments={"path": "a.py"},
                    ),
                )
            else:
                # After the tool result is in the conversation: final answer
                yield StreamEvent(type="text_delta", text="done reading")
            yield StreamEvent(type="done")

    provider = _ToolStreamProvider()
    agent = ReActAgent(provider=provider, registry=_registry_with_read_file(), streaming=True)

    tool_starts = []
    final = None
    async for ev in await agent.run("read a.py"):
        if ev.type == AgentEventType.TOOL_START:
            tool_starts.append(ev.tool_name)
        elif ev.type == AgentEventType.RESPONSE_COMPLETE:
            final = ev.text

    assert tool_starts == ["read_file"]
    assert final == "done reading"
