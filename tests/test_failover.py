"""
Tests for provider failover + conversation compaction + loop error handling.

Covers the real-world failure the user hit:
  Groq 413 rate_limit_exceeded → agent should fall back to another API
  (not crash), the conversation should stay under the token budget, and
  the loop should not print a bogus "exceeded maximum iterations" error.
"""

from __future__ import annotations

import asyncio

import pytest

from tracera.errors import ProviderError, ProviderRateLimitError
from tracera.providers.base import (
    LLMMessage,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    ToolCallRequest,
)
from tracera.tools.registry import ToolRegistry


class _FakeProvider:
    """Configurable fake provider — fails N times, then returns a response."""

    def __init__(self, name: str, fail_complete: bool = False,
                 fail_stream: bool = False, fail_always: bool = False,
                 response_text: str = "ok"):
        self.name = name
        self.default_model = f"{name}-model"
        self.fail_complete = fail_complete
        self.fail_stream = fail_stream
        self.fail_always = fail_always
        self.response_text = response_text
        self.complete_calls = 0
        self.stream_calls = 0
        self.last_message_count = 0

    async def complete(self, messages, **kwargs):
        self.complete_calls += 1
        self.last_message_count = len(messages)
        if self.fail_always or self.fail_complete:
            raise ProviderRateLimitError(f"{self.name} rate limited")
        return LLMResponse(
            content=f"{self.name}: {self.response_text}", tool_calls=None,
            usage=TokenUsage(), model=self.default_model, finish_reason="stop",
        )

    async def stream(self, messages, **kwargs):
        self.stream_calls += 1
        self.last_message_count = len(messages)
        if self.fail_always or self.fail_stream:
            raise ProviderRateLimitError(f"{self.name} rate limited")
        yield StreamEvent(type="text_delta", text=f"{self.name}: ")
        yield StreamEvent(type="text_delta", text=self.response_text)
        yield StreamEvent(type="done")


# ── FailoverProvider ──────────────────────────────────────────────────────────

def test_failover_falls_back_on_failure():
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_complete=True)
    openai = _FakeProvider("openai")
    provider = FailoverProvider([groq, openai])

    response = asyncio.run(provider.complete([LLMMessage.user("hi")]))

    assert response.content == "openai: ok"
    assert provider.name == "openai"  # active provider updated
    assert provider.failover_count == 1
    assert groq.complete_calls == 1 and openai.complete_calls == 1


def test_failover_all_fail_raises():
    from tracera.providers.failover import FailoverProvider

    provider = FailoverProvider([
        _FakeProvider("groq", fail_always=True),
        _FakeProvider("openai", fail_always=True),
    ])
    with pytest.raises(ProviderError, match="All 2 provider"):
        asyncio.run(provider.complete([LLMMessage.user("hi")]))


def test_failover_stream_falls_back():
    from tracera.providers.failover import FailoverProvider

    provider = FailoverProvider([
        _FakeProvider("groq", fail_stream=True),
        _FakeProvider("gemini"),
    ])
    events = asyncio.run(_collect_stream(provider))
    texts = [e.text for e in events if e.type == "text_delta"]
    assert "".join(t for t in texts if t) == "gemini: ok"
    assert provider.name == "gemini"


async def _collect_stream(provider):
    events = []
    async for ev in provider.stream([LLMMessage.user("hi")]):
        events.append(ev)
    return events


def test_failover_reuses_active_provider():
    """After a failover, the next call tries the active provider first."""
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_complete=True)
    openai = _FakeProvider("openai")
    provider = FailoverProvider([groq, openai])

    asyncio.run(provider.complete([LLMMessage.user("1")]))
    assert provider.name == "openai"

    asyncio.run(provider.complete([LLMMessage.user("2")]))
    assert openai.complete_calls == 2  # openai tried first (no groq call)
    assert groq.complete_calls == 1


# ── Conversation compaction (413 prevention) ──────────────────────────────────

def test_compact_history_fits_budget():
    from tracera.conversation.state import ConversationState

    conv = ConversationState(system_prompt="sys" * 20)
    for i in range(15):
        conv.add_user(f"turn {i} question " + "x" * 200)
        conv.add_assistant("a" * 300)
    assert conv.estimated_tokens() > 200

    conv.compact_history(200)
    assert conv.estimated_tokens() <= 210  # ~ budget (heuristic slack)
    # System kept, newest turn kept, older turns dropped
    assert len(conv.messages) < 31
    assert conv.messages[0].type.name == "SYSTEM"
    assert any("turn 14" in (m.content or "") for m in conv.messages)
    assert not any("turn 0" in (m.content or "") for m in conv.messages)


def test_compact_history_keeps_tool_pairs_intact():
    from tracera.conversation.state import ConversationState, MessageType

    conv = ConversationState(system_prompt="sys")
    conv.add_user("big old turn " + "z" * 400)
    conv.add_assistant("old answer")
    conv.add_user("current task " + "y" * 300)
    tc = ToolCallRequest(id="c1", name="read_file", arguments={"path": "a.py"})
    conv.add_tool_calls([tc])
    conv.add_tool_result("c1", "read_file", "file contents " + "v" * 300)

    conv.compact_history(50)

    types = [m.type for m in conv.messages]
    assert MessageType.TOOL_CALL in types
    assert MessageType.TOOL_RESULT in types
    # The tool round is complete (call followed by its result)
    last = types[-2:]
    assert last == [MessageType.TOOL_CALL, MessageType.TOOL_RESULT]
    # Old turn dropped, current user turn kept
    assert not any("big old turn" in (m.content or "") for m in conv.messages)
    assert any("current task" in (m.content or "") for m in conv.messages)


# ── Agent loop error handling ─────────────────────────────────────────────────

def test_loop_reports_error_without_bogus_max_iterations():
    from tracera.agent.react_loop import AgentEventType, ReActAgent

    agent = ReActAgent(provider=_FakeProvider("groq", fail_always=True),
                       registry=ToolRegistry(), streaming=True)

    errors = []
    types = []
    async def _run():
        async for ev in await agent.run("hello"):
            types.append(ev.type)
            if ev.type == AgentEventType.ERROR:
                errors.append(ev.text)
    asyncio.run(_run())

    assert any("LLM call failed" in e for e in errors)
    # The bogus "exceeded maximum iterations" error must NOT appear
    assert not any("maximum iterations" in e for e in errors)
    assert types[-1] == AgentEventType.DONE


def test_loop_compacts_conversation_before_call():
    from tracera.agent.react_loop import AgentEventType, ReActAgent
    from tracera.conversation.state import ConversationState

    provider = _FakeProvider("openai")
    conv = ConversationState(system_prompt="sys")
    for i in range(10):
        conv.add_user(f"old turn {i} " + "x" * 200)
        conv.add_assistant("y" * 200)

    agent = ReActAgent(provider=provider, registry=ToolRegistry(),
                       streaming=True, context_budget_tokens=100)
    asyncio.run(_drain(agent, conv))

    # The provider saw a compacted history (not all 21 pre-run messages)
    assert provider.last_message_count < 15


async def _drain(agent, conv):
    async for _ in await agent.run("final task", conversation=conv):
        pass


# ── main._build_provider wiring ───────────────────────────────────────────────

def test_build_provider_returns_failover(monkeypatch):
    from tracera.main import _build_provider
    from tracera.providers.failover import FailoverProvider

    settings = type("S", (), {
        "tracera_default_model": "",
    })()

    created = []

    def fake_create_provider(name=None, model=None, settings=None):
        created.append(name)
        return _FakeProvider(name)

    def fake_list(settings=None):
        return [
            {"name": "groq", "available": True},
            {"name": "openai", "available": True},
            {"name": "ollama", "available": True},
            {"name": "gemini", "available": False},  # no key
        ]

    monkeypatch.setattr("tracera.providers.list_available_providers", fake_list)
    monkeypatch.setattr("tracera.providers.create_provider", fake_create_provider)

    provider = _build_provider(settings)
    assert isinstance(provider, FailoverProvider)
    assert [p.name for p in provider.providers] == ["groq", "openai", "ollama"]
    assert created == ["groq", "openai", "ollama"]  # unavailable ones skipped
