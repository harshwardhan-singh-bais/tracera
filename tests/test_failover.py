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

from tracera.errors import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
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
                 fail_permanent: bool = False,
                 response_text: str = "ok"):
        self.name = name
        self.default_model = f"{name}-model"
        self.fail_complete = fail_complete
        self.fail_stream = fail_stream
        self.fail_always = fail_always
        self.fail_permanent = fail_permanent
        self.response_text = response_text
        self.complete_calls = 0
        self.stream_calls = 0
        self.last_message_count = 0
        self.last_model: str | None = "unset"

    def _raise(self, action: str) -> None:
        if self.fail_permanent:
            raise ProviderUnavailableError(f"{self.name} permanently unavailable (model 404)")
        raise ProviderRateLimitError(f"{self.name} {action} rate limited")

    async def complete(self, messages, model=None, **kwargs):
        self.complete_calls += 1
        self.last_message_count = len(messages)
        self.last_model = model
        if self.fail_always or self.fail_complete or self.fail_permanent:
            self._raise("complete")
        return LLMResponse(
            content=f"{self.name}: {self.response_text}", tool_calls=None,
            usage=TokenUsage(), model=self.default_model, finish_reason="stop",
        )

    async def stream(self, messages, model=None, **kwargs):
        self.stream_calls += 1
        self.last_message_count = len(messages)
        self.last_model = model
        if self.fail_always or self.fail_stream or self.fail_permanent:
            self._raise("stream")
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


def test_failover_skips_permanently_failed_providers():
    """
    A provider that fails permanently (dead key/model, payment required) is
    marked dead and never re-tried on subsequent calls — otherwise every LLM
    call would re-burn through the whole chain ("this error keeps coming").
    """
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_permanent=True)
    cerebras = _FakeProvider("cerebras")
    provider = FailoverProvider([groq, cerebras])

    asyncio.run(provider.complete([LLMMessage.user("1")]))
    assert provider.name == "cerebras"
    assert provider.dead_providers == ["groq"]

    # Second call starts at cerebras and never re-tries the dead groq.
    asyncio.run(provider.complete([LLMMessage.user("2")]))
    assert groq.complete_calls == 1  # not called again
    assert cerebras.complete_calls == 2


def test_failover_reports_each_provider_error():
    """When every provider fails, the raised error lists per-provider
    reasons (payment required, model 404, ...) instead of only the last one."""
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_permanent=True)
    samba = _FakeProvider("sambanova", fail_permanent=True)
    provider = FailoverProvider([groq, samba])

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(provider.complete([LLMMessage.user("hi")]))

    msg = str(exc_info.value)
    assert "All 2 provider(s) failed" in msg
    assert "groq" in msg and "sambanova" in msg
    assert "permanently unavailable" in msg


def test_failover_stream_skips_permanently_failed_provider():
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_permanent=True)
    gemini = _FakeProvider("gemini")
    provider = FailoverProvider([groq, gemini])

    events = asyncio.run(_collect_stream(provider))
    texts = [e.text for e in events if e.type == "text_delta"]
    assert "".join(t for t in texts if t) == "gemini: ok"
    assert provider.dead_providers == ["groq"]

    asyncio.run(_collect_stream(provider))
    assert groq.stream_calls == 1  # skipped on the second call


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


async def _collect_stream(provider, model=None):
    events = []
    async for ev in provider.stream([LLMMessage.user("hi")], model=model):
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


def test_failover_never_forwards_global_model_to_providers():
    """
    A single global model string (e.g. a Groq model like llama-3.3-70b-versatile)
    must NOT be forced onto every provider in the chain — it would 404 on
    every endpoint that doesn't serve it and the whole chain would fail.
    Each provider uses the model it was configured with.
    """
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq")
    cerebras = _FakeProvider("cerebras")
    provider = FailoverProvider([groq, cerebras])

    response = asyncio.run(provider.complete(
        [LLMMessage.user("hi")], model="llama-3.3-70b-versatile"
    ))

    assert response.content == "groq: ok"
    assert groq.last_model is None  # provider used its own default_model
    assert cerebras.complete_calls == 0  # first provider succeeded


def test_failover_fallback_provider_uses_own_model():
    """After a failure, the fallback provider must not inherit the caller's
    global model either — it falls back to its configured default."""
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_complete=True)
    cerebras = _FakeProvider("cerebras")
    provider = FailoverProvider([groq, cerebras])

    response = asyncio.run(provider.complete(
        [LLMMessage.user("hi")], model="llama-3.3-70b-versatile"
    ))

    assert response.content == "cerebras: ok"
    assert groq.last_model is None
    assert cerebras.last_model is None
    assert cerebras.complete_calls == 1


def test_failover_stream_never_forwards_global_model():
    """The streaming path must drop the global model the same way."""
    from tracera.providers.failover import FailoverProvider

    groq = _FakeProvider("groq", fail_stream=True)
    gemini = _FakeProvider("gemini")
    provider = FailoverProvider([groq, gemini])

    events = asyncio.run(_collect_stream(provider, model="llama-3.3-70b-versatile"))
    texts = [e.text for e in events if e.type == "text_delta"]
    assert "".join(t for t in texts if t) == "gemini: ok"
    assert groq.last_model is None
    assert gemini.last_model is None


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


def test_build_provider_does_not_apply_wrong_default_model(monkeypatch):
    """
    auto mode with a Groq default model must not push that model onto the
    first available non-Groq provider — each provider gets its own model
    (regression: "All 7 provider(s) failed ... model_not_found
    llama-3.3-70b-versatile" on every endpoint).
    """
    from tracera.main import _build_provider

    settings = type("S", (), {
        "tracera_default_provider": "auto",
        "tracera_default_model": "openai/gpt-oss-120b",  # a Groq model
    })()

    created = {}

    def fake_create_provider(name=None, model=None, settings=None):
        created[name] = model
        return _FakeProvider(name)

    def fake_list(settings=None):
        return [
            {"name": "cerebras", "available": True},  # first available, NOT groq
            {"name": "groq", "available": True},
        ]

    monkeypatch.setattr("tracera.providers.list_available_providers", fake_list)
    monkeypatch.setattr("tracera.providers.create_provider", fake_create_provider)

    _build_provider(settings)

    # Cerebras must get its own recommended model, not the Groq default.
    assert created["cerebras"] == "gpt-oss-120b"
    # Groq (the model's owner) keeps the recommended Groq model.
    assert created["groq"] == "openai/gpt-oss-120b"


def test_build_provider_explicit_provider_honours_default_model(monkeypatch):
    """When the user explicitly configures a default provider (non-auto),
    that provider still receives TRACERA_DEFAULT_MODEL."""
    from tracera.main import _build_provider

    settings = type("S", (), {
        "tracera_default_provider": "groq",
        "tracera_default_model": "llama-3.3-70b-versatile",
    })()

    created = {}

    def fake_create_provider(name=None, model=None, settings=None):
        created[name] = model
        return _FakeProvider(name)

    def fake_list(settings=None):
        return [
            {"name": "groq", "available": True},
            {"name": "openai", "available": True},
        ]

    monkeypatch.setattr("tracera.providers.list_available_providers", fake_list)
    monkeypatch.setattr("tracera.providers.create_provider", fake_create_provider)

    _build_provider(settings)

    # Explicit default provider → model="" → create_provider uses the default.
    assert created["groq"] == ""
    assert created["openai"] == "gpt-4o"  # fallback keeps its own model
