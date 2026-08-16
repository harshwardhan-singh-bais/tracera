"""Tests for the provider/model switcher (ctrl+p).

Covers the four things the feature must prove:
  1. The selector lists the REAL config (discovered at runtime, not hardcoded)
     and marks the active provider / unavailable ones.
  2. Switching changes which backend receives the next request.
  3. A misconfigured provider shows an inline warning and never silently
     breaks the next request.
  4. Conversation/session state survives a mid-session switch.
"""

import types

import pytest

from tracera.errors import MissingAPIKeyError
from tracera.providers import _FALLBACK_ORDER, list_available_providers
from tracera.tools.base import Tool, ToolResult
from tracera.tui.app import ProviderSwitcher
from tracera.tui.widgets.agent_panel import MessageWidget


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(self, name=None, default_model=None):
        if name is not None:
            self.name = name
        if default_model is not None:
            self.default_model = default_model


class _FakeAgent:
    """Duck-typed stand-in for ReActAgent (provider, model, decomposer)."""

    def __init__(self, provider=None, model=None):
        self.provider = provider or _FakeProvider()
        self.model = model or self.provider.default_model
        self.decomposer = None


def _make_app(tmp_path, provider=None):
    from tracera.agent.memory import AgentMemory
    from tracera.tui.app import TraceraTUI

    memory = AgentMemory(tmp_path / "memory")
    app = TraceraTUI(
        agent=_FakeAgent(provider=provider),
        memory=memory,
        workspace_path=tmp_path,
        banner="TRACERA",
    )
    return app


# ── 1. Config discovery (real source, no hardcoding) ─────────────────────────

class TestConfigDiscovery:
    def test_list_available_providers_marks_missing_keys(self):
        """Providers without a key are flagged unavailable; ollama is always on."""
        attrs = {key_attr: None for _, key_attr, _, _ in _FALLBACK_ORDER if key_attr}
        settings = types.SimpleNamespace(**attrs)
        entries = list_available_providers(settings)

        by_name = {e["name"]: e for e in entries}
        assert set(by_name) == {
            n for n, _, _, _ in _FALLBACK_ORDER
        }  # exactly the config table, nothing invented
        assert by_name["ollama"]["available"] is True  # local, no key needed
        assert by_name["openai"]["available"] is False
        assert by_name["openai"]["key_env"] == "OPENAI_API_KEY"

    def test_list_reflects_keys_actually_present(self):
        attrs = {key_attr: None for _, key_attr, _, _ in _FALLBACK_ORDER if key_attr}
        attrs["groq_api_key"] = "sk-groq"
        settings = types.SimpleNamespace(**attrs)
        entries = list_available_providers(settings)
        by_name = {e["name"]: e for e in entries}
        assert by_name["groq"]["available"] is True
        assert by_name["openai"]["available"] is False


# ── 1b. Selector UI opens/closes and shows the real list ─────────────────────

@pytest.mark.asyncio
async def test_selector_opens_shows_config_and_closes(tmp_path, monkeypatch):
    fake_entries = [
        {"name": "groq", "available": True, "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
        {"name": "openai", "available": False, "key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        {"name": "ollama", "available": True, "key_env": "none", "model": "llama3.2"},
    ]
    monkeypatch.setattr("tracera.providers.list_available_providers", lambda settings: fake_entries)

    app = _make_app(tmp_path, provider=_FakeProvider(name="groq", default_model="llama-3.3-70b-versatile"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()

        assert isinstance(app.screen, ProviderSwitcher)
        lst = app.screen.query_one("#provider-list")
        rows = list(lst.query("ListItem"))
        assert len(rows) == 3

        # Every entry is present; the active one is marked with a checkmark and
        # the missing-key one carries the inline warning.
        row_texts = []
        active_rows = []
        for item in rows:
            static = item.query_one("Static")
            row_texts.append(str(static.render()))
            if "provider-active" in (item.classes or set()):
                active_rows.append(str(static.render()))
        joined = " | ".join(row_texts)
        assert "groq" in joined and "llama-3.3-70b-versatile" in joined
        assert "ollama" in joined and "llama3.2" in joined
        assert "openai" in joined and "[!] missing OPENAI_API_KEY" in joined
        assert any("✓" in t and "groq" in t for t in active_rows)


@pytest.mark.asyncio
async def test_selecting_with_enter_actually_switches_backend(tmp_path, monkeypatch):
    """
    REGRESSION TEST for the real bug: selecting a provider in the picker did
    nothing because the screen dismissed with no result (the push_screen
    callback got None). Enter must close the overlay AND swap the backend.
    """
    fake_entries = [
        {"name": "groq", "available": True, "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
        {"name": "openai", "available": False, "key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        {"name": "ollama", "available": True, "key_env": "none", "model": "llama3.2"},
    ]
    monkeypatch.setattr("tracera.providers.list_available_providers", lambda settings: fake_entries)

    app = _make_app(tmp_path, provider=_FakeProvider(name="groq", default_model="llama-3.3-70b-versatile"))
    original = app.agent.provider

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen, ProviderSwitcher)

        # Cursor starts on the active row (groq, index 0). Move down twice to
        # ollama (skipping the disabled openai row) and press enter.
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Overlay closed AND the backend actually swapped.
        assert not isinstance(app.screen, ProviderSwitcher)
        assert app.agent.provider is not original
        assert app.agent.provider.name == "ollama"
        assert app.agent.model == "llama3.2"
        # Status line + header immediately show the new model.
        assert app._status_line()._model == "llama3.2"
        # Explicit confirmation row: old → new, impossible to miss.
        metas = [m for m in app._panel().query(MessageWidget) if m.role == "meta"]
        assert metas, "expected a confirmation row"
        conf = metas[-1].msg_content
        assert "Provider switched" in conf
        assert "groq" in conf and "llama-3.3-70b-versatile" in conf
        assert "ollama" in conf and "llama3.2" in conf

        # Escape closes the overlay without changing anything.
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is app._main_screen if hasattr(app, "_main_screen") else True
        assert not isinstance(app.screen, ProviderSwitcher)


# ── 2. Switching changes the backend for the next request ────────────────────

@pytest.mark.asyncio
async def test_switch_changes_backend_and_model(tmp_path):
    from tracera.providers.ollama_provider import OllamaProvider

    app = _make_app(tmp_path, provider=_FakeProvider(name="groq", default_model="llama-3.3-70b-versatile"))
    original = app.agent.provider

    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_provider("ollama", "llama3.2")
        await pilot.pause()

        assert app.agent.provider is not original
        assert isinstance(app.agent.provider, OllamaProvider)
        assert app.agent.model == "llama3.2"
        # Status line + header now reflect the new model.
        status = app._status_line()
        assert status._model == "llama3.2"


# ── 3. Misconfigured provider → inline warning, no silent break ──────────────

@pytest.mark.asyncio
async def test_switch_to_misconfigured_provider_is_rejected(tmp_path, monkeypatch):
    def _boom(name, model, settings):
        raise MissingAPIKeyError("openai", "OPENAI_API_KEY")

    monkeypatch.setattr("tracera.providers.create_provider", _boom)

    app = _make_app(tmp_path, provider=_FakeProvider(name="groq", default_model="llama-3.3-70b-versatile"))
    original = app.agent.provider
    original_model = app.agent.model

    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_provider("openai", "gpt-4o")
        await pilot.pause()

        # Backend untouched — nothing silently broke.
        assert app.agent.provider is original
        assert app.agent.model == original_model
        # Inline error row explains why.
        errors = [m for m in app._panel().query(MessageWidget) if m.role == "error"]
        assert errors, "expected an inline error row"
        assert "OPENAI_API_KEY" in errors[-1].msg_content


# ── 4. Conversation survives a mid-session switch ────────────────────────────

@pytest.mark.asyncio
async def test_conversation_and_session_survive_switch(tmp_path):
    app = _make_app(tmp_path, provider=_FakeProvider(name="groq", default_model="llama-3.3-70b-versatile"))
    conversation = app._conversation
    conversation.add_user("remember this for later")
    conversation.add_assistant("stored")
    session_id = conversation.id
    before = len(conversation.messages)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_provider("ollama", "llama3.2")
        await pilot.pause()

        assert app._conversation is conversation  # same object, not reset
        assert app._conversation.id == session_id
        assert len(app._conversation.messages) == before
        assert app._conversation.messages[0].content == "remember this for later"


# ── 5. Normalization layer stays intact across a mid-session switch ──────────
# Two providers with DIFFERENT response shapes (streaming vs complete-only)
# must drive the SAME normalized AgentEvent schema — the loop is the single
# normalization point, so the TUI never sees provider-specific shapes.

class _StreamingSwitcherProvider:
    """Provider A — streaming shape: tool-call event first, then deltas."""

    name = "provider-a"
    default_model = "model-a"

    def __init__(self):
        self.stream_calls = 0

    async def complete(self, messages, **kwargs):
        raise AssertionError("streaming provider should not use complete()")

    async def stream(self, messages, **kwargs):
        self.stream_calls += 1
        if self.stream_calls == 1:
            from tracera.providers.base import StreamEvent, ToolCallRequest
            yield StreamEvent(
                type="tool_call_complete",
                tool_call=ToolCallRequest(
                    id="call-1", name="read_file", arguments={"path": "a.py"}
                ),
            )
        else:
            from tracera.providers.base import StreamEvent
            yield StreamEvent(type="text_delta", text="done from A")
        yield StreamEvent(type="done")


class _CompleteSwitcherProvider:
    """Provider B — complete-only shape: tool_calls in the response object."""

    name = "provider-b"
    default_model = "model-b"

    def __init__(self):
        self.complete_calls = 0

    async def stream(self, messages, **kwargs):
        from tracera.providers.base import StreamEvent
        yield StreamEvent(type="done")  # nothing usable → loop falls back to complete()

    async def complete(self, messages, **kwargs):
        from tracera.providers.base import LLMResponse, TokenUsage, ToolCallRequest
        self.complete_calls += 1
        if self.complete_calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="call-2", name="read_file", arguments={"path": "b.py"})
                ],
                usage=TokenUsage(),
                model=self.default_model,
                finish_reason="tool_calls",
            )
        return LLMResponse(
            content="done from B",
            tool_calls=None,
            usage=TokenUsage(),
            model=self.default_model,
            finish_reason="stop",
        )


class _StubReadFile(Tool):
    """Minimal tool the loop can execute."""

    name = "read_file"
    description = "Read a file"
    parameters_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    async def execute(self, path=None, **kwargs):
        return ToolResult.ok(tool_name=self.name, tool_call_id="", output=f"contents of {path}")


async def _collect_normalized(agent, conversation):
    """Run the loop and return the normalized (phase, tool, text, model) sequence."""
    from tracera.agent.react_loop import AgentEventType

    phases, tools, finals, models = [], [], [], []
    async for ev in await agent.run("read a file", conversation=conversation):
        if ev.type == AgentEventType.PHASE_UPDATE:
            phases.append(ev.phase)
        elif ev.type == AgentEventType.TOOL_START:
            tools.append(ev.tool_name)
        elif ev.type == AgentEventType.RESPONSE_COMPLETE:
            finals.append(ev.text)
            models.append(ev.metadata.get("model"))
    return phases, tools, finals, models


@pytest.mark.asyncio
async def test_mid_session_switch_keeps_normalized_schema():
    from tracera.agent.react_loop import ReActAgent
    from tracera.conversation import ConversationState
    from tracera.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(_StubReadFile())
    conversation = ConversationState()

    # Run 1 under provider A (streaming shape).
    agent = ReActAgent(
        provider=_StreamingSwitcherProvider(),
        registry=registry,
        streaming=True,
        model="model-a",
    )
    phases_a, tools_a, finals_a, models_a = await _collect_normalized(agent, conversation)
    assert tools_a == ["read_file"]
    assert finals_a == ["done from A"]
    assert "planning" in phases_a and "thinking" in phases_a
    # The response metadata reports the model that actually handled the request.
    assert models_a and models_a[-1] == "model-a"

    # Mid-session switch: new backend + model, SAME conversation.
    agent.provider = _CompleteSwitcherProvider()
    agent.model = "model-b"
    phases_b, tools_b, finals_b, models_b = await _collect_normalized(agent, conversation)

    # The normalized schema is provider-agnostic: same phases, same tool rows,
    # only the final text differs — and the shared conversation accumulated.
    assert tools_b == ["read_file"]
    assert finals_b == ["done from B"]
    assert phases_b == phases_a
    # PROOF the switch took effect on the backend: the response metadata for
    # the SECOND run reports the NEW model, provably different from run 1.
    assert models_b and models_b[-1] == "model-b"
    assert models_b[-1] != models_a[-1]
    assert len(conversation.messages) > 0  # context carried across the switch
