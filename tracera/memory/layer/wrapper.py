"""
Memory Layer Wrapper — the transparent interceptor installed by
``MemoryLayer.register()``.

It wraps TRACERA's unified :class:`~tracera.providers.base.LLMProvider`
interface (used by every agent and the failover chain), so no call site in the
codebase needs to be rewritten. For each call it performs, in order:

  1. **Recall** — enrich the system prompt with the entity's top-k memories.
  2. **Forward** the (possibly enriched) request to the real provider.
  3. **Return** the response to the caller immediately — no added latency.
  4. **Enqueue** the turn for async background extraction (durable queue,
     processed by :class:`~tracera.memory.layer.queue.BackgroundWorker`).

If attribution is missing, or the process is disabled by policy, the wrapper
logs a clear warning and simply forwards the call untouched.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from tracera.logging import get_logger
from tracera.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    ToolSchema,
)

log = get_logger("memory.layer.wrapper")


class MemoryProvider(LLMProvider):
    """A transparent memory-aware decorator around any ``LLMProvider``."""

    def __init__(self, inner: LLMProvider, layer: Any) -> None:
        self._inner = inner
        self._layer = layer

    # ── Delegated identity ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def default_model(self) -> str:
        return self._inner.default_model

    @property
    def supports_vision(self) -> bool:
        return getattr(self._inner, "supports_vision", False)

    @property
    def inner(self) -> LLMProvider:
        """The unwrapped provider (used by tests / failover introspection)."""
        return self._inner

    # ── complete ─────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        scope, active = self._layer.prepare()
        if active and scope is not None:
            try:
                messages, system = self._layer.inject_recall(messages, system, scope)
            except Exception as e:  # noqa: BLE001 — recall must never break the call
                log.warning("Memory recall failed, forwarding unenriched: %s", str(e)[:150])

        response = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system=system,
        )

        if active and scope is not None and response.content:
            self._layer.enqueue_turn(
                user_message=_last_user_text(messages),
                assistant_message=response.content,
                scope=scope,
                session_id=_current_session(),
            )
        return response

    # ── stream ───────────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        scope, active = self._layer.prepare()
        if active and scope is not None:
            try:
                messages, system = self._layer.inject_recall(messages, system, scope)
            except Exception:  # noqa: BLE001
                pass  # forward unenriched

        assistant_parts: list[str] = []
        async for event in self._inner.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system=system,
        ):
            if event.type == "text_delta" and getattr(event, "text", None):
                assistant_parts.append(event.text)
            yield event

        if (
            active
            and scope is not None
            and assistant_parts
            and _last_user_text(messages)
        ):
            self._layer.enqueue_turn(
                user_message=_last_user_text(messages),
                assistant_message="".join(assistant_parts),
                scope=scope,
                session_id=_current_session(),
            )

    # ── token counting ───────────────────────────────────────────────────────

    async def count_tokens(
        self, messages: list[LLMMessage], *, model: str | None = None
    ) -> int:
        return await self._inner.count_tokens(messages, model=model)

    def __repr__(self) -> str:
        return f"<MemoryProvider wrapping {self._inner!r}>"


def _last_user_text(messages: list[LLMMessage]) -> str:
    from tracera.memory.layer.recall import last_user_text

    return last_user_text(messages)


def _current_session() -> str | None:
    from tracera.memory.layer.attribution import current_session_id

    return current_session_id()