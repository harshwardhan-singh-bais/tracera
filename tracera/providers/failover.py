"""
Provider failover — automatic fallback between LLM providers.

When a provider call fails (rate limit, auth error, overload, context too
large, network error), FailoverProvider transparently retries the request on
the next available provider in ranked order — e.g. Groq → OpenAI → Gemini →
Ollama. The agent never sees the transient failure and keeps working.

Used by ``_build_agent`` when more than one provider key is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from tracera.errors import ProviderError
from tracera.logging import get_logger
from tracera.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    StreamEvent,
    ToolSchema,
)

log = get_logger("providers.failover")

_FALLBACK_DELAY_SECONDS = 0.5


class FailoverProvider(LLMProvider):
    """
    Wraps a ranked list of providers and fails over between them.

    The first provider in the list is preferred. When a call raises (any
    exception — rate limit, auth, overload), the next provider is tried with
    a short delay. The last provider to succeed becomes the new active one,
    so subsequent calls reuse it without re-trying earlier failures first.
    """

    def __init__(self, providers: list[LLMProvider], *, delay: float = _FALLBACK_DELAY_SECONDS) -> None:
        if not providers:
            raise ProviderError("FailoverProvider requires at least one provider.")
        self._providers = list(providers)
        self._delay = delay
        self._active_index = 0
        self.failover_count = 0

    # ── Active provider ──────────────────────────────────────────────────────

    @property
    def active_provider(self) -> LLMProvider:
        return self._providers[self._active_index]

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    @property
    def name(self) -> str:
        return self.active_provider.name

    @property
    def default_model(self) -> str:
        return self.active_provider.default_model

    # ── Failover logic ───────────────────────────────────────────────────────

    def _candidates(self):
        """Try the current provider first, then the rest in ranked order."""
        n = len(self._providers)
        for offset in range(n):
            yield self._providers[(self._active_index + offset) % n]

    def _record_success(self, index_offset: int) -> None:
        if index_offset > 0:
            self.failover_count += 1
            log.warning(
                "Provider failover: switched to %s (failover #%d)",
                self.active_provider.name, self.failover_count,
            )
        self._active_index = (self._active_index + index_offset) % len(self._providers)

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
        last_error: Exception | None = None
        for offset, provider in enumerate(self._candidates()):
            try:
                response = await provider.complete(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    system=system,
                )
                self._record_success(offset)
                return response
            except Exception as e:  # noqa: BLE001 — any failure triggers failover
                last_error = e
                log.warning(
                    "Provider %s failed: %s — trying next available",
                    provider.name, str(e)[:200],
                )
                if offset < len(self._providers) - 1:
                    await asyncio.sleep(self._delay)

        raise ProviderError(
            f"All {len(self._providers)} provider(s) failed. "
            f"Last error: {last_error}"
        ) from last_error

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
        last_error: Exception | None = None
        for offset, provider in enumerate(self._candidates()):
            try:
                async for event in provider.stream(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    system=system,
                ):
                    yield event
                self._record_success(offset)
                return
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning(
                    "Provider %s stream failed: %s — trying next available",
                    provider.name, str(e)[:200],
                )
                if offset < len(self._providers) - 1:
                    await asyncio.sleep(self._delay)

        raise ProviderError(
            f"All {len(self._providers)} provider(s) failed streaming. "
            f"Last error: {last_error}"
        ) from last_error

    def __repr__(self) -> str:
        return (
            f"<FailoverProvider active={self.name} "
            f"providers={[p.name for p in self._providers]}>"
        )
