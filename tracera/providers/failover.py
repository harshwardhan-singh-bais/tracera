"""
Provider failover — automatic fallback between LLM providers.

When a provider call fails (rate limit, auth error, overload, context too
large, network error), FailoverProvider transparently retries the request on
the next available provider in ranked order — e.g. Groq → OpenAI → Gemini →
Ollama. The agent never sees the transient failure and keeps working.

Permanent failures (bad key, payment required, unknown model) raise
ProviderUnavailableError and the provider is marked dead for the rest of the
session — re-trying a 404 model or a 402 account on every LLM call is
pointless and only makes each call burn through the whole chain.

Used by ``_build_agent`` when more than one provider key is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from tracera.errors import ProviderError, ProviderUnavailableError
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

    Providers that fail permanently (ProviderUnavailableError — bad key,
    payment required, model not found) are marked dead and skipped for the
    rest of the session.
    """

    def __init__(self, providers: list[LLMProvider], *, delay: float = _FALLBACK_DELAY_SECONDS) -> None:
        if not providers:
            raise ProviderError("FailoverProvider requires at least one provider.")
        self._providers = list(providers)
        self._delay = delay
        self._active_index = 0
        #: Indices of providers that failed permanently — skipped from now on.
        self._dead: set[int] = set()
        #: provider name → last error, for the aggregate failure message.
        self._errors: dict[str, str] = {}
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

    @property
    def dead_providers(self) -> list[str]:
        return [self._providers[i].name for i in sorted(self._dead)]

    def clear_failures(self) -> None:
        """Forget permanently-failed providers (e.g. after fixing a key/credits)."""
        self._dead.clear()
        self._errors.clear()

    # ── Failover logic ───────────────────────────────────────────────────────

    def _candidates(self):
        """Try the current provider first, then the rest in ranked order,
        skipping providers that failed permanently."""
        n = len(self._providers)
        for offset in range(n):
            index = (self._active_index + offset) % n
            if index in self._dead:
                continue
            yield index, self._providers[index]

    def _record_success(self, index: int) -> None:
        if index != self._active_index:
            self.failover_count += 1
            log.warning(
                "Provider failover: switched to %s (failover #%d)",
                self._providers[index].name, self.failover_count,
            )
        self._active_index = index

    def _mark_dead(self, index: int, error: Exception) -> None:
        if index not in self._dead:
            self._dead.add(index)
            log.warning(
                "Provider %s marked permanently unavailable: %s",
                self._providers[index].name, str(error)[:200],
            )

    def _all_failed_error(self, last_error: Exception | None) -> ProviderError:
        lines = [f"All {len(self._providers)} provider(s) failed."]
        if last_error is not None:
            lines.append(f"Last error: {last_error}")
        if self._errors:
            lines.append("Per-provider errors:")
            for name, err in self._errors.items():
                lines.append(f"  - {name}: {err}")
        lines.append(
            "Tip: fix the failing accounts (add credits / valid keys), update "
            "model names, or run Ollama locally."
        )
        return ProviderError("\n".join(lines))

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
        candidates = list(self._candidates())
        last_error: Exception | None = None
        for offset, (index, provider) in enumerate(candidates):
            try:
                response = await provider.complete(
                    messages,
                    # Each provider in the chain is constructed with its own
                    # recommended model (see main._build_provider). A single
                    # global model string — e.g. a Groq model while the first
                    # available provider is Cerebras — must NOT be forced onto
                    # every provider: it 404s on every endpoint that doesn't
                    # serve it and the whole chain collapses. Providers fall
                    # back to the model they were configured with.
                    model=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    system=system,
                )
                self._record_success(index)
                return response
            except Exception as e:  # noqa: BLE001 — any failure triggers failover
                last_error = e
                self._errors[provider.name] = str(e)
                log.warning(
                    "Provider %s failed: %s — trying next available",
                    provider.name, str(e)[:200],
                )
                if isinstance(e, ProviderUnavailableError):
                    self._mark_dead(index, e)
                elif offset < len(candidates) - 1:
                    await asyncio.sleep(self._delay)

        raise self._all_failed_error(last_error)

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
        candidates = list(self._candidates())
        last_error: Exception | None = None
        for offset, (index, provider) in enumerate(candidates):
            try:
                async for event in provider.stream(
                    messages,
                    # See complete() — never force one global model onto
                    # providers that weren't configured to serve it.
                    model=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    system=system,
                ):
                    yield event
                self._record_success(index)
                return
            except Exception as e:  # noqa: BLE001
                last_error = e
                self._errors[provider.name] = str(e)
                log.warning(
                    "Provider %s stream failed: %s — trying next available",
                    provider.name, str(e)[:200],
                )
                if isinstance(e, ProviderUnavailableError):
                    self._mark_dead(index, e)
                elif offset < len(candidates) - 1:
                    await asyncio.sleep(self._delay)

        raise self._all_failed_error(last_error)

    def __repr__(self) -> str:
        dead = [self._providers[i].name for i in self._dead]
        suffix = f" dead={dead}" if dead else ""
        return (
            f"<FailoverProvider active={self.name} "
            f"providers={[p.name for p in self._providers]}{suffix}>"
        )
