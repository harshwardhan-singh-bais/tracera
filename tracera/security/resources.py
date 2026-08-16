"""
Phase 55 — Resource limits.

Enforces the resource ceilings that keep an agent run bounded:

    - max_iterations  — cap on ReAct loop iterations
    - max_tool_calls  — cap on total tool executions
    - max_context_tokens — cap on context sent to the LLM
    - command_timeout — cap on individual shell commands
    - indexing limits — max file size / files indexed

The settings layer already declares these (tracera/config/settings.py);
this module centralises the checks so any component (ReAct loop, MCP
client, TUI) can consult one source of truth. The ReAct loop already
enforces iterations/tool-calls; ResourceMonitor adds counters + a
single query point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tracera.logging import get_logger

log = get_logger("security.resources")


@dataclass
class ResourceSnapshot:
    """Current usage against configured limits."""

    iterations: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    max_iterations: int = 0
    max_tool_calls: int = 0
    max_context_tokens: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def exceeded(self) -> list[str]:
        """Names of limits that have been exceeded (empty = all fine)."""
        violations: list[str] = []
        if self.max_iterations and self.iterations >= self.max_iterations:
            violations.append("max_iterations")
        if self.max_tool_calls and self.tool_calls >= self.max_tool_calls:
            violations.append("max_tool_calls")
        if self.max_context_tokens and self.total_tokens >= self.max_context_tokens:
            violations.append("max_context_tokens")
        return violations

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_context_tokens": self.max_context_tokens,
            "elapsed_seconds": round(time.time() - self.started_at, 2),
            "exceeded": self.exceeded(),
        }


class ResourceMonitor:
    """
    Tracks agent-run resource usage against the configured ceilings.

        monitor = ResourceMonitor(settings)
        monitor.record_iteration()
        monitor.record_tool_call()
        monitor.record_tokens(prompt_tokens, completion_tokens)
        if monitor.check_iterations():  # exceeded → stop the loop
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        max_tool_calls: int = 200,
        max_context_tokens: int = 128_000,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_context_tokens = max_context_tokens
        self._iterations = 0
        self._tool_calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    @classmethod
    def from_settings(cls, settings) -> "ResourceMonitor":
        return cls(
            max_iterations=settings.tracera_max_iterations,
            max_tool_calls=settings.tracera_max_tool_calls,
            max_context_tokens=settings.tracera_max_context_tokens,
        )

    # ── Recording ────────────────────────────────────────────────────────────

    def record_iteration(self) -> None:
        self._iterations += 1

    def record_tool_call(self) -> None:
        self._tool_calls += 1

    def record_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out

    # ── Checks ───────────────────────────────────────────────────────────────

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def total_tokens(self) -> int:
        return self._tokens_in + self._tokens_out

    def check_iterations(self) -> bool:
        return self.max_iterations > 0 and self._iterations >= self.max_iterations

    def check_tool_calls(self) -> bool:
        return self.max_tool_calls > 0 and self._tool_calls >= self.max_tool_calls

    def check_context(self) -> bool:
        return self.max_context_tokens > 0 and self.total_tokens >= self.max_context_tokens

    def any_exceeded(self) -> bool:
        return self.check_iterations() or self.check_tool_calls() or self.check_context()

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            iterations=self._iterations,
            tool_calls=self._tool_calls,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            max_context_tokens=self.max_context_tokens,
        )

    def reset(self) -> None:
        self._iterations = 0
        self._tool_calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    def __repr__(self) -> str:
        return (
            f"<ResourceMonitor iter={self._iterations}/{self.max_iterations} "
            f"tools={self._tool_calls}/{self.max_tool_calls} "
            f"tokens={self.total_tokens}/{self.max_context_tokens}>"
        )
