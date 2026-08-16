"""
TRACERA LLM Provider Abstraction — Phase 4.

Defines the unified LLMProvider interface and all message/response types.
Every provider adapter must implement this interface.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Literal


# ── Message types ─────────────────────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCallRequest:
    """A single tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    """
    A single message in the conversation.
    Supports all roles including tool calls and tool results.
    """
    role: Role
    content: str | None = None
    tool_calls: list[ToolCallRequest] | None = None   # for assistant → tool
    tool_call_id: str | None = None                   # for tool → assistant
    tool_name: str | None = None                      # for tool results
    name: str | None = None                           # optional display name

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "LLMMessage":
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def assistant_tool_calls(cls, tool_calls: list[ToolCallRequest]) -> "LLMMessage":
        return cls(role=Role.ASSISTANT, content=None, tool_calls=tool_calls)

    @classmethod
    def tool_result(
        cls, tool_call_id: str, tool_name: str, content: str
    ) -> "LLMMessage":
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    def __repr__(self) -> str:
        preview = (self.content or "")[:60].replace("\n", "↵")
        if self.tool_calls:
            tools = [t.name for t in self.tool_calls]
            preview = f"[tool_calls: {tools}]"
        return f"<LLMMessage role={self.role.value} {preview!r}>"


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cost_estimate(self) -> float | None:
        """Rough cost estimate — None if pricing unknown."""
        return None


@dataclass
class LLMResponse:
    """The complete response from an LLM call."""
    content: str | None
    tool_calls: list[ToolCallRequest] | None
    usage: TokenUsage
    model: str
    finish_reason: str
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_complete(self) -> bool:
        return self.finish_reason in ("stop", "end_turn", "max_tokens", "length")

    def as_message(self) -> LLMMessage:
        """Convert this response into an LLMMessage for the conversation."""
        if self.has_tool_calls:
            return LLMMessage.assistant_tool_calls(self.tool_calls)  # type: ignore[arg-type]
        return LLMMessage.assistant(self.content or "")


# ── Streaming events ──────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    """A single event in a streaming LLM response."""
    type: Literal["text_delta", "tool_call_delta", "tool_call_complete", "usage", "done"]
    text: str | None = None
    tool_call: ToolCallRequest | None = None
    usage: TokenUsage | None = None


# ── Tool schema for LLM ───────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    """JSON schema representation of a tool for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema object

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_gemini_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ── Provider interface ────────────────────────────────────────────────────────

class LLMProvider(abc.ABC):
    """
    Abstract base for all TRACERA LLM providers.

    Implementations must be safe to call concurrently from async code.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'openai', 'anthropic'."""

    @property
    @abc.abstractmethod
    def default_model(self) -> str:
        """Default model ID for this provider."""

    #: Whether the provider's models accept image inputs. Providers that
    #: support vision set this to True; the TUI shows a badge on image
    #: attachments when the active provider cannot see them.
    supports_vision: bool = False

    @abc.abstractmethod
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
        """
        Send messages and return a complete LLMResponse.
        This is the primary method all agents use.
        """

    @abc.abstractmethod
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
        """
        Stream the response as StreamEvents.
        Must be an async generator.
        """
        yield StreamEvent(type="done")  # satisfies type checker

    async def count_tokens(
        self, messages: list[LLMMessage], *, model: str | None = None
    ) -> int:
        """
        Estimate token count for the given messages.
        Default implementation uses a rough char-based heuristic.
        """
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // 4  # ~4 chars per token

    async def health_check(self) -> bool:
        """
        Ping the provider to verify credentials and connectivity.
        Returns True on success.
        """
        try:
            response = await self.complete(
                [LLMMessage.user("ping")],
                max_tokens=5,
                temperature=0.0,
            )
            return response.content is not None or response.has_tool_calls
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} model={self.default_model}>"
