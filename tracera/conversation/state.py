"""
TRACERA Conversation State — Phase 5.

Provider-neutral conversation history management.
Supports all message types: user, assistant, tool calls, tool results, errors.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from tracera.providers.base import LLMMessage, Role, ToolCallRequest


# ── Message envelope ──────────────────────────────────────────────────────────

class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    SYSTEM = "system"
    PLANNING = "planning"


@dataclass
class ConversationMessage:
    """
    A typed, timestamped message envelope wrapping an LLMMessage.
    """
    id: str
    type: MessageType
    message: LLMMessage
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, content: str) -> "ConversationMessage":
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.USER,
            message=LLMMessage.user(content),
        )

    @classmethod
    def system(cls, content: str) -> "ConversationMessage":
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM,
            message=LLMMessage.system(content),
        )

    @classmethod
    def assistant_text(cls, content: str, **metadata: Any) -> "ConversationMessage":
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.ASSISTANT,
            message=LLMMessage.assistant(content),
            metadata=metadata,
        )

    @classmethod
    def assistant_tool_calls(
        cls, tool_calls: list[ToolCallRequest], **metadata: Any
    ) -> "ConversationMessage":
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.TOOL_CALL,
            message=LLMMessage.assistant_tool_calls(tool_calls),
            metadata=metadata,
        )

    @classmethod
    def tool_result(
        cls, tool_call_id: str, tool_name: str, content: str, **metadata: Any
    ) -> "ConversationMessage":
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.TOOL_RESULT,
            message=LLMMessage.tool_result(tool_call_id, tool_name, content),
            metadata={"tool_name": tool_name, **metadata},
        )

    @classmethod
    def error(cls, content: str, **metadata: Any) -> "ConversationMessage":
        """Record an error observation in the conversation."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            message=LLMMessage.user(f"[ERROR] {content}"),
            metadata={"error": True, **metadata},
        )

    @classmethod
    def planning(cls, content: str) -> "ConversationMessage":
        """Record a planning step (not sent to LLM, for UI display)."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.PLANNING,
            message=LLMMessage.assistant(content),
            metadata={"planning": True},
        )

    @property
    def content(self) -> str | None:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCallRequest] | None:
        return self.message.tool_calls

    def __repr__(self) -> str:
        preview = (self.content or "")[:50].replace("\n", "↵")
        return f"<ConversationMessage type={self.type.value} {preview!r}>"


# ── Conversation state ────────────────────────────────────────────────────────

@dataclass
class ConversationStats:
    """Aggregate statistics for a conversation."""
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    errors: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out


class ConversationState:
    """
    Provider-neutral conversation state container.

    Maintains the full history as ConversationMessage objects.
    Provides filtered views for LLM submission.
    Supports branching (snapshots).
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        *,
        conversation_id: str | None = None,
    ) -> None:
        self.id = conversation_id or str(uuid.uuid4())
        self._messages: list[ConversationMessage] = []
        self.stats = ConversationStats()
        self.created_at = time.time()

        if system_prompt:
            self._messages.append(ConversationMessage.system(system_prompt))

    # ── Adding messages ───────────────────────────────────────────────────────

    def add(self, msg: ConversationMessage) -> None:
        """Add a message and update stats."""
        self._messages.append(msg)
        self._update_stats(msg)

    def add_user(self, content: str) -> ConversationMessage:
        msg = ConversationMessage.user(content)
        self.add(msg)
        return msg

    def add_system(self, content: str) -> ConversationMessage:
        msg = ConversationMessage.system(content)
        self.add(msg)
        return msg

    def add_assistant(self, content: str, **metadata: Any) -> ConversationMessage:
        msg = ConversationMessage.assistant_text(content, **metadata)
        self.add(msg)
        return msg

    def add_tool_calls(
        self, tool_calls: list[ToolCallRequest], **metadata: Any
    ) -> ConversationMessage:
        msg = ConversationMessage.assistant_tool_calls(tool_calls, **metadata)
        self.add(msg)
        return msg

    def add_tool_result(
        self, tool_call_id: str, tool_name: str, content: str, **metadata: Any
    ) -> ConversationMessage:
        msg = ConversationMessage.tool_result(tool_call_id, tool_name, content, **metadata)
        self.add(msg)
        return msg

    def add_error(self, content: str, **metadata: Any) -> ConversationMessage:
        msg = ConversationMessage.error(content, **metadata)
        self.add(msg)
        return msg

    # ── Querying ──────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[ConversationMessage]:
        return list(self._messages)

    def llm_messages(self, *, exclude_planning: bool = True) -> list[LLMMessage]:
        """
        Return messages in the format expected by LLM providers.
        Excludes planning-only messages by default.
        """
        result = []
        for msg in self._messages:
            if exclude_planning and msg.metadata.get("planning"):
                continue
            result.append(msg.message)
        return result

    @property
    def last_user_message(self) -> ConversationMessage | None:
        for msg in reversed(self._messages):
            if msg.type == MessageType.USER:
                return msg
        return None

    @property
    def last_assistant_message(self) -> ConversationMessage | None:
        for msg in reversed(self._messages):
            if msg.type in (MessageType.ASSISTANT, MessageType.TOOL_CALL):
                return msg
        return None

    @property
    def has_errors(self) -> bool:
        return self.stats.errors > 0

    def iter_by_type(self, *types: MessageType) -> Iterator[ConversationMessage]:
        for msg in self._messages:
            if msg.type in types:
                yield msg

    # ── Truncation ────────────────────────────────────────────────────────────

    def truncate(self, max_messages: int, *, keep_system: bool = True) -> "ConversationState":
        """
        Return a new ConversationState with at most *max_messages* messages.
        Preserves system messages if *keep_system* is True.
        """
        new_state = ConversationState(conversation_id=self.id)
        system_msgs = [m for m in self._messages if m.type == MessageType.SYSTEM]
        recent_msgs = [m for m in self._messages if m.type != MessageType.SYSTEM]
        recent_msgs = recent_msgs[-max_messages:]

        if keep_system:
            for msg in system_msgs:
                new_state.add(msg)
        for msg in recent_msgs:
            new_state.add(msg)
        return new_state

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> "ConversationState":
        """Return a deep copy of the current state."""
        new_state = ConversationState.__new__(ConversationState)
        new_state.id = self.id
        new_state._messages = copy.deepcopy(self._messages)
        new_state.stats = copy.copy(self.stats)
        new_state.created_at = self.created_at
        return new_state

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "messages": [
                {
                    "id": m.id,
                    "type": m.type.value,
                    "role": m.message.role.value,
                    "content": m.message.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in (m.message.tool_calls or [])
                    ],
                    "tool_call_id": m.message.tool_call_id,
                    "tool_name": m.message.tool_name,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self._messages
            ],
            "stats": {
                "total_messages": self.stats.total_messages,
                "tool_calls": self.stats.tool_calls,
                "total_tokens": self.stats.total_tokens,
                "total_latency_ms": self.stats.total_latency_ms,
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_stats(self, msg: ConversationMessage) -> None:
        self.stats.total_messages += 1
        match msg.type:
            case MessageType.USER:
                self.stats.user_messages += 1
            case MessageType.ASSISTANT:
                self.stats.assistant_messages += 1
            case MessageType.TOOL_CALL:
                self.stats.tool_calls += len(msg.tool_calls or [])
            case MessageType.TOOL_RESULT:
                self.stats.tool_results += 1
            case MessageType.ERROR:
                self.stats.errors += 1

    def record_llm_usage(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Called after each LLM response to accumulate usage stats."""
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_latency_ms += latency_ms

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"<ConversationState id={self.id[:8]} "
            f"messages={len(self._messages)} "
            f"tokens={self.stats.total_tokens}>"
        )
