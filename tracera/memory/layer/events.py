"""
Memory Layer Event Pipeline — captures agent execution events for memory extraction.

This module defines the event types and a pipeline that intercepts agent execution
to extract memories from:
- LLM requests/responses
- Tool calls and results
- Agent decisions and plans
- File modifications
- Test results
- User corrections/preferences
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tracera.logging import get_logger
from tracera.memory.layer.store import MemoryStore
from tracera.providers.base import LLMMessage

log = get_logger("memory.layer.events")


class EventType(str, Enum):
    """Types of events that can trigger memory extraction."""

    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    AGENT_DECISION = "agent.decision"
    AGENT_PLAN = "agent.plan"
    AGENT_ERROR = "agent.error"
    AGENT_SUCCESS = "agent.success"
    FILE_CHANGED = "file.changed"
    TEST_FAILED = "test.failed"
    TEST_PASSED = "test.passed"
    USER_INSTRUCTION = "user.instruction"
    USER_PREFERENCE = "user.preference"
    USER_CORRECTION = "user.correction"
    REPOSITORY_DISCOVERY = "repository.discovery"
    CODE_ANALYSIS = "code.analysis"


@dataclass
class MemoryEvent:
    """A single event captured during agent execution."""

    type: EventType
    entity_id: str
    process_id: str
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEvent":
        return cls(
            type=EventType(d["type"]),
            entity_id=d["entity_id"],
            process_id=d["process_id"],
            session_id=d.get("session_id"),
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", time.time()),
            event_id=d.get("event_id", str(uuid.uuid4())),
        )


# Event handlers extract memory-worthy information from events
EventHandler = Callable[[MemoryEvent], list[dict[str, Any]] | None]


class EventPipeline:
    """
    Pipeline for capturing and processing agent execution events.

    Events flow through registered handlers which can extract structured
    memories. The pipeline is designed to be non-blocking - events are
    queued and processed asynchronously.
    """

    def __init__(self, store: MemoryStore, embed_fn: Callable[[str], list[float]]) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def register_global_handler(self, handler: EventHandler) -> None:
        """Register a handler that receives all events."""
        self._global_handlers.append(handler)

    def emit(self, event: MemoryEvent) -> None:
        """Emit an event to all registered handlers."""
        # Run global handlers
        for handler in self._global_handlers:
            try:
                result = handler(event)
                if result:
                    self._process_extracted_memories(event, result)
            except Exception as e:
                log.warning("Global event handler failed: %s", e)

        # Run type-specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if result:
                    self._process_extracted_memories(event, result)
            except Exception as e:
                log.warning("Event handler for %s failed: %s", event.type, e)

    def _process_extracted_memories(
        self, event: MemoryEvent, memories: list[dict[str, Any]]
    ) -> None:
        """Process extracted memories from an event."""
        for mem in memories:
            kind = mem.get("kind", "fact")
            subject = mem.get("subject", "user")
            predicate = mem.get("predicate", "")
            obj = mem.get("object", "")
            text = mem.get("text", "")
            confidence = mem.get("confidence", 0.8)
            importance = mem.get("importance", 0.5)

            if not predicate or not obj or not text:
                continue

            embedding = self._embed_fn(text)
            self._store.upsert_memory(
                entity_id=event.entity_id,
                process_id=event.process_id,
                kind=kind,
                subject=subject,
                predicate=predicate,
                object=obj,
                text=text,
                embedding=embedding,
                session_id=event.session_id,
                confidence=confidence,
                importance=importance,
                source_event=event.type.value,
                source_message_id=event.event_id,
            )


# Built-in event handlers for common patterns

def handle_llm_response(event: MemoryEvent) -> list[dict[str, Any]] | None:
    """Extract memories from LLM responses."""
    payload = event.payload
    user_message = payload.get("user_message", "")
    assistant_message = payload.get("assistant_message", "")
    if not user_message or not assistant_message:
        return None

    memories = []

    # Detect explicit user instructions ("Remember that...", "Always...", "Never...")
    user_lower = user_message.lower()
    if any(user_lower.startswith(p) for p in [
        "remember that", "remember to", "always ", "never ",
        "don't ", "do not ", "please ", "i prefer", "i like",
        "my preference", "we use", "we always"
    ]):
        # These are high-confidence memory candidates
        kind = "preference" if any(p in user_lower for p in ["prefer", "like"]) else "rule"
        memories.append({
            "kind": kind,
            "subject": "user",
            "predicate": "instruction" if kind == "rule" else "preference",
            "object": user_message[:200],
            "text": f"User instruction: {user_message[:300]}",
            "confidence": 0.9,
            "importance": 0.8,
        })

    # Detect decisions
    if any(p in user_lower for p in ["decided", "chose", "going with", "switched to"]):
        memories.append({
            "kind": "decision",
            "subject": "user",
            "predicate": "decided",
            "object": user_message[:200],
            "text": f"Decision: {user_message[:300]}",
            "confidence": 0.85,
            "importance": 0.7,
        })

    return memories if memories else None


def handle_tool_completed(event: MemoryEvent) -> list[dict[str, Any]] | None:
    """Extract memories from tool execution results."""
    payload = event.payload
    tool_name = payload.get("tool_name", "")
    success = payload.get("success", True)
    output = payload.get("output", "")
    file_path = payload.get("file_path")

    memories = []

    # Detect file modification patterns
    if tool_name in ("edit_file", "write_file") and file_path:
        if success:
            memories.append({
                "kind": "event",
                "subject": "agent",
                "predicate": "modified",
                "object": file_path,
                "text": f"Modified {file_path}",
                "confidence": 0.9,
                "importance": 0.6,
            })

    # Detect test results
    if tool_name in ("test_runner", "run_command") and "test" in str(payload.get("args", "")).lower():
        if success and "passed" in output.lower():
            memories.append({
                "kind": "event",
                "subject": "project",
                "predicate": "tests_pass",
                "object": "true",
                "text": "Tests passed after changes",
                "confidence": 0.85,
                "importance": 0.7,
            })
        elif not success and ("failed" in output.lower() or "error" in output.lower()):
            memories.append({
                "kind": "event",
                "subject": "project",
                "predicate": "tests_fail",
                "object": "true",
                "text": f"Tests failed: {output[:200]}",
                "confidence": 0.85,
                "importance": 0.8,
            })

    # Detect git discoveries
    if tool_name == "git_tool":
        if "status" in str(payload.get("args", "")):
            memories.append({
                "kind": "fact",
                "subject": "repository",
                "predicate": "has_changes",
                "object": "true",
                "text": "Repository has uncommitted changes",
                "confidence": 0.9,
                "importance": 0.5,
            })

    return memories if memories else None


def handle_agent_decision(event: MemoryEvent) -> list[dict[str, Any]] | None:
    """Extract memories from explicit agent decisions."""
    payload = event.payload
    decision = payload.get("decision", "")
    reason = payload.get("reason", "")
    scope = payload.get("scope", "task")

    if not decision:
        return None

    return [{
        "kind": "decision",
        "subject": "agent",
        "predicate": "decided",
        "object": decision[:200],
        "text": f"Decision ({scope}): {decision}. Reason: {reason}",
        "confidence": 0.85,
        "importance": 0.75,
    }]


def handle_file_changed(event: MemoryEvent) -> list[dict[str, Any]] | None:
    """Extract memories from file changes."""
    payload = event.payload
    file_path = payload.get("file_path", "")
    change_type = payload.get("change_type", "modified")  # created, modified, deleted
    symbols = payload.get("symbols_affected", [])

    memories = []

    if file_path:
        memories.append({
            "kind": "event",
            "subject": "repository",
            "predicate": change_type,
            "object": file_path,
            "text": f"{change_type.capitalize()} {file_path}",
            "confidence": 0.9,
            "importance": 0.6,
        })

    for symbol in symbols:
        memories.append({
            "kind": "relationship",
            "subject": symbol,
            "predicate": "defined_in",
            "object": file_path,
            "text": f"{symbol} is defined in {file_path}",
            "confidence": 0.8,
            "importance": 0.5,
        })

    return memories if memories else None


def handle_repository_discovery(event: MemoryEvent) -> list[dict[str, Any]] | None:
    """Extract memories from code analysis discoveries."""
    payload = event.payload
    discovery_type = payload.get("type", "")
    description = payload.get("description", "")
    related_files = payload.get("files", [])
    symbols = payload.get("symbols", [])

    if not description:
        return None

    kind_map = {
        "architecture": "fact",
        "pattern": "fact",
        "convention": "rule",
        "dependency": "relationship",
        "hotspot": "fact",
    }
    kind = kind_map.get(discovery_type, "fact")

    memories = [{
        "kind": kind,
        "subject": "repository",
        "predicate": discovery_type,
        "object": description[:200],
        "text": f"Code analysis: {description[:300]}",
        "confidence": 0.75,
        "importance": 0.6,
    }]

    for symbol in symbols:
        memories.append({
            "kind": "relationship",
            "subject": symbol,
            "predicate": "related_to",
            "object": discovery_type,
            "text": f"{symbol} relates to {discovery_type}: {description[:100]}",
            "confidence": 0.7,
            "importance": 0.5,
        })

    return memories


def build_default_pipeline(store: MemoryStore, embed_fn: Callable[[str], list[float]]) -> EventPipeline:
    """Build a pipeline with default handlers for common event types."""
    pipeline = EventPipeline(store, embed_fn)

    pipeline.register_handler(EventType.LLM_RESPONSE, handle_llm_response)
    pipeline.register_handler(EventType.TOOL_COMPLETED, handle_tool_completed)
    pipeline.register_handler(EventType.AGENT_DECISION, handle_agent_decision)
    pipeline.register_handler(EventType.FILE_CHANGED, handle_file_changed)
    pipeline.register_handler(EventType.REPOSITORY_DISCOVERY, handle_repository_discovery)

    return pipeline