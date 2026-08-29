"""
Enhanced Memory Taxonomy — structured memory types beyond flat entries.

Inspired by agent-native memory concepts where conversations are automatically
classified into rich, typed memories rather than unstructured text blobs.

Memory types:
  - Fact: objective knowledge extracted from conversations or code analysis
  - Rule: behavioral constraints ("always use edit_file, never write_file for small changes")
  - Relationship: connections between concepts ("AuthMiddleware calls UserService")
  - Skill: capabilities the agent has demonstrated or the user expects
  - Preference: how the user likes things done
  - Event: something that happened (decision made, bug fixed, test passed)

Each type carries its own metadata and retrieval characteristics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tracera.logging import get_logger

log = get_logger("memory.taxonomy")


class MemoryType(str, Enum):
    """The type of structured memory."""

    FACT = "fact"
    RULE = "rule"
    RELATIONSHIP = "relationship"
    SKILL = "skill"
    PREFERENCE = "preference"
    EVENT = "event"
    DECISION = "decision"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    EXPERIENCE = "experience"
    ATTRIBUTE = "attribute"


@dataclass
class StructuredMemory:
    """
    Base class for all structured memory entries.

    Each memory entry has:
      - A type (fact, rule, relationship, etc.)
      - Content (the actual memory)
      - Attribution (who/what created it)
      - Scoping (which session, entity, process)
      - Confidence (how sure we are this is correct)
      - Frequency (how many times observed)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.FACT
    content: str = ""
    # Attribution
    source: str = ""  # "conversation", "code_analysis", "user", "agent"
    entity_id: str = ""  # user who triggered this
    process_id: str = ""  # agent/process that created it
    session_id: str = ""  # which session this came from
    # Quality
    confidence: float = 0.8  # 0.0–1.0
    importance: float = 0.5  # 0.0–1.0
    frequency: int = 1  # how many times observed
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float | None = None
    access_count: int = 0
    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    def observe(self) -> None:
        """Called when this memory is confirmed/observed again."""
        self.frequency += 1
        self.updated_at = time.time()
        self.confidence = min(1.0, self.confidence + 0.05)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "source": self.source,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "confidence": self.confidence,
            "importance": self.importance,
            "frequency": self.frequency,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredMemory":
        return cls(
            id=d["id"],
            memory_type=MemoryType(d["memory_type"]),
            content=d["content"],
            source=d.get("source", ""),
            entity_id=d.get("entity_id", ""),
            process_id=d.get("process_id", ""),
            session_id=d.get("session_id", ""),
            confidence=d.get("confidence", 0.8),
            importance=d.get("importance", 0.5),
            frequency=d.get("frequency", 1),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            last_accessed=d.get("last_accessed"),
            access_count=d.get("access_count", 0),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", "↵")
        return f"<{self.__class__.__name__} [{self.memory_type.value}] {preview!r}>"


# ── Specific memory types ─────────────────────────────────────────────────────


@dataclass
class MemoryFact(StructuredMemory):
    """An objective fact extracted from conversation or code analysis.

    Examples:
      - "The project uses pytest for testing"
      - "JWT tokens expire after 24 hours"
      - "The auth module is in tracera/security/"
    """

    memory_type: MemoryType = MemoryType.FACT
    # Facts can reference code symbols
    symbol: str = ""  # related code symbol name
    file_path: str = ""  # related file path


@dataclass
class MemoryRule(StructuredMemory):
    """A behavioral constraint or convention the agent should follow.

    Examples:
      - "Always use edit_file for small changes, not write_file"
      - "Run pytest after every edit to verify correctness"
      - "Never modify .env files directly"
    """

    memory_type: MemoryType = MemoryType.RULE
    scope: str = "global"  # global | session | task
    priority: int = 0  # higher = more important


@dataclass
class MemoryRelationship(StructuredMemory):
    """A connection between two concepts or code entities.

    Examples:
      - "AuthMiddleware depends on UserService"
      - "The test suite covers the retrieval pipeline"
      - "User prefers pytest over unittest"
    """

    memory_type: MemoryType = MemoryType.RELATIONSHIP
    subject: str = ""  # first entity
    predicate: str = ""  # relationship type
    object: str = ""  # second entity


@dataclass
class MemorySkill(StructuredMemory):
    """A capability the agent has demonstrated or the user expects.

    Examples:
      - "Can run pytest and parse failure output"
      - "Knows how to use tree-sitter for Python AST parsing"
      - "Has successfully fixed import errors before"
    """

    memory_type: MemoryType = MemoryType.SKILL
    proficiency: float = 0.5  # 0.0–1.0, increases with success


@dataclass
class MemoryPreference(StructuredMemory):
    """How the user likes things done.

    Examples:
      - "Prefers TypeScript over JavaScript"
      - "Wants concise commit messages"
      - "Likes tests to be in a separate tests/ directory"
    """

    memory_type: MemoryType = MemoryType.PREFERENCE
    strength: float = 0.7  # 0.0–1.0, how strongly held


@dataclass
class MemoryEvent(StructuredMemory):
    """Something that happened — a decision, a fix, a discovery.

    Examples:
      - "Decided to use LanceDB for vector storage"
      - "Fixed the failing test_auth.py test"
      - "Discovered the config is loaded from .env"
    """

    memory_type: MemoryType = MemoryType.EVENT
    event_type: str = ""  # decision | fix | discovery | failure | success
    related_files: list[str] = field(default_factory=list)


@dataclass
class MemoryDecision(StructuredMemory):
    """A decision made during a coding session.

    Examples:
      - "Chose PostgreSQL over SQLite for the database"
      - "Decided to use dependency injection pattern"
    """

    memory_type: MemoryType = MemoryType.DECISION
    rationale: str = ""
    alternatives_considered: list[str] = field(default_factory=list)


@dataclass
class MemoryGoal(StructuredMemory):
    """A goal or objective for the agent or project.

    Examples:
      - "Implement user authentication"
      - "Achieve 90% test coverage"
    """

    memory_type: MemoryType = MemoryType.GOAL
    target_date: str = ""
    progress: float = 0.0  # 0.0–1.0


@dataclass
class MemoryConstraint(StructuredMemory):
    """A constraint or limitation that must be respected.

    Examples:
      - "Cannot modify production database directly"
      - "Must maintain backward compatibility with API v1"
    """

    memory_type: MemoryType = MemoryType.CONSTRAINT
    scope: str = "global"  # global | project | file
    severity: str = "hard"  # hard | soft


@dataclass
class MemoryExperience(StructuredMemory):
    """A learned experience from past execution.

    Examples:
      - "Refactoring auth module caused test failures in payment module"
      - "Using async/await pattern improved throughput by 40%"
    """

    memory_type: MemoryType = MemoryType.EXPERIENCE
    outcome: str = ""  # success | failure | partial
    lessons_learned: str = ""


@dataclass
class MemoryAttribute(StructuredMemory):
    """An attribute or characteristic of an entity.

    Examples:
      - "Project uses FastAPI framework"
      - "Team prefers functional programming style"
    """

    memory_type: MemoryType = MemoryType.ATTRIBUTE
    entity: str = ""  # what this attribute applies to
    value: str = ""   # the attribute value


# ── Factory helpers ────────────────────────────────────────────────────────────

def create_fact(
    content: str,
    *,
    symbol: str = "",
    file_path: str = "",
    session_id: str = "",
    confidence: float = 0.8,
    **kwargs: Any,
) -> MemoryFact:
    return MemoryFact(
        content=content,
        symbol=symbol,
        file_path=file_path,
        session_id=session_id,
        confidence=confidence,
        source="conversation",
        **kwargs,
    )


def create_rule(
    content: str,
    *,
    scope: str = "global",
    priority: int = 0,
    session_id: str = "",
    **kwargs: Any,
) -> MemoryRule:
    return MemoryRule(
        content=content,
        scope=scope,
        priority=priority,
        session_id=session_id,
        source="conversation",
        **kwargs,
    )


def create_relationship(
    subject: str,
    predicate: str,
    obj: str,
    *,
    session_id: str = "",
    **kwargs: Any,
) -> MemoryRelationship:
    return MemoryRelationship(
        content=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        session_id=session_id,
        source="conversation",
        **kwargs,
    )


def create_skill(
    content: str,
    *,
    proficiency: float = 0.5,
    session_id: str = "",
    **kwargs: Any,
) -> MemorySkill:
    return MemorySkill(
        content=content,
        proficiency=proficiency,
        session_id=session_id,
        source="agent",
        **kwargs,
    )


def create_preference(
    content: str,
    *,
    strength: float = 0.7,
    session_id: str = "",
    **kwargs: Any,
) -> MemoryPreference:
    return MemoryPreference(
        content=content,
        strength=strength,
        session_id=session_id,
        source="user",
        **kwargs,
    )


def create_event(
    content: str,
    *,
    event_type: str = "decision",
    related_files: list[str] | None = None,
    session_id: str = "",
    **kwargs: Any,
) -> MemoryEvent:
    return MemoryEvent(
        content=content,
        event_type=event_type,
        related_files=related_files or [],
        session_id=session_id,
        source="agent",
        **kwargs,
    )
