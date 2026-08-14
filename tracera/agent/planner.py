"""
TRACERA Planning System — Phase 9.

Provides task decomposition, TODO state tracking, and replanning on failure.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tracera.logging import get_logger

log = get_logger("agent.planner")


# ── Todo item ─────────────────────────────────────────────────────────────────

class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """A single actionable step in a plan."""
    id: str
    title: str
    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    priority: int = 0          # lower = higher priority
    depends_on: list[str] = field(default_factory=list)   # IDs of prerequisite items
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True if this item has no pending dependencies."""
        return not self.depends_on

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def start(self) -> None:
        self.status = TodoStatus.IN_PROGRESS
        self.started_at = time.time()

    def complete(self, result: str | None = None) -> None:
        self.status = TodoStatus.DONE
        self.completed_at = time.time()
        self.result = result

    def fail(self, error: str) -> None:
        self.status = TodoStatus.FAILED
        self.completed_at = time.time()
        self.error = error

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "error": self.error,
            "result": self.result,
        }

    def __repr__(self) -> str:
        icon = {"pending": "○", "in_progress": "◎", "done": "●", "failed": "✗", "skipped": "⊘"}
        return f"{icon.get(self.status.value, '?')} [{self.id[:6]}] {self.title}"


# ── Plan ──────────────────────────────────────────────────────────────────────

class Plan:
    """
    An ordered list of TodoItems representing a decomposed task.
    
    Supports:
    - Item lifecycle management (start/complete/fail)
    - Progress tracking
    - Dependency resolution
    - Serialisation for persistence
    """

    def __init__(self, task: str, *, plan_id: str | None = None) -> None:
        self.id = plan_id or str(uuid.uuid4())
        self.task = task
        self._items: dict[str, TodoItem] = {}
        self._order: list[str] = []
        self.created_at = time.time()
        self.replanned_count = 0

    # ── Item management ───────────────────────────────────────────────────────

    def add_item(
        self,
        title: str,
        description: str = "",
        *,
        priority: int = 0,
        depends_on: list[str] | None = None,
        item_id: str | None = None,
    ) -> TodoItem:
        item = TodoItem(
            id=item_id or str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority,
            depends_on=depends_on or [],
        )
        self._items[item.id] = item
        self._order.append(item.id)
        return item

    def get_item(self, item_id: str) -> TodoItem | None:
        return self._items.get(item_id)

    def start_item(self, item_id: str) -> TodoItem:
        item = self._items[item_id]
        item.start()
        log.debug("Plan item started: %s", item.title)
        return item

    def complete_item(self, item_id: str, result: str | None = None) -> TodoItem:
        item = self._items[item_id]
        item.complete(result)
        # Remove from dependents
        for other in self._items.values():
            if item_id in other.depends_on:
                other.depends_on.remove(item_id)
        log.debug("Plan item done: %s", item.title)
        return item

    def fail_item(self, item_id: str, error: str) -> TodoItem:
        item = self._items[item_id]
        item.fail(error)
        log.warning("Plan item failed: %s — %s", item.title, error)
        return item

    # ── Querying ──────────────────────────────────────────────────────────────

    @property
    def items(self) -> list[TodoItem]:
        return [self._items[id_] for id_ in self._order]

    @property
    def pending(self) -> list[TodoItem]:
        return [i for i in self.items if i.status == TodoStatus.PENDING]

    @property
    def in_progress(self) -> list[TodoItem]:
        return [i for i in self.items if i.status == TodoStatus.IN_PROGRESS]

    @property
    def done(self) -> list[TodoItem]:
        return [i for i in self.items if i.status == TodoStatus.DONE]

    @property
    def failed(self) -> list[TodoItem]:
        return [i for i in self.items if i.status == TodoStatus.FAILED]

    @property
    def ready(self) -> list[TodoItem]:
        """Items that are pending and have no blocking dependencies."""
        return [i for i in self.pending if i.is_ready]

    @property
    def is_complete(self) -> bool:
        return all(
            i.status in (TodoStatus.DONE, TodoStatus.SKIPPED)
            for i in self.items
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.failed)

    @property
    def progress(self) -> tuple[int, int]:
        """Return (done, total) counts."""
        done = len(self.done)
        total = len(self._items)
        return done, total

    @property
    def progress_pct(self) -> float:
        done, total = self.progress
        return (done / total * 100) if total else 0.0

    def next_item(self) -> TodoItem | None:
        """Return the next ready item by priority, or None."""
        ready = self.ready
        if not ready:
            return None
        return min(ready, key=lambda i: i.priority)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "created_at": self.created_at,
            "replanned_count": self.replanned_count,
            "items": [item.to_dict() for item in self.items],
        }

    def to_markdown(self) -> str:
        """Render the plan as a markdown checklist."""
        lines = [f"## Plan: {self.task}\n"]
        for item in self.items:
            check = {
                TodoStatus.DONE: "x",
                TodoStatus.IN_PROGRESS: "/",
                TodoStatus.FAILED: "!",
                TodoStatus.SKIPPED: "-",
            }.get(item.status, " ")
            lines.append(f"- [{check}] {item.title}")
            if item.description:
                lines.append(f"  > {item.description}")
        done, total = self.progress
        lines.append(f"\n**Progress: {done}/{total} ({self.progress_pct:.0f}%)**")
        return "\n".join(lines)

    def __repr__(self) -> str:
        done, total = self.progress
        return f"<Plan task={self.task[:40]!r} progress={done}/{total}>"


# ── Task Decomposer ───────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """You are a software engineering planning assistant.

Given a task, decompose it into a list of concrete, ordered steps.
Each step should be specific and actionable.
Steps should be ordered so that earlier steps don't depend on later ones.

Return ONLY a JSON array of objects, each with:
- "title": short step title (max 60 chars)
- "description": one-sentence description of what to do
- "priority": integer 0-9 (0 = first)

Example:
[
  {{"title": "Read existing auth code", "description": "Read auth/middleware.py to understand current implementation", "priority": 0}},
  {{"title": "Add JWT validation", "description": "Edit the validate() method to check JWT expiry", "priority": 1}}
]

Task: {task}"""


class TaskDecomposer:
    """
    Uses the LLM to decompose a task into a Plan of TodoItems.
    Falls back to a single-item plan if decomposition fails.
    """

    def __init__(self, provider, *, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    async def decompose(self, task: str) -> Plan:
        """
        Decompose *task* into a Plan.
        
        Returns a Plan with TodoItems ordered by priority.
        """
        from tracera.providers.base import LLMMessage

        plan = Plan(task)

        try:
            response = await self.provider.complete(
                [LLMMessage.user(_DECOMPOSE_PROMPT.format(task=task))],
                model=self.model,
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.content or "[]"

            # Extract JSON from the response
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            steps = json.loads(raw)
            if not isinstance(steps, list):
                raise ValueError("Expected a JSON array")

            for i, step in enumerate(steps):
                plan.add_item(
                    title=step.get("title", f"Step {i+1}"),
                    description=step.get("description", ""),
                    priority=step.get("priority", i),
                )
            log.info("Decomposed task into %d steps", len(plan.items))

        except Exception as e:
            log.warning("Task decomposition failed (%s), using single-step plan", e)
            plan.add_item(title=task[:80], description="Execute the task directly.", priority=0)

        return plan

    async def replan(self, plan: Plan, reason: str) -> Plan:
        """
        Replan after a failure. Adds recovery steps to the existing plan.
        """
        from tracera.providers.base import LLMMessage

        plan.replanned_count += 1
        failed_items = "\n".join(f"- {item.title}: {item.error}" for item in plan.failed)
        prompt = (
            f"The following steps of a plan failed:\n{failed_items}\n\n"
            f"Failure reason: {reason}\n\n"
            f"Original task: {plan.task}\n\n"
            "List recovery steps as a JSON array (same format as before)."
        )
        try:
            response = await self.provider.complete(
                [LLMMessage.user(prompt)],
                model=self.model,
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.content or "[]"
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            steps = json.loads(raw)
            for i, step in enumerate(steps):
                plan.add_item(
                    title=f"[Recovery] {step.get('title', f'Step {i+1}')}",
                    description=step.get("description", ""),
                    priority=100 + i,
                )
            log.info("Replanned with %d recovery steps", len(steps))
        except Exception as e:
            log.warning("Replan failed: %s", e)
            plan.add_item(
                title="[Recovery] Analyze and retry",
                description=f"Review the failure and attempt a different approach: {reason}",
                priority=100,
            )
        return plan
