"""
Session Manager — groups coding conversations into trackable sessions.

Each session represents a single coding interaction (a task, a question,
a fix loop, etc.) and carries metadata about what happened: which files
were touched, which tools were used, what the outcome was.

Sessions enable:
  - Scoped memory recall (recall memories from sessions like this one)
  - Session summaries (what was accomplished in this session)
  - Cross-session learning (patterns across sessions)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.logging import get_logger

log = get_logger("memory.session")


@dataclass
class SessionTurn:
    """A single turn within a session (user message + agent response + tool calls)."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_success: bool | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content[:2000],  # truncate for storage
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_success": self.tool_success,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionTurn":
        return cls(
            role=d["role"],
            content=d["content"],
            tool_name=d.get("tool_name"),
            tool_args=d.get("tool_args"),
            tool_success=d.get("tool_success"),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Session:
    """
    A coding session — one task or interaction from start to finish.

    Sessions track:
      - The task description
      - All turns (user messages, agent responses, tool calls)
      - Files touched during the session
      - Tools used and their success rates
      - Outcome (success, failure, partial)
      - Duration
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    entity_id: str = ""  # who initiated (user identifier)
    process_id: str = ""  # which agent/process handled it
    created_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    turns: list[SessionTurn] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tools_used: dict[str, int] = field(default_factory=dict)  # tool_name → count
    outcome: str = "in_progress"  # in_progress | success | failure | partial
    tags: list[str] = field(default_factory=list)
    summary: str = ""  # filled after session closes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.closed_at:
            return self.closed_at - self.created_at
        return None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None

    def add_turn(self, turn: SessionTurn) -> None:
        """Add a turn to the session."""
        self.turns.append(turn)
        if turn.tool_name:
            self.tools_used[turn.tool_name] = self.tools_used.get(turn.tool_name, 0) + 1
        if turn.metadata.get("file_path"):
            fp = turn.metadata["file_path"]
            if fp not in self.files_touched:
                self.files_touched.append(fp)

    def close(self, outcome: str = "success", summary: str = "") -> None:
        """Close the session with an outcome."""
        self.closed_at = time.time()
        self.outcome = outcome
        self.summary = summary

    def conversation_text(self) -> str:
        """Render the session as a conversation text for extraction."""
        lines = []
        for turn in self.turns:
            if turn.role == "user":
                lines.append(f"User: {turn.content}")
            elif turn.role == "assistant":
                lines.append(f"Assistant: {turn.content}")
            elif turn.role == "tool":
                status = "OK" if turn.tool_success else "FAIL"
                lines.append(f"Tool({turn.tool_name} [{status}]): {turn.content[:500]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "turns": [t.to_dict() for t in self.turns],
            "files_touched": self.files_touched,
            "tools_used": self.tools_used,
            "outcome": self.outcome,
            "tags": self.tags,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        session = cls(
            id=d["id"],
            task=d.get("task", ""),
            entity_id=d.get("entity_id", ""),
            process_id=d.get("process_id", ""),
            created_at=d.get("created_at", 0.0),
            closed_at=d.get("closed_at"),
            files_touched=d.get("files_touched", []),
            tools_used=d.get("tools_used", {}),
            outcome=d.get("outcome", "unknown"),
            tags=d.get("tags", []),
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
        )
        session.turns = [SessionTurn.from_dict(t) for t in d.get("turns", [])]
        return session


class SessionManager:
    """
    Manages coding sessions — creation, tracking, persistence, and recall.

    The manager owns a list of sessions and handles:
      - Creating new sessions
      - Adding turns to the active session
      - Closing sessions with outcomes
      - Persisting sessions to disk
      - Finding similar past sessions for recall
    """

    SESSIONS_FILE = "sessions.json"
    MAX_SESSIONS = 200  # keep last N sessions

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.memory_dir / self.SESSIONS_FILE
        self._sessions: dict[str, Session] = {}
        self._active_session_id: str | None = None
        self._load()

    # ── Active session ────────────────────────────────────────────────────────

    @property
    def active_session(self) -> Session | None:
        if self._active_session_id and self._active_session_id in self._sessions:
            return self._sessions[self._active_session_id]
        return None

    def new_session(
        self,
        task: str = "",
        entity_id: str = "",
        process_id: str = "tracera",
        tags: list[str] | None = None,
    ) -> Session:
        """Create and activate a new session."""
        # Close any existing active session
        if self.active_session and not self.active_session.is_closed:
            self.active_session.close(outcome="interrupted")

        session = Session(
            task=task,
            entity_id=entity_id,
            process_id=process_id,
            tags=tags or [],
        )
        self._sessions[session.id] = session
        self._active_session_id = session.id
        log.info("New session: %s (task: %s)", session.id[:8], task[:60])
        self._save()
        return session

    def set_session(self, session_id: str) -> Session | None:
        """Switch to an existing session by ID."""
        if session_id in self._sessions:
            self._active_session_id = session_id
            return self._sessions[session_id]
        return None

    def close_session(
        self,
        outcome: str = "success",
        summary: str = "",
    ) -> Session | None:
        """Close the active session."""
        session = self.active_session
        if session:
            session.close(outcome=outcome, summary=summary)
            self._active_session_id = None
            self._prune()
            self._save()
            log.info(
                "Closed session %s: %s (%s, %d turns)",
                session.id[:8], outcome, summary[:40], session.turn_count,
            )
        return session

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(self, turn: SessionTurn) -> None:
        """Add a turn to the active session (creates one if needed)."""
        session = self.active_session
        if session is None:
            session = self.new_session(task="(auto)")
        session.add_turn(turn)

    def record_user_message(self, content: str) -> None:
        """Record a user message in the active session."""
        self.add_turn(SessionTurn(role="user", content=content))

    def record_agent_response(self, content: str) -> None:
        """Record the agent's final response."""
        self.add_turn(SessionTurn(role="assistant", content=content))

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        output: str,
        success: bool,
        file_path: str | None = None,
    ) -> None:
        """Record a tool call in the active session."""
        metadata = {}
        if file_path:
            metadata["file_path"] = file_path
        self.add_turn(
            SessionTurn(
                role="tool",
                content=output[:2000],
                tool_name=tool_name,
                tool_args=args,
                tool_success=success,
                metadata=metadata,
            )
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    @property
    def sessions(self) -> list[Session]:
        """All sessions, most recent first."""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )

    def find_similar_sessions(
        self,
        task: str,
        *,
        k: int = 5,
        outcome: str | None = None,
    ) -> list[Session]:
        """
        Find sessions with similar task descriptions.
        Simple keyword overlap — good enough for session recall.
        """
        task_tokens = set(task.lower().split())
        scored = []
        for session in self._sessions.values():
            if outcome and session.outcome != outcome:
                continue
            session_tokens = set(session.task.lower().split())
            if not session_tokens or not task_tokens:
                continue
            overlap = len(task_tokens & session_tokens) / max(len(task_tokens | session_tokens), 1)
            # Boost by recency
            recency_boost = 1.0 + (1.0 / (1.0 + (time.time() - session.created_at) / 86400))
            scored.append((overlap * recency_boost, session))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k] if scored[0][0] > 0]

    def get_session_summaries(self, k: int = 10) -> list[str]:
        """Get recent session summaries for context."""
        summaries = []
        for session in self.sessions[:k]:
            if session.summary:
                summaries.append(
                    f"[{session.outcome}] {session.task[:60]}: {session.summary[:100]}"
                )
            else:
                summaries.append(
                    f"[{session.outcome}] {session.task[:60]} "
                    f"({session.turn_count} turns, {len(session.files_touched)} files)"
                )
        return summaries

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for d in data.get("sessions", []):
                    session = Session.from_dict(d)
                    self._sessions[session.id] = session
                self._active_session_id = data.get("active_session_id")
                log.debug("Loaded %d sessions", len(self._sessions))
            except Exception as e:
                log.warning("Failed to load sessions: %s", e)

    def _save(self) -> None:
        try:
            data = {
                "sessions": [s.to_dict() for s in self._sessions.values()],
                "active_session_id": self._active_session_id,
            }
            self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.error("Failed to save sessions: %s", e)

    def _prune(self) -> None:
        """Keep only the most recent sessions."""
        if len(self._sessions) <= self.MAX_SESSIONS:
            return
        sorted_sessions = self.sessions  # most recent first
        to_remove = sorted_sessions[self.MAX_SESSIONS:]
        for session in to_remove:
            del self._sessions[session.id]
        log.debug("Pruned %d old sessions", len(to_remove))

    def __repr__(self) -> str:
        active = self.active_session
        active_str = f" active={active.id[:8]}" if active else ""
        return f"<SessionManager sessions={len(self._sessions)}{active_str}>"
