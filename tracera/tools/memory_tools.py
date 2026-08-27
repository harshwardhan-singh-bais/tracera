"""
Memory Tools — let the agent interact with the enhanced memory system.

Tools:
  - recall_memory: search across all memory sources for relevant context
  - remember_memory: explicitly store a memory (fact, rule, preference, etc.)
  - forget_memory: delete a memory by ID or content match
  - list_sessions: show past coding sessions and their outcomes
"""

from __future__ import annotations

from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.memory")


class RecallMemoryTool(Tool):
    """Search across all memory sources for relevant context."""

    name = "recall_memory"
    description = (
        "Search the agent's memory system for relevant context about the project, "
        "past decisions, user preferences, rules, and relationships. Use this when "
        "you need to recall prior knowledge before making decisions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory (e.g. 'authentication pattern', 'user preferences').",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, context_recall: Any) -> None:
        self._recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, query: str, k: int = 10) -> ToolResult:
        try:
            context = self._recall.recall(
                query, k=k, max_chars=8000,
                include_sessions=True, include_triples=True, include_legacy=True,
            )
            if not context:
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="",
                    output="No relevant memories found for this query.",
                    query=query,
                )
            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output=context,
                query=query,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), query=query)


class RememberMemoryTool(Tool):
    """Explicitly store a memory (fact, rule, preference, relationship)."""

    name = "remember_memory"
    description = (
        "Store a piece of information in the agent's persistent memory. "
        "Use this for facts about the project, rules to follow, user preferences, "
        "or relationships between code entities."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store.",
            },
            "memory_type": {
                "type": "string",
                "enum": ["fact", "rule", "preference", "relationship", "skill", "event"],
                "description": "Type of memory (default: fact).",
                "default": "fact",
            },
            "importance": {
                "type": "number",
                "description": "How important this memory is, 0.0-1.0 (default 0.7).",
                "default": 0.7,
            },
        },
        "required": ["content"],
    }

    def __init__(self, enhanced_memory: Any) -> None:
        self._memory = enhanced_memory

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.7,
    ) -> ToolResult:
        try:
            from tracera.memory.taxonomy import MemoryType, create_fact
            from tracera.memory.taxonomy import (
                create_rule, create_preference, create_relationship,
                create_skill, create_event,
            )

            factories = {
                "fact": create_fact,
                "rule": create_rule,
                "preference": create_preference,
                "relationship": create_relationship,
                "skill": create_skill,
                "event": create_event,
            }
            factory = factories.get(memory_type, create_fact)
            memory = factory(content, importance=importance)
            self._memory.add(memory)

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output=f"Memory stored ({memory_type}): {content[:100]}",
                memory_id=memory.id,
                memory_type=memory_type,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), content=content)


class ForgetMemoryTool(Tool):
    """Delete a memory by ID or content match."""

    name = "forget_memory"
    description = (
        "Delete a memory from the agent's memory system. "
        "Provide either a memory_id or a content fragment to match."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The ID of the memory to delete.",
            },
            "content_match": {
                "type": "string",
                "description": "A content fragment to search for and delete.",
            },
        },
    }

    def __init__(self, enhanced_memory: Any) -> None:
        self._memory = enhanced_memory

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        memory_id: str = "",
        content_match: str = "",
    ) -> ToolResult:
        try:
            if memory_id:
                deleted = self._memory.delete(memory_id)
                if deleted:
                    return ToolResult.ok(
                        tool_name=self.name, tool_call_id="",
                        output=f"Memory {memory_id[:8]} deleted.",
                    )
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="",
                    output=f"Memory {memory_id[:8]} not found.",
                )

            if content_match:
                # Search and delete matching memories
                results = self._memory.recall(content_match, k=5)
                deleted_count = 0
                for mem in results:
                    if content_match.lower() in mem.content.lower():
                        self._memory.delete(mem.id)
                        deleted_count += 1
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="",
                    output=f"Deleted {deleted_count} memory matching '{content_match[:50]}'.",
                )

            return ToolResult.fail(
                self.name, "",
                "Provide either memory_id or content_match.",
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class ListSessionsTool(Tool):
    """Show past coding sessions and their outcomes."""

    name = "list_sessions"
    description = (
        "List recent coding sessions with their outcomes, tasks, and files touched. "
        "Use this to understand what work has been done recently."
    )
    parameters = {
        "type": "object",
        "properties": {
            "k": {
                "type": "integer",
                "description": "Number of sessions to show (default 5).",
                "default": 5,
            },
        },
    }

    def __init__(self, session_manager: Any) -> None:
        self._sessions = session_manager

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, k: int = 5) -> ToolResult:
        try:
            sessions = self._sessions.sessions[:k]
            if not sessions:
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="",
                    output="No past sessions found.",
                )

            lines = ["## Recent Sessions\n"]
            for i, session in enumerate(sessions, 1):
                duration = ""
                if session.duration_seconds:
                    mins = int(session.duration_seconds / 60)
                    duration = f" ({mins}m)" if mins > 0 else f" ({int(session.duration_seconds)}s)"
                icon = {"success": "✅", "failure": "❌", "partial": "⚠️"}.get(
                    session.outcome, "📋"
                )
                files = f", {len(session.files_touched)} files" if session.files_touched else ""
                tools = ", ".join(
                    f"{name}×{count}" for name, count in session.tools_used.items()
                ) if session.tools_used else "no tools"
                lines.append(
                    f"{i}. {icon} [{session.outcome}] {session.task[:70]}{duration}{files}"
                )
                lines.append(f"   Tools: {tools}")
                if session.summary:
                    lines.append(f"   Summary: {session.summary[:120]}")
                lines.append("")

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output="\n".join(lines),
                session_count=len(sessions),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))
