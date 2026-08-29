"""
Memory Tools — let the agent interact with the enhanced memory system.

Tools:
  - recall_memory: search across all memory sources for relevant context
  - remember_memory: explicitly store a memory (fact, rule, preference, etc.)
  - forget_memory: delete a memory by ID or content match
  - list_sessions: show past coding sessions and their outcomes
  - memory_stats: show memory system statistics
  - memory_consolidate: run consolidation to merge near-duplicates
  - memory_graph: query the knowledge graph
  - memory_export: export memories to file
  - memory_import: import memories from file
  - memory_worker_status: show background worker statistics
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
            "use_graph_expansion": {
                "type": "boolean",
                "description": "Whether to use graph-backed query expansion (default true).",
                "default": True,
            },
        },
        "required": ["query"],
    }

    def __init__(self, context_recall: Any) -> None:
        self._recall = context_recall

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, query: str, k: int = 10, use_graph_expansion: bool = True) -> ToolResult:
        try:
            context = self._recall.recall(
                query, k=k, max_chars=8000,
                include_sessions=True, include_triples=True, include_legacy=True,
                use_graph_expansion=use_graph_expansion,
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
                "enum": ["fact", "rule", "preference", "relationship", "skill", "event",
                         "decision", "goal", "constraint", "experience", "attribute"],
                "description": "Type of memory (default: fact).",
                "default": "fact",
            },
            "importance": {
                "type": "number",
                "description": "How important this memory is, 0.0-1.0 (default 0.7).",
                "default": 0.7,
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this memory, 0.0-1.0 (default 0.8).",
                "default": 0.8,
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
        confidence: float = 0.8,
    ) -> ToolResult:
        try:
            from tracera.memory.taxonomy import MemoryType, create_fact
            from tracera.memory.taxonomy import (
                create_rule, create_preference, create_relationship,
                create_skill, create_event,
            )
            from tracera.memory.taxonomy import MemoryRule, MemoryPreference
            from tracera.memory.taxonomy import MemoryDecision, MemoryGoal
            from tracera.memory.taxonomy import MemoryConstraint, MemoryExperience, MemoryAttribute

            # Factory functions for types that have dedicated creators
            factories = {
                "fact": create_fact,
                "rule": create_rule,
                "preference": create_preference,
                "relationship": create_relationship,
                "skill": create_skill,
                "event": create_event,
            }

            # For types without dedicated factories, create directly
            if memory_type in factories:
                factory = factories[memory_type]
                memory = factory(content, importance=importance)
            elif memory_type == "decision":
                memory = MemoryDecision(content=content, importance=importance, session_id="", source="tool")
            elif memory_type == "goal":
                memory = MemoryGoal(content=content, importance=importance, session_id="", source="tool")
            elif memory_type == "constraint":
                memory = MemoryConstraint(content=content, importance=importance, session_id="", source="tool")
            elif memory_type == "experience":
                memory = MemoryExperience(content=content, importance=importance, session_id="", source="tool")
            elif memory_type == "attribute":
                memory = MemoryAttribute(content=content, importance=importance, session_id="", source="tool")
            else:
                memory = create_fact(content, importance=importance)

            memory.confidence = confidence
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


class MemoryStatsTool(Tool):
    """Show memory system statistics."""

    name = "memory_stats"
    description = (
        "Show statistics about the memory system including total memories, "
        "breakdown by type, and storage usage."
    )
    parameters = {
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "description": "Show detailed breakdown by type.",
                "default": True,
            },
        },
    }

    def __init__(self, enhanced_memory: Any, triple_store: Any = None) -> None:
        self._memory = enhanced_memory
        self._triple_store = triple_store

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, detailed: bool = True) -> ToolResult:
        try:
            stats = self._memory.stats()
            lines = ["## Memory Statistics\n"]
            lines.append(f"**Total Memories**: {stats['total']}")

            if detailed:
                by_type = stats.get("by_type", {})
                if by_type:
                    lines.append("\n**By Type**:")
                    for mtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                        lines.append(f"  {mtype}: {count}")

            if self._triple_store:
                lines.append(f"\n**Knowledge Graph**:")
                lines.append(f"  Triples: {self._triple_store.triple_count}")
                lines.append(f"  Nodes: {self._triple_store.node_count}")
                lines.append(f"  Edges: {self._triple_store.edge_count}")

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output="\n".join(lines),
                stats=stats,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryConsolidateTool(Tool):
    """Run memory consolidation to merge near-duplicates."""

    name = "memory_consolidate"
    description = (
        "Run consolidation pass to find and merge near-duplicate memories. "
        "This helps keep the memory store clean and accurate."
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity to consolidate (optional, defaults to all).",
            },
            "threshold": {
                "type": "number",
                "description": "Similarity threshold for merging (0.0-1.0, default 0.92).",
                "default": 0.92,
            },
        },
    }

    def __init__(self, memory_store: Any) -> None:
        self._store = memory_store

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, entity: str = "", threshold: float = 0.92) -> ToolResult:
        try:
            result = self._store.run_consolidation(
                entity_id=entity if entity else None,
                similarity_threshold=threshold,
            )
            lines = ["## Consolidation Complete\n"]
            lines.append(f"**Scanned**: {result['scanned']} memories")
            lines.append(f"**Merged**: {result['merged']} pairs")
            lines.append(f"**Superseded**: {result['superseded']} memories")
            lines.append(f"**Errors**: {result['errors']}")

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output="\n".join(lines),
                stats=result,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryGraphTool(Tool):
    """Query the knowledge graph for relationships."""

    name = "memory_graph"
    description = (
        "Query the knowledge graph for relationships between concepts. "
        "Use this to understand how code entities, decisions, and preferences are connected."
    )
    parameters = {
        "type": "object",
        "properties": {
            "concept": {
                "type": "string",
                "description": "Concept to explore relationships for.",
            },
            "depth": {
                "type": "integer",
                "description": "Graph traversal depth (default 2).",
                "default": 2,
            },
            "action": {
                "type": "string",
                "enum": ["neighbors", "paths", "central", "clusters"],
                "description": "What to query (default: neighbors).",
                "default": "neighbors",
            },
            "target": {
                "type": "string",
                "description": "Target concept for path finding (used with action=paths).",
            },
        },
        "required": ["concept"],
    }

    def __init__(self, triple_store: Any) -> None:
        self._store = triple_store

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        concept: str,
        depth: int = 2,
        action: str = "neighbors",
        target: str = "",
    ) -> ToolResult:
        try:
            if action == "neighbors":
                subgraph = self._store.get_entity_subgraph(concept, max_depth=depth)
                lines = [f"## Knowledge Graph: {concept}\n"]
                if subgraph["outgoing"]:
                    lines.append(f"**Outgoing ({len(subgraph['outgoing'])})**:")
                    for t in subgraph["outgoing"][:15]:
                        lines.append(f"  {concept} → {t.predicate} → {t.object} (conf: {t.confidence:.2f})")
                if subgraph["incoming"]:
                    lines.append(f"\n**Incoming ({len(subgraph['incoming'])})**:")
                    for t in subgraph["incoming"][:15]:
                        lines.append(f"  {t.subject} → {t.predicate} → {concept} (conf: {t.confidence:.2f})")

            elif action == "paths" and target:
                paths = self._store.find_paths(concept, target, max_depth=depth)
                lines = [f"## Paths: {concept} → {target}\n"]
                if paths:
                    for i, path in enumerate(paths[:5], 1):
                        lines.append(f"\n**Path {i}** ({len(path)} hops):")
                        for t in path:
                            lines.append(f"  {t.subject} → {t.predicate} → {t.object}")
                else:
                    lines.append("No paths found.")

            elif action == "central":
                central = self._store.get_central_concepts(20)
                lines = ["## Most Central Concepts\n"]
                for c, d in central:
                    lines.append(f"  {c}: {d} connections")

            elif action == "clusters":
                clusters = self._store.get_concept_clusters(3)
                lines = [f"## Concept Clusters ({len(clusters)})\n"]
                for i, cluster in enumerate(clusters[:10], 1):
                    lines.append(f"  Cluster {i}: {', '.join(cluster[:15])}")

            else:
                return ToolResult.fail(self.name, "", f"Unknown action: {action}")

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output="\n".join(lines),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryWorkerStatusTool(Tool):
    """Show background worker statistics."""

    name = "memory_worker_status"
    description = "Show the background memory worker's processing statistics."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self) -> ToolResult:
        try:
            if not hasattr(self._layer, '_worker') or not self._layer._worker:
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="",
                    output="Memory worker not running.",
                )

            stats = self._layer._worker.get_stats()
            lines = ["## Memory Worker Status\n"]
            for key, value in stats.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

            return ToolResult.ok(
                tool_name=self.name, tool_call_id="",
                output="\n".join(lines),
                stats=stats,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))
