"""
Memory Tools — let the agent interact with the enhanced memory system.

Legacy (JSON-backed) surface:
  - recall_memory: search across all memory sources for relevant context
  - remember_memory: explicitly store a memory (fact, rule, preference, etc.)
  - forget_memory: delete a memory by ID or content match
  - list_sessions: show past coding sessions and their outcomes
  - memory_stats: show memory system statistics
  - memory_consolidate: run consolidation to merge near-duplicates
  - memory_graph: query the knowledge graph
  - memory_worker_status: show background worker statistics

Agent-native layer (v2) surface — bi-temporal, graph, feedback, maintenance:
  - memory_timeline: how a belief changed over time
  - memory_entities: canonical entity graph
  - memory_update: correct a memory, keeping the previous version
  - memory_feedback: mark a recalled memory useful / harmful / irrelevant
  - memory_maintenance: decay + garbage-collect + consolidate
  - memory_explain: why a memory was (or wasn't) recalled
  - memory_export: export an entity's memories to a JSON file
  - memory_import: import a previously exported JSON file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.memory")


def _resolve_agent_memory(layer: Any) -> tuple[Any, str | None]:
    """
    Build an ``AgentMemory`` facade for the layer's current attribution scope.

    Returns ``(facade, error)``. Every v2 tool funnels through here so a missing
    or disabled layer degrades into a readable message instead of a traceback.
    """
    if layer is None:
        return None, "Agent-native memory layer is not available."
    try:
        from tracera.memory.layer.facade import AgentMemory

        return AgentMemory(layer), None
    except Exception as e:  # noqa: BLE001
        return None, f"Agent-native memory layer unavailable: {e}"


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

    async def execute(
        self, query: str, k: int = 10, use_graph_expansion: bool = True
    ) -> ToolResult:
        try:
            context = self._recall.recall(
                query,
                k=k,
                max_chars=8000,
                include_sessions=True,
                include_triples=True,
                include_legacy=True,
                use_graph_expansion=use_graph_expansion,
            )
            if not context:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output="No relevant memories found for this query.",
                    query=query,
                )
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
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
                "enum": [
                    "fact",
                    "rule",
                    "preference",
                    "relationship",
                    "skill",
                    "event",
                    "decision",
                    "goal",
                    "constraint",
                    "experience",
                    "attribute",
                ],
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
            from tracera.memory.taxonomy import (
                MemoryAttribute,
                MemoryConstraint,
                MemoryDecision,
                MemoryExperience,
                MemoryGoal,
                create_event,
                create_fact,
                create_preference,
                create_relationship,
                create_rule,
                create_skill,
            )

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
                memory = MemoryDecision(
                    content=content, importance=importance, session_id="", source="tool"
                )
            elif memory_type == "goal":
                memory = MemoryGoal(
                    content=content, importance=importance, session_id="", source="tool"
                )
            elif memory_type == "constraint":
                memory = MemoryConstraint(
                    content=content, importance=importance, session_id="", source="tool"
                )
            elif memory_type == "experience":
                memory = MemoryExperience(
                    content=content, importance=importance, session_id="", source="tool"
                )
            elif memory_type == "attribute":
                memory = MemoryAttribute(
                    content=content, importance=importance, session_id="", source="tool"
                )
            else:
                memory = create_fact(content, importance=importance)

            memory.confidence = confidence
            self._memory.add(memory)

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
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
                        tool_name=self.name,
                        tool_call_id="",
                        output=f"Memory {memory_id[:8]} deleted.",
                    )
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
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
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Deleted {deleted_count} memory matching '{content_match[:50]}'.",
                )

            return ToolResult.fail(
                self.name,
                "",
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
                    tool_name=self.name,
                    tool_call_id="",
                    output="No past sessions found.",
                )

            lines = ["## Recent Sessions\n"]
            for i, session in enumerate(sessions, 1):
                duration = ""
                if session.duration_seconds:
                    mins = int(session.duration_seconds / 60)
                    duration = f" ({mins}m)" if mins > 0 else f" ({int(session.duration_seconds)}s)"
                icon = {"success": "✅", "failure": "❌", "partial": "⚠️"}.get(session.outcome, "📋")
                files = f", {len(session.files_touched)} files" if session.files_touched else ""
                tools = (
                    ", ".join(f"{name}×{count}" for name, count in session.tools_used.items())
                    if session.tools_used
                    else "no tools"
                )
                lines.append(
                    f"{i}. {icon} [{session.outcome}] {session.task[:70]}{duration}{files}"
                )
                lines.append(f"   Tools: {tools}")
                if session.summary:
                    lines.append(f"   Summary: {session.summary[:120]}")
                lines.append("")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
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
                lines.append("\n**Knowledge Graph**:")
                lines.append(f"  Triples: {self._triple_store.triple_count}")
                lines.append(f"  Nodes: {self._triple_store.node_count}")
                lines.append(f"  Edges: {self._triple_store.edge_count}")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
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
                tool_name=self.name,
                tool_call_id="",
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
                "description": "Concept to explore relationships for. Omit for a graph overview.",
            },
            "depth": {
                "type": "integer",
                "description": "Graph traversal depth (default 2).",
                "default": 2,
            },
            "action": {
                "type": "string",
                "enum": ["neighbors", "paths", "central", "clusters", "overview"],
                "description": "What to query (default: neighbors when a concept is given, overview otherwise).",
                "default": "neighbors",
            },
            "target": {
                "type": "string",
                "description": "Target concept for path finding (used with action=paths).",
            },
        },
        "required": [],
    }

    def __init__(self, triple_store: Any) -> None:
        self._store = triple_store

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        concept: str = "",
        depth: int = 2,
        action: str = "neighbors",
        target: str = "",
    ) -> ToolResult:
        try:
            if not concept:
                # Argless call (/memgraph2): graph overview instead of an error.
                total = getattr(self._store, "triple_count", 0)
                central = self._store.get_central_concepts(10)
                lines = ["## Knowledge Graph Overview\n"]
                lines.append(f"  Triples: {total}")
                if central:
                    lines.append("\n**Most connected concepts:**")
                    for c, d in central:
                        lines.append(f"  {c}: {d} connections")
                else:
                    lines.append("  No concepts yet — memories build the graph as you chat.")
                return ToolResult.ok(tool_name=self.name, tool_call_id="", output="\n".join(lines))

            if action == "neighbors":
                subgraph = self._store.get_entity_subgraph(concept, max_depth=depth)
                lines = [f"## Knowledge Graph: {concept}\n"]
                if subgraph["outgoing"]:
                    lines.append(f"**Outgoing ({len(subgraph['outgoing'])})**:")
                    for t in subgraph["outgoing"][:15]:
                        lines.append(
                            f"  {concept} → {t.predicate} → {t.object} (conf: {t.confidence:.2f})"
                        )
                if subgraph["incoming"]:
                    lines.append(f"\n**Incoming ({len(subgraph['incoming'])})**:")
                    for t in subgraph["incoming"][:15]:
                        lines.append(
                            f"  {t.subject} → {t.predicate} → {concept} (conf: {t.confidence:.2f})"
                        )

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
                tool_name=self.name,
                tool_call_id="",
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
            if not hasattr(self._layer, "_worker") or not self._layer._worker:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output="Memory worker not running.",
                )

            stats = self._layer._worker.get_stats()
            lines = ["## Memory Worker Status\n"]
            for key, value in stats.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                stats=stats,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Agent-native memory layer (v2) tools
#
# These reach the SQLite bi-temporal store rather than the legacy JSON files:
# temporal history, canonical entity graph, relevance feedback, decay/GC, and
# portable export/import.
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryTimelineTool(Tool):
    """Show how a belief changed over time."""

    name = "memory_timeline"
    description = (
        "Show the chronological history of a memory, including versions that are "
        "no longer current. Use this to answer 'what did we believe before?' or "
        "to see when a fact was superseded."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Limit to one subject (e.g. 'user'). Optional.",
            },
            "predicate": {
                "type": "string",
                "description": "Limit to one relation (e.g. 'employer'). Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum versions to return (default 50).",
                "default": 50,
            },
        },
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self, subject: str = "", predicate: str = "", limit: int = 50
    ) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            rows = facade.timeline(
                subject=subject or None, predicate=predicate or None
            )[:limit]
            if not rows:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output="No memory history found for this scope.",
                )

            lines = ["## Memory Timeline\n"]
            for rec in rows:
                marker = "●" if rec.is_current else "○"
                when = rec.valid_at or rec.first_seen_at
                stamp = (
                    __import__("datetime")
                    .datetime.fromtimestamp(when)
                    .strftime("%Y-%m-%d")
                    if when
                    else "unknown"
                )
                state = "current" if rec.is_current else (rec.status or "past")
                lines.append(f"{marker} [{stamp}] ({state}) {rec.text}")
                if rec.invalid_at and rec.superseded_by:
                    lines.append(f"    └ superseded by memory {rec.superseded_by}")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                count=len(rows),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryEntitiesTool(Tool):
    """Inspect the canonical entity graph."""

    name = "memory_entities"
    description = (
        "Show the memory graph: canonical entities and the relationships between "
        "them. Use this to see what concepts the agent knows and how they connect."
    )
    parameters = {
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
                "description": "Explore one node's edges. Omit for a graph overview.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum edges/nodes to show (default 50).",
                "default": 50,
            },
        },
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, node: str = "", limit: int = 50) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            graph = facade.entities(node=node or None)

            if node:
                # ``entity_graph`` returns asymmetric edge dicts: an outgoing edge
                # names only its ``dst`` (the ``src`` is the queried node) and an
                # incoming edge names only its ``src``. Render from ``graph["node"]``
                # so neither side prints as None.
                focus = graph.get("node") or node
                lines = [f"## Entity: {focus}\n"]
                outgoing = graph.get("outgoing", [])[:limit]
                incoming = graph.get("incoming", [])[:limit]
                if outgoing:
                    lines.append(f"**Outgoing ({len(outgoing)})**:")
                    for e in outgoing:
                        lines.append(
                            f"  {focus} \u2192 {e.get('predicate')} \u2192 {e.get('dst')}"
                        )
                if incoming:
                    lines.append(f"\n**Incoming ({len(incoming)})**:")
                    for e in incoming:
                        lines.append(
                            f"  {e.get('src')} \u2192 {e.get('predicate')} \u2192 {focus}"
                        )
                if not outgoing and not incoming:
                    lines.append("  No relationships recorded for this node.")
                return ToolResult.ok(
                    tool_name=self.name, tool_call_id="", output="\n".join(lines), graph=graph
                )

            nodes = graph.get("nodes", [])[:limit]
            lines = [f"## Memory Graph ({len(nodes)} entities)\n"]
            for n in nodes:
                label = n.get("canonical") or n.get("name")
                kind = n.get("kind") or "concept"
                mentions = n.get("mentions", n.get("degree", 0))
                lines.append(f"  {label} ({kind}) \u2014 {mentions} mentions")
            if not nodes:
                lines.append("  No entities yet \u2014 memories build the graph as you chat.")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                graph=graph,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryUpdateTool(Tool):
    """Correct a memory, keeping the previous version."""

    name = "memory_update"
    description = (
        "Correct a stored memory. The previous value is retired, not erased, so "
        "the history stays queryable via memory_timeline. Use this when a fact "
        "has changed (e.g. the user switched editors)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "The memory to correct."},
            "text": {"type": "string", "description": "The corrected content."},
            "reason": {
                "type": "string",
                "description": "Why it changed (stored in the audit trail).",
            },
            "object": {
                "type": "string",
                "description": "Corrected object value (e.g. the new employer). Optional.",
            },
            "predicate": {
                "type": "string",
                "description": "Corrected relation, if the slot itself was wrong. Optional.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the corrected value, 0.0-1.0 (default 0.8).",
                "default": 0.8,
            },
        },
        "required": ["memory_id", "text"],
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        memory_id: int,
        text: str,
        reason: str = "explicit update",
        object: str = "",
        predicate: str = "",
        confidence: float = 0.8,
    ) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            new = facade.update(
                int(memory_id),
                text=text,
                reason=reason,
                object=object or None,
                predicate=predicate or None,
                confidence=confidence,
            )
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=(
                    f"Memory {memory_id} corrected → new memory {new.id}. "
                    "The previous version is retained in the timeline."
                ),
                memory_id=new.id,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryFeedbackTool(Tool):
    """Record whether a recalled memory actually helped."""

    name = "memory_feedback"
    description = (
        "Tell the memory system whether a recalled memory was useful, harmful, or "
        "irrelevant. This adjusts its importance so good memories rank higher and "
        "noise fades out."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "The memory to rate."},
            "signal": {
                "type": "string",
                "enum": ["useful", "harmful", "irrelevant"],
                "description": "How the memory performed in this turn.",
            },
            "query": {
                "type": "string",
                "description": "The query it was recalled for. Optional.",
            },
        },
        "required": ["memory_id", "signal"],
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self, memory_id: int, signal: str, query: str = ""
    ) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            result = facade.feedback(
                int(memory_id), signal, query=query or None
            )
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=(
                    f"Feedback recorded: memory {memory_id} marked '{signal}'. "
                    f"Importance is now {result['importance']:.2f}."
                ),
                stats=result,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryMaintenanceTool(Tool):
    """Run decay, garbage collection and consolidation."""

    name = "memory_maintenance"
    description = (
        "Run a maintenance pass: recompute memory decay, archive stale memories, "
        "and merge near-duplicates. Keeps long-lived memory from growing without "
        "bound. Use dry_run to preview."
    )
    parameters = {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "Report what would change without changing it.",
                "default": False,
            },
            "consolidate": {
                "type": "boolean",
                "description": "Also merge near-duplicate memories (default true).",
                "default": True,
            },
        },
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self, dry_run: bool = False, consolidate: bool = True
    ) -> ToolResult:
        if self._layer is None:
            return ToolResult.fail(
                self.name, "", "Agent-native memory layer is not available."
            )
        try:
            report = self._layer.run_maintenance(
                dry_run=dry_run, consolidate=consolidate
            )
            gc = report.get("gc", {})
            lines = ["## Memory Maintenance\n"]
            lines.append(f"**Decay scores updated**: {report.get('decay_updated', 0)}")
            lines.append(
                f"**GC candidates**: {gc.get('candidates', 0)}"
                f" ({'would archive' if dry_run else 'archived'}: "
                f"{gc.get('candidates', 0) if dry_run else gc.get('archived', 0)})"
            )
            consolidation = report.get("consolidation")
            if consolidation:
                lines.append(
                    f"**Consolidation**: scanned {consolidation.get('scanned', 0)}, "
                    f"merged {consolidation.get('merged', 0)}"
                )
            if dry_run:
                lines.append("\n_Dry run — nothing was changed._")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                stats=report,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryExplainTool(Tool):
    """Explain why a memory was (or wasn't) recalled."""

    name = "memory_explain"
    description = (
        "Explain the recall decision for a query: every candidate inspected, its "
        "score breakdown, and why it was injected or rejected. Use this when the "
        "agent recalled the wrong thing (or nothing)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to explain recall for.",
            },
            "k": {
                "type": "integer",
                "description": "Number of candidates to inspect (default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, query: str, k: int = 10) -> ToolResult:
        if self._layer is None:
            return ToolResult.fail(
                self.name, "", "Agent-native memory layer is not available."
            )
        try:
            from types import SimpleNamespace

            from tracera.memory.layer.attribution import (
                current_attribution,
            )

            scope = current_attribution() or self._layer._default_attribution
            if scope is None:
                return ToolResult.fail(
                    self.name, "", "No attribution scope set for memory recall."
                )

            debug = self._layer._recaller.debug_recall(
                query,
                SimpleNamespace(
                    entity_id=scope.entity_id, process_id=scope.process_id
                ),
                k=k,
            )

            lines = [
                "## Recall Explanation\n",
                f"**Query**: {query}",
                f"**Inspected**: {debug.get('returned', 0)} candidates  "
                f"| **Injected**: {debug.get('would_inject', 0)}",
                "",
            ]
            for i, r in enumerate(debug.get("results", [])[:k], 1):
                flag = "INJECT" if r.get("above_threshold") else "reject"
                lines.append(
                    f"{i}. [{flag}] score={r.get('score', 0.0):.3f}  {r.get('text', '')[:90]}"
                )
                reason = r.get("why_recalled") or ""
                if reason:
                    lines.append(f"    why: {reason}")
            if not debug.get("results"):
                lines.append("  No candidates in this scope.")

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output="\n".join(lines),
                debug=debug,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryExportTool(Tool):
    """Export this scope's memories and graph to a JSON file."""

    name = "memory_export"
    description = (
        "Export the agent's memories and entity graph for this scope to a JSON "
        "file. Embeddings are omitted (they are regenerated on import), so the "
        "file stays small and diffable."
    )
    parameters = {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "Where to write the JSON file.",
            },
            "include_invalidated": {
                "type": "boolean",
                "description": "Include superseded/invalidated versions (default true).",
                "default": True,
            },
        },
        "required": ["output_path"],
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self, output_path: str, include_invalidated: bool = True
    ) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            payload = facade.export(include_invalidated=include_invalidated)
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=(
                    f"Exported {len(payload.get('memories', []))} memories to {path}"
                ),
                path=str(path),
                count=len(payload.get("memories", [])),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))


class MemoryImportTool(Tool):
    """Import memories from a previously exported JSON file."""

    name = "memory_import"
    description = (
        "Import memories and graph edges from a JSON file produced by "
        "memory_export. Import is idempotent — re-importing the same file adds "
        "nothing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "The JSON file to import.",
            },
        },
        "required": ["input_path"],
    }

    def __init__(self, memory_layer: Any) -> None:
        self._layer = memory_layer

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(self, input_path: str) -> ToolResult:
        facade, err = _resolve_agent_memory(self._layer)
        if err:
            return ToolResult.fail(self.name, "", err)
        try:
            path = Path(input_path).expanduser()
            payload = json.loads(path.read_text(encoding="utf-8"))
            stats = facade.import_data(payload)
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=(
                    f"Imported {stats.get('imported', 0)} memories "
                    f"from {path} ({stats.get('skipped', 0)} already present)."
                ),
                stats=stats,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e))
