"""
Context Recall — retrieves relevant memories and injects them into LLM context.

This is the "recall" side of the memory system. Before each LLM call, the
recall system:
  1. Takes the current query/task
  2. Searches across all memory types (facts, rules, relationships, skills,
     preferences, sessions, triples)
  3. Ranks results by relevance and confidence
  4. Assembles a memory context block for the system prompt

This replaces the simple TF-IDF `build_context()` in Phase 10 with a
multi-source, ranked recall system.
"""

from __future__ import annotations

from typing import Any

from tracera.logging import get_logger

log = get_logger("memory.recall")


class ContextRecall:
    """
    Multi-source memory recall for LLM context injection.

    Searches across:
      - Structured memories (facts, rules, relationships, skills, preferences, events)
      - Session history (past sessions with similar tasks)
      - Semantic triples (knowledge graph relationships)

    Assembles results into a formatted context block.
    """

    def __init__(
        self,
        *,
        memory_store: Any = None,  # EnhancedMemoryStore (below)
        session_manager: Any = None,  # SessionManager
        triple_store: Any = None,  # TripleStore
        legacy_memory: Any = None,  # original AgentMemory (Phase 10)
    ) -> None:
        self._memory_store = memory_store
        self._session_manager = session_manager
        self._triple_store = triple_store
        self._legacy_memory = legacy_memory

    def recall(
        self,
        query: str,
        *,
        k: int = 15,
        max_chars: int = 6000,
        include_sessions: bool = True,
        include_triples: bool = True,
        include_legacy: bool = True,
    ) -> str:
        """
        Recall relevant memories for a query and assemble them into context.

        Args:
            query: The current task or question.
            k: Maximum number of memories to include.
            max_chars: Maximum characters for the assembled context.
            include_sessions: Whether to include session history.
            include_triples: Whether to include knowledge graph triples.
            include_legacy: Whether to include Phase 10 memories.

        Returns:
            Formatted context string ready for system prompt injection.
        """
        parts: list[str] = []
        chars_used = 0

        # 1. Structured memories (highest priority)
        if self._memory_store is not None:
            memories = self._memory_store.recall(query, k=k // 2)
            if memories:
                section = self._format_memories(memories)
                parts.append(section)
                chars_used += len(section)

        # 2. Session history (past similar sessions)
        if include_sessions and self._session_manager is not None:
            sessions = self._session_manager.find_similar_sessions(query, k=3)
            if sessions:
                section = self._format_sessions(sessions)
                if chars_used + len(section) <= max_chars:
                    parts.append(section)
                    chars_used += len(section)

        # 3. Knowledge graph triples
        if include_triples and self._triple_store is not None:
            triples = self._triple_store.search(query)
            if triples:
                section = self._format_triples(triples[:10])
                if chars_used + len(section) <= max_chars:
                    parts.append(section)
                    chars_used += len(section)

        # 4. Legacy Phase 10 memories (fallback)
        if include_legacy and self._legacy_memory is not None:
            legacy_ctx = self._legacy_memory.build_context(
                query, k=5, max_chars=max_chars - chars_used
            )
            if legacy_ctx:
                parts.append(legacy_ctx)

        if not parts:
            return ""

        result = "## Agent Memory\n\n" + "\n\n".join(parts)
        log.debug(
            "Recalled memories for %r: %d chars across %d sources",
            query[:40], len(result), len(parts),
        )
        return result[:max_chars]

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_memories(self, memories: list) -> str:
        """Format structured memories into a readable section."""
        lines = ["### Facts & Knowledge"]
        for mem in memories:
            icon = {
                "fact": "📌",
                "rule": "📏",
                "relationship": "🔗",
                "skill": "🛠️",
                "preference": "⭐",
                "event": "📋",
            }.get(mem.memory_type.value, "•")
            conf = f" ({mem.confidence:.0%})" if mem.confidence < 0.9 else ""
            lines.append(f"- {icon} {mem.content}{conf}")
        return "\n".join(lines)

    def _format_sessions(self, sessions: list) -> str:
        """Format past session summaries."""
        lines = ["### Past Sessions"]
        for session in sessions:
            duration = ""
            if session.duration_seconds:
                mins = int(session.duration_seconds / 60)
                duration = f" ({mins}m)" if mins > 0 else ""
            files = f", {len(session.files_touched)} files" if session.files_touched else ""
            summary = session.summary[:80] if session.summary else session.task[:80]
            lines.append(
                f"- [{session.outcome}] {summary}{duration}{files}"
            )
        return "\n".join(lines)

    def _format_triples(self, triples: list) -> str:
        """Format knowledge graph triples."""
        lines = ["### Relationships"]
        for triple in triples[:8]:
            lines.append(f"- {triple.subject} → {triple.predicate} → {triple.object}")
        return "\n".join(lines)


# ── Enhanced Memory Store ──────────────────────────────────────────────────────
# Bridges the new taxonomy with persistence, sitting alongside the legacy
# AgentMemory but using the richer MemoryType system.


class EnhancedMemoryStore:
    """
    Persistent store for structured memories using the enhanced taxonomy.

    Wraps a JSON file with CRUD operations, recall via TF-IDF similarity,
    and deduplication. This replaces the flat MemoryEntry system with
    typed, attributed, session-scoped memories.
    """

    MEMORY_FILE = "enhanced_memory.json"
    MAX_ENTRIES = 3000

    def __init__(self, memory_dir: Any) -> None:
        from pathlib import Path
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / self.MEMORY_FILE
        self._memories: dict[str, StructuredMemory] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                import json
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for d in data.get("memories", []):
                    mem = _memory_from_dict(d)
                    if mem:
                        self._memories[mem.id] = mem
                log.debug("Loaded %d enhanced memories", len(self._memories))
            except Exception as e:
                log.warning("Failed to load enhanced memory: %s", e)

    def _save(self) -> None:
        try:
            import json
            data = {
                "memories": [m.to_dict() for m in self._memories.values()]
            }
            self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.error("Failed to save enhanced memory: %s", e)

    def add(self, memory: StructuredMemory, *, dedup: bool = True) -> StructuredMemory:
        """Add a memory entry. If dedup is True, merges with existing similar entries."""
        if dedup:
            existing = self._find_duplicate(memory)
            if existing:
                existing.observe()
                existing.content = memory.content  # update content
                existing.updated_at = memory.updated_at
                self._save()
                return existing

        self._memories[memory.id] = memory
        self._prune()
        self._save()
        return memory

    def add_many(self, memories: list[StructuredMemory]) -> int:
        """Bulk add memories. Returns count actually added."""
        count = 0
        for mem in memories:
            added = self.add(mem)
            if added.id == mem.id:
                count += 1
        return count

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        memory_type: Any = None,
        min_confidence: float = 0.3,
    ) -> list[StructuredMemory]:
        """Retrieve the most relevant memories for a query using TF-IDF."""
        from tracera.agent.memory import _tokenize, _tfidf_score

        query_tokens = _tokenize(query)
        if not query_tokens:
            return list(self._memories.values())[:k]

        candidates = [
            m for m in self._memories.values()
            if m.confidence >= min_confidence
            and (memory_type is None or m.memory_type == memory_type)
        ]

        scored = []
        # Build IDF across all memories
        n = len(candidates) or 1
        df: dict[str, int] = {}
        for mem in candidates:
            tokens = set(_tokenize(mem.content + " " + " ".join(mem.tags)))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        idf = {t: __import__("math").log(n / c + 1) for t, c in df.items()}

        for mem in candidates:
            doc_tokens = _tokenize(mem.content + " " + " ".join(mem.tags))
            score = _tfidf_score(query_tokens, doc_tokens, idf)
            # Boost by importance, confidence, and frequency
            score *= (1 + mem.importance) * mem.confidence * (1 + mem.frequency * 0.1)
            if score > 0:
                mem.touch()
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:k]]
        if results:
            self._save()
        return results

    def get_by_type(self, memory_type: Any) -> list[StructuredMemory]:
        """Get all memories of a specific type."""
        return [m for m in self._memories.values() if m.memory_type == memory_type]

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save()
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._memories)

    def stats(self) -> dict:
        from collections import Counter
        by_type = Counter(m.memory_type.value for m in self._memories.values())
        return {
            "total": self.count,
            "by_type": dict(by_type),
        }

    def _find_duplicate(self, memory: StructuredMemory) -> StructuredMemory | None:
        """Find a very similar existing memory."""
        from tracera.agent.memory import _tokenize
        query_tokens = set(_tokenize(memory.content))
        for existing in self._memories.values():
            if existing.memory_type != memory.memory_type:
                continue
            doc_tokens = set(_tokenize(existing.content))
            if not doc_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            if overlap > 0.80:
                return existing
        return None

    def _prune(self) -> None:
        if len(self._memories) <= self.MAX_ENTRIES:
            return
        sorted_mems = sorted(
            self._memories.values(),
            key=lambda m: m.importance + m.confidence + (m.access_count * 0.05),
        )
        to_remove = sorted_mems[: len(sorted_mems) - self.MAX_ENTRIES + 100]
        for mem in to_remove:
            del self._memories[mem.id]


def _memory_from_dict(d: dict) -> StructuredMemory | None:
    """Reconstruct a StructuredMemory from a dict."""
    from tracera.memory.taxonomy import MemoryType

    mem_type = d.get("memory_type", "fact")
    try:
        mt = MemoryType(mem_type)
    except ValueError:
        return None

    cls_map = {
        MemoryType.FACT: MemoryFact,
        MemoryType.RULE: MemoryRule,
        MemoryType.RELATIONSHIP: MemoryRelationship,
        MemoryType.SKILL: MemorySkill,
        MemoryType.PREFERENCE: MemoryPreference,
        MemoryType.EVENT: MemoryEvent,
    }
    cls = cls_map.get(mt, StructuredMemory)
    return cls.from_dict(d)


# Import the specific memory classes
from tracera.memory.taxonomy import (
    MemoryEvent,
    MemoryFact,
    MemoryPreference,
    MemoryRelationship,
    MemoryRule,
    MemorySkill,
    StructuredMemory,
)
