"""
TRACERA Persistent Agent Memory — Phase 10.

JSON-backed memory store for project facts, user preferences, and past decisions.
Supports semantic retrieval via TF-IDF similarity (upgradeable to dense embeddings in phase 17+).
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tracera.logging import get_logger, log_memory

log = get_logger("agent.memory")


# ── Memory types ──────────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    PROJECT_FACT = "project_fact"         # static facts about the codebase
    USER_PREFERENCE = "user_preference"   # how the user likes things done
    PAST_DECISION = "past_decision"       # architectural/design decisions made
    TASK_CONTEXT = "task_context"         # context relevant to current task
    ERROR_PATTERN = "error_pattern"       # recurring errors and their fixes
    CODE_LOCATION = "code_location"       # where specific things are in the codebase


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    category: MemoryCategory
    content: str
    source: str = ""          # where this memory came from (e.g. "user", "agent", "file")
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5   # 0.0–1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update access stats."""
        self.access_count += 1
        self.last_accessed = time.time()

    def update_content(self, new_content: str) -> None:
        self.content = new_content
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "content": self.content,
            "source": self.source,
            "tags": self.tags,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d["id"],
            category=MemoryCategory(d["category"]),
            content=d["content"],
            source=d.get("source", ""),
            tags=d.get("tags", []),
            importance=d.get("importance", 0.5),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed"),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", "↵")
        return f"<MemoryEntry [{self.category.value}] {preview!r}>"


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """Compute TF-IDF dot product between query and document."""
    doc_counter = Counter(doc_tokens)
    doc_len = len(doc_tokens) or 1
    score = 0.0
    for token in query_tokens:
        tf = doc_counter.get(token, 0) / doc_len
        score += tf * idf.get(token, 0.0)
    return score


# ── Agent Memory ──────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Persistent memory store for TRACERA agents.
    
    Storage: JSON file at {memory_dir}/memory.json
    Retrieval: TF-IDF similarity (upgradeable to dense embeddings in phase 17+)
    
    The memory system answers questions like:
    - "What do I know about this project?"
    - "What has the user asked me not to do?"
    - "Have I solved this kind of problem before?"
    """

    MEMORY_FILE = "memory.json"
    MAX_ENTRIES = 5000

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.memory_dir / self.MEMORY_FILE
        self._entries: dict[str, MemoryEntry] = {}
        self._idf: dict[str, float] = {}
        self._load()

    # ── Storage ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                entries = data.get("entries", [])
                for d in entries:
                    entry = MemoryEntry.from_dict(d)
                    self._entries[entry.id] = entry
                log.debug("Loaded %d memory entries", len(self._entries))
            except Exception as e:
                log.warning("Failed to load memory: %s", e)
        self._rebuild_idf()

    def _save(self) -> None:
        try:
            data = {"entries": [e.to_dict() for e in self._entries.values()]}
            self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.error("Failed to save memory: %s", e)

    def _rebuild_idf(self) -> None:
        """Recompute IDF scores across all entries."""
        if not self._entries:
            self._idf = {}
            return
        n = len(self._entries)
        df: dict[str, int] = Counter()
        for entry in self._entries.values():
            tokens = set(_tokenize(entry.content + " " + " ".join(entry.tags)))
            df.update(tokens)
        self._idf = {token: math.log(n / count + 1) for token, count in df.items()}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.PROJECT_FACT,
        *,
        source: str = "agent",
        tags: list[str] | None = None,
        importance: float = 0.5,
        deduplicate: bool = True,
        **metadata: Any,
    ) -> MemoryEntry:
        """
        Add a new memory entry.
        
        If *deduplicate* is True and a very similar entry exists, updates it instead.
        """
        if deduplicate:
            existing = self._find_duplicate(content, category)
            if existing:
                existing.update_content(content)
                existing.importance = max(existing.importance, importance)
                log_memory("update", existing.id[:8])
                self._rebuild_idf()
                self._save()
                return existing

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            category=category,
            content=content,
            source=source,
            tags=tags or [],
            importance=importance,
            metadata=metadata,
        )
        self._entries[entry.id] = entry
        log_memory("add", f"{category.value}:{entry.id[:8]}")

        # Prune if too large
        if len(self._entries) > self.MAX_ENTRIES:
            self._prune()

        self._rebuild_idf()
        self._save()
        return entry

    def update(self, entry_id: str, content: str) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.update_content(content)
            self._rebuild_idf()
            self._save()
        return entry

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._rebuild_idf()
            self._save()
            return True
        return False

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        k: int = 10,
        category: MemoryCategory | None = None,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """
        Retrieve the top-k most relevant memory entries for *query*.
        
        Uses TF-IDF similarity. Upgradeable to dense embeddings in phase 17.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return list(self._entries.values())[:k]

        candidates = [
            e for e in self._entries.values()
            if e.importance >= min_importance
            and (category is None or e.category == category)
        ]

        scored = []
        for entry in candidates:
            doc_tokens = _tokenize(entry.content + " " + " ".join(entry.tags))
            score = _tfidf_score(query_tokens, doc_tokens, self._idf)
            # Boost by importance and recency
            score *= (1 + entry.importance)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, entry in scored[:k]:
            if score > 0:
                entry.touch()
                results.append(entry)

        if results:
            self._save()  # persist access stats
        return results

    def get_by_category(self, category: MemoryCategory) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.category == category]

    # ── Context building ──────────────────────────────────────────────────────

    def build_context(
        self,
        query: str,
        *,
        k: int = 8,
        max_chars: int = 4000,
    ) -> str:
        """
        Build a memory context string for inclusion in LLM prompts.
        """
        entries = self.retrieve(query, k=k)
        if not entries:
            return ""

        parts = ["## Relevant Memory\n"]
        chars_used = len(parts[0])

        for entry in entries:
            line = f"[{entry.category.value}] {entry.content}\n"
            if chars_used + len(line) > max_chars:
                break
            parts.append(line)
            chars_used += len(line)

        return "".join(parts)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        by_category: dict[str, int] = Counter(
            e.category.value for e in self._entries.values()
        )
        return {
            "total": self.count,
            "by_category": dict(by_category),
            "memory_file": str(self._file),
        }

    # ── Convenience methods ───────────────────────────────────────────────────

    def remember_project_fact(self, fact: str, **kw: Any) -> MemoryEntry:
        return self.add(fact, MemoryCategory.PROJECT_FACT, importance=0.7, **kw)

    def remember_preference(self, pref: str, **kw: Any) -> MemoryEntry:
        return self.add(pref, MemoryCategory.USER_PREFERENCE, importance=0.9, source="user", **kw)

    def remember_decision(self, decision: str, **kw: Any) -> MemoryEntry:
        return self.add(decision, MemoryCategory.PAST_DECISION, importance=0.8, **kw)

    def remember_error_pattern(self, pattern: str, **kw: Any) -> MemoryEntry:
        return self.add(pattern, MemoryCategory.ERROR_PATTERN, importance=0.6, **kw)

    def remember_code_location(self, location: str, **kw: Any) -> MemoryEntry:
        return self.add(location, MemoryCategory.CODE_LOCATION, importance=0.5, **kw)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_duplicate(
        self, content: str, category: MemoryCategory
    ) -> MemoryEntry | None:
        """Find an entry with very similar content in the same category."""
        query_tokens = set(_tokenize(content))
        for entry in self._entries.values():
            if entry.category != category:
                continue
            doc_tokens = set(_tokenize(entry.content))
            if not doc_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            if overlap > 0.85:
                return entry
        return None

    def _prune(self) -> None:
        """Remove least-important, least-accessed entries."""
        entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance + (e.access_count * 0.1)),
        )
        to_remove = entries[:len(entries) - self.MAX_ENTRIES + 100]
        for entry in to_remove:
            del self._entries[entry.id]
        log.info("Pruned %d memory entries", len(to_remove))

    def __repr__(self) -> str:
        return f"<AgentMemory entries={self.count} file={self._file}>"
