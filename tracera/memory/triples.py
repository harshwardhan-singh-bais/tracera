"""
Semantic Triple Store — builds a knowledge graph from extracted relationships.

Triples are subject→predicate→object statements:
  - "AuthMiddleware" → "calls" → "UserService"
  - "test_auth.py" → "tests" → "AuthMiddleware"
  - "user" → "prefers" → "pytest"

The triple store builds on NetworkX (already used by the symbol graph) and
provides:
  - Triple insertion and deduplication
  - Forward/backward traversal
  - Concept-to-code-symbol bridging
  - Persistence to JSON

This is a *conceptual* knowledge graph about the project — different from
the symbol graph which tracks code-level relationships (imports, calls).
The triple store captures higher-level understanding: conventions, preferences,
architectural decisions, and cross-cutting concerns.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.logging import get_logger

log = get_logger("memory.triples")


@dataclass
class Triple:
    """A single subject→predicate→object statement."""

    subject: str
    predicate: str
    object: str
    # Metadata
    confidence: float = 0.8
    source: str = ""  # where this triple came from
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    frequency: int = 1  # how many times observed

    @property
    def key(self) -> str:
        """Dedup key: normalized triple."""
        return f"{self.subject.lower()}|{self.predicate.lower()}|{self.object.lower()}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Triple":
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            confidence=d.get("confidence", 0.8),
            source=d.get("source", ""),
            session_id=d.get("session_id", ""),
            created_at=d.get("created_at", 0.0),
            frequency=d.get("frequency", 1),
        )


# Common predicates (for normalization)
_PREDICATE_ALIASES: dict[str, str] = {
    "calls": "calls",
    "uses": "uses",
    "depends_on": "depends_on",
    "depends": "depends_on",
    "imports": "imports",
    "inherits": "inherits",
    "extends": "inherits",
    "implements": "implements",
    "contains": "contains",
    "has": "contains",
    "tests": "tests",
    "tests_for": "tests",
    "covers": "tests",
    "configures": "configures",
    "configured_by": "configured_by",
    "prefers": "prefers",
    "likes": "prefers",
    "requires": "requires",
    "produces": "produces",
    "consumes": "consumes",
    "relates_to": "relates_to",
    "related_to": "relates_to",
}


def _normalize_predicate(pred: str) -> str:
    """Normalize a predicate string."""
    return _PREDICATE_ALIASES.get(pred.lower().strip(), pred.lower().strip())


class TripleStore:
    """
    A knowledge graph of semantic triples.

    Uses NetworkX for graph operations (same as the symbol graph).
    Nodes are concepts (strings), edges are typed predicates.

    Usage:
        store = TripleStore()
        store.add_triple(Triple("AuthMiddleware", "calls", "UserService"))
        callers = store.get_subjects("UserService", predicate="calls")
        all_related = store.get_neighbors("AuthMiddleware")
    """

    TRIPLES_FILE = "memory_triples.json"

    def __init__(self) -> None:
        try:
            import networkx as nx
            self._g: Any = nx.MultiDiGraph()
        except ImportError:
            raise RuntimeError("networkx not installed. Run: uv add networkx")
        self._triples: dict[str, Triple] = {}  # key → Triple

    # ── Adding triples ────────────────────────────────────────────────────────

    def add_triple(self, triple: Triple, *, dedup: bool = True) -> None:
        """
        Add a triple to the store.

        If dedup is True and an identical triple exists, increments its
        frequency and updates confidence.
        """
        triple.predicate = _normalize_predicate(triple.predicate)
        key = triple.key

        if dedup and key in self._triples:
            existing = self._triples[key]
            existing.frequency += 1
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.source = triple.source or existing.source
            return

        self._triples[key] = triple
        self._g.add_edge(
            triple.subject,
            triple.object,
            predicate=triple.predicate,
            confidence=triple.confidence,
            key=key,
        )

    def add_triples(self, triples: list[Triple]) -> int:
        """Bulk-add triples. Returns the number actually added (after dedup)."""
        before = len(self._triples)
        for triple in triples:
            self.add_triple(triple)
        return len(self._triples) - before

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_subjects(
        self,
        object_name: str,
        predicate: str | None = None,
    ) -> list[Triple]:
        """Find all triples where object_name is the object (who uses/calls/contains it)."""
        results = []
        for triple in self._triples.values():
            if triple.object.lower() == object_name.lower():
                if predicate is None or triple.predicate == predicate:
                    results.append(triple)
        return sorted(results, key=lambda t: t.confidence, reverse=True)

    def get_objects(
        self,
        subject_name: str,
        predicate: str | None = None,
    ) -> list[Triple]:
        """Find all triples where subject_name is the subject (what it uses/calls/contains)."""
        results = []
        for triple in self._triples.values():
            if triple.subject.lower() == subject_name.lower():
                if predicate is None or triple.predicate == predicate:
                    results.append(triple)
        return sorted(results, key=lambda t: t.confidence, reverse=True)

    def get_neighbors(self, concept: str, max_depth: int = 2) -> list[Triple]:
        """
        Get all triples within max_depth hops of a concept.
        Returns triples where the concept appears as subject or object.
        """
        visited = set()
        results = []
        frontier = [concept]

        for _ in range(max_depth):
            next_frontier = []
            for name in frontier:
                if name.lower() in visited:
                    continue
                visited.add(name.lower())
                # Forward: subject → object
                for triple in self.get_objects(name):
                    results.append(triple)
                    if triple.object.lower() not in visited:
                        next_frontier.append(triple.object)
                # Backward: object → subject
                for triple in self.get_subjects(name):
                    results.append(triple)
                    if triple.subject.lower() not in visited:
                        next_frontier.append(triple.subject)
            frontier = next_frontier

        return results

    def find_by_predicate(self, predicate: str) -> list[Triple]:
        """Find all triples with a specific predicate."""
        pred = _normalize_predicate(predicate)
        return [t for t in self._triples.values() if t.predicate == pred]

    def search(self, query: str) -> list[Triple]:
        """Simple keyword search across all triples."""
        query_lower = query.lower()
        results = []
        for triple in self._triples.values():
            if (
                query_lower in triple.subject.lower()
                or query_lower in triple.predicate.lower()
                or query_lower in triple.object.lower()
            ):
                results.append(triple)
        return sorted(results, key=lambda t: t.confidence, reverse=True)

    @property
    def all_triples(self) -> list[Triple]:
        return list(self._triples.values())

    @property
    def triple_count(self) -> int:
        return len(self._triples)

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_memory_triples(self) -> list[Triple]:
        """Return all triples (for persistence)."""
        return list(self._triples.values())

    def to_text(self) -> str:
        """Render triples as readable text for LLM context."""
        lines = []
        for triple in sorted(self._triples.values(), key=lambda t: t.confidence, reverse=True):
            lines.append(f"- {triple.subject} → {triple.predicate} → {triple.object}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save triples to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.to_dict() for t in self._triples.values()]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.debug("Saved %d triples to %s", len(self._triples), path)

    @classmethod
    def load(cls, path: Path) -> "TripleStore":
        """Load triples from JSON."""
        store = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for d in data:
                    store.add_triple(Triple.from_dict(d))
                log.debug("Loaded %d triples from %s", len(store._triples), path)
            except Exception as e:
                log.warning("Failed to load triples: %s", e)
        return store

    def __repr__(self) -> str:
        return f"<TripleStore triples={self.triple_count} nodes={self.node_count}>"

    # ── Graph-Backed Recall (Phase 9) ──────────────────────────────────────────

    def expand_query_with_graph(
        self,
        query: str,
        *,
        max_hops: int = 2,
        max_results: int = 20,
        min_confidence: float = 0.5,
    ) -> list[Triple]:
        """
        Expand a query using graph traversal.

        Given a query, find relevant nodes and traverse the graph to find
        related concepts. This enables "graph-backed recall" where memories
        connected via the knowledge graph are also retrieved.
        """
        # First, find direct matches
        direct_matches = self.search(query)
        if not direct_matches:
            return []

        # Collect seed concepts from matches
        seed_concepts = set()
        for triple in direct_matches[:5]:
            seed_concepts.add(triple.subject.lower())
            seed_concepts.add(triple.object.lower())

        # Traverse from each seed concept
        expanded: list[Triple] = []
        visited = set()

        for concept in seed_concepts:
            if concept in visited:
                continue
            visited.add(concept)

            neighbors = self.get_neighbors(concept, max_depth=max_hops)
            for triple in neighbors:
                if triple.confidence >= min_confidence:
                    expanded.append(triple)

        # Deduplicate and sort by confidence
        seen = set()
        unique = []
        for triple in expanded:
            if triple.key not in seen:
                seen.add(triple.key)
                unique.append(triple)

        unique.sort(key=lambda t: t.confidence, reverse=True)
        return unique[:max_results]

    def get_entity_subgraph(
        self,
        entity: str,
        *,
        max_depth: int = 2,
        min_confidence: float = 0.5,
    ) -> dict[str, list[Triple]]:
        """
        Get the subgraph centered on an entity.

        Returns a dict with 'outgoing', 'incoming', and 'neighbors' keys
        containing the relevant triples for building context about an entity.
        """
        outgoing = [
            t for t in self.get_objects(entity)
            if t.confidence >= min_confidence
        ]
        incoming = [
            t for t in self.get_subjects(entity)
            if t.confidence >= min_confidence
        ]
        neighbors = self.get_neighbors(entity, max_depth=max_depth)
        neighbors = [t for t in neighbors if t.confidence >= min_confidence]

        return {
            "outgoing": outgoing[:50],
            "incoming": incoming[:50],
            "neighbors": neighbors[:100],
        }

    def find_paths(
        self,
        source: str,
        target: str,
        *,
        max_depth: int = 4,
    ) -> list[list[Triple]]:
        """
        Find paths between two concepts in the knowledge graph.

        Returns a list of paths (each path is a list of triples).
        Useful for explaining how two concepts are related.
        """
        try:
            import networkx as nx
        except ImportError:
            return []

        source_l = source.lower()
        target_l = target.lower()

        if source_l not in self._g or target_l not in self._g:
            return []

        paths = []
        try:
            for path in nx.all_simple_paths(
                self._g, source=source_l, target=target_l, cutoff=max_depth
            ):
                if len(path) < 2:
                    continue
                triple_path = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # Find the triple connecting these nodes
                    for triple in self._triples.values():
                        if (triple.subject.lower() == u and triple.object.lower() == v) or \
                           (triple.subject.lower() == v and triple.object.lower() == u):
                            triple_path.append(triple)
                            break
                if triple_path:
                    paths.append(triple_path)
        except nx.NetworkXNoPath:
            pass

        return paths

    def get_central_concepts(self, top_n: int = 20) -> list[tuple[str, int]]:
        """Get the most connected concepts in the graph (by degree)."""
        degrees = [(node, self._g.degree(node)) for node in self._g.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_n]

    def get_concept_clusters(self, min_cluster_size: int = 3) -> list[list[str]]:
        """Find clusters of closely related concepts using connected components."""
        try:
            import networkx as nx
        except ImportError:
            return []

        # Convert to undirected for clustering
        undirected = self._g.to_undirected()
        clusters = list(nx.connected_components(undirected))
        clusters = [list(c) for c in clusters if len(c) >= min_cluster_size]
        clusters.sort(key=len, reverse=True)
        return clusters
