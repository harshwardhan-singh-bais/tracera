"""
Phase 25 — Symbol Relationship Graph.

Builds a directed graph where nodes are code symbols and edges represent
relationships: imports, calls, inherits, implements, references.

Uses NetworkX under the hood for graph operations.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from tracera.indexer.schema import Symbol, SymbolType
from tracera.logging import get_logger

log = get_logger("graph.symbol_graph")


class RelationType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    REFERENCES = "references"
    CONTAINS = "contains"     # class contains method


class SymbolGraph:
    """
    A directed graph of code symbols and their relationships.

    Nodes: symbol qualified names (e.g. "auth.py::AuthMiddleware")
    Edges: typed relationships (imports, calls, inherits, ...)

    Usage:
        graph = SymbolGraph()
        graph.add_symbol("auth.py", Symbol(...))
        graph.add_relation("auth.py::login", "auth.py::AuthMiddleware", RelationType.CALLS)
        callers = graph.get_callers("auth.py::AuthMiddleware")
    """

    def __init__(self) -> None:
        try:
            import networkx as nx
            self._g: Any = nx.DiGraph()
        except ImportError:
            raise RuntimeError("networkx not installed. Run: uv add networkx")

    def _node_id(self, file_path: str, symbol_name: str) -> str:
        return f"{file_path}::{symbol_name}"

    # ── Adding nodes ──────────────────────────────────────────────────────────

    def add_symbol(self, file_path: str, symbol: Symbol) -> str:
        """Add a Symbol as a graph node. Returns the node ID."""
        node_id = self._node_id(file_path, symbol.name)
        self._g.add_node(
            node_id,
            name=symbol.name,
            symbol_type=symbol.type.value,
            file_path=file_path,
            start_line=symbol.range.start_line,
            end_line=symbol.range.end_line,
            parent=symbol.parent_symbol,
        )
        # Add containment edge: parent class → method
        if symbol.parent_symbol:
            parent_id = self._node_id(file_path, symbol.parent_symbol)
            self._g.add_edge(parent_id, node_id, relation=RelationType.CONTAINS.value)
        return node_id

    def add_relation(
        self,
        from_symbol: str,
        to_symbol: str,
        relation: RelationType,
        file_path: str = "",
    ) -> None:
        """Add a directed relationship edge between two symbol node IDs."""
        self._g.add_edge(from_symbol, to_symbol, relation=relation.value)

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> dict | None:
        if self._g.has_node(node_id):
            return dict(self._g.nodes[node_id])
        return None

    def get_callers(self, node_id: str) -> list[str]:
        """Return all node IDs that call or reference this symbol."""
        return [
            src for src, dst, data in self._g.in_edges(node_id, data=True)
            if data.get("relation") in (RelationType.CALLS.value, RelationType.REFERENCES.value)
        ]

    def get_callees(self, node_id: str) -> list[str]:
        """Return all node IDs called by this symbol."""
        return [
            dst for src, dst, data in self._g.out_edges(node_id, data=True)
            if data.get("relation") in (RelationType.CALLS.value, RelationType.REFERENCES.value)
        ]

    def get_children(self, node_id: str) -> list[str]:
        """Return all methods/properties contained by this class."""
        return [
            dst for src, dst, data in self._g.out_edges(node_id, data=True)
            if data.get("relation") == RelationType.CONTAINS.value
        ]

    def get_descendants(self, node_id: str, max_depth: int = 3) -> list[str]:
        """BFS traversal of all descendants up to max_depth."""
        try:
            import networkx as nx
            paths = nx.single_source_shortest_path(self._g, node_id, cutoff=max_depth)
            return [n for n in paths if n != node_id]
        except Exception:
            return []

    def get_ancestors(self, node_id: str, max_depth: int = 3) -> list[str]:
        """BFS traversal of all ancestors (what uses this symbol)."""
        try:
            import networkx as nx
            reverse_g = self._g.reverse()
            paths = nx.single_source_shortest_path(reverse_g, node_id, cutoff=max_depth)
            return [n for n in paths if n != node_id]
        except Exception:
            return []

    def find_by_name(self, symbol_name: str) -> list[str]:
        """Find all node IDs with a matching symbol name."""
        return [
            n for n, data in self._g.nodes(data=True)
            if data.get("name") == symbol_name
        ]

    def find_by_file(self, file_path: str) -> list[str]:
        """Find all nodes belonging to a specific file."""
        return [
            n for n, data in self._g.nodes(data=True)
            if data.get("file_path") == file_path
        ]

    # ── Building from extracted symbols ───────────────────────────────────────

    def build_from_file_symbols(
        self,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        """
        Bulk-add all symbols from a file and their containment relationships.
        """
        for symbol in symbols:
            self.add_symbol(file_path, symbol)

    def remove_file(self, file_path: str) -> None:
        """Remove all nodes (and their edges) belonging to a file."""
        for node_id in self.find_by_file(file_path):
            self._g.remove_node(node_id)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize the graph to disk as JSON (node-link format)."""
        import networkx as nx
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g, edges="edges")
        path.write_text(json.dumps(data), encoding="utf-8")
        log.debug("Symbol graph saved: %s (%d nodes, %d edges)", path, self.node_count, self.edge_count)

    @classmethod
    def load(cls, path: Path) -> "SymbolGraph":
        """Load a graph previously saved with save()."""
        import networkx as nx
        graph = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        graph._g = nx.node_link_graph(data, edges="edges")
        return graph

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()
