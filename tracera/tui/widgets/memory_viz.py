"""
TRACERA Memory Graph Visualization — text-based graph display.

Shows memory connections and knowledge graph structure with
color-coded nodes and relationship links.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical

from rich.text import Text


# Memory type colors
_TYPE_COLORS = {
    "fact": "#4ac26b",
    "preference": "#d2a8ff",
    "rule": "#ffd700",
    "decision": "#6cb6ff",
    "skill": "#ff6b6b",
    "relationship": "#ff9f43",
    "unknown": "#9a9aa3",
}


class MemoryGraphWidget(Widget):
    """
    Text-based visualization of the memory/knowledge graph.

    ┌─ MEMORY GRAPH ──────────────────────────────────────┐
    │                                                      │
    │         ◆ auth                                       │
    │        ╱    ╲                                        │
    │   ◆ jwt ─── ◆ middleware                             │
    │        ╲    ╱                                        │
    │         ◆ validation                                 │
    │                                                      │
    │  Nodes: 12  │  Edges: 18  │  Types: 6               │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, str]] = []  # (from, to, relation)
        self._central_concepts: list[tuple[str, int]] = []
        self._type_counts: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-graph-container"):
            yield Static(self._render_header(), id="memory-graph-header")
            yield Static(self._render_graph(), id="memory-graph-content")

    def _render_header(self) -> Text:
        text = Text()
        text.append(" 🧠 ", style="bold #d2a8ff")
        text.append("MEMORY GRAPH", style="bold #d2a8ff")
        text.append("  ", style="dim")
        text.append("─" * 28, style="dim #3a3a4a")
        return text

    def _render_graph(self) -> Text:
        text = Text()

        if not self._nodes and not self._central_concepts:
            text.append("  No memory data loaded", style="dim #55555e")
            return text

        # Render central concepts as a simple graph visualization
        if self._central_concepts:
            text.append("\n", style="dim")

            # Find the most central concept
            top_concepts = self._central_concepts[:5]

            if len(top_concepts) >= 3:
                # Create a simple radial visualization
                center = top_concepts[0]
                text.append(f"         ◆ {center[0]}", style="bold #d2a8ff")
                text.append("\n", style="dim")
                text.append("        ╱    ╲", style="dim #3a3a4a")
                text.append("\n", style="dim")

                if len(top_concepts) >= 2:
                    left = top_concepts[1]
                    text.append(f"   ◆ {left[0]}", style="bold #6cb6ff")
                    text.append(" ─── ", style="dim #3a3a4a")

                if len(top_concepts) >= 3:
                    right = top_concepts[2]
                    text.append(f"◆ {right[0]}", style="bold #4ac26b")
                    text.append("\n", style="dim")
                    text.append("        ╲    ╱", style="dim #3a3a4a")
                    text.append("\n", style="dim")

                if len(top_concepts) >= 4:
                    bottom = top_concepts[3]
                    text.append(f"         ◆ {bottom[0]}", style="bold #ffd700")
                    text.append("\n", style="dim")

            elif len(top_concepts) == 2:
                text.append(f"   ◆ {top_concepts[0][0]}", style="bold #d2a8ff")
                text.append(" ──── ", style="dim #3a3a4a")
                text.append(f"◆ {top_concepts[1][0]}", style="bold #6cb6ff")
                text.append("\n", style="dim")

            elif len(top_concepts) == 1:
                text.append(f"         ◆ {top_concepts[0][0]}", style="bold #d2a8ff")
                text.append("\n", style="dim")

        # Statistics
        text.append("\n  ", style="dim")
        text.append("Nodes: ", style="dim #9a9aa3")
        text.append(str(len(self._nodes)), style="bold #e0e0ff")
        text.append("  │  ", style="dim #3a3a4a")
        text.append("Edges: ", style="dim #9a9aa3")
        text.append(str(len(self._edges)), style="bold #e0e0ff")
        text.append("  │  ", style="dim #3a3a4a")
        text.append("Types: ", style="dim #9a9aa3")
        text.append(str(len(self._type_counts)), style="bold #e0e0ff")

        # Type breakdown
        if self._type_counts:
            text.append("\n  ", style="dim")
            for mtype, count in sorted(self._type_counts.items(), key=lambda x: -x[1])[:5]:
                color = _TYPE_COLORS.get(mtype, "#9a9aa3")
                text.append(f"{mtype}:{count} ", style=f"dim {color}")

        return text

    def update_graph(
        self,
        nodes: dict[str, dict] | None = None,
        edges: list[tuple[str, str, str]] | None = None,
        central_concepts: list[tuple[str, int]] | None = None,
        type_counts: dict[str, int] | None = None,
    ) -> None:
        """Update the graph visualization with new data."""
        if nodes is not None:
            self._nodes = nodes
        if edges is not None:
            self._edges = edges
        if central_concepts is not None:
            self._central_concepts = central_concepts
        if type_counts is not None:
            self._type_counts = type_counts
        self._refresh()

    def add_node(self, node_id: str, node_data: dict) -> None:
        self._nodes[node_id] = node_data
        self._refresh()

    def add_edge(self, from_node: str, to_node: str, relation: str = "") -> None:
        self._edges.append((from_node, to_node, relation))
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.query_one("#memory-graph-content", Static).update(self._render_graph())
        except Exception:
            pass
