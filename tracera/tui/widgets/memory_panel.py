"""
TRACERA Memory Panel widget.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer
from rich.text import Text

from tracera.agent.memory import AgentMemory, MemoryCategory


_CATEGORY_COLORS = {
    MemoryCategory.PROJECT_FACT: "#00d4ff",
    MemoryCategory.USER_PREFERENCE: "#ffd700",
    MemoryCategory.PAST_DECISION: "#bd00ff",
    MemoryCategory.TASK_CONTEXT: "#00ff88",
    MemoryCategory.ERROR_PATTERN: "#ff4466",
    MemoryCategory.CODE_LOCATION: "#ff8800",
}

_CATEGORY_ICONS = {
    MemoryCategory.PROJECT_FACT: "◈",
    MemoryCategory.USER_PREFERENCE: "★",
    MemoryCategory.PAST_DECISION: "◉",
    MemoryCategory.TASK_CONTEXT: "◌",
    MemoryCategory.ERROR_PATTERN: "✗",
    MemoryCategory.CODE_LOCATION: "⌖",
}


class MemoryEntryWidget(Static):
    def __init__(self, entry, **kwargs):
        super().__init__(**kwargs)
        self._entry = entry

    def render(self) -> Text:
        text = Text()
        cat = self._entry.category
        color = _CATEGORY_COLORS.get(cat, "white")
        icon = _CATEGORY_ICONS.get(cat, "◈")
        text.append(f" {icon} ", style=f"bold {color}")
        text.append(self._entry.content[:70], style="#a0a0c0")
        return text


class MemoryPanel(Widget):
    """Left sidebar panel showing agent memory."""

    DEFAULT_CSS = """
    MemoryPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-panel"):
            yield Static(
                " ◈  MEMORY ",
                classes="panel-title panel-title-purple",
            )
            with ScrollableContainer(id="memory-list"):
                yield Static(
                    "[dim]No memories yet.[/]",
                    id="memory-empty",
                    markup=True,
                )

    def refresh_memory(self, memory: AgentMemory, query: str = "") -> None:
        container = self.query_one("#memory-list", ScrollableContainer)
        for child in list(container.children):
            child.remove()

        entries = memory.retrieve(query, k=20) if query else list(memory._entries.values())[-20:]
        if not entries:
            container.mount(Static("[dim]No memories yet.[/]", markup=True))
            return

        for entry in entries:
            container.mount(MemoryEntryWidget(entry))

        container.mount(Static(
            f"\n[dim]Total: [bold cyan]{memory.count}[/] memories[/]",
            markup=True,
        ))
