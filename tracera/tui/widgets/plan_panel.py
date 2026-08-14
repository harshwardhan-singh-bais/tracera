"""
TRACERA Plan Panel widget.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, ProgressBar
from textual.containers import Vertical, ScrollableContainer
from rich.text import Text

from tracera.agent.planner import Plan, TodoStatus


class PlanItemWidget(Static):
    """Display a single plan item."""

    def __init__(self, item, **kwargs):
        super().__init__(**kwargs)
        self._item = item

    def render(self) -> Text:
        text = Text()
        icons = {
            TodoStatus.PENDING: ("○", "#606090"),
            TodoStatus.IN_PROGRESS: ("◎", "#ffd700"),
            TodoStatus.DONE: ("●", "#00ff88"),
            TodoStatus.FAILED: ("✗", "#ff4466"),
            TodoStatus.SKIPPED: ("⊘", "#606090"),
        }
        icon, color = icons.get(self._item.status, ("?", "white"))
        text.append(f" {icon} ", style=f"bold {color}")
        title_style = "dim" if self._item.status == TodoStatus.DONE else "bold"
        if self._item.status == TodoStatus.IN_PROGRESS:
            title_style = "bold #ffd700"
        elif self._item.status == TodoStatus.FAILED:
            title_style = "bold #ff4466"
        text.append(self._item.title[:55], style=title_style)
        if self._item.status == TodoStatus.IN_PROGRESS:
            text.append("  ◌", style="dim #ffd700")
        return text


class PlanPanel(Widget):
    """Left sidebar panel showing the current task plan."""

    DEFAULT_CSS = """
    PlanPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plan: Plan | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-panel"):
            yield Static(
                " ◈  PLAN ",
                classes="panel-title panel-title-gold",
            )
            with ScrollableContainer(id="plan-list"):
                yield Static(
                    "[dim]No active plan.[/]\n"
                    "[dim]Send a task to get started.[/]",
                    markup=True,
                )

    def set_plan(self, plan: Plan) -> None:
        """Render a new plan."""
        self._plan = plan
        container = self.query_one("#plan-list", ScrollableContainer)
        for child in list(container.children):
            child.remove()

        for item in plan.items:
            container.mount(PlanItemWidget(item))

        done, total = plan.progress
        pct = int(plan.progress_pct)
        container.mount(Static(
            f"\n[dim]Progress: [bold cyan]{done}/{total}[/] ({pct}%)[/]",
            markup=True,
        ))

        # Show the plan from the top
        container.scroll_home(animate=False)

    def update_plan(self, plan: Plan) -> None:
        """Refresh plan display."""
        self.set_plan(plan)

    def clear(self) -> None:
        container = self.query_one("#plan-list", ScrollableContainer)
        for child in list(container.children):
            child.remove()
        self._plan = None
        container.mount(Static(
            "[dim]No active plan.[/]",
            markup=True,
        ))
