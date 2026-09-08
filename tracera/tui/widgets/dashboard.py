"""
TRACERA Welcome Dashboard — system overview and quick actions.

Shows provider, model, workspace, memory count, recent activity,
and feature status grid on startup.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Horizontal, Vertical

from rich.text import Text


class DashboardWidget(Widget):
    """
    Premium welcome dashboard showing system overview.

    ┌─────────────────────────────────────────────────────────────────┐
    │  ◆ TRACERA DASHBOARD                                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  Provider: openai │ Model: gpt-4o │ Workspace: /path/to/project │
    ├─────────────────────────────────────────────────────────────────┤
    │  [◉ MEM] [◉ RET] [◎ RAG] [◉ MCP] [◉ IDX] [◉ SBX]            │
    ├─────────────────────────────────────────────────────────────────┤
    │  Quick Actions: /search · /index · /test · /review             │
    └─────────────────────────────────────────────────────────────────┘
    """

    class ActionTriggered(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def __init__(
        self,
        provider: str = "—",
        model: str = "—",
        workspace: str = "—",
        memory_count: int = 0,
        tool_count: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._model = model
        self._workspace = workspace
        self._memory_count = memory_count
        self._tool_count = tool_count
        self._features = {
            "memory": True,
            "retrieval": False,
            "rag": False,
            "mcp": False,
            "index": False,
            "sandbox": True,
        }
        self._recent_activity: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="dashboard-container"):
            yield Static(self._render_header(), id="dash-header")
            yield Static(self._render_system_info(), id="dash-system")
            yield Static(self._render_features(), id="dash-features")
            yield Static(self._render_actions(), id="dash-actions")

    def _render_header(self) -> Text:
        text = Text()
        text.append(" ◆ ", style="bold #d2a8ff")
        text.append("TRACERA DASHBOARD", style="bold #6cb6ff")
        text.append("  ", style="dim")
        text.append("─" * 40, style="dim #3a3a4a")
        return text

    def _render_system_info(self) -> Text:
        text = Text()
        text.append("  Provider: ", style="dim #9a9aa3")
        text.append(self._provider, style="bold #6cb6ff")
        text.append("  │  ", style="dim #3a3a4a")
        text.append("Model: ", style="dim #9a9aa3")
        text.append(self._model[:20], style="bold #d2a8ff")
        text.append("  │  ", style="dim #3a3a4a")
        text.append("Tools: ", style="dim #9a9aa3")
        text.append(str(self._tool_count), style="bold #4ac26b")
        return text

    def _render_workspace(self) -> Text:
        text = Text()
        text.append("  Workspace: ", style="dim #9a9aa3")
        text.append(self._workspace[:50], style="bold #e0e0ff")
        text.append("  │  ", style="dim #3a3a4a")
        text.append("Memory: ", style="dim #9a9aa3")
        text.append(str(self._memory_count), style="bold #d2a8ff")
        return text

    def _render_features(self) -> Text:
        text = Text()
        text.append("  ", style="dim")
        feature_labels = {
            "memory": ("MEM", "#6cb6ff"),
            "retrieval": ("RET", "#6cb6ff"),
            "rag": ("RAG", "#6cb6ff"),
            "mcp": ("MCP", "#6cb6ff"),
            "index": ("IDX", "#6cb6ff"),
            "sandbox": ("SBX", "#6cb6ff"),
        }
        for feat, (label, color) in feature_labels.items():
            active = self._features.get(feat, False)
            icon = "◉" if active else "◎"
            style = f"bold {color}" if active else f"dim {color}"
            text.append(f"[{icon} {label}] ", style=style)
        return text

    def _render_actions(self) -> Text:
        text = Text()
        text.append("  Quick Actions: ", style="dim #9a9aa3")
        actions = [
            ("/search", "#6cb6ff"),
            ("/index", "#d2a8ff"),
            ("/test", "#4ac26b"),
            ("/review", "#ffd700"),
            ("/memory", "#d2a8ff"),
            ("/inspect", "#6cb6ff"),
        ]
        for i, (action, color) in enumerate(actions):
            text.append(action, style=f"bold {color}")
            if i < len(actions) - 1:
                text.append("  ·  ", style="dim #3a3a4a")
        return text

    def update_info(
        self,
        provider: str | None = None,
        model: str | None = None,
        workspace: str | None = None,
        memory_count: int | None = None,
        tool_count: int | None = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        if model is not None:
            self._model = model
        if workspace is not None:
            self._workspace = workspace
        if memory_count is not None:
            self._memory_count = memory_count
        if tool_count is not None:
            self._tool_count = tool_count
        self._refresh_all()

    def set_feature_status(self, feature: str, active: bool) -> None:
        if feature in self._features:
            self._features[feature] = active
            self._refresh_all()

    def _refresh_all(self) -> None:
        try:
            self.query_one("#dash-system", Static).update(self._render_system_info())
            self.query_one("#dash-features", Static).update(self._render_features())
        except Exception:
            pass
