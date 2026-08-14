"""
TRACERA Status Bar widget.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from textual.containers import Horizontal
from rich.text import Text


class StatusBar(Widget):
    """Bottom status bar showing provider, model, tokens, tool calls, and iteration."""

    provider: reactive[str] = reactive("—")
    model: reactive[str] = reactive("—")
    tokens: reactive[int] = reactive(0)
    tool_calls: reactive[int] = reactive(0)
    iteration: reactive[int] = reactive(0)
    status: reactive[str] = reactive("idle")
    latency_ms: reactive[float] = reactive(0.0)
    workspace: reactive[str] = reactive(".")

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        layout: horizontal;
        background: #0d0d1e;
        border-top: solid #007a9a;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-left")
        yield Static("", id="sb-mid")
        yield Static("", id="sb-right")

    def on_mount(self) -> None:
        self._refresh_display()
        self.set_interval(1.0, self._refresh_display)

    def _refresh_display(self) -> None:
        import time

        status_colors = {
            "idle": "#606090",
            "thinking": "#ffd700",
            "running": "#00d4ff",
            "done": "#00ff88",
            "error": "#ff4466",
        }
        sc = status_colors.get(self.status, "#606090")

        left = Text()
        left.append("  TRACERA", style="bold #00d4ff")
        left.append(f"  {self.workspace[:25]}", style="dim #606090")

        mid = Text()
        mid.append(f"  {self.provider}", style="dim #606090")
        mid.append("  /  ", style="dim #303050")
        mid.append(f"{self.model}", style="bold #bd00ff")
        if self.tokens:
            mid.append(f"  ·  {self.tokens:,} tok", style="dim #606090")
        if self.latency_ms:
            mid.append(f"  ·  {self.latency_ms:.0f}ms", style="dim #606090")

        right = Text()
        right.append(f"⚙ {self.tool_calls}", style="dim #00ff88")
        right.append("  ", style="")
        right.append(f"↻ {self.iteration}", style="dim #ffd700")
        right.append("  ", style="")
        right.append(f"● {self.status.upper()}", style=f"bold {sc}")
        right.append("  ", style="")

        self.query_one("#sb-left", Static).update(left)
        self.query_one("#sb-mid", Static).update(mid)
        self.query_one("#sb-right", Static).update(right)

    def update_stats(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        tokens: int | None = None,
        tool_calls: int | None = None,
        iteration: int | None = None,
        status: str | None = None,
        latency_ms: float | None = None,
        workspace: str | None = None,
    ) -> None:
        if provider is not None:
            self.provider = provider
        if model is not None:
            self.model = model
        if tokens is not None:
            self.tokens = tokens
        if tool_calls is not None:
            self.tool_calls = tool_calls
        if iteration is not None:
            self.iteration = iteration
        if status is not None:
            self.status = status
        if latency_ms is not None:
            self.latency_ms = latency_ms
        if workspace is not None:
            self.workspace = workspace
        self._refresh_display()
