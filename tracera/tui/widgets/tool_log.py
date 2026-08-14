"""
TRACERA Tool Log Panel.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Log
from textual.containers import Vertical, ScrollableContainer
from rich.text import Text
import time


class ToolLogEntry(Static):
    """A single tool execution entry in the log."""

    def __init__(self, name: str, args: dict, success: bool = True,
                 output: str = "", duration_ms: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = name
        self.tool_args = args
        self.success = success
        self.tool_output = output
        self.duration_ms = duration_ms

    def render(self) -> Text:
        text = Text()
        icon = "✓" if self.success else "✗"
        icon_style = "bold green" if self.success else "bold red"
        text.append(f" {icon} ", style=icon_style)
        text.append(self.tool_name, style="bold #00ff88")

        # Show key args
        arg_parts = []
        for k, v in list(self.tool_args.items())[:2]:
            sv = str(v)
            if len(sv) > 30:
                sv = sv[:27] + "..."
            arg_parts.append(f"{k}={sv!r}")
        if arg_parts:
            text.append(f"  ({', '.join(arg_parts)})", style="dim #606090")

        if self.duration_ms:
            text.append(f"  {self.duration_ms:.0f}ms", style="dim cyan")

        # Output preview
        if self.tool_output:
            lines = self.tool_output.strip().splitlines()
            preview = lines[0][:80] if lines else ""
            if preview:
                style = "#606090" if self.success else "red"
                text.append(f"\n   {preview}", style=style)

        return text


class ToolLogPanel(Widget):
    """
    Right-side panel showing a live log of all tool executions.
    """

    DEFAULT_CSS = """
    ToolLogPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entries: list[ToolLogEntry] = []
        self._total_calls = 0
        self._total_ms = 0.0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                " ⚙  TOOL LOG ",
                classes="panel-title panel-title-green",
            )
            with ScrollableContainer(id="tool-log"):
                yield Static(
                    "[dim]No tools executed yet.[/]",
                    id="tool-log-empty",
                    markup=True,
                )

    def add_entry(
        self,
        name: str,
        args: dict,
        *,
        success: bool = True,
        output: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        container = self.query_one("#tool-log", ScrollableContainer)

        # Remove empty placeholder
        try:
            placeholder = self.query_one("#tool-log-empty", Static)
            placeholder.remove()
        except Exception:
            pass

        entry = ToolLogEntry(name, args, success=success, output=output, duration_ms=duration_ms)
        self._entries.append(entry)
        self._total_calls += 1
        self._total_ms += duration_ms

        container.mount(entry)
        container.scroll_end(animate=False)

    def clear(self) -> None:
        container = self.query_one("#tool-log", ScrollableContainer)
        for child in list(container.children):
            child.remove()
        self._entries.clear()
        self._total_calls = 0
        self._total_ms = 0.0
        container.mount(Static(
            "[dim]No tools executed yet.[/]",
            id="tool-log-empty",
            markup=True,
        ))
