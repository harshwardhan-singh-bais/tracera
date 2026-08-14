"""
TRACERA Thinking Panel — real-time agent trace (Claude Code / Gemini CLI style).

A collapsible panel that shows the agent *thinking live*:
  ⠋ thinking (iteration 2)          ← dim, animated
  ⠹ read_file(path="main.py")       ← pending tool, animated + running detail
  ✓ read_file  12ms                 ← completed tool
  ✗ grep  404                       ← failed tool

Click the panel header (▾/▸ THINKING) to expand or collapse it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class ThinkingEntry(Static):
    """A single trace line. Pending (tool) entries animate their spinner."""

    def __init__(self, kind: str, text: str = "", detail: str = "",
                 pending: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kind = kind  # think | tool | done | error
        self.text = text
        self.detail = detail
        self.pending = pending
        self._frame = 0

    def on_mount(self) -> None:
        if self.pending:
            self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
        self.refresh()

    def finish(self, ok: bool = True, duration_ms: float = 0.0) -> None:
        """Mark a pending tool entry as completed/failed."""
        self.pending = False
        self.kind = "done" if ok else "error"
        self.detail = f"{duration_ms:.0f}ms" if duration_ms else ""
        self.refresh()

    def render(self) -> Text:
        text = Text()
        if self.kind == "think":
            text.append(f" {_SPINNER_FRAMES[self._frame]} ", style="bold cyan")
            text.append(self.text, style="italic #60a0c0")
        elif self.kind == "tool":
            text.append(f" {_SPINNER_FRAMES[self._frame]} ", style="bold #00d4ff")
            text.append(self.text, style="bold #00d4ff")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #606090")
        elif self.kind == "done":
            text.append(" ✓ ", style="bold green")
            text.append(self.text, style="green")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #00a05a")
        elif self.kind == "error":
            text.append(" ✗ ", style="bold red")
            text.append(self.text, style="red")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #ff6688")
        return text


class ThinkingPanel(Widget):
    """
    Live trace of the agent's reasoning and tool activity.

    The header row is a clickable toggle (▾ expanded / ▸ collapsed).
    Entries auto-scroll; pending tool calls animate with a spinner.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.expanded = True
        self._pending: list[ThinkingEntry] = []

    def compose(self) -> ComposeResult:
        yield Static("▾ THINKING", id="thinking-toggle", classes="panel-title panel-title-cyan")
        with ScrollableContainer(id="thinking-trace"):
            yield Static("[dim]Waiting for the agent…[/]", id="thinking-empty", markup=True)

    # ── Toggle ───────────────────────────────────────────────────────────────

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "thinking-toggle":
            self.toggle()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        header = self.query_one("#thinking-toggle", Static)
        trace = self.query_one("#thinking-trace", ScrollableContainer)
        header.update("▾ THINKING" if self.expanded else "▸ THINKING")
        trace.display = self.expanded

    def expand(self) -> None:
        if not self.expanded:
            self.toggle()

    def collapse(self) -> None:
        if self.expanded:
            self.toggle()

    # ── Feed ─────────────────────────────────────────────────────────────────

    def _prepare(self) -> ScrollableContainer:
        self.expand()
        trace = self.query_one("#thinking-trace", ScrollableContainer)
        try:
            empty = self.query_one("#thinking-empty", Static)
            empty.remove()
        except Exception:
            pass
        return trace

    def _append(self, entry: ThinkingEntry) -> None:
        trace = self._prepare()
        trace.mount(entry)
        trace.scroll_end(animate=False)

    def add_thinking(self, text: str) -> None:
        self._append(ThinkingEntry("think", text))

    def tool_start(self, name: str, detail: str = "") -> None:
        entry = ThinkingEntry("tool", name, detail, pending=True)
        self._pending.append(entry)
        self._append(entry)

    def tool_end(self, name: str, success: bool, duration_ms: float = 0.0) -> None:
        if self._pending:
            entry = self._pending.pop(0)
            entry.finish(success, duration_ms)
        else:
            self._append(ThinkingEntry("done" if success else "error", name,
                                       f"{duration_ms:.0f}ms"))

    def add_error(self, text: str) -> None:
        self._append(ThinkingEntry("error", text))

    def add_tokens(self, tokens: int) -> None:
        """Brief live ticker of tokens streamed so far this turn."""
        self._append(ThinkingEntry("think", f"streaming — {tokens} tokens"))

    def clear(self) -> None:
        trace = self.query_one("#thinking-trace", ScrollableContainer)
        for child in list(trace.children):
            child.remove()
        self._pending.clear()
        trace.mount(Static("[dim]Waiting for the agent…[/]", id="thinking-empty", markup=True))
