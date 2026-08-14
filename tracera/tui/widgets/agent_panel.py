"""
TRACERA Agent Conversation Panel.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Input
from textual.reactive import reactive
from textual.containers import Horizontal, Vertical, ScrollableContainer
from rich.text import Text


class MessageWidget(Static):
    """A single message bubble in the conversation."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.msg_content = content

    def render(self) -> Text:
        text = Text()
        if self.role == "user":
            text.append("  YOU  ", style="bold black on cyan")
            text.append("  ")
            text.append(self.msg_content, style="#e0e0ff")
        elif self.role == "assistant":
            text.append("  TRACERA  ", style="bold black on #bd00ff")
            text.append("  ")
            text.append(self.msg_content, style="#e0e0ff")
        elif self.role == "tool":
            text.append("  ⚙  ", style="#00ff88")
            text.append(self.msg_content[:200], style="dim #60c060")
        elif self.role == "error":
            text.append("  ✗  ", style="bold red")
            text.append(self.msg_content, style="red")
        elif self.role == "thinking":
            text.append("  ◌  ", style="dim cyan")
            text.append(self.msg_content, style="dim #606090 italic")
        return text


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class ActivityLine(Static):
    """
    Claude Code-style live status line: animated spinner + current action.

    Shows the agent's current activity in a single row that updates in place:
      ⠋  Thinking…
      ⠹  read_file(path="tracera/main.py")
      ✓  read_file  12ms      ← auto-clears after a moment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.status = "idle"  # idle | thinking | running | done | error
        self.action = ""
        self.detail = ""
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        if self.status in ("thinking", "running"):
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self.refresh()

    def show(self, status: str, action: str = "", detail: str = "") -> None:
        """Display a status line. Terminal states auto-clear, like Claude Code."""
        self.status = status
        self.action = action
        self.detail = detail
        self._frame = 0
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if status in ("done", "error"):
            self._timer = self.set_timer(2.0, self.clear)
        self.refresh()

    def clear(self) -> None:
        self.status = "idle"
        self.action = ""
        self.detail = ""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.refresh()

    def render(self) -> Text:
        text = Text()
        if self.status == "idle":
            return text
        if self.status == "thinking":
            text.append(f" {_SPINNER_FRAMES[self._frame]}  ", style="bold cyan")
            text.append(self.action or "Thinking…", style="italic #60a0c0")
        elif self.status == "running":
            text.append(f" {_SPINNER_FRAMES[self._frame]}  ", style="bold #00d4ff")
            text.append(self.action, style="bold #00d4ff")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #606090")
        elif self.status == "done":
            text.append(" ✓  ", style="bold green")
            text.append(self.action, style="green")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #00a05a")
        elif self.status == "error":
            text.append(" ✗  ", style="bold red")
            text.append(self.action, style="red")
            if self.detail:
                text.append(f"  {self.detail}", style="dim #ff6688")
        return text


class AgentPanel(Widget):
    """
    Central agent conversation panel.
    
    Shows the conversation history and provides an input field.
    Emits 'submit_task' message when user sends input.
    """

    DEFAULT_CSS = """
    AgentPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    class SubmitTask(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._message_widgets: list[MessageWidget] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-panel"):
            yield Static(
                " ◈  CONVERSATION ",
                id="agent-panel-title",
                classes="panel-title",
            )
            with ScrollableContainer(id="agent-log"):
                yield Static(
                    "\n[dim cyan]Welcome to TRACERA.[/]\n"
                    "[dim]Ask me anything about your codebase, or give me a task to complete.[/]\n"
                    "[dim]Type [bold cyan]/help[/] for available commands.[/]\n",
                    markup=True,
                )
            yield ActivityLine(id="activity-line")
            with Horizontal(id="agent-input-area"):
                yield Static("❯", id="agent-prompt-icon")
                yield Input(
                    placeholder="Ask TRACERA anything... (/help for commands)",
                    id="agent-input",
                )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.query_one("#agent-input", Input).value = ""
            self.post_message(self.SubmitTask(text))

    def add_message(self, role: str, content: str) -> None:
        """Add a message bubble to the conversation."""
        log_container = self.query_one("#agent-log", ScrollableContainer)
        widget = MessageWidget(role, content)
        widget.add_class(f"msg-{role}")
        log_container.mount(widget)
        log_container.scroll_end(animate=True)

    def add_user_message(self, text: str) -> None:
        self.add_message("user", text)

    def add_assistant_message(self, text: str) -> None:
        self.add_message("assistant", text)

    def add_tool_message(self, tool_name: str, output: str | None = None) -> None:
        preview = f"{tool_name}"
        if output:
            first_line = output.strip().splitlines()[0] if output.strip() else ""
            if first_line:
                preview += f"  →  {first_line[:80]}"
        self.add_message("tool", preview)

    def add_thinking(self, text: str) -> None:
        self.add_message("thinking", text)

    def add_error(self, text: str) -> None:
        self.add_message("error", text)

    def set_activity(self, status: str, action: str = "", detail: str = "") -> None:
        """Drive the live Claude Code-style status line (spinner + action)."""
        try:
            self.query_one("#activity-line", ActivityLine).show(status, action, detail)
        except Exception:
            pass

    def clear_activity(self) -> None:
        try:
            self.query_one("#activity-line", ActivityLine).clear()
        except Exception:
            pass

    def clear(self) -> None:
        log_container = self.query_one("#agent-log", ScrollableContainer)
        for child in list(log_container.children):
            child.remove()
        self.clear_activity()
