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

    def clear(self) -> None:
        log_container = self.query_one("#agent-log", ScrollableContainer)
        for child in list(log_container.children):
            child.remove()
