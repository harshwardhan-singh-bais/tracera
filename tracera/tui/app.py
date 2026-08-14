"""
TRACERA Main Textual Application — Futuristic TUI.

Layout:
┌─────────────────────────────────────────────────────────┐
│  TRACERA  ◈  /workspace           model · tokens        │ ← header
├──────────────┬──────────────────────────┬───────────────┤
│  ◈ PLAN      │  ◈ CONVERSATION          │  ⚙ TOOL LOG   │
│              │                          │               │
│  ○ Step 1    │  YOU: add auth           │  ✓ read_file  │
│  ◎ Step 2 ◌  │  TRACERA: ...            │  ✓ grep       │
│  ● Step 3    │                          │  ✓ edit_file  │
│              │  [Tabs: Memory | Stats]  │               │
├──────────────┴──────────────────────────┴───────────────┤
│  TRACERA · openai/gpt-4o · 2,341tok · ⚙3 · ↻5 · RUNNING│ ← status
└─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from tracera.tui.widgets.agent_panel import AgentPanel
from tracera.tui.widgets.tool_log import ToolLogPanel
from tracera.tui.widgets.plan_panel import PlanPanel
from tracera.tui.widgets.memory_panel import MemoryPanel
from tracera.tui.widgets.status_bar import StatusBar
from tracera.agent.react_loop import AgentEvent, AgentEventType, ReActAgent
from tracera.agent.memory import AgentMemory
from tracera.agent.planner import TaskDecomposer
from tracera.conversation.state import ConversationState


_HELP_TEXT = """
[bold cyan]TRACERA Commands[/]

  [bold]/help[/]       Show this help
  [bold]/clear[/]      Clear conversation
  [bold]/status[/]     Show system status
  [bold]/memory[/]     Show memory contents
  [bold]/model[/] [name]  Switch model
  [bold]/plan[/] [task]   Decompose a task into steps
  [bold]/reset[/]      Reset conversation state
  [bold]ctrl+q[/]      Quit TRACERA

[dim]For anything else, just type your request and press Enter.[/]
"""


class TraceraTUI(App):
    """
    TRACERA — Agentic Code Intelligence Terminal UI.
    
    A futuristic cyberpunk-themed TUI for interacting with the coding agent.
    """

    TITLE = "TRACERA — CodePilotX"
    CSS_PATH = "styles/tracera.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_conversation", "Clear", show=True),
        Binding("ctrl+p", "focus_input", "Focus Input", show=False),
        Binding("ctrl+m", "show_memory", "Memory", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("escape", "cancel_task", "Cancel", show=False),
        # Keyboard scrolling — works like any other CLI tool (less, htop):
        # PgUp/PgDn scroll the panel under the mouse cursor, falling back to
        # the conversation log. (Home/End are left to the input field, where
        # they move the cursor — the standard terminal behaviour.)
        Binding("pageup", "scroll_active(-1)", "Scroll Up", show=True),
        Binding("pagedown", "scroll_active(1)", "Scroll Down", show=True),
    ]

    def __init__(
        self,
        agent: ReActAgent,
        memory: AgentMemory,
        workspace_path: Path = Path("."),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.memory = memory
        self.workspace_path = workspace_path
        self._conversation = ConversationState()
        self._running_task: asyncio.Task | None = None
        self._hovered_scrollable: ScrollableContainer | None = None

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield self._build_header()
        with Horizontal(id="main-layout"):
            # Left sidebar: Plan + Memory tabs
            with Vertical(id="left-sidebar"):
                with TabbedContent():
                    with TabPane("Plan", id="tab-plan"):
                        yield PlanPanel(id="plan-panel-widget")
                    with TabPane("Memory", id="tab-memory"):
                        yield MemoryPanel(id="memory-panel-widget")

            # Center: Conversation
            yield AgentPanel(id="agent-panel-widget")

            # Right sidebar: Tool log
            with Vertical(id="right-sidebar"):
                yield ToolLogPanel(id="tool-log-widget")
                yield self._build_stats_panel()

        yield StatusBar(id="status-bar")
        yield Footer()

    def _build_header(self) -> Static:
        from rich.text import Text
        text = Text()
        text.append("  ██  ", style="bold #00d4ff")
        text.append("TRACERA", style="bold #00d4ff")
        text.append("  ·  ", style="dim #303050")
        text.append("CodePilotX", style="dim #606090")
        text.append("  ·  ", style="dim #303050")
        text.append(str(self.workspace_path), style="dim #606090")
        return Static(text, id="app-header")

    def _build_stats_panel(self) -> Vertical:
        return Vertical(
            Static(" ◈  STATS ", classes="panel-title panel-title-purple"),
            Static(
                "[dim]Iterations:[/] [bold cyan]0[/]\n"
                "[dim]Tool calls:[/] [bold green]0[/]\n"
                "[dim]Tokens in:[/] [bold cyan]0[/]\n"
                "[dim]Tokens out:[/] [bold cyan]0[/]\n"
                "[dim]Latency:[/] [bold gold1]—[/]",
                id="stats-display",
                markup=True,
            ),
            id="stats-panel",
        )

    # ── Scroll handling ───────────────────────────────────────────────────────

    def _find_scrollable(self, widget) -> ScrollableContainer | None:
        """Walk up the DOM from *widget* to find the enclosing scrollable, if any."""
        node = widget
        while node is not None:
            if isinstance(node, ScrollableContainer):
                return node
            node = node.parent
        return None

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Track which scrollable panel the mouse is currently over."""
        try:
            widget, _ = self.screen.get_widget_at(event.x, event.y)
        except Exception:
            return
        scrollable = self._find_scrollable(widget)
        if scrollable is not None:
            self._hovered_scrollable = scrollable

    def _active_scrollable(self) -> ScrollableContainer:
        """The scrollable to act on: hovered panel, else the conversation log."""
        if self._hovered_scrollable is not None and self._hovered_scrollable.is_attached:
            return self._hovered_scrollable
        return self.query_one("#agent-log", ScrollableContainer)

    def action_scroll_active(self, direction: int) -> None:
        """Scroll the active panel by one page (direction: -1 up, +1 down)."""
        target = self._active_scrollable()
        if direction < 0:
            target.scroll_page_up(animate=False)
        else:
            target.scroll_page_down(animate=False)

    # ── Event handlers ────────────────────────────────────────────────────────

    @on(AgentPanel.SubmitTask)
    def on_submit_task(self, event: AgentPanel.SubmitTask) -> None:
        """Handle user input from the agent panel."""
        text = event.text.strip()
        if not text:
            return

        # Handle slash commands
        if text.startswith("/"):
            self._handle_command(text)
            return

        # Start agent task
        agent_panel = self.query_one("#agent-panel-widget", AgentPanel)
        agent_panel.add_user_message(text)
        self._run_agent_task(text)

    def _handle_command(self, text: str) -> None:
        panel = self.query_one("#agent-panel-widget", AgentPanel)
        cmd = text.split()[0].lower()

        if cmd == "/help":
            panel.add_assistant_message(_HELP_TEXT)
        elif cmd == "/clear":
            self.action_clear_conversation()
        elif cmd == "/status":
            self._show_status(panel)
        elif cmd == "/memory":
            self.action_show_memory()
        elif cmd == "/reset":
            self._conversation = ConversationState()
            panel.add_assistant_message("[dim]Conversation reset.[/]")
        elif cmd == "/plan":
            task = text[6:].strip()
            if task:
                self._run_planning(task)
            else:
                panel.add_error("Usage: /plan <task description>")
        elif cmd == "/model":
            parts = text.split()
            if len(parts) > 1:
                new_model = parts[1]
                self.agent.model = new_model
                panel.add_assistant_message(f"Model switched to: [bold cyan]{new_model}[/]")
            else:
                panel.add_error("Usage: /model <model-name>")
        else:
            panel.add_error(f"Unknown command: {cmd}. Type /help for available commands.")

    def _show_status(self, panel: AgentPanel) -> None:
        provider = self.agent.provider
        stats = self._conversation.stats
        status = (
            f"[bold cyan]System Status[/]\n\n"
            f"Provider:    [bold]{provider.name}[/]\n"
            f"Model:       [bold]{provider.default_model}[/]\n"
            f"Workspace:   [bold]{self.workspace_path}[/]\n"
            f"Messages:    [bold cyan]{stats.total_messages}[/]\n"
            f"Tool calls:  [bold green]{stats.tool_calls}[/]\n"
            f"Tokens:      [bold cyan]{stats.total_tokens:,}[/]\n"
            f"Memory:      [bold orchid]{self.memory.count}[/] entries\n"
        )
        panel.add_assistant_message(status)

    # ── Agent execution ───────────────────────────────────────────────────────

    @work(exclusive=False)
    async def _run_agent_task(self, task: str) -> None:
        """Run the agent loop in a background worker."""
        agent_panel = self.query_one("#agent-panel-widget", AgentPanel)
        tool_log = self.query_one("#tool-log-widget", ToolLogPanel)
        status_bar = self.query_one("#status-bar", StatusBar)

        provider = self.agent.provider
        status_bar.update_stats(
            provider=provider.name,
            model=provider.default_model or "—",
            status="thinking",
            workspace=str(self.workspace_path),
        )

        agent_panel.add_thinking("Thinking…")

        total_iterations = 0
        total_tool_calls = 0

        try:
            async for event in await self.agent.run(task, conversation=self._conversation):
                match event.type:
                    case AgentEventType.THINKING:
                        agent_panel.add_thinking(f"◌  Iteration {event.iteration + 1}")
                        status_bar.update_stats(
                            status="thinking",
                            iteration=event.iteration + 1,
                        )

                    case AgentEventType.TOOL_START:
                        status_bar.update_stats(status="running")
                        agent_panel.add_tool_message(event.tool_name or "tool")

                    case AgentEventType.TOOL_END:
                        total_tool_calls += 1
                        tool_log.add_entry(
                            event.tool_name or "tool",
                            event.tool_args or {},
                            success=event.tool_success,
                            output=event.tool_output or "",
                            duration_ms=event.metadata.get("duration_ms", 0.0),
                        )
                        status_bar.update_stats(tool_calls=total_tool_calls)

                    case AgentEventType.RESPONSE_COMPLETE:
                        total_iterations = event.metadata.get("iterations", 0)
                        total_tokens = event.metadata.get("total_tokens", 0)
                        total_latency = event.metadata.get("total_latency_ms", 0.0)

                        agent_panel.add_assistant_message(event.text or "")
                        status_bar.update_stats(
                            tokens=total_tokens,
                            iteration=total_iterations,
                            status="done",
                            latency_ms=total_latency,
                        )
                        self._update_stats_panel(
                            total_iterations, total_tool_calls,
                            self._conversation.stats.total_tokens_in,
                            self._conversation.stats.total_tokens_out,
                            total_latency,
                        )

                    case AgentEventType.ERROR:
                        agent_panel.add_error(event.text or "Unknown error")
                        status_bar.update_stats(status="error")

                    case AgentEventType.DONE:
                        if status_bar.status != "done" and status_bar.status != "error":
                            status_bar.update_stats(status="idle")

        except Exception as e:
            agent_panel.add_error(f"Agent error: {e}")
            status_bar.update_stats(status="error")

    @work(exclusive=False)
    async def _run_planning(self, task: str) -> None:
        """Run task decomposition and display the plan."""
        agent_panel = self.query_one("#agent-panel-widget", AgentPanel)
        plan_panel = self.query_one("#plan-panel-widget", PlanPanel)

        agent_panel.add_thinking("Decomposing task into steps…")
        try:
            decomposer = TaskDecomposer(self.agent.provider)
            plan = await decomposer.decompose(task)
            plan_panel.set_plan(plan)
            agent_panel.add_assistant_message(
                f"[bold]Plan ready[/]: {len(plan.items)} steps\n\n"
                + plan.to_markdown()
            )
        except Exception as e:
            agent_panel.add_error(f"Planning failed: {e}")

    def _update_stats_panel(
        self,
        iterations: int,
        tool_calls: int,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
    ) -> None:
        display = self.query_one("#stats-display", Static)
        display.update(
            f"[dim]Iterations:[/] [bold cyan]{iterations}[/]\n"
            f"[dim]Tool calls:[/] [bold green]{tool_calls}[/]\n"
            f"[dim]Tokens in:[/] [bold cyan]{tokens_in:,}[/]\n"
            f"[dim]Tokens out:[/] [bold cyan]{tokens_out:,}[/]\n"
            f"[dim]Latency:[/] [bold gold1]{latency_ms:.0f}ms[/]"
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_conversation(self) -> None:
        self.query_one("#agent-panel-widget", AgentPanel).clear()
        self.query_one("#tool-log-widget", ToolLogPanel).clear()
        self._conversation = ConversationState()

    def action_focus_input(self) -> None:
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass

    def action_show_memory(self) -> None:
        memory_panel = self.query_one("#memory-panel-widget", MemoryPanel)
        memory_panel.refresh_memory(self.memory)

    def action_show_help(self) -> None:
        panel = self.query_one("#agent-panel-widget", AgentPanel)
        panel.add_assistant_message(_HELP_TEXT)

    def action_cancel_task(self) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_stats(status="idle")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Initialise status bar on mount."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_stats(
            provider=self.agent.provider.name,
            model=self.agent.provider.default_model,
            status="idle",
            workspace=str(self.workspace_path),
        )
        # Focus the input
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass
