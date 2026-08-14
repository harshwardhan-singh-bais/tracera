"""
TRACERA Main Textual Application — modern terminal-agent TUI
(Claude Code / Gemini CLI style).

Layout:
┌──────────────────────────────────────────────────────────────┐
│  TRACERA · workspace             (slim header)               │
├───────────────────────────────────────────────┬──────────────┤
│  ◈ CONVERSATION (single column)               │  ▾ THINKING  │
│  YOU: add auth                               │   ⠋ thinking │
│  ▸ reasoning (3)  ← collapsible              │   ⠹ read_file│
│  TRACERA: streamed answer…                   │  ▾ TOOLS     │
│  ⏱ 2 iter · 3 tools · 1,234 tok · 456ms     │  [Plan|Memory]│
│                                               │  STATS       │
│  ⠹ read_file(path="main.py")  ← activity line│              │
│  ❯ _                                          │              │
├───────────────────────────────────────────────┴──────────────┤
│  ⚙3 ↻2 · 1,234 tok · 456ms · ● DONE · ctrl+p commands      │
└──────────────────────────────────────────────────────────────┘
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

from textual.command import Hit, Provider

from tracera.tui.widgets.agent_panel import AgentPanel
from tracera.tui.widgets.tool_log import ToolLogPanel
from tracera.tui.widgets.plan_panel import PlanPanel
from tracera.tui.widgets.memory_panel import MemoryPanel
from tracera.tui.widgets.status_bar import StatusBar
from tracera.tui.widgets.thinking_panel import ThinkingPanel
from tracera.agent.react_loop import AgentEvent, AgentEventType, ReActAgent
from tracera.agent.memory import AgentMemory
from tracera.agent.planner import TaskDecomposer
from tracera.conversation.state import ConversationState


def _format_args(args: dict) -> str:
    """Compact, single-line rendering of tool arguments for the activity line."""
    parts = []
    for k, v in list(args.items())[:3]:
        sv = str(v)
        if len(sv) > 30:
            sv = sv[:27] + "…"
        parts.append(f"{k}={sv}")
    return f"({', '.join(parts)})" if parts else ""


_HELP_TEXT = """
[bold cyan]TRACERA — Commands[/]

[bold]/help[/]         Show this help
[bold]/clear[/]        Clear conversation
[bold]/status[/]       Show system status
[bold]/memory[/]       Show memory contents
[bold]/model[/] [name]  Switch model
[bold]/plan[/] [task]   Decompose a task into steps
[bold]/reset[/]        Reset conversation state

[bold cyan]Keys[/]

[bold]ctrl+q[/]        Quit
[bold]ctrl+l[/]        Clear conversation
[bold]ctrl+t[/]        Toggle the live THINKING panel
[bold]ctrl+b[/]        Toggle the activity sidebar
[bold]ctrl+p[/]        Command palette
[bold]ctrl+m[/]        Show memory
[bold]f1[/]            Help
[bold]esc[/]           Cancel running task
[bold]pgup/pgdn[/]     Scroll the panel under the cursor

[bold cyan]Thinking[/]

Click [bold]▾ THINKING[/] in the right sidebar for the real-time agent trace.
Click [bold]▸ Thinking… (n)[/] under any answer to expand that turn's steps.
"""


class TraceraCommands(Provider):
    """Command-palette entries (ctrl+p)."""

    _COMMANDS = [
        ("Toggle thinking panel", "action_toggle_thinking", "Show/hide the live agent trace"),
        ("Toggle sidebar", "action_toggle_sidebar", "Show/hide the activity sidebar"),
        ("Clear conversation", "action_clear_conversation", "Reset the chat"),
        ("Show memory", "action_show_memory", "List persistent memory entries"),
        ("Show help", "action_show_help", "List commands and key bindings"),
        ("Focus input", "action_focus_input", "Move focus to the prompt"),
        ("Cancel task", "action_cancel_task", "Stop the running agent task"),
    ]

    async def search(self, query: str):
        matcher = self.matcher(query)
        for name, action, help_text in self._COMMANDS:
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, score, getattr(self.app, action), name, help_text)


class TraceraTUI(App):
    """
    TRACERA — Agentic Code Intelligence Terminal UI.
    
    A futuristic cyberpunk-themed TUI for interacting with the coding agent.
    """

    TITLE = "TRACERA — CodePilotX"
    CSS_PATH = "styles/tracera.tcss"

    COMMANDS = {TraceraCommands}

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_conversation", "Clear", show=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking", show=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+m", "show_memory", "Memory", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("escape", "cancel_task", "Cancel", show=False),
        # Keyboard scrolling — works like any other CLI tool (less, htop):
        # PgUp/PgDn scroll the panel under the mouse cursor, falling back to
        # the conversation log.
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
        self._splash_done = False
        self._history: list[str] = []

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield self._build_header()
        with Horizontal(id="main-layout"):
            # Single-column conversation (the terminal) — full focus
            yield AgentPanel(id="agent-panel-widget")

            # Right column: live activity — thinking trace, tools, plan, stats
            with Vertical(id="right-sidebar"):
                yield ThinkingPanel(id="thinking-panel-widget")
                yield ToolLogPanel(id="tool-log-widget")
                with TabbedContent():
                    with TabPane("Plan", id="tab-plan"):
                        yield PlanPanel(id="plan-panel-widget")
                    with TabPane("Memory", id="tab-memory"):
                        yield MemoryPanel(id="memory-panel-widget")
                    with TabPane("Tools", id="tab-tools"):
                        yield self._build_tools_panel()
                yield self._build_stats_panel()
                yield self._build_context_panel()
                yield self._build_history_panel()

        yield StatusBar(id="status-bar")
        yield Footer()
        # Typewriter splash overlay — covers the screen while it animates
        with Vertical(id="splash-overlay"):
            yield Static("", id="splash-logo")
            yield Static("", id="splash-status")

    def _build_header(self) -> Horizontal:
        from rich.text import Text
        model = self.agent.provider.default_model or "model"
        left = Text()
        left.append(" ■ ", style="bold #6cb6ff")
        left.append("TRACERA", style="bold")
        left.append("   your terminal coding agent", style="dim #6f6f78")
        left.append(f"   {self.workspace_path}", style="dim #55555e")
        right = Text()
        right.append("● Active", style="bold #4ac26b")
        right.append(f"  {model}", style="bold")
        right.append("   /help", style="dim #6f6f78")
        return Horizontal(
            Static(left, id="header-left"),
            Static(right, id="header-right"),
            id="app-header",
        )

    def _build_context_panel(self) -> Vertical:
        """CONTEXT card — working dir, active file, mode (mockup layout)."""
        rows = [
            ("Working Dir", str(self.workspace_path)[:30] or "."),
            ("Active File", "—"),
            ("Mode", "Code"),
        ]
        children: list[Any] = [Static(" CONTEXT ", classes="panel-title")]
        for key, val in rows:
            children.append(
                Horizontal(
                    Static(key, classes="context-key"),
                    Static(val, classes="context-val"),
                    classes="context-row",
                )
            )
        return Vertical(*children, id="context-panel")

    def _build_tools_panel(self) -> Vertical:
        """TOOLS tab — the available tools the agent can call (mockup card)."""
        names = []
        try:
            if self.agent.registry is not None:
                names = [t.name for t in self.agent.registry.tools]
        except Exception:
            pass
        if not names:
            names = ["read_file", "write_file", "edit_file", "list_dir", "grep", "run_command"]
        lines = "\n".join(f"[dim]▪[/] {n}" for n in names)
        return Vertical(
            Static(lines or "[dim]No tools[/]", id="tools-list", markup=True),
            id="tools-panel",
        )

    def _build_history_panel(self) -> Vertical:
        """HISTORY card — recent tasks from this session (mockup layout)."""
        return Vertical(
            Static(" HISTORY ", classes="panel-title"),
            Static(
                "[dim]No previous messages[/]",
                id="history-display",
                markup=True,
            ),
            id="history-panel",
        )

    def _build_stats_panel(self) -> Vertical:
        return Vertical(
            Static(" STATUS ", classes="panel-title"),
            Static(
                "[#4ac26b]●[/] Active\n"
                "[dim]Session:[/] [bold]———\n"
                f"[dim]Model:[/] [bold]{self.agent.provider.default_model or '—'}\n"
                "[dim]Iterations:[/] [bold]0\n"
                "[dim]Tokens:[/] [bold]0\n"
                "[dim]Time:[/] [bold]—",
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
        self._append_history(text)
        self._run_agent_task(text)

    def _append_history(self, task: str) -> None:
        """Keep the HISTORY card showing the most recent tasks."""
        short = task[:36]
        if short in self._history:
            self._history.remove(short)
        self._history.insert(0, short)
        self._history = self._history[:5]
        try:
            display = self.query_one("#history-display", Static)
            lines = "\n".join(f"[dim]▪[/] {t}" for t in self._history)
            display.update(lines)
        except Exception:
            pass

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
        thinking = self.query_one("#thinking-panel-widget", ThinkingPanel)
        status_bar = self.query_one("#status-bar", StatusBar)

        provider = self.agent.provider
        status_bar.update_stats(
            provider=provider.name,
            model=provider.default_model or "—",
            status="thinking",
            workspace=str(self.workspace_path),
            activity="Thinking…",
        )

        agent_panel.set_activity("thinking", "Thinking…")
        thinking.expand()
        thinking.add_thinking("task received")

        total_iterations = 0
        total_tool_calls = 0
        turn_trace: list[tuple[str, str]] = []

        try:
            async for event in await self.agent.run(task, conversation=self._conversation):
                match event.type:
                    case AgentEventType.THINKING:
                        text = f"thinking (iteration {event.iteration + 1})"
                        status_bar.update_stats(
                            provider=self.agent.provider.name,
                            status="thinking",
                            iteration=event.iteration + 1,
                            activity=text,
                        )
                        agent_panel.set_activity("thinking", text)
                        thinking.add_thinking(text)
                        turn_trace.append(("think", text))

                    case AgentEventType.TOOL_START:
                        name = event.tool_name or "tool"
                        args_preview = _format_args(event.tool_args or {})
                        status_bar.update_stats(
                            status="running",
                            activity=f"{name}{args_preview}",
                        )
                        agent_panel.set_activity("running", name, args_preview)
                        agent_panel.add_tool_message(name)
                        thinking.tool_start(name, args_preview)
                        turn_trace.append(("tool", f"{name}{args_preview}"))

                    case AgentEventType.TOOL_END:
                        total_tool_calls += 1
                        name = event.tool_name or "tool"
                        duration_ms = event.metadata.get("duration_ms", 0.0)
                        tool_log.add_entry(
                            name,
                            event.tool_args or {},
                            success=event.tool_success,
                            output=event.tool_output or "",
                            duration_ms=duration_ms,
                        )
                        status_bar.update_stats(
                            tool_calls=total_tool_calls,
                            activity="",
                        )
                        thinking.tool_end(name, event.tool_success, duration_ms)
                        if event.tool_success:
                            agent_panel.set_activity(
                                "done", name, f"{duration_ms:.0f}ms"
                            )
                            turn_trace.append(("done", f"{name}  {duration_ms:.0f}ms"))
                        else:
                            agent_panel.set_activity(
                                "error", name, (event.tool_output or "")[:60]
                            )
                            turn_trace.append(("error", f"{name} failed"))

                    case AgentEventType.RESPONSE_DELTA:
                        # Live token-by-token rendering, Claude Code style
                        if event.text:
                            agent_panel.stream_delta(event.text)

                    case AgentEventType.RESPONSE_COMPLETE:
                        total_iterations = event.metadata.get("iterations", 0)
                        total_tokens = event.metadata.get("total_tokens", 0)
                        total_latency = event.metadata.get("total_latency_ms", 0.0)

                        agent_panel.clear_activity()
                        agent_panel.stream_end(event.text or "")
                        thinking.add_thinking("response complete")
                        # Collapsible per-turn Thinking… block — mockup-style
                        # step checklist + gum-glyph trace lines
                        steps = self._build_turn_steps(turn_trace)
                        agent_panel.add_thinking_disclosure(steps + turn_trace)
                        # Compact turn statistics line
                        agent_panel.add_meta(
                            f"⏱ {total_iterations} iter · ⚙ {total_tool_calls} tools · "
                            f"{total_tokens:,} tok · {total_latency:.0f}ms"
                        )
                        status_bar.update_stats(
                            tokens=total_tokens,
                            iteration=total_iterations,
                            status="done",
                            latency_ms=total_latency,
                            activity="",
                        )
                        self._update_stats_panel(
                            total_iterations, total_tool_calls,
                            self._conversation.stats.total_tokens_in,
                            self._conversation.stats.total_tokens_out,
                            total_latency,
                        )

                    case AgentEventType.ERROR:
                        agent_panel.set_activity("error", "Error", event.text or "")
                        agent_panel.add_error(event.text or "Unknown error")
                        thinking.add_error(event.text or "Unknown error")
                        turn_trace.append(("error", event.text or "error"))
                        status_bar.update_stats(status="error", activity="")

                    case AgentEventType.DONE:
                        if status_bar.status != "done" and status_bar.status != "error":
                            status_bar.update_stats(status="idle", activity="")

        except Exception as e:
            agent_panel.set_activity("error", "Agent error", str(e))
            agent_panel.add_error(f"Agent error: {e}")
            thinking.add_error(str(e))
            status_bar.update_stats(status="error", activity="")

    @work(exclusive=False)
    async def _run_planning(self, task: str) -> None:
        """Run task decomposition and display the plan."""
        agent_panel = self.query_one("#agent-panel-widget", AgentPanel)
        plan_panel = self.query_one("#plan-panel-widget", PlanPanel)

        agent_panel.set_activity("thinking", "Decomposing task into steps…")
        try:
            thinking = self.query_one("#thinking-panel-widget", ThinkingPanel)
            thinking.add_thinking("planning — decomposing task")
            decomposer = TaskDecomposer(self.agent.provider)
            plan = await decomposer.decompose(task)
            agent_panel.clear_activity()
            thinking.add_thinking(f"plan ready — {len(plan.items)} steps")
            plan_panel.set_plan(plan)
            agent_panel.add_assistant_message(
                f"[bold]Plan ready[/]: {len(plan.items)} steps\n\n"
                + plan.to_markdown()
            )
        except Exception as e:
            agent_panel.set_activity("error", "Planning failed", str(e))
            agent_panel.add_error(f"Planning failed: {e}")

    @staticmethod
    def _build_turn_steps(trace: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        The mockup's Thinking… checklist, derived from what actually happened:
        Understanding request → Analyzing tools → Planning → Preparing response.
        Rows are (kind, text); kinds are step-done / step-active / step-pending.
        """
        kinds = {k for k, _ in trace}
        ran_tools = bool(kinds & {"tool", "done", "error"})
        return [
            ("step-done", "Understanding user request"),
            (
                "step-done" if ran_tools else "step-pending",
                "Analyzing available tools",
            ),
            ("step-done", "Planning next action"),
            ("step-done", "Preparing response"),
        ]

    def _update_stats_panel(
        self,
        iterations: int,
        tool_calls: int,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
    ) -> None:
        display = self.query_one("#stats-display", Static)
        model = self.agent.provider.default_model or "—"
        time_str = f"{latency_ms / 1000:.1f}s" if latency_ms else "—"
        display.update(
            "[#4ac26b]●[/] Active\n"
            f"[dim]Session:[/] [bold]{self._conversation.id[:8]}\n"
            f"[dim]Model:[/] [bold]{model}\n"
            f"[dim]Iterations:[/] [bold]{iterations}\n"
            f"[dim]Tokens:[/] [bold]{tokens_in + tokens_out:,}\n"
            f"[dim]Time:[/] [bold]{time_str}"
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_conversation(self) -> None:
        self.query_one("#agent-panel-widget", AgentPanel).clear()
        self.query_one("#tool-log-widget", ToolLogPanel).clear()
        try:
            self.query_one("#thinking-panel-widget", ThinkingPanel).clear()
        except Exception:
            pass
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

    def action_toggle_thinking(self) -> None:
        """Expand/collapse the live thinking panel (ctrl+t or palette)."""
        try:
            self.query_one("#thinking-panel-widget", ThinkingPanel).toggle()
        except Exception:
            pass

    def action_toggle_sidebar(self) -> None:
        """Show/hide the activity sidebar — true single-terminal mode (ctrl+b)."""
        try:
            sidebar = self.query_one("#right-sidebar", Vertical)
            sidebar.display = not sidebar.display
        except Exception:
            pass

    def action_cancel_task(self) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_stats(status="idle")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Initialise status bar, then play the typewriter splash animation."""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_stats(
            provider=self.agent.provider.name,
            model=self.agent.provider.default_model,
            status="idle",
            workspace=str(self.workspace_path),
        )
        self.run_worker(self._animate_splash(), exclusive=True)

    # ── Typewriter splash (gum-style) ────────────────────────────────────────

    async def _animate_splash(self) -> None:
        """
        Startup animation: types 'TRACERA', then a tagline, then cycles
        through init steps with a spinner — like a live boot sequence.
        """
        try:
            logo = self.query_one("#splash-logo", Static)
            status = self.query_one("#splash-status", Static)

            word = "TRACERA"
            buf = ""
            for ch in word:
                buf += ch
                logo.update(f"\n  {buf}▌\n")
                await asyncio.sleep(0.09)
                if self._splash_done:
                    return
            logo.update(f"\n  {word}\n")
            await asyncio.sleep(0.12)
            if self._splash_done:
                return

            sub = "your terminal coding agent"
            sbuf = ""
            for ch in sub:
                sbuf += ch
                status.update(f"  {sbuf}▌")
                await asyncio.sleep(0.02)
                if self._splash_done:
                    return

            for step in ("checking providers", "loading memory", "loading code index", "ready"):
                status.update(f"  ⠋ {step}")
                await asyncio.sleep(0.18)
                if self._splash_done:
                    return
            status.update("  ✓ ready")
            await asyncio.sleep(0.15)
        except Exception:
            pass
        self._splash_done = True
        await self._finish_splash()

    async def _finish_splash(self) -> None:
        """Remove the overlay and type the welcome message into the terminal."""
        try:
            self.query_one("#splash-overlay").remove()
        except Exception:
            pass
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass
        await self._type_welcome()

    async def _type_welcome(self) -> None:
        """Type the greeting character-by-character, gum-style."""
        try:
            panel = self.query_one("#agent-panel-widget", AgentPanel)
            await panel.type_message(
                "I'm ready to help with your coding tasks.\n"
                "Read a file, edit code, search the codebase, run tests, "
                "or plan a change — just ask.\n"
            )
        except Exception:
            pass

    def on_key(self, event) -> None:
        """Any key during the splash skips straight into the terminal."""
        if not self._splash_done:
            try:
                if self.query_one("#splash-overlay"):
                    self._splash_done = True
                    self.run_worker(self._finish_splash())
            except Exception:
                pass
