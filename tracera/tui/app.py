"""
TRACERA Main Textual Application — single-stream terminal-agent TUI
(Claude Code style).

Layout:
┌──────────────────────────────────────────────────────────────┐
│  ■ TRACERA  your terminal coding agent  /path      ● model  │  header
├──────────────────────────────────────────────────────────────┤
│  ┌─ YOU ───────────────────────────────────────────────────┐ │
│  │  add jwt validation to the middleware                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ✓ search_code (query='jwt auth')                      8ms │  ← one
│  ✓ read_file   (path='auth/middleware.py')             2ms │    stream,
│  ✗ run_command (command='pytest')                      0ms │    auto-
│    └ pytest: error: unrecognized arguments                 │    scrolls
│  → Memory: recalled architecture notes                     │
│  ┌─ TRACERA ─────────────────────────────────────────────┐ │
│  │  Done. All 23 tests pass.                             │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ● DONE  session 8f2c · model gemini · 5 tools · 3 iter  ...│  status
│  ❯ [ input pill .................... ]                     │
│  Enter send · /help commands · ctrl+t verbose rows          │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Provider
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import DirectoryTree, ListItem, ListView, Static

from tracera.agent.memory import AgentMemory
from tracera.agent.planner import TaskDecomposer
from tracera.agent.react_loop import AgentEventType, ReActAgent
from tracera.conversation.state import ConversationState
from tracera.observability import get_telemetry
from tracera.tui import theme as tui_theme
from tracera.tui.diffutil import DIFFABLE_TOOLS, MAX_DIFF_BYTES, compute_diff, is_image
from tracera.tui.splash import SplashScreen
from tracera.tui.widgets.agent_panel import (
    _PHASE_LABELS,
    AgentPanel,
    CollapsibleRow,
    InlineStatus,
    LoaderPill,
    format_args,
)
from tracera.tui.widgets.command_input import CommandInput
from tracera.tui.widgets.dashboard import DashboardWidget
from tracera.tui.widgets.file_context import FileContextPanel
from tracera.tui.widgets.memory_viz import MemoryGraphWidget
from tracera.tui.widgets.slash_actions import (
    SLASH_TOOLS,
    coerce_args,
    parse_tool_invocation,
    single_arg_kwargs,
)

#: Attached text files larger than this (bytes) are not injected into context.
_ATTACH_TEXT_MAX_BYTES = 200_000


_HELP_TEXT = """\
[bold cyan]TRACERA — Commands[/]

[bold]/help[/]          Show this help
[bold]/clear[/]         Clear conversation
[bold]/status[/]        Show system status
[bold]/memory[/]        Show memory contents
[bold]/model[/] [name]   Switch model
[bold]/models[/]        List all provider/model options
[bold]/plan[/] [task]    Decompose a task into steps
[bold]/code[/] [task]    Run a coding task (same as plain input)
[bold]/search[/] <q>     Search the code index (hybrid)
[bold]/debug[/] <q>      Compare retrieval strategies (BM25/Dense/Hybrid/Reranker)
[bold]/index[/]          Index the workspace (Phase 16-24 pipeline)
[bold]/test[/]           Run the project's test suite
[bold]/review[/]         Ask the agent to review current changes
[bold]/fix[/] <task>     Autonomous fix loop (plan → retrieve → edit → test)
[bold]/selfreview[/]     Independent LLM review of uncommitted changes
[bold]/regression[/]     Baseline vs current test comparison
[bold]/tools[/]          List available tools
[bold]/mcp[/]            Show MCP status & config
[bold]/cost[/]           Show session token/cost estimate
[bold]/observability[/]  Show live telemetry (LLM/tool/retrieval/cost)
[bold]/inspect[/]        Repository inspection (files, symbols, git)
[bold]/deps[/] <symbol>   Show a symbol's dependency chain
[bold]/dashboard[/]      Show system overview panel
[bold]/theme[/]         Cycle accent theme (claude · crush · nord)
[bold]/memgraph[/]       Show the memory knowledge graph
[bold]/files[/]          Show recently touched files
[bold]/phases[/]         Show the phase map + verified checklist
[bold]/phases done <n>[/]  Mark a phase as verified (persisted)
[bold]/features[/]       List every feature as a slash command
[bold]/delegate[/] [task]  Decompose a task across sub-agents and aggregate results
[bold]/agents[/]         Sub-agent fleet overview
[bold]/tool[/] [name args]  Run any registry tool directly
[bold]/reset[/]          Reset conversation state

[bold cyan]Keys[/]

[bold]ctrl+q[/]        Quit
[bold]ctrl+l[/]        Clear conversation
[bold]ctrl+t[/]        Toggle verbose tool rows (show/hide args)
[bold]ctrl+p[/]        Switch provider/model (rounded dropdown, live)
[bold]ctrl+shift+p[/]  Command palette
[bold]ctrl+m[/]        Show memory
[bold]f1[/]            Help
[bold]esc[/]           Cancel running task
[bold]pgup/pgdn[/]     Scroll the stream

[bold cyan]Stream[/]

Everything the agent does streams inline: phase markers (◇ Thinking,
◇ Generating), tool calls, file reads, command runs — ✓ success, ✗ failure
(with the error line beneath), an animated spinner while in flight.
[bold]File edits[/] collapse to a [bold]📝 path +N -M[/] summary row — click
it to expand the full inline diff (green added, red removed). Click any
[bold]→ row[/] (memory, search results, plans, repo info) to expand it.

[bold cyan]Attachments[/]

Click [bold]＋[/] next to the input to attach files (text files are read and
injected into the agent's context; images show a [bold][!][/] badge when the
active model cannot view them). While a request runs, the input is replaced
by a loader pill showing the live phase — click [bold]●[/] to stop it.
"""


class FilePicker(Screen):
    """Modal directory-tree picker for attachments. Dismisses with the path."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, root: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = root

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-panel"):
            yield Static(f"Attach file from {self._root}", id="picker-title")
            yield DirectoryTree(str(self._root), id="picker-tree")
            yield Static(
                "enter: attach · esc: cancel",
                id="picker-hint",
            )

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(event.path)


class ProviderSwitcher(Screen):
    """
    Modal dropdown listing every configured provider/model.

    The list is discovered at runtime from ``list_available_providers`` — the
    same source the CLI uses — so config changes show up with no code change.
    Providers without an API key are listed dimmed with a warning marker and
    cannot be selected (no silent failures later).
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

    def __init__(self, entries: list[dict], *, active_name: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries = entries
        self._active_name = active_name

    def compose(self) -> ComposeResult:
        with Vertical(id="provider-panel"):
            yield Static(" Provider / Model ", id="provider-title")
            yield ListView(id="provider-list")
            yield Static("↑↓ navigate · enter select · esc close", id="provider-hint")

    def on_mount(self) -> None:
        from rich.text import Text

        from tracera.tui.theme import get_theme as _gt
        th = _gt()
        lst = self.query_one("#provider-list", ListView)
        active_index = 0
        for i, info in enumerate(self._entries):
            name = str(info.get("name", "?"))
            model = str(info.get("model") or "")
            available = bool(info.get("available", False))
            is_active = name == self._active_name
            if is_active:
                active_index = i

            row = Text()
            row.append(" ✓ " if is_active else "   ", style=f"bold #{th.success}")
            row.append(
                name,
                style=f"bold #{th.text}" if available else f"dim #{th.muted}",
            )
            row.append(
                f"   {model}",
                style=f"dim #{th.secondary}" if available else f"dim #{th.faint}",
            )
            if not available:
                env = str(info.get("key_env") or "API_KEY").upper()
                row.append(f"   [!] missing {env}", style=f"bold #{th.warning}")
            item = ListItem(Static(row), disabled=not available)
            if is_active:
                item.add_class("provider-active")
            lst.append(item)
        # Open with the active provider highlighted.
        if self._entries:
            lst.index = active_index
        lst.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        lst = self.query_one("#provider-list", ListView)
        idx = lst.index
        if idx is None or not (0 <= idx < len(self._entries)):
            return
        info = self._entries[idx]
        if not info.get("available"):
            return  # disabled rows can't be selected — never fail silently
        # The dismissal RESULT is what the push_screen callback receives —
        # this is the (name, model) that flows into _apply_provider.
        self.dismiss((str(info["name"]), str(info.get("model") or "")))


class TraceraCommands(Provider):
    """Command-palette entries (ctrl+p)."""

    _COMMANDS = [
        ("Switch provider/model", "action_switch_provider", "Open the provider/model selector (ctrl+p)"),
        ("Cycle theme", "action_cycle_theme", "Cycle accent theme presets (/theme)"),
        ("Toggle verbose rows", "action_toggle_verbose", "Show/hide tool call arguments"),
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
    """TRACERA — single-stream terminal UI (Claude Code style)."""

    TITLE = "TRACERA"
    CSS_PATH = "styles/tracera.tcss"

    COMMANDS = {TraceraCommands}

    def get_css_variables(self) -> dict[str, str]:
        """Inject the active theme preset into the stylesheet as CSS vars."""
        return {**super().get_css_variables(), **tui_theme.get_theme().css_variables()}

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_conversation", "Clear", show=True),
        Binding("ctrl+t", "toggle_verbose", "Rows", show=True),
        Binding("ctrl+p", "switch_provider", "Provider", show=True),
        Binding("ctrl+shift+p", "command_palette", "Commands", show=True),
        Binding("ctrl+m", "show_memory", "Memory", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("escape", "cancel_task", "Cancel", show=False),
        Binding("pageup", "scroll_active(-1)", "Scroll Up", show=True),
        Binding("pagedown", "scroll_active(1)", "Scroll Down", show=True),
    ]

    def __init__(
        self,
        agent: ReActAgent,
        memory: AgentMemory,
        workspace_path: Path = Path("."),
        retrieval_pipeline=None,
        banner: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.memory = memory
        self.workspace_path = workspace_path
        self.retrieval_pipeline = retrieval_pipeline
        # Wire the memory layer to the agent so /memory, /memgraph, and memory
        # tools can reach it through agent._enhanced_memory etc.
        self._wire_memory_to_agent()
        # The CLI already printed this banner to scrollback; the app reproduces
        # it at the top of its own first frame so the (unavoidable on Windows)
        # alt-screen switch looks continuous rather than like a new screen.
        self._banner = banner
        self._conversation = ConversationState()
        self._running_worker: Any = None
        self._hovered_scrollable: ScrollableContainer | None = None
        self._plan_row: CollapsibleRow | None = None
        self._recent_files: list[tuple[str, str]] = []
        # Restore the persisted theme preset (process-wide Rich colors too).
        try:
            from tracera.config.settings import get_settings
            tui_theme.set_theme(tui_theme.load_saved_theme(get_settings().tracera_data_dir))
        except Exception:
            tui_theme.set_theme(tui_theme.DEFAULT_THEME)

    # ── Memory wiring ─────────────────────────────────────────────────────────

    def _wire_memory_to_agent(self) -> None:
        """Connect the memory facade to the agent so TUI commands and tools can reach it.

        The legacy ``AgentMemory`` (JSON-backed) is enhanced with the new
        memory-layer attributes when a ``MemoryLayer``-based facade is available.
        This makes ``/memory``, ``/memgraph``, ``/triples``, and the memory tools
        work without requiring the agent constructor to change.
        """
        mem = self.memory

        # If the memory object already has the new-style attributes, use them.
        if hasattr(mem, "_layer"):
            # MemoryLayer-based AgentMemory — extract the stores.
            layer = mem._layer
            self.agent._enhanced_memory = mem
            self.agent._triple_store = layer.store.triple_store
            self.agent._session_manager = layer.store.session_manager
            return

        # Check if the memory object exposes triple_store / session_manager directly.
        ts = getattr(mem, "triple_store", None)
        sm = getattr(mem, "session_manager", None)
        if ts is not None or sm is not None:
            self.agent._enhanced_memory = mem
            self.agent._triple_store = ts
            self.agent._session_manager = sm
            return

        # Legacy JSON-backed AgentMemory — attach a TripleStore and SessionManager
        # so the TUI displays gracefully even without the full memory layer.
        try:
            from tracera.memory.triples import TripleStore
            from tracera.memory.session import SessionManager

            self.agent._triple_store = TripleStore()
            self.agent._session_manager = SessionManager()
            self.agent._enhanced_memory = mem
        except Exception:
            # If even those can't be created, leave the attributes unset.
            # action_show_memory will fall back to the legacy display path.
            pass

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # The ASCII banner was already printed to scrollback by the CLI before
        # the app started — the TUI renders directly below it, no clear.
        yield self._build_header()
        # The single main panel — conversation stream, status line, input.
        yield AgentPanel(id="agent-panel-widget")

    def _build_header(self) -> Horizontal:
        from rich.text import Text

        from tracera.tui.theme import get_theme as _gt
        th = _gt()
        model = self.agent.provider.default_model or "model"
        left = Text()
        left.append(" ◆ ", style=f"bold #{th.accent}")
        left.append("TRACERA", style=f"bold #{th.text}")
        left.append("  your terminal coding agent", style=f"dim #{th.muted}")
        left.append("  ", style="dim")
        left.append(str(self.workspace_path), style=f"dim #{th.faint}")
        right = Text()
        git_info = self._get_git_status()
        if git_info:
            right.append(f"{git_info}  ", style=f"bold #{th.meta}")
        right.append("● ", style=f"bold #{th.success}")
        right.append(model, style=f"bold #{th.text}")
        right.append("  /help", style=f"dim #{th.faint}")
        return Horizontal(
            Static(left, id="header-left"),
            Static(right, id="header-right"),
            id="app-header",
        )

    def _get_git_status(self) -> str:
        """Get git branch and status info."""
        try:
            from tracera.git.operations import GitRepo
            repo = GitRepo(self.workspace_path)
            status = repo.status()
            branch = status.branch or "detached"
            dirty = "●" if status.is_dirty else "○"
            return f"{dirty} {branch}"
        except Exception:
            return ""

    def _update_header_git(self) -> None:
        """Update header with git status."""
        try:
            from rich.text import Text

            from tracera.tui.theme import get_theme as _gt
            th = _gt()
            git_info = self._get_git_status()
            right = Text()
            right.append("● ", style=f"bold #{th.success}")
            if git_info:
                right.append(f"{git_info}  ", style=f"dim #{th.meta}")
            right.append(self.agent.provider.default_model or "model", style=f"bold #{th.text}")
            right.append("  /help", style=f"dim #{th.faint}")
            self.query_one("#header-right", Static).update(right)
        except Exception:
            pass

    # ── Scroll handling ───────────────────────────────────────────────────────

    def _find_scrollable(self, widget) -> ScrollableContainer | None:
        node = widget
        while node is not None:
            if isinstance(node, ScrollableContainer):
                return node
            node = node.parent
        return None

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Track which scrollable the mouse is over (for pgup/pgdn)."""
        try:
            widget, _ = self.screen.get_widget_at(event.x, event.y)
        except Exception:
            return
        scrollable = self._find_scrollable(widget)
        if scrollable is not None:
            self._hovered_scrollable = scrollable

    def _active_scrollable(self) -> ScrollableContainer:
        if self._hovered_scrollable is not None and self._hovered_scrollable.is_attached:
            return self._hovered_scrollable
        return self._panel()._stream()

    def action_scroll_active(self, direction: int) -> None:
        target = self._active_scrollable()
        if direction < 0:
            target.scroll_page_up(animate=False)
        else:
            target.scroll_page_down(animate=False)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _panel(self) -> AgentPanel:
        return self.query_one("#agent-panel-widget", AgentPanel)

    def _status_line(self) -> InlineStatus:
        return self._panel().query_one("#status-line", InlineStatus)

    @on(AgentPanel.SubmitTask)
    async def on_submit_task(self, event: AgentPanel.SubmitTask) -> None:
        text = event.text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._handle_command(text)
            return
        panel = self._panel()
        panel.add_user_message(text)
        self._start_agent_task(text)

    @on(AgentPanel.AttachRequested)
    def on_attach_requested(self, event: AgentPanel.AttachRequested) -> None:
        self._open_picker()

    @on(LoaderPill.StopRequested)
    def on_loader_stop_requested(self, event: LoaderPill.StopRequested) -> None:
        self.action_cancel_task()

    async def _handle_command(self, text: str) -> None:
        panel = self._panel()
        cmd = text.split()[0].lower()

        if cmd == "/help":
            panel.add_assistant_message(_HELP_TEXT, trusted=True)
        elif cmd == "/clear":
            self.action_clear_conversation()
        elif cmd == "/status":
            self._show_status(panel)
        elif cmd == "/memory":
            self.action_show_memory()
        elif cmd == "/reset":
            self._conversation = ConversationState()
            panel.add_assistant_message("[dim]Conversation reset.[/]", trusted=True)
        elif cmd == "/plan":
            task = text[6:].strip()
            if task:
                await self._run_planning(task)
            else:
                panel.add_error("Usage: /plan <task description>")
        elif cmd == "/model":
            parts = text.split()
            if len(parts) > 1:
                new_model = parts[1]
                self.agent.model = new_model
                panel.add_assistant_message(
                    f"[bold]Model switched to:[/] [bold cyan]{escape(new_model)}[/]",
                    trusted=True,
                )
            else:
                self._show_models()
        elif cmd == "/models":
            self._show_models()
        elif cmd in ("/code", "/ask"):
            task = text[len(cmd):].strip()
            if task:
                panel.add_user_message(task)
                self._start_agent_task(task)
            else:
                panel.add_error(f"Usage: {escape(cmd)} <task description>")
        elif cmd == "/search":
            query = text[len(cmd):].strip()
            if query:
                await self._run_search(query)
            else:
                panel.add_error("Usage: /search <query>")
        elif cmd == "/debug":
            query = text[len(cmd):].strip()
            if query:
                await self._run_debug(query)
            else:
                panel.add_error("Usage: /debug <query>")
        elif cmd == "/index":
            await self._run_indexing()
        elif cmd == "/test":
            await self._run_tests()
        elif cmd == "/review":
            await self._run_review()
        elif cmd == "/tools":
            self._show_tools(panel)
        elif cmd == "/mcp":
            self._show_mcp(panel)
        elif cmd == "/cost":
            self._show_cost(panel)
        elif cmd == "/inspect":
            await self._run_inspect()
        elif cmd == "/deps":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_deps(symbol)
            else:
                panel.add_error("Usage: /deps <symbol>")
        elif cmd == "/phases":
            self._show_phases(text)
        elif cmd == "/observability":
            self._show_observability(panel)
        elif cmd == "/theme":
            self._cycle_theme()
        elif cmd == "/dashboard":
            self._show_dashboard()
        elif cmd == "/memgraph":
            self._show_memory_graph()
        elif cmd == "/files":
            self._show_file_context()
        elif cmd == "/features":
            self._show_features(panel)
        elif cmd == "/tool":
            spec = text[len(cmd):].strip()
            if spec:
                await self._run_tool_command(spec)
            else:
                panel.add_error("Usage: /tool <name> [key=value ...]")
        elif cmd == "/delegate":
            task = text[len(cmd):].strip()
            if task:
                await self._run_delegate(task)
            else:
                panel.add_error("Usage: /delegate <task description>")
        elif cmd == "/agents":
            self._show_agents(panel)
        elif cmd == "/fix":
            task = text[len(cmd):].strip()
            if task:
                await self._run_fix_loop(task)
            else:
                panel.add_error("Usage: /fix <failing task or test description>")
        elif cmd == "/selfreview":
            await self._run_selfreview()
        elif cmd == "/regression":
            await self._run_regression_check()
        # ── Code intelligence retrieval aliases (require index) ──────────────
        elif cmd == "/symbol":
            name = text[len(cmd):].strip()
            if name:
                await self._run_symbol(name)
            else:
                panel.add_error("Usage: /symbol <name>")
        elif cmd == "/symbols":
            query = text[len(cmd):].strip()
            if query:
                await self._run_symbols(query)
            else:
                panel.add_error("Usage: /symbols <query>")
        elif cmd == "/source":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_source(symbol)
            else:
                panel.add_error("Usage: /source <symbol>")
        elif cmd == "/definition":
            name = text[len(cmd):].strip()
            if name:
                await self._run_definition(name)
            else:
                panel.add_error("Usage: /definition <name>")
        elif cmd == "/outline":
            file = text[len(cmd):].strip()
            if file:
                await self._run_outline(file)
            else:
                panel.add_error("Usage: /outline <file>")
        elif cmd == "/repomap":
            await self._run_repomap()
        elif cmd == "/assemble":
            task = text[len(cmd):].strip()
            if task:
                await self._run_assemble(task)
            else:
                panel.add_error("Usage: /assemble <task>")
        elif cmd == "/context":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_context(symbol)
            else:
                panel.add_error("Usage: /context <symbol>")
        elif cmd == "/deps":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_deps(symbol)
            else:
                panel.add_error("Usage: /deps <symbol>")
        elif cmd == "/refs":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_refs(symbol)
            else:
                panel.add_error("Usage: /refs <symbol>")
        elif cmd == "/callers":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_callers(symbol)
            else:
                panel.add_error("Usage: /callers <symbol>")
        elif cmd == "/blast":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_blast(symbol)
            else:
                panel.add_error("Usage: /blast <symbol>")
        elif cmd == "/changed":
            await self._run_changed()
        elif cmd == "/freshness":
            await self._run_freshness()
        elif cmd == "/importers":
            file = text[len(cmd):].strip()
            if file:
                await self._run_importers(file)
            else:
                panel.add_error("Usage: /importers <file>")
        elif cmd == "/classhier":
            class_name = text[len(cmd):].strip()
            if class_name:
                await self._run_classhier(class_name)
            else:
                panel.add_error("Usage: /classhier <class>")
        elif cmd == "/cycles":
            await self._run_cycles()
        elif cmd == "/coupling":
            await self._run_coupling()
        elif cmd == "/endpoint":
            route = text[len(cmd):].strip()
            if route:
                await self._run_endpoint(route)
            else:
                panel.add_error("Usage: /endpoint <route>")
        elif cmd == "/deadcode":
            await self._run_deadcode()
        elif cmd == "/hotspots":
            await self._run_hotspots()
        elif cmd == "/pagerank":
            await self._run_pagerank()
        elif cmd == "/refactor":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_refactor(symbol)
            else:
                panel.add_error("Usage: /refactor <symbol>")
        elif cmd == "/editsafe":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_editsafe(symbol)
            else:
                panel.add_error("Usage: /editsafe <symbol>")
        elif cmd == "/deletesafe":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_deletesafe(symbol)
            else:
                panel.add_error("Usage: /deletesafe <symbol>")
        elif cmd == "/impls":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_impls(symbol)
            else:
                panel.add_error("Usage: /impls <symbol>")
        elif cmd == "/provenance":
            symbol = text[len(cmd):].strip()
            if symbol:
                await self._run_provenance(symbol)
            else:
                panel.add_error("Usage: /provenance <symbol>")
        elif cmd == "/risk":
            target = text[len(cmd):].strip()
            if target:
                await self._run_risk(target)
            else:
                panel.add_error("Usage: /risk <target>")
        elif cmd == "/prrisk":
            await self._run_prrisk()
        elif cmd == "/auditconfig":
            await self._run_auditconfig()
        elif cmd == "/ast":
            pattern = text[len(cmd):].strip()
            if pattern:
                await self._run_ast(pattern)
            else:
                panel.add_error("Usage: /ast <pattern>")
        elif cmd == "/sessionstats":
            await self._run_sessionstats()
        elif cmd == "/plantask":
            task = text[len(cmd):].strip()
            if task:
                panel.add_meta(f"→ [bold]PlanTask[/] {escape(task[:60])}")
                await self._run_plantask(task)
            else:
                panel.add_error("Usage: /plantask <task description>")
        elif cmd == "/planturn":
            query = text[len(cmd):].strip()
            if query:
                await self._run_planturn(query)
            else:
                panel.add_error("Usage: /planturn <query>")
        elif cmd == "/ranked":
            query = text[len(cmd):].strip()
            if query:
                await self._run_ranked(query)
            else:
                panel.add_error("Usage: /ranked <query>")
        elif cmd == "/taskcontext":
            task = text[len(cmd):].strip()
            if task:
                await self._run_taskcontext(task)
            else:
                panel.add_error("Usage: /taskcontext <task>")
        # ── Memory aliases ────────────────────────────────────────────────────
        elif cmd == "/recall":
            query = text[len(cmd):].strip()
            if query:
                await self._run_recall(query)
            else:
                panel.add_error("Usage: /recall <query>")
        elif cmd == "/remember":
            text_content = text[len(cmd):].strip()
            if text_content:
                await self._run_remember(text_content)
            else:
                panel.add_error("Usage: /remember <text>")
        elif cmd == "/forget":
            text_content = text[len(cmd):].strip()
            if text_content:
                await self._run_forget(text_content)
            else:
                panel.add_error("Usage: /forget <text>")
        elif cmd == "/sessions":
            await self._run_sessions()
        elif cmd == "/memstats":
            await self._run_memstats()
        elif cmd == "/consolidate":
            await self._run_consolidate()
        elif cmd == "/memgraph2":
            await self._run_memgraph2()
        elif cmd == "/memworker":
            await self._run_memworker()
        elif cmd == "/memsearch":
            query = text[len(cmd):].strip()
            if query:
                await self._run_memsearch(query)
            else:
                panel.add_error("Usage: /memsearch <query>")
        elif cmd == "/triples":
            await self._run_triples()
        # ── Git & repo operations ────────────────────────────────────────────
        elif cmd == "/git":
            subcommand = text[len(cmd):].strip()
            if subcommand:
                await self._run_git(subcommand)
            else:
                panel.add_error("Usage: /git <subcommand>")
        elif cmd == "/inspectrepo":
            await self._run_inspectrepo()
        elif cmd == "/tests":
            framework = text[len(cmd):].strip()
            await self._run_tests_tool(framework)
        elif cmd == "/read":
            path = text[len(cmd):].strip()
            if path:
                await self._run_read(path)
            else:
                panel.add_error("Usage: /read <path>")
        elif cmd == "/write":
            path = text[len(cmd):].strip()
            if path:
                await self._run_write(path)
            else:
                panel.add_error("Usage: /write <path>")
        elif cmd == "/edit":
            path = text[len(cmd):].strip()
            if path:
                await self._run_edit(path)
            else:
                panel.add_error("Usage: /edit <path>")
        elif cmd == "/ls":
            path = text[len(cmd):].strip()
            if path:
                await self._run_ls(path)
            else:
                panel.add_error("Usage: /ls <path>")
        elif cmd == "/grep":
            pattern = text[len(cmd):].strip()
            if pattern:
                await self._run_grep(pattern)
            else:
                panel.add_error("Usage: /grep <pattern>")
        elif cmd == "/run":
            command = text[len(cmd):].strip()
            if command:
                await self._run_run(command)
            else:
                panel.add_error("Usage: /run <command>")
        elif cmd[1:] in SLASH_TOOLS:
            await self._run_tool_alias(SLASH_TOOLS[cmd[1:]], text[len(cmd):].strip())
        else:
            panel.add_error(
                f"Unknown command: {escape(cmd)}. Type /help for available commands."
            )

    # ── Theme switching ─────────────────────────────────────────────────────

    def _cycle_theme(self) -> None:
        """/theme — cycle accent presets live and persist the choice."""
        new_theme = tui_theme.next_theme()
        try:
            from tracera.config.settings import get_settings
            tui_theme.save_theme(get_settings().tracera_data_dir, new_theme.name)
        except Exception:
            pass
        # Refresh CSS-variable-driven colors immediately.
        self.refresh_css()
        # Rich-Text-rendering widgets read the preset each render; force a
        # repaint of the stream rows so inline colors match the new preset.
        for row in self._panel().query("ToolRow, PhaseRow, InlineStatus"):
            row.refresh()
        self._update_header_git()
        self._panel().add_meta(
            f"Theme switched to [bold]{new_theme.label}[/] "
            f"— persisted. (Cycle again with /theme)"
        )

    # ── REPL command implementations ─────────────────────────────────────────

    def _show_tools(self, panel: AgentPanel) -> None:
        names = [t.name for t in self.agent.registry.tools]
        if not names:
            panel.add_error("No tools registered.")
            return
        lines = "\n".join(f"  [dim]▪[/] {escape(n)}" for n in sorted(names))
        panel.add_assistant_message(
            f"[bold]Available tools ({len(names)})[/]\n{lines}", trusted=True
        )

    def _show_mcp(self, panel: AgentPanel) -> None:
        config_path = Path(self.workspace_path) / ".tracera" / "mcp_servers.json"
        lines = ["[bold]MCP[/]"]
        if config_path.exists():
            lines.append(f"  config: [cyan]{escape(str(config_path))}[/]")
        else:
            lines.append(
                "  [dim]No mcp_servers.json yet — see MCP_CONNECTIONS.md for "
                "server configs and required credentials.[/]"
            )
        lines.append(
            "  [dim]Use `tracera mcp serve` (server) or "
            "`tracera mcp connect <file>` (client).[/]"
        )
        panel.add_assistant_message("\n".join(lines), trusted=True)

    def _show_cost(self, panel: AgentPanel) -> None:
        stats = self._conversation.stats
        tokens_in = stats.total_tokens_in
        tokens_out = stats.total_tokens_out
        cost_in = tokens_in / 1_000_000 * 0.30
        cost_out = tokens_out / 1_000_000 * 1.20
        panel.add_assistant_message(
            f"[bold]Session cost estimate[/]\n\n"
            f"  Tokens in:   [cyan]{tokens_in:,}[/]\n"
            f"  Tokens out:  [cyan]{tokens_out:,}[/]\n"
            f"  Total:       [bold]{tokens_in + tokens_out:,}[/]\n"
            f"  Est. cost:   [bold green]${cost_in + cost_out:.4f}[/]\n"
            f"[dim](estimate @ $0.30/$1.20 per 1M tokens)[/]",
            trusted=True,
        )

    def _show_models(self) -> None:
        """List every provider/model discovered at runtime, with availability."""
        panel = self._panel()
        try:
            from tracera.config.settings import get_settings
            from tracera.providers import list_available_providers
            entries = list_available_providers(get_settings())
            active_name = getattr(self.agent.provider, "name", None)
            lines = ["[bold]Models[/] (from your config, nothing hardcoded)\n"]
            for info in entries:
                name = info["name"]
                model = info["model"] or "—"
                if info["available"]:
                    marker = "✓" if name == active_name else "·"
                    lines.append(
                        f"  [bold green]{marker}[/] [bold]{escape(name)}[/]  [dim]{escape(model)}[/]"
                    )
                else:
                    env = info.get("key_env") or "API_KEY"
                    lines.append(
                        f"  [dim]✗[/] [dim]{escape(name)}[/]  [dim]{escape(model)}  "
                        f"\\[!] missing {escape(env)}[/]"
                    )
            lines.append("\n[dim]switch with /model <id> or ctrl+p[/]")
            panel.add_assistant_message("\n".join(lines), trusted=True)
        except Exception as e:
            panel.add_error(f"Cannot list models: {escape(str(e))}")

    def _show_observability(self, panel: AgentPanel) -> None:
        """Phase 60 — live telemetry as an expandable row."""
        try:
            from tracera.observability import get_telemetry
            snap = get_telemetry().snapshot()
            llm = snap["llm"]
            tools = snap["tools"]
            retrieval = snap["retrieval"]
            agent = snap["agent"]
            cost = snap["cost"]

            lines = [
                "[bold]LLM[/]",
                f"  calls: {llm['calls']} · errors: {llm['errors']} "
                f"· {llm['total_tokens']:,} tok · avg {llm['avg_latency_ms']}ms",
                "",
                "[bold]Tools[/]  (total {})".format(tools["calls"]),
            ]
            lines = [
                "[bold]LLM[/]",
                f"  calls: {llm['calls']} · errors: {llm['errors']} "
                f"· {llm['total_tokens']:,} tok · avg {llm['avg_latency_ms']}ms",
                "",
                "[bold]Tools[/]  (total {})".format(tools["calls"]),
            ]
            lines += [
                f"  {name}: {count}" for name, count in list(tools["per_tool"].items())[:12]
            ]
            lines.append("")
            lines.append("[bold]Retrieval[/]")
            lines += [f"  {k}: {v}" for k, v in retrieval["by_kind"].items()]
            lines.append("")
            lines.append(
                f"[bold]Agent[/]  {agent['iterations']} iterations · {agent['errors']} errors"
            )
            lines.append(
                f"[bold]Cost[/]  [green]${cost['estimate_usd']:.4f}[/] "
                f"({snap['elapsed_seconds']}s elapsed)"
            )
            panel.add_info_row(
                f"Observability: {llm['calls']} LLM · {tools['calls']} tools",
                "\n".join(lines),
                trusted=True,
            )
        except Exception as e:
            panel.add_error(f"Observability failed: {escape(str(e))}")

    def _show_dashboard(self) -> None:
        """/dashboard — system overview as an inline widget block."""
        panel = self._panel()
        provider = self.agent.provider
        widget = DashboardWidget(
            provider=getattr(provider, "name", "—"),
            model=provider.default_model or "—",
            workspace=str(self.workspace_path),
            memory_count=self.memory.count,
            tool_count=len(self.agent.registry.tools),
        )
        widget.set_feature_status("retrieval", self.retrieval_pipeline is not None)
        widget.set_feature_status("rag", self.retrieval_pipeline is not None)
        widget.set_feature_status("index", self.retrieval_pipeline is not None)
        try:
            import mcp  # noqa: F401
            widget.set_feature_status("mcp", True)
        except ImportError:
            widget.set_feature_status("mcp", False)
        panel._append(widget)

    def _show_memory_graph(self) -> None:
        """/memgraph — knowledge graph visualization."""
        panel = self._panel()
        triple_store = getattr(self.agent, "_triple_store", None)
        widget = MemoryGraphWidget()
        if triple_store is not None and triple_store.triple_count:
            central = triple_store.get_central_concepts(10)
            type_counts: dict[str, int] = {}
            for t in triple_store.all_triples:
                pred = t.predicate
                type_counts[pred] = type_counts.get(pred, 0) + 1
            widget.update_graph(
                central_concepts=central,
                type_counts=type_counts,
            )
        panel._append(widget)

    def _show_file_context(self) -> None:
        """/files — recently touched files panel."""
        panel = self._panel()
        widget = FileContextPanel()
        for path, action in getattr(self, "_recent_files", []):
            widget.add_file(path, action=action)
        panel._append(widget)

    # ── Unified slash surface (every tool/feature is a slash command) ────────

    def _show_features(self, panel: AgentPanel) -> None:
        """/features — list every capability as a slash command."""
        from tracera.tui.widgets.slash_actions import FEATURE_GROUPS
        lines = ["[bold]Features[/] — every capability is a slash command\n"]
        for group, items in FEATURE_GROUPS.items():
            lines.append(f"[bold cyan]{group}[/]")
            lines.extend(f"  [bold]{usage}[/]  [dim]{desc}[/]" for usage, desc in items)
            lines.append("")
        lines.append(
            "[dim]/tool <name> [key=value ...] runs any registry tool directly. "
            "Type / for autocomplete.[/]"
        )
        panel.add_assistant_message("\n".join(lines), trusted=True)

    async def _run_tool_command(self, spec: str) -> None:
        """/tool <name> [key=value ...] — run any registered tool inline."""
        panel = self._panel()
        try:
            name, raw_args = parse_tool_invocation(spec)
        except ValueError as e:
            panel.add_error(str(e))
            return
        if not self.agent.registry.has(name):
            panel.add_error(f"Unknown tool: {escape(name)}. Try /tools or /features.")
            return
        tool = self.agent.registry.get(name)
        try:
            args = coerce_args(tool, raw_args)
        except ValueError as e:
            panel.add_error(f"Bad arguments for {escape(name)}: {escape(str(e))}")
            return
        await self._execute_tool_inline(name, args)

    async def _run_search(self, query: str) -> None:
        """Hybrid search — results as a collapsible inline row."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            if self.retrieval_pipeline is None:
                panel.add_error("Code index not loaded — run /index first.")
                return
            symbol_retriever = self.retrieval_pipeline[1]
            hits = symbol_retriever.search(query, k=8)
            if not hits:
                panel.add_info_row(f"Search: {escape(query)}", "[dim]No results.[/]")
                return
            lines = []
            for i, hit in enumerate(hits[:8], 1):
                path = hit.get("file_path") or hit.get("id") or "?"
                symbol = hit.get("symbol") or ""
                score = hit.get("_relevance_score") or hit.get("_rrf_score") or ""
                line = f"  {i}. [bold]{escape(str(path))}[/]"
                if symbol:
                    line += f" [dim]({escape(str(symbol))})[/]"
                if score:
                    line += f" [dim]· {float(score):.3f}[/]"
                lines.append(line)
                content = (hit.get("content") or "").strip().splitlines()
                if content:
                    lines.append("     [dim]" + escape(content[0][:80]) + "[/]")
            panel.add_info_row(
                f"Search: {escape(query)} ({len(hits)} hits)",
                "\n".join(lines),
                trusted=True,
            )
        except Exception as e:
            panel.add_error(f"Search failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_debug(self, query: str) -> None:
        """Phase 59: retrieval debugging — per-strategy comparison row."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            if self.retrieval_pipeline is None:
                panel.add_error("Code index not loaded — run /index first.")
                return
            from tracera.evaluation.strategies import (
                build_doc_resolver,
                build_strategies,
            )
            from tracera.retrieval.dense import DenseRetriever
            from tracera.retrieval.hybrid import HybridRetriever
            (
                _, _, _, reranker, _, _, embedder, vector_store, bm25, _,
            ) = self.retrieval_pipeline
            dense_retriever = DenseRetriever(embedder, vector_store)
            hybrid = HybridRetriever(bm25, dense_retriever)
            strategies = build_strategies(
                workspace=self.workspace_path,
                bm25=bm25,
                dense=dense_retriever,
                hybrid=hybrid,
                reranker=reranker,
                resolve_doc=build_doc_resolver(vector_store),
            )
            lines = [f"[bold cyan]Query:[/] {query}\n"]
            for name, strategy in strategies.items():
                hits = strategy.retrieve(query, k=5)
                lines.append(f"[bold]{name.upper()}[/]")
                if not hits:
                    lines.append("  [dim]— no results —[/]")
                for i, hit in enumerate(hits[:5], 1):
                    path = hit.file_path or hit.doc_id or "?"
                    lines.append(f"  {i}. [bold]{path}[/]")
                    if hit.content:
                        first = hit.content.strip().splitlines()
                        if first:
                            lines.append("     [dim]" + first[0][:70] + "[/]")
                lines.append("")
            panel.add_info_row(
                f"Debug: {query} ({len(strategies)} strategies)",
                "\n".join(lines),
            )
        except Exception as e:
            panel.add_error(f"Debug failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_indexing(self) -> None:
        """/index — run the Phase 16-24 indexing pipeline."""
        panel = self._panel()
        status = self._status_line()
        panel.add_assistant_message(
            "[dim]Indexing workspace… this may take a while.[/]", trusted=True
        )
        status.update_stats(state="running")
        try:
            from tracera.config.settings import get_settings
            from tracera.main import _build_retrieval_pipeline
            settings = get_settings()
            pipeline = _build_retrieval_pipeline(settings, self.workspace_path)
            indexer = pipeline[0]
            stats = await asyncio.to_thread(indexer.run, full_rebuild=False)
            self.retrieval_pipeline = pipeline
            panel.add_assistant_message(
                f"[bold green]✓ Index complete[/]\n"
                f"  new: {stats.get('new', 0)} · modified: {stats.get('modified', 0)} · "
                f"deleted: {stats.get('deleted', 0)} · skipped: {stats.get('skipped', 0)}\n"
                f"  chunks: {stats.get('chunks_indexed', 0)}",
                trusted=True,
            )
        except Exception as e:
            panel.add_error(f"Indexing failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_tests(self) -> None:
        """/test — run the project test suite."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            import sys

            from tracera.tools.test_runner import TestRunner
            runner = TestRunner(self.workspace_path, python=sys.executable)
            report = await asyncio.to_thread(runner.run)
            lines = [report.summary, ""]
            for f in report.failures[:10]:
                location = f"{f.file_path}:{f.line_number}" if f.file_path else f.test_name
                lines.append(
                    f"  [red]✗[/] {escape(location)}: {escape(f.error_type)}: "
                    f"{escape(f.error_message[:120])}"
                )
            if not report.failures and not report.success and report.raw_output:
                lines.append(report.raw_output[:800])
            panel.add_assistant_message(
                "\n".join(lines) or "[dim]No tests detected.[/]", trusted=True
            )
        except Exception as e:
            panel.add_error(f"Test run failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_review(self) -> None:
        """/review — ask the agent to review current changes."""
        panel = self._panel()
        panel.add_user_message("Review the current uncommitted changes and report issues.")
        self._start_agent_task(
            "Review the current uncommitted changes in the workspace: "
            "check git diff for bugs, security issues, and style problems. "
            "Report findings with file locations."
        )

    async def _run_inspect(self) -> None:
        """/inspect — repository overview as a collapsible row."""
        panel = self._panel()
        from tracera.config.settings import get_settings
        from tracera.workspace.sandbox import WorkspaceSandbox
        root = Path(self.workspace_path)
        lines = [f"[bold cyan]Repository:[/] {root}\n"]
        try:
            sandbox = WorkspaceSandbox(root)
            entries = await sandbox.list_directory(".", max_depth=1)
            dirs: list[str] = []
            files: list[str] = []
            for e in entries:
                if len(e.relative.parts) == 1:
                    (dirs if e.is_dir else files).append(str(e.relative))
            lines.append("[bold]Structure[/]")
            if dirs:
                lines.append("  [dim]dirs:[/] " + ", ".join(sorted(dirs)[:15]))
            if files:
                lines.append("  [dim]files:[/] " + ", ".join(sorted(files)[:15]))
            lines.append("")
        except Exception as e:
            lines.append(f"[dim]Structure unavailable: {e}[/]")
        try:
            from tracera.git.operations import GitRepo
            repo = GitRepo(root)
            status = repo.status()
            lines.append(
                f"[bold]Git:[/] branch `{status.branch}` — "
                f"{'dirty' if status.is_dirty else 'clean'}"
            )
            for c in repo.log(max_count=2):
                lines.append(f"  [dim]• {c.hexsha[:7]} {c.summary[:60]}[/]")
        except Exception:
            lines.append("[dim]Git: not a repository[/]")
        settings = get_settings()
        manifest = settings.index_dir / "index_manifest.json"
        lines.append(
            "[bold]Code index:[/] "
            + ("[green]indexed[/]" if manifest.exists() else "[yellow]not indexed[/]")
        )
        panel.add_info_row(f"Repository: {escape(root.name or str(root))}", "\n".join(lines))

    def _run_deps(self, symbol: str) -> None:
        """/deps — symbol dependency chain as a collapsible row."""
        panel = self._panel()
        from tracera.config.settings import get_settings
        graph_path = get_settings().index_dir / "symbol_graph.json"
        if not graph_path.exists():
            panel.add_info_row(f"Dependencies: {escape(symbol)}", "[dim]No symbol graph — run /index.[/]")
            return
        try:
            from tracera.graph.symbol_graph import SymbolGraph
            graph = SymbolGraph.load(graph_path)
            neighbors = graph.neighbors_of(symbol)
            if not neighbors:
                panel.add_info_row(f"Dependencies: {escape(symbol)}", "[dim]No dependencies found.[/]")
                return
            lines = "\n".join(f"  [dim]•[/] {n}" for n in neighbors[:25])
            panel.add_info_row(
                f"Dependencies: {escape(symbol)} ({len(neighbors)})", lines
            )
        except Exception as e:
            panel.add_info_row(f"Dependencies: {escape(symbol)}", f"[dim]Failed: {escape(str(e))}[/]")

    # ── Multi-agent delegation (Phases 42–44) ─────────────────────────────

    def _show_agents(self, panel: AgentPanel) -> None:
        """/agents — sub-agent fleet overview (Phases 42–44)."""
        try:
            from tracera.agent.subagents import SubAgentRole
            roles = [r.value for r in SubAgentRole]
            lines = [
                "[bold]Sub-agent fleet[/] — Researcher · Coder · Tester · Reviewer · Debugger\n",
                "  [dim]roles:[/] " + ", ".join(roles),
                "  [dim]usage:[/] [bold]/delegate <task>[/] decomposes the task, runs each "
                "role, and aggregates results into a report.",
                "  [dim]status:[/] available (delegation runs on demand — nothing "
                "stays alive between runs).",
            ]
            panel.add_assistant_message("\n".join(lines), trusted=True)
        except Exception as e:
            panel.add_error(f"Sub-agent framework unavailable: {escape(str(e))}")

    async def _run_delegate(self, task: str) -> None:
        """/delegate — decompose a task across the sub-agent fleet."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            from tracera.agent.orchestrator import TaskOrchestrator
            from tracera.agent.subagents import build_sub_agent_fleet

            fleet = build_sub_agent_fleet(
                self.agent.provider,
                self.agent.registry,
                model=getattr(self.agent, "model", None),
                max_iterations=getattr(self.agent, "max_iterations", 12),
                max_tool_calls=getattr(self.agent, "max_tool_calls", 30),
            )
            decomposer = getattr(self.agent, "decomposer", None)
            orchestrator = TaskOrchestrator(fleet, decomposer=decomposer, parallel=False)

            lines: list[str] = []
            async for event in orchestrator.delegate(task):
                etype = event["type"]
                if etype == "plan_ready":
                    plan = event["plan"]
                    lines.append("[bold]Delegation plan[/]")
                    for step in getattr(plan, "steps", []):
                        role = getattr(step, "role", None)
                        lines.append(
                            f"  [dim]→[/] {getattr(role, 'value', role)}: "
                            f"{escape(str(getattr(step, 'task', ''))[:80])}"
                        )
                    lines.append("")
                elif etype == "agent_end":
                    step, result = event["step"], event["result"]
                    ok = getattr(result, "status", None)
                    ok = getattr(ok, "value", ok)
                    icon = "[green]✓[/]" if ok == "success" else "[red]✗[/]"
                    lines.append(
                        f"  {icon} {getattr(step, 'role', '?')} — "
                        f"{getattr(result, 'iterations', 0)} iter · "
                        f"{getattr(result, 'tool_calls', 0)} tools"
                    )
                elif etype == "report":
                    report = event["report"]
                    body = getattr(report, "summary", None) or str(report)
                    lines.append("")
                    lines.append("[bold]Aggregated report[/]")
                    lines.append(escape(str(body)[:2000]))
            panel.add_info_row(f"Delegate: {escape(task[:60])}", "\n".join(lines))
        except Exception as e:
            panel.add_error(f"Delegation failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    # ── Autonomous loop (Phases 35-38) ─────────────────────────────────────────

    def _build_test_runner(self):
        import sys
        from tracera.tools.test_runner import TestRunner
        return TestRunner(self.workspace_path, python=sys.executable)

    async def _run_fix_loop(self, task: str) -> None:
        """/fix — autonomous fix loop: Plan → Retrieve → Edit → Test → repeat."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger
            pipeline = self.retrieval_pipeline
            debugger = RetrievalDebugger(
                retriever=pipeline[1] if pipeline else None,
                context_engine=pipeline[4] if pipeline else None,
                compressor=pipeline[5] if pipeline and len(pipeline) > 5 else None,
            )
            loop = AutonomousFixLoop(
                self.workspace_path,
                self._build_test_runner(),
                debugger,
                decomposer=getattr(self.agent, "decomposer", None),
            )
            lines: list[str] = [f"[bold]Autonomous fix loop[/] — {escape(task)}"]
            result = await loop.run(task, self.agent.provider, self.agent)
            icon = "[green]✓ resolved[/]" if getattr(result, "final_success", False) else "[red]✗ unresolved[/]"
            lines.append(f"{icon} · {getattr(result, 'total_iterations', 0)} iterations · {len(getattr(result, 'attempts', []))} fix attempts")
            for att in list(getattr(result, "attempts", []))[-3:]:
                att_ok = getattr(att, "success", None)
                att_icon = "[green]✓[/]" if att_ok else "[red]✗[/]"
                lines.append(f"  {att_icon} iter {getattr(att, 'iteration', '?')}: {escape(str(getattr(att, 'patch_description', '') or '')[:100])}")
            panel.add_info_row(f"/fix: {escape(task[:60])}", "\n".join(lines))
        except Exception as e:
            panel.add_error(f"Fix loop failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_selfreview(self) -> None:
        """/selfreview — independent LLM review of the current uncommitted diff."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            from tracera.agent.autonomous import SelfReviewer
            pipeline = self.retrieval_pipeline
            reviewer = SelfReviewer(
                self.workspace_path,
                retriever=pipeline[1] if pipeline else None,
            )
            panel.add_meta("→ [bold]SelfReviewer[/] reviewing uncommitted changes…")
            review = await reviewer.review(self.agent.provider)
            panel.add_assistant_message(str(review)[:4000], trusted=True)
        except Exception as e:
            panel.add_error(f"Self-review failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    async def _run_regression_check(self) -> None:
        """/regression — baseline snapshot vs current tests (Phase 38)."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            from tracera.agent.autonomous import RegressionProtector
            protector = RegressionProtector(self.workspace_path, self._build_test_runner())
            panel.add_meta("→ [bold]RegressionProtector[/] running baseline tests…")
            pre = await asyncio.to_thread(protector.snapshot_before)
            panel.add_info_row("Baseline", pre.summary)
            report = await asyncio.to_thread(protector.verify_after)
            ok = report.get("overall_success", False)
            icon = "[green]✓ no regressions[/]" if ok else "[red]✗ regressions detected[/]"
            lines = [
                icon,
                f"Baseline passed: {report.get('pre_passed', '?')} · "
                f"Post: {report.get('post_passed', '?')}/{report.get('post_passed', 0) + report.get('post_failed', 0)} "
                f"({report.get('post_failed', 0)} failed)",
                report.get("summary", ""),
            ]
            changed = report.get("changed_files") or []
            if changed:
                lines.append("")
                lines.append("[bold]Changed files:[/]")
                for f in changed[:10]:
                    lines.append(f"  - {escape(str(f))}")
            panel.add_info_row("Regression check", "\n".join(lines))
        except Exception as e:
            panel.add_error(f"Regression check failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    # ── Code intelligence tool stubs (Phase 11-28) ─────────────────────────
    # These tools require a full retrieval pipeline + SymbolGraph to be useful.
    # The stubs dispatch to the registered tool via _run_tool_alias so they
    # work immediately when /index has been run; otherwise they show a clear
    # "run /index first" message (handled inside _run_tool_alias).

    async def _run_code_tool(self, tool_name: str, arg: str) -> None:
        """Generic dispatcher for every code-intelligence slash alias."""
        await self._run_tool_alias(tool_name, arg)

    # ── Memory tool stubs (Phase 10) ─────────────────────────────────────────
    # These require the enhanced memory layer (AgentMemory + triple store).
    # They dispatch through _run_tool_alias; the tools themselves show
    # fallbacks when the memory layer isn't initialized.

    async def _run_memory_tool(self, tool_name: str, arg: str) -> None:
        """Generic dispatcher for every memory slash alias."""
        await self._run_tool_alias(tool_name, arg)

    # ── Session / task context stubs (Phase 29-31) ──────────────────────────

    async def _run_session_tool(self, tool_name: str, arg: str) -> None:
        """Generic dispatcher for session/context slash aliases."""
        await self._run_tool_alias(tool_name, arg)

    # ── Dedicated retrieval commands (require index) ────────────────────────

    async def _run_symbol(self, name: str) -> None:
        """/symbol <name> — find a symbol by name (requires index)."""
        await self._run_tool_alias("find_symbol", name)

    async def _run_symbols(self, query: str) -> None:
        """/symbols <query> — search symbols (requires index)."""
        await self._run_tool_alias("search_symbols", query)

    async def _run_source(self, symbol: str) -> None:
        """/source <symbol> — exact source of a symbol (requires index)."""
        await self._run_tool_alias("get_symbol_source", symbol)

    async def _run_definition(self, name: str) -> None:
        """/definition <name> — jump to a definition (requires index)."""
        await self._run_tool_alias("find_definition", name)

    async def _run_outline(self, file: str) -> None:
        """/outline <file> — file outline (requires index)."""
        await self._run_tool_alias("get_file_outline", file)

    async def _run_repomap(self) -> None:
        """/repomap — repository overview (requires index)."""
        await self._run_tool_alias("get_repo_map", "")

    async def _run_assemble(self, task: str) -> None:
        """/assemble <task> — task context capsule (requires index)."""
        await self._run_tool_alias("assemble_code_context", task)

    async def _run_context(self, symbol: str) -> None:
        """/context <symbol> — expanded context (requires index)."""
        await self._run_tool_alias("get_context", symbol)

    async def _run_deps(self, symbol: str) -> None:
        """/deps <symbol> — dependency chain (requires index)."""
        await self._run_tool_alias("get_dependencies", symbol)

    async def _run_refs(self, symbol: str) -> None:
        """/refs <symbol> — find references (requires index)."""
        await self._run_tool_alias("find_references", symbol)

    async def _run_callers(self, symbol: str) -> None:
        """/callers <symbol> — call hierarchy (requires index)."""
        await self._run_tool_alias("get_call_hierarchy", symbol)

    async def _run_blast(self, symbol: str) -> None:
        """/blast <symbol> — blast radius (requires index)."""
        await self._run_tool_alias("get_blast_radius", symbol)

    async def _run_changed(self) -> None:
        """/changed — git diff → affected symbols (requires index + git)."""
        await self._run_tool_alias("get_changed_symbols", "")

    async def _run_freshness(self) -> None:
        """/freshness — index freshness vs filesystem (requires index)."""
        await self._run_tool_alias("get_index_freshness", "")

    async def _run_importers(self, file: str) -> None:
        """/importers <file> — what imports a file (requires index)."""
        await self._run_tool_alias("find_importers", file)

    async def _run_classhier(self, class_name: str) -> None:
        """/classhier <class> — inheritance chain (requires index)."""
        await self._run_tool_alias("get_class_hierarchy", class_name)

    async def _run_cycles(self) -> None:
        """/cycles — circular dependency cycles (requires index)."""
        await self._run_tool_alias("get_dependency_cycles", "")

    async def _run_coupling(self) -> None:
        """/coupling — module coupling + instability (requires index)."""
        await self._run_tool_alias("get_coupling_metrics", "")

    async def _run_endpoint(self, route: str) -> None:
        """/endpoint <route> — endpoint blast radius (requires index)."""
        await self._run_tool_alias("get_endpoint_impact", route)

    async def _run_deadcode(self) -> None:
        """/deadcode — unreachable symbols (requires index)."""
        await self._run_tool_alias("find_dead_code", "")

    async def _run_hotspots(self) -> None:
        """/hotspots — risky code by complexity × churn (requires index)."""
        await self._run_tool_alias("get_hotspots", "")

    async def _run_pagerank(self) -> None:
        """/pagerank — symbol importance (requires index)."""
        await self._run_tool_alias("calculate_pagerank", "")

    async def _run_refactor(self, symbol: str) -> None:
        """/refactor <symbol> — edit-ready refactor plan (requires index)."""
        await self._run_tool_alias("plan_refactoring", symbol)

    async def _run_editsafe(self, symbol: str) -> None:
        """/editsafe <symbol> — pre-modification safety check (requires index)."""
        await self._run_tool_alias("check_edit_safe", symbol)

    async def _run_deletesafe(self, symbol: str) -> None:
        """/deletesafe <symbol> — pre-deletion safety check (requires index)."""
        await self._run_tool_alias("check_delete_safe", symbol)

    async def _run_impls(self, symbol: str) -> None:
        """/impls <symbol> — find implementations (requires index)."""
        await self._run_tool_alias("find_implementations", symbol)

    async def _run_provenance(self, symbol: str) -> None:
        """/provenance <symbol> — git archaeology (requires index)."""
        await self._run_tool_alias("get_code_provenance", symbol)

    async def _run_risk(self, target: str) -> None:
        """/risk <target> — composite change-risk score (requires index)."""
        await self._run_tool_alias("assess_change_risk", target)

    async def _run_prrisk(self) -> None:
        """/prrisk — PR risk profile (requires git)."""
        await self._run_tool_alias("get_pr_risk_profile", "")

    async def _run_auditconfig(self) -> None:
        """/auditconfig — scan config for token waste (requires index)."""
        await self._run_tool_alias("audit_agent_config", "")

    async def _run_ast(self, pattern: str) -> None:
        """/ast <pattern> — cross-language AST pattern search (requires index)."""
        await self._run_tool_alias("structural_search", pattern)

    async def _run_sessionstats(self) -> None:
        """/sessionstats — session economics + token savings."""
        await self._run_tool_alias("get_session_stats", "")

    async def _run_plantask(self, task: str = "") -> None:
        """/plantask <task> — plan a code task (intent + anchors + route)."""
        panel = self._panel()
        if not task:
            panel.add_error("Usage: /plantask <task description>")
            return
        await self._run_tool_alias("plan_code_task", task)

    async def _run_planturn(self, query: str) -> None:
        """/planturn <query> — confidence-guided routing."""
        await self._run_tool_alias("plan_turn", query)

    async def _run_ranked(self, query: str) -> None:
        """/ranked <query> — token-budgeted ranked context."""
        await self._run_tool_alias("get_ranked_context", query)

    async def _run_taskcontext(self, task: str) -> None:
        """/taskcontext <task> — full task context assembly."""
        await self._run_tool_alias("assemble_task_context", task)

    # ── Memory commands (require memory layer) ──────────────────────────────

    async def _run_recall(self, query: str) -> None:
        """/recall <query> — recall relevant memories."""
        await self._run_tool_alias("recall_memory", query)

    async def _run_remember(self, text: str) -> None:
        """/remember <text> — store a memory."""
        await self._run_tool_alias("remember_memory", text)

    async def _run_forget(self, text: str) -> None:
        """/forget <text> — forget a memory by content fragment."""
        tool = self.agent.registry.get("forget_memory")
        if tool is None:
            self._panel().add_error("Tool 'forget_memory' not available.")
            return
        await self._run_tool_with_args("forget_memory", {"content_match": text})

    async def _run_sessions(self) -> None:
        """/sessions — list past sessions."""
        await self._run_tool_alias("list_sessions", "")

    async def _run_memstats(self) -> None:
        """/memstats — memory statistics."""
        await self._run_tool_alias("memory_stats", "")

    async def _run_consolidate(self) -> None:
        """/consolidate — merge near-duplicate memories."""
        await self._run_tool_alias("memory_consolidate", "")

    async def _run_memgraph2(self) -> None:
        """/memgraph2 — knowledge graph (tool form)."""
        await self._run_tool_alias("memory_graph", "")

    async def _run_memworker(self) -> None:
        """/memworker — background memory worker stats."""
        await self._run_tool_alias("memory_worker_status", "")

    async def _run_memsearch(self, query: str) -> None:
        """/memsearch <query> — search the memory store."""
        await self._run_tool_alias("search_memory", query)

    async def _run_triples(self) -> None:
        """/triples — semantic triples in the knowledge graph."""
        await self._run_tool_alias("get_memory_graph", "")

    # ── Git & repo operations ───────────────────────────────────────────────

    async def _run_git(self, subcommand: str) -> None:
        """/git <subcommand> — run a git operation."""
        await self._run_tool_alias("git", subcommand)

    async def _run_inspectrepo(self) -> None:
        """/inspectrepo — repository overview."""
        await self._run_tool_alias("inspect_repository", "")

    async def _run_tests_tool(self, framework: str) -> None:
        """/tests <framework> — run tests (pytest/unittest/npm/cargo)."""
        await self._run_tool_alias("run_tests", framework)

    async def _run_read(self, path: str) -> None:
        """/read <path> — read a file."""
        await self._run_tool_alias("read_file", path)

    async def _run_write(self, path: str) -> None:
        """/write <path> — write a file."""
        await self._run_tool_alias("write_file", path)

    async def _run_edit(self, path: str) -> None:
        """/edit <path> — edit a file."""
        await self._run_tool_alias("edit_file", path)

    async def _run_ls(self, path: str) -> None:
        """/ls <path> — list a directory."""
        await self._run_tool_alias("list_dir", path)

    async def _run_grep(self, pattern: str) -> None:
        """/grep <pattern> — regex search file contents."""
        await self._run_tool_alias("grep", pattern)

    async def _run_run(self, command: str) -> None:
        """/run <command> — run a shell command."""
        await self._run_tool_alias("run_command", command)

    # ── Tool profile helpers ─────────────────────────────────────────────────

    async def _run_tool_alias(self, tool_name: str, arg: str) -> None:
        """Dispatch a short slash alias (e.g. /blast foo) to its tool."""
        panel = self._panel()
        if not self.agent.registry.has(tool_name):
            panel.add_error(
                f"Tool '{escape(tool_name)}' not available — run /index first."
            )
            return
        tool = self.agent.registry.get(tool_name)
        await self._run_tool_with_args(tool_name, single_arg_kwargs(tool, arg))

    async def _run_tool_with_args(self, name: str, args: dict) -> None:
        await self._execute_tool_inline(name, args)

    async def _execute_tool_inline(self, name: str, args: dict) -> None:
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        panel.add_meta(f"→ [bold]{escape(name)}[/] {escape(format_args(args))}")
        try:
            result = await self.agent.registry.execute(name, "tui-cmd", args)
            body = result.output if result.success else (result.error or result.output)
            panel.add_info_row(
                f"{escape(name)} {'✓' if result.success else '✗'} "
                f"({result.duration_ms:.0f}ms)",
                body,
            )
        except Exception as e:
            panel.add_error(f"{escape(name)} failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    # ── Phase map (roadmap coverage + verified checklist) ─────────────────────────

    def _phases_progress_path(self) -> Path:
        """Where verified-phase progress is persisted (per data dir)."""
        from tracera.config.settings import get_settings
        return get_settings().tracera_data_dir / "phases_progress.json"

    def _load_verified_phases(self) -> set[int]:
        path = self._phases_progress_path()
        try:
            if path.exists():
                import json
                data = json.loads(path.read_text(encoding="utf-8"))
                return {int(n) for n in data.get("verified", [])}
        except Exception:
            pass
        return set()

    def _save_verified_phases(self, verified: set[int]) -> None:
        import json
        path = self._phases_progress_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"verified": sorted(verified)}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self._panel().add_error(f"Could not save phase progress: {escape(str(e))}")

    def _show_phases(self, text: str) -> None:
        """
        /phases — render the roadmap phase map with verification status.

        Usage:
          /phases            show the map + verified checklist
          /phases done <n>   mark an implemented phase as verified (persisted)
          /phases undo <n>   unmark a phase
          /phases reset      clear all verification progress
        """
        from tracera.phase_map import (
            PHASES,
            STATUS_EXCLUDED,
            STATUS_IMPLEMENTED,
            STATUS_ROADMAP,
            counts,
            get_phase,
        )

        panel = self._panel()
        parts = text.split()
        sub = parts[1] if len(parts) > 1 else ""
        verified = self._load_verified_phases()

        if sub == "reset":
            self._save_verified_phases(set())
            panel.add_assistant_message(
                "[bold green]✓[/] Phase verification progress cleared. "
                "Follow [cyan]tests/PROBLEM_STATEMENT.md[/] to re-verify.",
                trusted=True,
            )
            return

        if sub in ("done", "undo") and len(parts) >= 3:
            try:
                number = int(parts[2])
            except ValueError:
                panel.add_error(f"Usage: /phases {sub} <phase-number>")
                return
            phase = get_phase(number)
            if phase is None or phase.status != STATUS_IMPLEMENTED:
                status = phase.status if phase else "unknown"
                panel.add_error(
                    f"Phase {number} is not testable (status: {status}). "
                    "Only implemented phases (1–41, 42–61) can be verified."
                )
                return
            if sub == "done":
                verified.add(number)
            else:
                verified.discard(number)
            self._save_verified_phases(verified)
            action = "verified" if sub == "done" else "unmarked"
            panel.add_assistant_message(
                f"[bold green]✓[/] Phase {number} {action} — {escape(phase.title)}",
                trusted=True,
            )
            return

        if sub not in ("", "list", "show"):
            panel.add_error(
                "Usage: /phases | /phases done <n> | /phases undo <n> | /phases reset"
            )
            return

        c = counts()
        lines = [
            "[bold cyan]Phase Map[/] — "
            f"[green]{c[STATUS_IMPLEMENTED]} implemented[/] · "
            f"[dim red]{c[STATUS_EXCLUDED]} excluded[/] · "
            f"[dim]{c[STATUS_ROADMAP]} roadmap[/] — "
            f"[bold]{len(verified)} verified[/]\n"
        ]

        def _row(p) -> str:
            if p.status == STATUS_IMPLEMENTED:
                if p.number in verified:
                    return f"  [bold green]✓[/] [bold]{p.number:>2}[/] {p.title} [green]verified[/]"
                return f"  [cyan]·[/] [bold]{p.number:>2}[/] {p.title}"
            if p.status == STATUS_EXCLUDED:
                return f"  [dim red]✖[/] [bold]{p.number:>2}[/] {p.title}"
            return f"  [dim]➤[/] [bold]{p.number:>2}[/] {p.title}"

        #: (label, inclusive range) — mirrors the README roadmap structure.
        groups = [
            ("Core (1–10)", (1, 10)),
            ("Indexing (11–15)", (11, 15)),
            ("Retrieval (16–24)", (16, 24)),
            ("Graph & code-search tools (25–28)", (25, 28)),
            ("Context & repo-aware agent (29–31)", (29, 31)),
            ("Testing & autonomy (32–38)", (32, 38)),
            ("MCP server & client (39–41)", (39, 41)),
            ("Multi-agent delegation (42–44)", (42, 44)),
            ("Evaluation (45–50)", (45, 50)),
            ("Security (51–55)", (51, 55)),
            ("Terminal UI (56–59)", (56, 59)),
            ("Observability & config (60–61)", (60, 61)),
        ]

        for label, (lo, hi) in groups:
            rows = [p for p in PHASES if lo <= p.number <= hi]
            if not rows:
                continue
            lines.append(f"[bold]{label}[/]")
            lines.extend(_row(p) for p in rows)
            lines.append("")

        excluded_phases = [p for p in PHASES if p.status == STATUS_EXCLUDED]
        if excluded_phases:
            lines.append("[bold]Excluded — not implemented (62–66)[/]")
            lines.extend(_row(p) for p in excluded_phases)
            lines.append("")

        roadmap_phases = [p for p in PHASES if p.status == STATUS_ROADMAP]
        if roadmap_phases:
            lines.append("[bold]Roadmap — not implemented (67–72)[/]")
            lines.extend(_row(p) for p in roadmap_phases)
            lines.append("")

        lines.append(
            "[dim]Tick phases off as you verify them: /phases done <n> · "
            "the scenario lives in tests/PROBLEM_STATEMENT.md[/]"
        )
        panel.add_assistant_message("\n".join(lines), trusted=True)

    # ── Rich execution display (Phase 57) ─────────────────────────────────────

    @staticmethod
    def _phase_for_tool(name: str) -> str | None:
        if name in ("search_code", "find_symbol", "find_definition", "grep"):
            return "Searching"
        if name in ("get_context", "get_dependencies", "find_references"):
            return "Analyzing"
        if name in ("read_file", "list_dir"):
            return "Reading"
        if name in ("write_file", "edit_file", "delete_file"):
            return "Editing"
        if name == "run_command":
            return "Running command"
        if name == "git":
            return "Git"
        return None

    def _track_touched_file(self, name: str, args: dict) -> None:
        """Record files the agent read/wrote/edited so /files can show them."""
        action_map = {
            "read_file": "read",
            "list_dir": "search",
            "write_file": "write",
            "edit_file": "edit",
        }
        action = action_map.get(name)
        if not action:
            return
        path = args.get("path")
        if not path:
            return
        seen = {p for p, _ in self._recent_files}
        if str(path) in seen:
            return
        self._recent_files.insert(0, (str(path), action))
        del self._recent_files[8:]

    @staticmethod
    def _count_tests_passed(output: str | None) -> str | None:
        if not output:
            return None
        match = re.search(r"(\d+) passed", output)
        if not match:
            return None
        total = match.group(1)
        failed = re.search(r"(\d+) failed", output)
        suffix = f", {failed.group(1)} failed" if failed else ""
        return f"{total} passed{suffix}"

    def _show_status(self, panel: AgentPanel) -> None:
        provider = self.agent.provider
        stats = self._conversation.stats
        panel.add_assistant_message(
            f"[bold cyan]System Status[/]\n\n"
            f"Provider:    [bold]{escape(provider.name)}[/]\n"
            f"Model:       [bold]{escape(provider.default_model)}[/]\n"
            f"Workspace:   [bold]{escape(str(self.workspace_path))}[/]\n"
            f"Messages:    [bold cyan]{stats.total_messages}[/]\n"
            f"Tool calls:  [bold green]{stats.tool_calls}[/]\n"
            f"Tokens:      [bold cyan]{stats.total_tokens:,}[/]\n"
            f"Memory:      [bold orchid]{self.memory.count}[/] entries\n",
            trusted=True,
        )

    # ── Agent execution ───────────────────────────────────────────────────────

    def _start_agent_task(self, text: str) -> None:
        """Kick off an agent run, injecting attachments and tracking the worker."""
        panel = self._panel()
        if self._running_worker is not None and self._running_worker.is_running:
            panel.add_error("A task is already running — press esc to stop it first.")
            return
        task_text = self._build_task_with_attachments(text)
        if panel.attachments:
            panel.clear_attachments()
        self._running_worker = self._run_agent_task(task_text)

    def _build_task_with_attachments(self, text: str) -> str:
        """Append attached files to the task prompt the agent actually sees."""
        panel = self._panel()
        parts = [text]
        vision = bool(getattr(self.agent.provider, "supports_vision", False))
        model = self.agent.provider.default_model or "?"
        for path in panel.attachments:
            p = Path(path)
            if is_image(path):
                if vision:
                    parts.append(f"\n[Image attachment: {path}]")
                else:
                    parts.append(
                        f"\n[Image attachment: {path} — the active model ({model}) "
                        f"cannot view images, so only its path was attached]"
                    )
                continue
            try:
                size = p.stat().st_size
                if size > _ATTACH_TEXT_MAX_BYTES:
                    parts.append(
                        f"\n[Attached file: {path} — skipped: {size} bytes exceeds the "
                        f"{_ATTACH_TEXT_MAX_BYTES}-byte limit]"
                    )
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                parts.append(f"\n\n--- Attached file: {path} ---\n{content}")
            except Exception as e:
                parts.append(f"\n[Attached file: {path} — failed to read: {e}]")
        return "\n".join(parts)

    def _open_picker(self) -> None:
        root = Path(self.workspace_path)
        if not root.is_dir():
            root = Path(".")
        self.push_screen(FilePicker(root), callback=self._on_file_picked)

    def _on_file_picked(self, path: Any) -> None:
        if not path:
            return
        path = Path(path)
        try:
            root = Path(self.workspace_path).resolve()
            resolved = path.resolve()
            if not str(resolved).startswith(str(root)) or not resolved.is_file():
                self._panel().add_error(f"Not a workspace file: {escape(str(path))}")
                return
        except Exception as e:
            self._panel().add_error(f"Cannot attach {escape(str(path))}: {escape(str(e))}")
            return
        vision = bool(getattr(self.agent.provider, "supports_vision", False))
        warning = is_image(str(resolved)) and not vision
        self._panel().add_attachment(str(resolved), warning=warning)
        if warning:
            self._panel().add_meta(
                f"[dim]⚠ {escape(path.name)} is an image — the active model "
                f"({escape(self.agent.provider.default_model or '?')}) can't view images, "
                f"only its path will reach the agent.[/]"
            )

    def _show_loader(self) -> None:
        try:
            self.query_one("#loader-pill").display = True
            self.query_one("#agent-input-area").display = False
        except Exception:
            pass

    def _reset_loader(self) -> None:
        try:
            self.query_one("#loader-pill").display = False
            self.query_one("#agent-input-area").display = True
        except Exception:
            pass

    async def _read_file_snapshot(self, path: str) -> str | None:
        """Read a workspace file for diff snapshots; None if unreadable/too big."""
        try:
            root = Path(self.workspace_path).resolve()
            target = (root / path).resolve()
            if not str(target).startswith(str(root)):
                return None
            if not target.is_file() or target.stat().st_size > MAX_DIFF_BYTES:
                return None
            return await asyncio.to_thread(
                target.read_text, encoding="utf-8", errors="replace"
            )
        except Exception:
            return None

    @work(exclusive=False)
    async def _run_agent_task(self, task: str) -> None:
        """Run the agent loop, streaming every phase, tool call and diff inline."""
        panel = self._panel()
        status = self._status_line()
        provider = self.agent.provider
        pill = self.query_one("#loader-pill", LoaderPill)

        status.update_stats(
            state="thinking",
            session=self._conversation.id[:8],
            model=provider.default_model or "—",
        )
        pill.set_phase("planning")
        self._show_loader()

        total_iterations = 0
        total_tool_calls = 0
        turn_trace: list[tuple[str, str]] = []

        try:
            async for event in await self.agent.run(task, conversation=self._conversation):
                match event.type:
                    case AgentEventType.THINKING:
                        status.update_stats(
                            state="thinking",
                            iterations=event.iteration + 1,
                        )
                        turn_trace.append(("think", f"iteration {event.iteration + 1}"))

                    case AgentEventType.PHASE_UPDATE:
                        # v3: normalized phase from the agent loop → loader pill,
                        # status line, and stream marker.
                        phase = event.phase or "thinking"
                        pill.set_phase(phase)
                        status.update_stats(
                            state="running" if phase == "running" else "thinking"
                        )
                        if phase in ("planning", "thinking", "generating"):
                            panel.add_phase(
                                _PHASE_LABELS.get(phase, phase.title())
                            )

                    case AgentEventType.TOOL_START:
                        name = event.tool_name or "tool"
                        args = event.tool_args or {}
                        args_str = format_args(args)
                        status.update_stats(state="running")
                        # Phase 57: remember the phase for the turn trace.
                        phase = self._phase_for_tool(name)
                        if phase:
                            turn_trace.append(("think", phase))
                        row = panel.tool_start(name, args_str)
                        # v3: snapshot the file before a write/edit runs so the
                        # tool row can show an inline diff once it finishes.
                        if name in DIFFABLE_TOOLS and args.get("path"):
                            before = await self._read_file_snapshot(str(args["path"]))
                            if before is not None:
                                row.set_snapshot(str(args["path"]), before)
                        # Track recently touched files for /files.
                        self._track_touched_file(name, args)
                        turn_trace.append(("tool", f"{name} {args_str}"))

                    case AgentEventType.TOOL_END:
                        total_tool_calls += 1
                        name = event.tool_name or "tool"
                        duration_ms = event.metadata.get("duration_ms", 0.0)
                        row = panel.tool_end(
                            name,
                            event.tool_success,
                            duration_ms=duration_ms,
                            output=event.tool_output or "",
                        )
                        # v3: compute the diff for file-touching tools and render
                        # the 📝 path +N -M summary (expandable on click).
                        if (
                            event.tool_success
                            and row.snapshot is not None
                            and row.snapshot_path
                        ):
                            after = await self._read_file_snapshot(row.snapshot_path)
                            if after is not None:
                                lines, added, removed = compute_diff(
                                    row.snapshot, after, row.snapshot_path
                                )
                                if lines:
                                    row.set_diff(row.snapshot_path, lines, added, removed)
                        status.update_stats(tool_calls=total_tool_calls)
                        if event.tool_success:
                            turn_trace.append(("done", f"{name}  {duration_ms:.0f}ms"))
                            # Phase 57: "Running N tests… → ✓ N passed"
                            if name == "run_command" and event.tool_output:
                                passed = self._count_tests_passed(event.tool_output)
                                if passed is not None:
                                    turn_trace.append(("done", f"✓ {passed}"))
                        else:
                            turn_trace.append(("error", f"{name} failed"))

                    case AgentEventType.RESPONSE_DELTA:
                        if event.text:
                            panel.stream_delta(event.text)

                    case AgentEventType.PLAN_UPDATE:
                        plan_data = event.metadata.get("plan")
                        if plan_data:
                            from tracera.agent.planner import Plan
                            try:
                                plan = Plan.from_dict(plan_data)
                                done, total = plan.progress
                                body = plan.to_markdown()
                                if self._plan_row is None:
                                    self._plan_row = panel.add_info_row(
                                        f"Plan: {done}/{total} steps",
                                        body,
                                        prefix="▸",
                                    )
                                else:
                                    self._plan_row.set_title(f"Plan: {done}/{total} steps")
                                    self._plan_row.set_body(body)
                            except Exception:
                                pass
                        turn_trace.append(("think", "plan updated"))

                    case AgentEventType.MEMORY_UPDATE:
                        panel.add_info_row(
                            f"Memory: {event.text or 'saved'}",
                            prefix="→",
                        )
                        turn_trace.append(("think", event.text or "memory saved"))

                    case AgentEventType.RESPONSE_COMPLETE:
                        total_iterations = event.metadata.get("iterations", 0)
                        total_tokens = event.metadata.get("total_tokens", 0)
                        total_latency = event.metadata.get("total_latency_ms", 0.0)
                        # The model the backend ACTUALLY reported for this
                        # response — proof of which provider really answered.
                        actual_model = event.metadata.get("model") or provider.default_model or "—"

                        panel.stream_end(event.text or "")
                        # Only REAL events go in the disclosure — every entry in
                        # turn_trace corresponds to an actual loop event (real
                        # iteration, real tool call, real plan/memory update).
                        panel.add_thinking_disclosure(turn_trace)
                        panel.add_meta(
                            f"⏱ {total_iterations} iter · ⚙ {total_tool_calls} tools · "
                            f"{total_tokens:,} tok · {total_latency:.0f}ms · "
                            f"model {escape(str(actual_model))}"
                        )
                        status.update_stats(
                            state="done",
                            model=actual_model,
                            iterations=total_iterations,
                            tool_calls=total_tool_calls,
                            tokens=total_tokens,
                            elapsed_ms=total_latency,
                            # Phase 60: live session cost + retrieval hits.
                            cost_estimate=get_telemetry().snapshot()["cost"]["estimate_usd"],
                            retrieval_hits=get_telemetry().snapshot()["retrieval"]["calls"],
                        )

                    case AgentEventType.ERROR:
                        panel.add_error(event.text or "Unknown error")
                        turn_trace.append(("error", event.text or "error"))
                        status.update_stats(state="error")

                    case AgentEventType.DONE:
                        if status._state not in ("done", "error"):
                            status.update_stats(state="idle")

        except Exception as e:
            panel.add_error(f"Agent error: {escape(str(e))}")
            status.update_stats(state="error")
        finally:
            # Back to the input the instant the run ends (or is cancelled).
            panel.freeze_phase()
            pill.set_phase("done")
            self._reset_loader()
            self._running_worker = None

    async def _run_planning(self, task: str) -> None:
        """/plan — decompose a task and show the plan as a collapsible row."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="thinking")
        try:
            decomposer = TaskDecomposer(self.agent.provider)
            plan = await decomposer.decompose(task)
            body = plan.to_markdown()
            panel.add_info_row(
                f"Plan: {len(plan.items)} steps", body, prefix="▸", trusted=True
            )
            panel.add_assistant_message(
                f"[bold]Plan ready[/]: {len(plan.items)} steps — click the "
                f"[bold]▸ Plan[/] row above to expand it.",
                trusted=True,
            )
        except Exception as e:
            panel.add_error(f"Planning failed: {escape(str(e))}")
        finally:
            status.update_stats(state="idle")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_conversation(self) -> None:
        self._panel().clear()
        self._conversation = ConversationState()
        self._plan_row = None

    def action_focus_input(self) -> None:
        try:
            self.query_one("#agent-input", CommandInput).focus()
        except Exception:
            pass

    def action_show_memory(self) -> None:
        """Show enhanced memory layer contents with structured memories, triples, and stats."""
        panel = self._panel()

        # Try to access enhanced memory layer via agent
        enhanced_memory = getattr(self.agent, "_enhanced_memory", None)
        triple_store = getattr(self.agent, "_triple_store", None)
        session_manager = getattr(self.agent, "_session_manager", None)

        if enhanced_memory is None:
            # Fallback to legacy memory
            entries = self.memory.entries()
            if not entries:
                panel.add_info_row("Memory: no entries", "[dim]Nothing stored yet.[/]")
                return
            lines = []
            for e in entries[:20]:
                kind = getattr(e, "category", "general")
                content = getattr(e, "content", str(e))[:140]
                lines.append(f"  [dim]{kind}[/] {content}")
            panel.add_info_row(f"Memory: {len(entries)} entries", "\n".join(lines))
            return

        # Enhanced memory display
        lines = []

        # Stats
        stats = enhanced_memory.stats()
        lines.append(f"[bold cyan]Enhanced Memory[/] — {stats['total']} total")
        for mtype, count in stats.get("by_type", {}).items():
            lines.append(f"  [dim]{mtype}[/]: {count}")

        # Recent memories by type
        from tracera.memory.taxonomy import MemoryType
        for mtype in [MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.RULE,
                      MemoryType.DECISION, MemoryType.SKILL, MemoryType.RELATIONSHIP]:
            memories = enhanced_memory.get_by_type(mtype)
            if memories:
                lines.append(f"\n  [bold]{mtype.value.upper()}[/] ({len(memories)}):")
                for mem in memories[:5]:
                    conf = f" (conf: {mem.confidence:.0%})" if mem.confidence < 0.9 else ""
                    lines.append(f"  • {mem.content[:120]}{conf}")

        # Triple store info
        if triple_store:
            lines.append(f"\n  [bold]KNOWLEDGE GRAPH[/] — {triple_store.triple_count} triples, {triple_store.node_count} nodes")
            central = triple_store.get_central_concepts(5)
            if central:
                lines.append("  Central concepts: " + ", ".join(f"{c} ({d})" for c, d in central))

        # Session info
        if session_manager:
            active = session_manager.active_session
            if active:
                lines.append(f"\n  [bold]ACTIVE SESSION[/]: {active.id[:8]} ({active.task[:60]})")

        content = "\n".join(lines) if lines else "[dim]No enhanced memories yet[/]"
        panel.add_info_row("Enhanced Memory", content)

    def action_show_help(self) -> None:
        self._panel().add_assistant_message(_HELP_TEXT, trusted=True)

    def action_toggle_verbose(self) -> None:
        """ctrl+t — toggle showing tool-call arguments in the stream rows."""
        panel = self._panel()
        panel.verbose = not panel.verbose
        for row in panel.query("ToolRow"):
            row.verbose = panel.verbose
            row.refresh()
        self._status_line()._refresh()
        self._panel().add_assistant_message(
            "[dim]Verbose tool rows "
            + ("[green]on[/]" if panel.verbose else "[red]off[/]")
            + " — new rows show/hide their arguments.[/]",
            trusted=True,
        )

    # ── Provider / model switching ────────────────────────────────────────────

    def action_cycle_theme(self) -> None:
        """Command-palette entry for /theme."""
        self._cycle_theme()

    def action_switch_provider(self) -> None:
        """Open the provider/model selector (ctrl+p)."""
        if self._running_worker is not None and self._running_worker.is_running:
            self._panel().add_error("A task is running — press esc to stop it first.")
            return
        try:
            from tracera.config.settings import get_settings
            from tracera.providers import list_available_providers
            entries = list_available_providers(get_settings())
            active = getattr(self.agent.provider, "name", None)
            self.push_screen(
                ProviderSwitcher(entries, active_name=active),
                callback=self._on_provider_picked,
            )
        except Exception as e:
            self._panel().add_error(f"Cannot open provider switcher: {escape(str(e))}")

    def _on_provider_picked(self, result: Any) -> None:
        if not result:
            return
        name, model = result
        self._apply_provider(name, model)

    def _apply_provider(self, name: str, model: str) -> None:
        """
        Swap the backend that handles the NEXT request.

        The conversation and memory are untouched — only the provider object
        (and the model the loop passes to it) changes. The status line and
        header update immediately, and an explicit old → new confirmation row
        is streamed so the switch is impossible to miss.
        """
        old_name = getattr(self.agent.provider, "name", "?")
        old_model = getattr(self.agent.provider, "default_model", "") or self.agent.model or "?"
        try:
            from tracera.config.settings import get_settings
            from tracera.providers import create_provider
            new_provider = create_provider(
                name=name, model=model, settings=get_settings()
            )
        except Exception as e:
            self._panel().add_error(f"Provider {escape(name)} unavailable: {escape(str(e))}")
            return

        self.agent.provider = new_provider
        self.agent.model = model
        # Keep the task planner on the same backend.
        decomposer = getattr(self.agent, "decomposer", None)
        if decomposer is not None and hasattr(decomposer, "provider"):
            try:
                decomposer.provider = new_provider
            except Exception:
                pass

        # Read the NEW provider for the header/status line — not whatever was
        # already displayed.
        self._update_header_model(model)
        self._status_line().update_stats(state="idle", model=model)
        self._panel().add_meta(
            f"→ [bold cyan]Provider switched:[/] {escape(old_name)} ({escape(old_model)}) "
            f"→ [bold cyan]{escape(name)}[/] ({escape(model)})"
        )

    def _update_header_model(self, model: str) -> None:
        from rich.text import Text

        from tracera.tui.theme import get_theme as _gt
        th = _gt()
        right = Text()
        right.append("● ", style=f"bold #{th.success}")
        right.append(model, style=f"bold #{th.text}")
        right.append("  /help", style=f"dim #{th.faint}")
        try:
            self.query_one("#header-right", Static).update(right)
        except Exception:
            pass

    def action_cancel_task(self) -> None:
        """Stop the in-flight request: cancel the worker, fail open rows, reset."""
        if self._running_worker is not None:
            self._running_worker.cancel()
            self._running_worker = None
        self._panel().finalize_pending("cancelled")
        self._status_line().update_stats(state="idle")
        self._reset_loader()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Initialize feature status, attach external MCP tools, and welcome."""
        panel = self._panel()

        # Enable all available core features
        panel.set_feature_status("memory", True)      # Memory layer always available
        panel.set_feature_status("sandbox", True)     # Sandbox execution available

        # Activate retrieval/rag/index if pipeline exists
        if self.retrieval_pipeline is not None:
            panel.set_feature_status("retrieval", True)
            panel.set_feature_status("rag", True)
            panel.set_feature_status("index", True)
        else:
            panel.set_feature_status("retrieval", False)
            panel.set_feature_status("rag", False)
            panel.set_feature_status("index", False)

        # MCP (Model Context Protocol) - enable if the SDK is importable
        try:
            import mcp  # noqa: F401
            panel.set_feature_status("mcp", True)
            self.run_worker(self._attach_mcp_worker())
        except ImportError:
            panel.set_feature_status("mcp", False)

        self._update_header_git()

        status = self._status_line()
        status.update_stats(
            state="idle",
            session=self._conversation.id[:8],
            model=self.agent.provider.default_model,
        )
        try:
            self.query_one("#agent-input", CommandInput).focus()
        except Exception:
            pass
        self.run_worker(self._type_welcome())
        # Splash: decrypt/materialize boot animation above the main layout.
        # Any keypress skips it; TRACERA_NO_ANIMATION / NO_COLOR / headless
        # runs skip automatically; it pops back to this screen when done.
        self.push_screen(SplashScreen())

    async def _attach_mcp_worker(self) -> None:
        """Phase 41 — merge external MCP tools into the runtime registry."""
        try:
            from tracera.config.settings import get_settings
            from tracera.main import _attach_external_mcp
            await _attach_external_mcp(
                self.agent, get_settings(), self.workspace_path
            )
        except Exception:
            pass

    async def _type_welcome(self) -> None:
        try:
            panel = self._panel()
            if self._banner:
                panel.add_banner(self._banner)
                panel.add_meta(f"workspace {escape(str(self.workspace_path))}")
            await panel.type_message(
                "Ready. What would you like to work on?\n"
                "Tool calls, file edits, and command output stream inline as I work. "
                "Type /help for all commands.\n"
            )
        except Exception:
            pass
