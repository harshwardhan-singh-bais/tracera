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

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import DirectoryTree, Input, ListItem, ListView, Static
from textual.command import Hit, Provider

from tracera.tui.diffutil import DIFFABLE_TOOLS, MAX_DIFF_BYTES, compute_diff, is_image
from tracera.tui.widgets.agent_panel import (
    AgentPanel,
    CollapsibleRow,
    InlineStatus,
    LoaderPill,
    _PHASE_LABELS,
    format_args,
)
from tracera.agent.react_loop import AgentEvent, AgentEventType, ReActAgent
from tracera.agent.memory import AgentMemory
from tracera.agent.planner import TaskDecomposer
from tracera.conversation.state import ConversationState

#: Attached text files larger than this (bytes) are not injected into context.
_ATTACH_TEXT_MAX_BYTES = 200_000


_HELP_TEXT = """\
[bold cyan]TRACERA — Commands[/]

[bold]/help[/]          Show this help
[bold]/clear[/]         Clear conversation
[bold]/status[/]        Show system status
[bold]/memory[/]        Show memory contents
[bold]/model[/] [name]   Switch model
[bold]/plan[/] [task]    Decompose a task into steps
[bold]/code[/] [task]    Run a coding task (same as plain input)
[bold]/search[/] <q>     Search the code index (hybrid)
[bold]/debug[/] <q>      Compare retrieval strategies (BM25/Dense/Hybrid/Reranker)
[bold]/index[/]          Index the workspace (Phase 16-24 pipeline)
[bold]/test[/]           Run the project's test suite
[bold]/review[/]         Ask the agent to review current changes
[bold]/tools[/]          List available tools
[bold]/mcp[/]            Show MCP status & config
[bold]/cost[/]           Show session token/cost estimate
[bold]/inspect[/]        Repository inspection (files, symbols, git)
[bold]/deps[/] <symbol>   Show a symbol's dependency chain
[bold]/phases[/]         Show the phase map + verified checklist
[bold]/phases done <n>[/]  Mark a phase as verified (persisted)
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
            row.append(" ✓ " if is_active else "   ", style="bold #4ac26b")
            row.append(
                name,
                style="bold #dcdcf5" if available else "dim #9a9aa3",
            )
            row.append(f"   {model}", style="dim #6cb6ff" if available else "dim #55555e")
            if not available:
                env = str(info.get("key_env") or "API_KEY").upper()
                row.append(f"   [!] missing {env}", style="bold #d4a72c")
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

    TITLE = "TRACERA — CodePilotX"
    CSS_PATH = "styles/tracera.tcss"

    COMMANDS = {TraceraCommands}

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
        # The CLI already printed this banner to scrollback; the app reproduces
        # it at the top of its own first frame so the (unavoidable on Windows)
        # alt-screen switch looks continuous rather than like a new screen.
        self._banner = banner
        self._conversation = ConversationState()
        self._running_worker: Any = None
        self._hovered_scrollable: ScrollableContainer | None = None
        self._plan_row: CollapsibleRow | None = None

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # The ASCII banner was already printed to scrollback by the CLI before
        # the app started — the TUI renders directly below it, no clear.
        yield self._build_header()
        # The single main panel — conversation stream, status line, input.
        yield AgentPanel(id="agent-panel-widget")

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
    def on_submit_task(self, event: AgentPanel.SubmitTask) -> None:
        text = event.text.strip()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
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

    def _handle_command(self, text: str) -> None:
        panel = self._panel()
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
        elif cmd in ("/code", "/ask"):
            task = text[len(cmd):].strip()
            if task:
                panel.add_user_message(task)
                self._start_agent_task(task)
            else:
                panel.add_error(f"Usage: {cmd} <task description>")
        elif cmd == "/search":
            query = text[len(cmd):].strip()
            if query:
                self._run_search(query)
            else:
                panel.add_error("Usage: /search <query>")
        elif cmd == "/debug":
            query = text[len(cmd):].strip()
            if query:
                self._run_debug(query)
            else:
                panel.add_error("Usage: /debug <query>")
        elif cmd == "/index":
            self._run_indexing()
        elif cmd == "/test":
            self._run_tests()
        elif cmd == "/review":
            self._run_review()
        elif cmd == "/tools":
            self._show_tools(panel)
        elif cmd == "/mcp":
            self._show_mcp(panel)
        elif cmd == "/cost":
            self._show_cost(panel)
        elif cmd == "/inspect":
            self._run_inspect()
        elif cmd == "/deps":
            symbol = text[len(cmd):].strip()
            if symbol:
                self._run_deps(symbol)
            else:
                panel.add_error("Usage: /deps <symbol>")
        elif cmd == "/phases":
            self._show_phases(text)
        else:
            panel.add_error(f"Unknown command: {cmd}. Type /help for available commands.")

    # ── REPL command implementations ─────────────────────────────────────────

    def _show_tools(self, panel: AgentPanel) -> None:
        names = [t.name for t in self.agent.registry.tools]
        if not names:
            panel.add_error("No tools registered.")
            return
        lines = "\n".join(f"  [dim]▪[/] {n}" for n in sorted(names))
        panel.add_assistant_message(f"[bold]Available tools ({len(names)})[/]\n{lines}")

    def _show_mcp(self, panel: AgentPanel) -> None:
        config_path = Path(self.workspace_path) / ".tracera" / "mcp_servers.json"
        lines = ["[bold]MCP[/]"]
        if config_path.exists():
            lines.append(f"  config: [cyan]{config_path}[/]")
        else:
            lines.append(
                "  [dim]No mcp_servers.json yet — see MCP_CONNECTIONS.md for "
                "server configs and required credentials.[/]"
            )
        lines.append(
            "  [dim]Use `tracera mcp serve` (server) or "
            "`tracera mcp connect <file>` (client).[/]"
        )
        panel.add_assistant_message("\n".join(lines))

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
            f"[dim](estimate @ $0.30/$1.20 per 1M tokens)[/]"
        )

    @work(exclusive=False)
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
                panel.add_info_row(f"Search: {query}", "[dim]No results.[/]")
                return
            lines = []
            for i, hit in enumerate(hits[:8], 1):
                path = hit.get("file_path") or hit.get("id") or "?"
                symbol = hit.get("symbol") or ""
                score = hit.get("_relevance_score") or hit.get("_rrf_score") or ""
                line = f"  {i}. [bold]{path}[/]"
                if symbol:
                    line += f" [dim]({symbol})[/]"
                if score:
                    line += f" [dim]· {float(score):.3f}[/]"
                lines.append(line)
                content = (hit.get("content") or "").strip().splitlines()
                if content:
                    lines.append("     [dim]" + content[0][:80] + "[/]")
            panel.add_info_row(f"Search: {query} ({len(hits)} hits)", "\n".join(lines))
        except Exception as e:
            panel.add_error(f"Search failed: {e}")
        finally:
            status.update_stats(state="idle")

    @work(exclusive=False)
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
            panel.add_error(f"Debug failed: {e}")
        finally:
            status.update_stats(state="idle")

    @work(exclusive=False)
    async def _run_indexing(self) -> None:
        """/index — run the Phase 16-24 indexing pipeline."""
        panel = self._panel()
        status = self._status_line()
        panel.add_assistant_message("[dim]Indexing workspace… this may take a while.[/]")
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
                f"  chunks: {stats.get('chunks_indexed', 0)}"
            )
        except Exception as e:
            panel.add_error(f"Indexing failed: {e}")
        finally:
            status.update_stats(state="idle")

    @work(exclusive=False)
    async def _run_tests(self) -> None:
        """/test — run the project test suite."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="running")
        try:
            from tracera.tools.test_runner import TestRunner
            import sys
            runner = TestRunner(self.workspace_path, python=sys.executable)
            report = await asyncio.to_thread(runner.run)
            lines = [report.summary, ""]
            for f in report.failures[:10]:
                location = f"{f.file_path}:{f.line_number}" if f.file_path else f.test_name
                lines.append(f"  [red]✗[/] {location}: {f.error_type}: {f.error_message[:120]}")
            if not report.failures and not report.success and report.raw_output:
                lines.append(report.raw_output[:800])
            panel.add_assistant_message("\n".join(lines) or "[dim]No tests detected.[/]")
        except Exception as e:
            panel.add_error(f"Test run failed: {e}")
        finally:
            status.update_stats(state="idle")

    @work(exclusive=False)
    async def _run_review(self) -> None:
        """/review — ask the agent to review current changes."""
        panel = self._panel()
        panel.add_user_message("Review the current uncommitted changes and report issues.")
        self._start_agent_task(
            "Review the current uncommitted changes in the workspace: "
            "check git diff for bugs, security issues, and style problems. "
            "Report findings with file locations."
        )

    @work(exclusive=False)
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
        panel.add_info_row(f"Repository: {root.name or root}", "\n".join(lines))

    def _run_deps(self, symbol: str) -> None:
        """/deps — symbol dependency chain as a collapsible row."""
        panel = self._panel()
        from tracera.config.settings import get_settings
        graph_path = get_settings().index_dir / "symbol_graph.json"
        if not graph_path.exists():
            panel.add_info_row(f"Dependencies: {symbol}", "[dim]No symbol graph — run /index.[/]")
            return
        try:
            from tracera.graph.symbol_graph import SymbolGraph
            graph = SymbolGraph.load(graph_path)
            neighbors = graph.neighbors_of(symbol)
            if not neighbors:
                panel.add_info_row(f"Dependencies: {symbol}", "[dim]No dependencies found.[/]")
                return
            lines = "\n".join(f"  [dim]•[/] {n}" for n in neighbors[:25])
            panel.add_info_row(
                f"Dependencies: {symbol} ({len(neighbors)})", lines
            )
        except Exception as e:
            panel.add_info_row(f"Dependencies: {symbol}", f"[dim]Failed: {e}[/]")

    # ── Phase map (roadmap coverage + verified checklist) ─────────────────────

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
            self._panel().add_error(f"Could not save phase progress: {e}")

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
                "Follow [cyan]tests/PROBLEM_STATEMENT.md[/] to re-verify."
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
                    "Only implemented phases (1–40, 42–59) can be verified."
                )
                return
            if sub == "done":
                verified.add(number)
            else:
                verified.discard(number)
            self._save_verified_phases(verified)
            action = "verified" if sub == "done" else "unmarked"
            panel.add_assistant_message(
                f"[bold green]✓[/] Phase {number} {action} — {phase.title}"
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
            ("MCP server & client (39–40)", (39, 40)),
            ("Multi-agent delegation (42–44)", (42, 44)),
            ("Evaluation (45–50)", (45, 50)),
            ("Security (51–55)", (51, 55)),
            ("Terminal UI (56–59)", (56, 59)),
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
            lines.append("[bold]Excluded — not implemented (41, 60–66)[/]")
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
        panel.add_assistant_message("\n".join(lines))

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
            f"Provider:    [bold]{provider.name}[/]\n"
            f"Model:       [bold]{provider.default_model}[/]\n"
            f"Workspace:   [bold]{self.workspace_path}[/]\n"
            f"Messages:    [bold cyan]{stats.total_messages}[/]\n"
            f"Tool calls:  [bold green]{stats.tool_calls}[/]\n"
            f"Tokens:      [bold cyan]{stats.total_tokens:,}[/]\n"
            f"Memory:      [bold orchid]{self.memory.count}[/] entries\n"
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
                self._panel().add_error(f"Not a workspace file: {path}")
                return
        except Exception as e:
            self._panel().add_error(f"Cannot attach {path}: {e}")
            return
        vision = bool(getattr(self.agent.provider, "supports_vision", False))
        warning = is_image(str(resolved)) and not vision
        self._panel().add_attachment(str(resolved), warning=warning)
        if warning:
            self._panel().add_meta(
                f"[dim]⚠ {path.name} is an image — the active model "
                f"({self.agent.provider.default_model or '?'}) can't view images, "
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
                            f"model {actual_model}"
                        )
                        status.update_stats(
                            state="done",
                            model=actual_model,
                            iterations=total_iterations,
                            tool_calls=total_tool_calls,
                            tokens=total_tokens,
                            elapsed_ms=total_latency,
                        )

                    case AgentEventType.ERROR:
                        panel.add_error(event.text or "Unknown error")
                        turn_trace.append(("error", event.text or "error"))
                        status.update_stats(state="error")

                    case AgentEventType.DONE:
                        if status._state not in ("done", "error"):
                            status.update_stats(state="idle")

        except Exception as e:
            panel.add_error(f"Agent error: {e}")
            status.update_stats(state="error")
        finally:
            # Back to the input the instant the run ends (or is cancelled).
            panel.freeze_phase()
            pill.set_phase("done")
            self._reset_loader()
            self._running_worker = None

    @work(exclusive=False)
    async def _run_planning(self, task: str) -> None:
        """/plan — decompose a task and show the plan as a collapsible row."""
        panel = self._panel()
        status = self._status_line()
        status.update_stats(state="thinking")
        try:
            decomposer = TaskDecomposer(self.agent.provider)
            plan = await decomposer.decompose(task)
            body = plan.to_markdown()
            panel.add_info_row(f"Plan: {len(plan.items)} steps", body, prefix="▸")
            panel.add_assistant_message(
                f"[bold]Plan ready[/]: {len(plan.items)} steps — click the "
                f"[bold]▸ Plan[/] row above to expand it."
            )
        except Exception as e:
            panel.add_error(f"Planning failed: {e}")
        finally:
            status.update_stats(state="idle")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_conversation(self) -> None:
        self._panel().clear()
        self._conversation = ConversationState()
        self._plan_row = None

    def action_focus_input(self) -> None:
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass

    def action_show_memory(self) -> None:
        panel = self._panel()
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

    def action_show_help(self) -> None:
        self._panel().add_assistant_message(_HELP_TEXT)

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
            + " — new rows show/hide their arguments.[/]"
        )

    # ── Provider / model switching ────────────────────────────────────────────

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
            self._panel().add_error(f"Cannot open provider switcher: {e}")

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
            self._panel().add_error(f"Provider {name} unavailable: {e}")
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
            f"→ [bold cyan]Provider switched:[/] {old_name} ({old_model}) "
            f"→ [bold cyan]{name}[/] ({model})"
        )

    def _update_header_model(self, model: str) -> None:
        from rich.text import Text
        right = Text()
        right.append("● Active", style="bold #4ac26b")
        right.append(f"  {model}", style="bold")
        right.append("   /help", style="dim #6f6f78")
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
        status = self._status_line()
        status.update_stats(
            state="idle",
            session=self._conversation.id[:8],
            model=self.agent.provider.default_model,
        )
        # Show the whole UI immediately — the banner already sits in scrollback
        # above us (printed once by the CLI). No splash, no clears.
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass
        self.run_worker(self._type_welcome())

    async def _type_welcome(self) -> None:
        try:
            panel = self._panel()
            # First frame: reproduce the CLI banner at the top of the app's own
            # screen, then the welcome message below it.
            if self._banner:
                panel.add_banner(self._banner)
                panel.add_meta(f"booted from {self.workspace_path}")
            await panel.type_message(
                "I'm ready to help with your coding tasks.\n"
                "Everything I do streams inline — tool calls, file reads, "
                "command runs. Type /help for commands.\n"
            )
        except Exception:
            pass
