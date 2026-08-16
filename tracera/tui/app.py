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
from textual.widgets import Input, Static
from textual.command import Hit, Provider

from tracera.tui.widgets.agent_panel import (
    AgentPanel,
    CollapsibleRow,
    InlineStatus,
    format_args,
)
from tracera.agent.react_loop import AgentEvent, AgentEventType, ReActAgent
from tracera.agent.memory import AgentMemory
from tracera.agent.planner import TaskDecomposer
from tracera.conversation.state import ConversationState


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
[bold]/reset[/]          Reset conversation state

[bold cyan]Keys[/]

[bold]ctrl+q[/]        Quit
[bold]ctrl+l[/]        Clear conversation
[bold]ctrl+t[/]        Toggle verbose tool rows (show/hide args)
[bold]ctrl+p[/]        Command palette
[bold]ctrl+m[/]        Show memory
[bold]f1[/]            Help
[bold]esc[/]           Cancel running task
[bold]pgup/pgdn[/]     Scroll the stream

[bold cyan]Stream[/]

Everything the agent does streams inline: tool calls, file reads, command
runs — ✓ success, ✗ failure (with the error line beneath), an animated
spinner while in flight. Click any [bold]→ row[/] (memory, search results,
plans, repo info) to expand its content inline.
"""


class TraceraCommands(Provider):
    """Command-palette entries (ctrl+p)."""

    _COMMANDS = [
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
        Binding("ctrl+p", "command_palette", "Commands", show=True),
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.memory = memory
        self.workspace_path = workspace_path
        self.retrieval_pipeline = retrieval_pipeline
        self._conversation = ConversationState()
        self._running_task: asyncio.Task | None = None
        self._hovered_scrollable: ScrollableContainer | None = None
        self._splash_done = False
        self._plan_row: CollapsibleRow | None = None

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # Typewriter ASCII banner at the top — the UI assembles below it.
        with Vertical(id="splash-banner"):
            yield Static("", id="splash-logo")
            yield Static("", id="splash-status")
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
        self._run_agent_task(text)

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
                self._run_agent_task(task)
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
        self._run_agent_task(
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

    @work(exclusive=False)
    async def _run_agent_task(self, task: str) -> None:
        """Run the agent loop, streaming every action inline."""
        panel = self._panel()
        status = self._status_line()
        provider = self.agent.provider

        status.update_stats(
            state="thinking",
            session=self._conversation.id[:8],
            model=provider.default_model or "—",
        )

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

                    case AgentEventType.TOOL_START:
                        name = event.tool_name or "tool"
                        args_str = format_args(event.tool_args or {})
                        status.update_stats(state="running")
                        # Phase 57: remember the phase for the turn trace.
                        phase = self._phase_for_tool(name)
                        if phase:
                            turn_trace.append(("think", phase))
                        panel.tool_start(name, args_str)
                        turn_trace.append(("tool", f"{name} {args_str}"))

                    case AgentEventType.TOOL_END:
                        total_tool_calls += 1
                        name = event.tool_name or "tool"
                        duration_ms = event.metadata.get("duration_ms", 0.0)
                        panel.tool_end(
                            name,
                            event.tool_success,
                            duration_ms=duration_ms,
                            output=event.tool_output or "",
                        )
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

                        panel.stream_end(event.text or "")
                        steps = self._build_turn_steps(turn_trace)
                        panel.add_thinking_disclosure(steps + turn_trace)
                        panel.add_meta(
                            f"⏱ {total_iterations} iter · ⚙ {total_tool_calls} tools · "
                            f"{total_tokens:,} tok · {total_latency:.0f}ms"
                        )
                        status.update_stats(
                            state="done",
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

    @staticmethod
    def _build_turn_steps(trace: list[tuple[str, str]]) -> list[tuple[str, str]]:
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

    def action_cancel_task(self) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
        self._status_line().update_stats(state="idle")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    _SPLASH_ART = [
        " ████████╗██████╗  █████╗  ██████╗███████╗██████╗  █████╗ ",
        " ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗",
        "    ██║   ██████╔╝███████║██║     █████╗  ██████╔╝███████║",
        "    ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗██╔══██║",
        "    ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║██║  ██║",
        "    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝",
    ]

    def on_mount(self) -> None:
        status = self._status_line()
        status.update_stats(
            state="idle",
            session=self._conversation.id[:8],
            model=self.agent.provider.default_model,
        )
        # Hide the interface initially — it builds in below the banner.
        for selector in ("#app-header", "#agent-panel-widget"):
            try:
                self.query_one(selector).display = False
            except Exception:
                pass
        self.run_worker(self._animate_splash(), exclusive=True)

    def _show(self, selector: str) -> None:
        try:
            self.query_one(selector).display = True
        except Exception:
            pass

    async def _animate_splash(self) -> None:
        try:
            logo = self.query_one("#splash-logo", Static)
            status = self.query_one("#splash-status", Static)

            completed: list[str] = []
            for line in self._SPLASH_ART:
                buf = ""
                for ch in line:
                    buf += ch
                    logo.update("\n".join(completed + [buf + "▌"]))
                    await asyncio.sleep(0.004)
                    if self._splash_done:
                        return
                completed.append(line)
                logo.update("\n".join(completed))
                await asyncio.sleep(0.06)
                if self._splash_done:
                    return

            sub = "your terminal coding agent"
            sbuf = ""
            for ch in sub:
                sbuf += ch
                status.update(f"  {sbuf}▌")
                await asyncio.sleep(0.018)
                if self._splash_done:
                    return
            status.update(f"  {sub}")
            await asyncio.sleep(0.1)
            if self._splash_done:
                return

            boot_steps = [
                ("checking providers", "#app-header"),
                ("loading code index", "#agent-panel-widget"),
                ("ready", None),
            ]
            for step, selector in boot_steps:
                status.update(f"  ⠋ {step}")
                if selector:
                    self._show(selector)
                await asyncio.sleep(0.2)
                if self._splash_done:
                    return
            status.update("  ✓ ready")
            await asyncio.sleep(0.15)
        except Exception:
            pass
        self._splash_done = True
        await self._finish_splash()

    async def _finish_splash(self) -> None:
        try:
            self.query_one("#splash-banner").remove()
        except Exception:
            pass
        for selector in ("#app-header", "#agent-panel-widget"):
            self._show(selector)
        try:
            self.query_one("#agent-input", Input).focus()
        except Exception:
            pass
        await self._type_welcome()

    async def _type_welcome(self) -> None:
        try:
            await self._panel().type_message(
                "I'm ready to help with your coding tasks.\n"
                "Everything I do streams inline — tool calls, file reads, "
                "command runs. Type /help for commands.\n"
            )
        except Exception:
            pass

    def on_key(self, event) -> None:
        if not self._splash_done:
            try:
                if self.query_one("#splash-banner"):
                    self._splash_done = True
                    self.run_worker(self._finish_splash())
            except Exception:
                pass
