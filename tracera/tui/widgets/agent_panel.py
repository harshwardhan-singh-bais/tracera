"""
TRACERA Conversation Stream — the single main panel.

Claude Code style: one continuous, auto-scrolling stream of everything the
agent does, inline in the main panel:

    ┌─ YOU ────────────────────────────────┐
    │  add jwt validation to the middleware│
    └──────────────────────────────────────┘
    ✓ search_code (query='jwt auth')   8ms
    ✓ read_file  (path='auth/middleware.py')  2ms
    ✗ run_command  (command='pytest')  0ms
      └ pytest: error: unrecognized arguments
    → Memory: recalled architecture notes          ← collapsible
    ┌─ TRACERA ──────────────────────────┐
    │  Done. All tests pass.            │
    └────────────────────────────────────┘
    ● ACTIVE  session 8f2c · model gemini · 5 tools · 3 iter · 1.2k tok · 0:42
    ❯ [ input pill ............. ]
    Enter send · /help commands · ctrl+t verbose rows

Every row auto-scrolls into view while the user is at the bottom; scrolling
up during a run pauses the auto-scroll until the user returns to the bottom.
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Input
from textual.containers import Horizontal, Vertical, ScrollableContainer
from rich.text import Text

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def format_args(args: dict) -> str:
    """Compact, single-line rendering of tool arguments."""
    parts = []
    for k, v in list(args.items())[:3]:
        sv = str(v)
        if len(sv) > 34:
            sv = sv[:31] + "…"
        parts.append(f"{k}={sv!r}")
    return f"({', '.join(parts)})" if parts else ""


# ── Message bubbles ──────────────────────────────────────────────────────────

class MessageWidget(Static):
    """A single message bubble in the stream.

    User and assistant turns get a rounded border with the role as the
    border title (┌─ YOU ─ / ┌─ TRACERA ─); tool/meta lines stay borderless.
    Content is rendered as Rich markup (Static) and colored per role by CSS.
    """

    _BORDER_TITLES = {"user": " YOU ", "assistant": " TRACERA "}
    _PREFIXES = {"tool": "⚙  ", "error": "✗  "}

    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self.msg_content = content
        display = f"{self._PREFIXES.get(role, '')}{content}"
        super().__init__(display, markup=True, classes=f"msg-{role}", **kwargs)
        title = self._BORDER_TITLES.get(role)
        if title:
            self.border_title = title

    def set_content(self, content: str) -> None:
        """Replace the message content (used by streaming)."""
        self.msg_content = content
        self.update(f"{self._PREFIXES.get(self.role, '')}{content}")


_GLYPH = {
    "think": "◇",
    "tool": "◆",
    "done": "✓",
    "error": "✗",
    "step-done": "✓",
    "step-active": "⠋",
    "step-pending": "○",
}


class ThinkingDisclosure(Widget):
    """Collapsible per-turn 'Thinking…' block attached to an assistant turn."""

    def __init__(self, entries: list[tuple[str, str]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries = entries
        self.expanded = False

    def compose(self) -> ComposeResult:
        yield Static(f"▸ Thinking… ({len(self._entries)})", id="reasoning-toggle")
        with Vertical(id="reasoning-body"):
            for kind, text in self._entries:
                glyph = _GLYPH.get(kind, "·")
                yield Static(
                    f"{glyph} {text}",
                    classes=f"reasoning-line reasoning-{kind}",
                )

    def on_mount(self) -> None:
        self.query_one("#reasoning-body", Vertical).display = False

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "reasoning-toggle":
            self.toggle()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        toggle = self.query_one("#reasoning-toggle", Static)
        body = self.query_one("#reasoning-body", Vertical)
        if self.expanded:
            toggle.update(f"▾ Thinking… ({len(self._entries)})")
        else:
            toggle.update(f"▸ Thinking… ({len(self._entries)})")
        body.display = self.expanded


# ── Inline tool rows ─────────────────────────────────────────────────────────

class ToolRow(Static):
    """
    One compact inline row per tool call, in the stream:

        ⠋ run_command (command='pytest')        ← in-flight (animated)
        ✓ read_file  (path='a.py')   12ms
        ✗ edit_file  (path='b.py')   3ms
          └ ERROR: merge conflict               ← auto-expanded on failure
    """

    def __init__(
        self,
        name: str,
        args_str: str = "",
        *,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = name
        self.args_str = args_str
        self.verbose = verbose
        self.success = True
        self.duration_ms: float | None = None
        self.output = ""
        self._frame = 0
        # NOTE: not named ``_running`` — Textual's MessagePump owns that.
        self._spinning = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        if self._spinning:
            self.set_interval(0.1, self._tick)

    def start_spinner(self) -> None:
        self._spinning = True
        if self.is_mounted:
            self.set_interval(0.1, self._tick)

    def finish(self, success: bool, duration_ms: float, output: str = "") -> None:
        self._spinning = False
        self.success = success
        self.duration_ms = duration_ms
        self.output = output
        self.refresh()

    def _tick(self) -> None:
        if self._spinning:
            self._frame = (self._frame + 1) % len(_SPINNER)
            self.refresh()

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self) -> Text:
        text = Text()
        if self._spinning:
            text.append(f" {_SPINNER[self._frame]} ", style="bold #6cb6ff")
            text.append(self.tool_name, style="bold #6cb6ff")
            if self.verbose and self.args_str:
                text.append(f"  {self.args_str}", style="dim #8a8a96")
            return text

        icon = "✓" if self.success else "✗"
        icon_style = "bold #4ac26b" if self.success else "bold #f47067"
        text.append(f" {icon} ", style=icon_style)
        text.append(self.tool_name, style="bold #dcdcf5")
        if self.verbose and self.args_str:
            text.append(f"  {self.args_str}", style="dim #8a8a96")
        if self.duration_ms is not None:
            text.append(f"  {self.duration_ms:.0f}ms", style="dim #6cb6ff")

        if not self.success and self.output:
            preview = self.output.strip().splitlines()
            first = preview[0][:90] if preview else ""
            if first:
                text.append(f"\n   └ {first}", style="dim #f47067")
        return text


# ── Collapsible info rows (memory / repo / debug / plan content) ─────────────

class CollapsibleRow(Widget):
    """
    A titled, collapsed-by-default row:

        → Memory: recalled architecture notes

    Clicking the title expands the content inline — no separate panel.
    """

    def __init__(
        self,
        title: str,
        body: str = "",
        *,
        prefix: str = "→",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = body
        self._prefix = prefix
        self.expanded = False

    def compose(self) -> ComposeResult:
        yield Static(
            f" {self._prefix} {self._title}",
            id="info-row-toggle",
            markup=True,
        )
        with Vertical(id="info-row-body"):
            yield Static(
                self._body or "[dim](empty)[/]",
                markup=True,
                classes="info-row-content",
            )

    def on_mount(self) -> None:
        self.query_one("#info-row-body", Vertical).display = False

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "info-row-toggle":
            self.toggle()

    def set_title(self, title: str) -> None:
        self._title = title
        try:
            self.query_one("#info-row-toggle", Static).update(
                f" {self._prefix} {title}"
            )
        except Exception:
            pass

    def set_body(self, body: str) -> None:
        self._body = body
        try:
            self.query_one(".info-row-content", Static).update(body)
        except Exception:
            pass

    def toggle(self) -> None:
        self.expanded = not self.expanded
        body = self.query_one("#info-row-body", Vertical)
        if self.expanded:
            self.query_one("#info-row-toggle", Static).update(
                f" ▾ {self._title}"
            )
        else:
            self.query_one("#info-row-toggle", Static).update(
                f" {self._prefix} {self._title}"
            )
        body.display = self.expanded


# ── Inline status line (thin, above the input) ───────────────────────────────

_STATE_GLYPHS = {
    "idle": ("●", "#9a9aa3"),
    "active": ("●", "#4ac26b"),
    "thinking": ("⠋", "#d4a72c"),
    "running": ("⠋", "#6cb6ff"),
    "done": ("●", "#4ac26b"),
    "error": ("●", "#f47067"),
}


class InlineStatus(Static):
    """
    Thin single-line status, pinned above the input:

        ● IDLE  session — · model —   0 tools · 0 iter   tokens 0   elapsed 0:00
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = "idle"
        self._session = "—"
        self._model = "—"
        self._tool_calls = 0
        self._iterations = 0
        self._tokens = 0
        self._elapsed_ms = 0.0
        self._started_at: float | None = None
        self._frame = 0

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.25, self._tick)

    def _tick(self) -> None:
        if self._state in ("thinking", "running"):
            self._frame += 1
        self._refresh()

    def update_stats(
        self,
        *,
        state: str | None = None,
        session: str | None = None,
        model: str | None = None,
        tool_calls: int | None = None,
        iterations: int | None = None,
        tokens: int | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        if state is not None:
            self._state = state
            self._started_at = (
                time.time() if state in ("thinking", "running", "active") else None
            )
        if session is not None:
            self._session = session
        if model is not None:
            self._model = model
        if tool_calls is not None:
            self._tool_calls = tool_calls
        if iterations is not None:
            self._iterations = iterations
        if tokens is not None:
            self._tokens = tokens
        if elapsed_ms is not None:
            self._elapsed_ms = elapsed_ms
        self._refresh()

    def _elapsed_text(self) -> str:
        ms = self._elapsed_ms
        if self._started_at is not None:
            ms += (time.time() - self._started_at) * 1000
        total = int(ms // 1000)
        return f"{total // 60}:{total % 60:02d}"

    def _refresh(self) -> None:
        glyph, color = _STATE_GLYPHS.get(self._state, ("●", "#9a9aa3"))
        if self._state in ("thinking", "running"):
            glyph = "⠋" if self._frame % 2 == 0 else "⠙"
        text = Text()
        text.append(f" {glyph} ", style=f"bold {color}")
        text.append(self._state.upper(), style=f"bold {color}")
        text.append(f"   session {self._session[:10]}", style="dim #9a9aa3")
        text.append(f" · model {self._model[:16]}", style="dim #9a9aa3")
        text.append(
            f"   {self._tool_calls} tools · {self._iterations} iter",
            style="dim #9a9aa3",
        )
        text.append(f"   tokens {self._tokens:,}", style="dim #6cb6ff")
        text.append(f"   elapsed {self._elapsed_text()}", style="dim #d2a8ff")
        self.update(text)


# ── The stream panel ─────────────────────────────────────────────────────────

class AgentPanel(Widget):
    """
    The single main panel: conversation stream + inline status + input.

    Emits 'submit_task' message when the user sends input.
    """

    DEFAULT_CSS = """
    AgentPanel {
        width: 1fr;
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
        self._stream_widget: MessageWidget | None = None
        self._pending_tools: list[ToolRow] = []
        self.verbose = True

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-panel"):
            with ScrollableContainer(id="stream"):
                pass
            yield InlineStatus(id="status-line")
            with Horizontal(id="agent-input-area"):
                yield Static("❯", id="agent-prompt-icon")
                yield Input(
                    placeholder="Ask TRACERA anything... (/help for commands)",
                    id="agent-input",
                )
            yield Static(
                "Enter send · /help commands · ctrl+t verbose rows",
                id="input-hints",
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.query_one("#agent-input", Input).value = ""
            self.post_message(self.SubmitTask(text))

    # ── Stream helpers ───────────────────────────────────────────────────────

    def _stream(self) -> ScrollableContainer:
        return self.query_one("#stream", ScrollableContainer)

    def _append(self, widget: Widget) -> None:
        """Mount *widget* into the stream, snapping to bottom if pinned."""
        stream = self._stream()
        stream.mount(widget)
        self._scroll_bottom_if_pinned(stream)

    def _scroll_bottom_if_pinned(self, stream: ScrollableContainer | None = None) -> None:
        """Auto-scroll to the latest row only while the user is at the bottom."""
        stream = stream or self._stream()
        try:
            pinned = stream.scroll_y >= stream.max_scroll_y - 1
        except Exception:
            pinned = True
        if pinned:
            stream.scroll_end(animate=False)

    # ── Messages ─────────────────────────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        self._append(MessageWidget("user", text))

    def add_assistant_message(self, text: str) -> None:
        self._append(MessageWidget("assistant", text))

    def add_tool_message(self, tool_name: str, output: str | None = None) -> None:
        preview = tool_name
        if output:
            first_line = output.strip().splitlines()
            if first_line:
                preview += f"  →  {first_line[0][:80]}"
        self._append(MessageWidget("tool", preview))

    def add_error(self, text: str) -> None:
        self._append(MessageWidget("error", text))

    def add_meta(self, text: str) -> None:
        self._append(MessageWidget("meta", text))

    def add_thinking_disclosure(self, entries: list[tuple[str, str]]) -> None:
        if not entries:
            return
        self._append(ThinkingDisclosure(entries))

    async def type_message(self, text: str) -> None:
        """Type a message into the conversation character-by-character."""
        import asyncio
        stream = self._stream()
        widget = MessageWidget("assistant", "")
        stream.mount(widget)
        for ch in text:
            widget.set_content(widget.msg_content + ch)
            self._scroll_bottom_if_pinned(stream)
            await asyncio.sleep(0.012)

    def stream_delta(self, text: str) -> None:
        """Append a token delta to the live assistant message (streaming)."""
        stream = self._stream()
        if self._stream_widget is None:
            widget = MessageWidget("assistant", text)
            stream.mount(widget)
            self._stream_widget = widget
        else:
            self._stream_widget.set_content(self._stream_widget.msg_content + text)
        self._scroll_bottom_if_pinned(stream)

    def stream_end(self, full_text: str | None = None) -> None:
        if self._stream_widget is not None:
            if full_text is not None:
                self._stream_widget.set_content(full_text)
            self._stream_widget = None
        elif full_text:
            self.add_assistant_message(full_text)

    # ── Inline tool rows ─────────────────────────────────────────────────────

    def tool_start(self, name: str, detail: str = "") -> None:
        """Open an in-flight tool row with an animated spinner."""
        row = ToolRow(name, detail, verbose=self.verbose)
        row.start_spinner()
        self._pending_tools.append(row)
        self._append(row)

    def tool_end(
        self,
        name: str,
        success: bool,
        duration_ms: float = 0.0,
        output: str = "",
    ) -> None:
        """Finalize the most recent pending row for *name*."""
        for i in range(len(self._pending_tools) - 1, -1, -1):
            row = self._pending_tools[i]
            if row.tool_name == name:
                row.finish(success, duration_ms, output)
                self._pending_tools.pop(i)
                break
        else:
            # No pending row (event raced ahead) — append a finished one.
            row = ToolRow(name, "", verbose=self.verbose)
            row.finish(success, duration_ms, output)
            self._append(row)
        self._scroll_bottom_if_pinned()

    # ── Collapsible info rows (memory / repo / debug / plan) ─────────────────

    def add_info_row(
        self,
        title: str,
        body: str = "",
        *,
        prefix: str = "→",
    ) -> CollapsibleRow:
        row = CollapsibleRow(title, body, prefix=prefix)
        self._append(row)
        return row

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        stream = self._stream()
        for child in list(stream.children):
            child.remove()
        self._stream_widget = None
        self._pending_tools.clear()
