"""
TRACERA Conversation Stream — the single main panel (Claude Code style).

One continuous, auto-scrolling stream of everything the agent does, inline:

    ┌─ YOU ────────────────────────────────┐
    │  add jwt validation to the middleware│
    └──────────────────────────────────────┘
    ◇ Planning
    ⠋ Thinking
    ✓ search_code (query='jwt auth')   8ms
    ✓ edit_file  📝 auth/middleware.py  +12 -4   8ms   ← click → inline diff
    ✗ run_command  (command='pytest')  0ms
      └ pytest: error: unrecognized arguments
    ┌─ TRACERA ──────────────────────────┐
    │  Done. All tests pass.            │
    └────────────────────────────────────┘
    ● DONE  session 8f2c · model gemini · 5 tools · 3 iter · 1.2k tok · 0:42
    ⠋ Thinking  ●            ← loader pill while a request is in flight
    ＋ ❯ [ input pill ......... ]
    Enter send · /help commands · ctrl+t verbose rows

While a task runs the loader pill replaces the input; attachments show as
removable chips above the pill.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from pygments.lexers import get_lexer_for_filename
from pygments.token import (
    Comment,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    String,
    Token,
)
from pygments.util import ClassNotFound
from rich.markup import escape
from rich.style import Style
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Horizontal, Vertical, ScrollableContainer
from rich.text import Text

from tracera.tui.diffutil import is_image
from tracera.tui.theme import get_theme
from tracera.tui.widgets.command_input import CommandInput
from tracera.tui.widgets.command_registry import SLASH_COMMANDS

# Premium multi-style spinner collections for different states
_SPINNERS = {
    "thinking": ["◐", "◓", "◑", "◒"],  # Smooth circle rotation
    "running": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],  # Classic braille
    "loading": ["▰▱▱▱▱", "▰▰▱▱▱", "▰▰▰▱▱", "▰▰▰▰▱", "▰▰▰▰▰", "▱▰▰▰▰", "▱▱▰▰▰", "▱▱▱▰▰", "▱▱▱▱▰"],  # Progress wave
    "pulse": ["●", "○", "◎", "◉", "●", "○", "◎", "◉"],  # Pulsing effect
    "wave": ["⎽", "⎼", "⎻", "⎺", "⎼", "⎻"],  # Vertical wave
    "diamond": ["◇", "◆", "◇", "◆"],  # Blinking diamond
    "arrows": ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"],  # Spinning arrows
    "blocks": ["█", "▉", "▊", "▋", "▌", "▍", "▎", "▏"],  # Filling block
}

# Default to thinking spinner for backward compatibility
_SPINNER = _SPINNERS["thinking"]


def _t(key: str) -> str:
    """Active theme color as an inline-Rich hex string ('#da8548')."""
    return f"#{getattr(get_theme(), key)}"


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

    Content is **escaped by default**: model output, tool output, and file
    content routinely contain ``[``/``]`` (``data[0]``, ``List[int]``,
    markdown links, code snippets) which Rich markup would eat or crash on.
    Pass ``trusted=True`` only for strings this codebase itself authored as
    Rich markup (banners, command output tables…).
    """

    _BORDER_TITLES = {"user": " > ", "assistant": " ◆ TRACERA "}
    _PREFIXES = {"tool": "⚙  ", "error": "✗  "}

    def __init__(self, role: str, content: str, *, trusted: bool = False, **kwargs):
        self.role = role
        self.msg_content = content
        self.trusted = trusted
        display = self._compose_display(content)
        super().__init__(display, markup=True, classes=f"msg-{role}", **kwargs)
        title = self._BORDER_TITLES.get(role)
        if title:
            self.border_title = title

    def _compose_display(self, content: str) -> str:
        """Prefix + (escape unless trusted) — the single render path.

        NOTE: deliberately NOT named ``_render`` — that name is a Textual
        ``Static`` hook invoked by the compositor with its own signature.
        """
        prefix = self._PREFIXES.get(self.role, "")
        body = content if self.trusted else escape(content)
        return f"{prefix}{body}"

    def set_content(self, content: str) -> None:
        """Replace the message content (used by streaming)."""
        self.msg_content = content
        self.update(self._compose_display(content))

    def set_markdown(self, content: str) -> None:
        """Replace the message content with a Markdown renderable.

        Used once the stream completes so the final assistant turn is rendered
        as full Markdown (headers, code blocks with syntax highlighting, lists,
        tables, blockquotes) instead of raw markup. Falls back to escaped plain
        markup if Markdown rendering fails for any reason. Markdown source is
        never treated as Rich markup, so brackets survive verbatim.
        """
        self.msg_content = content
        try:
            from rich.markdown import Markdown

            self.add_class("msg-markdown")
            md = Markdown(
                content,
                code_theme="monokai",
                inline_code_lexer="python",
                inline_code_theme="monokai",
            )
            self.update(md)
        except Exception:
            self.update(escape(content))


_GLYPH = {
    "think": "◇",
    "tool": "◆",
    "done": "✓",
    "error": "✗",
}


class ThinkingDisclosure(Widget):
    """Collapsible per-turn 'Thinking…' block attached to an assistant turn."""

    def __init__(self, entries: list[tuple[str, str]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries = entries
        self.expanded = False

    def compose(self) -> ComposeResult:
        yield Static(self._trusted_label(), id="reasoning-toggle")
        with Vertical(id="reasoning-body"):
            for kind, text in self._entries:
                glyph = _GLYPH.get(kind, "·")
                yield Static(
                    f"{glyph} {escape(text)}",
                    classes=f"reasoning-line reasoning-{kind}",
                )

    def on_mount(self) -> None:
        self.query_one("#reasoning-body", Vertical).display = False
        self._refresh_toggle()

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "reasoning-toggle":
            self.toggle()

    def _trusted_label(self) -> Text:
        """Render the toggle label as styled Text (no markup parsing)."""
        t = Text()
        glyph = "▾" if self.expanded else "▸"
        t.append(f" {glyph} Thinking… ({len(self._entries)})", style=f"dim italic {_t('muted')}")
        return t

    def _refresh_toggle(self) -> None:
        try:
            self.query_one("#reasoning-toggle", Static).update(self._trusted_label())
        except Exception:
            pass

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self._refresh_toggle()
        body = self.query_one("#reasoning-body", Vertical)
        body.display = self.expanded


# ── Phase marker rows (full agent-loop visualization) ────────────────────────

class PhaseRow(Static):
    """A premium phase marker with dynamic spinner selection based on phase:

        ◐ Planning     ← active (animated circle)
        ◇ Planning     ← superseded by the next phase (dim)
    """

    # Map phases to appropriate spinners — color resolved per render so the
    # theme preset applies live.
    _PHASE_CONFIG = {
        "planning": "thinking",   # Circle rotation - planning
        "thinking": "pulse",      # Pulsing - thinking
        "searching": "running",   # Classic - searching
        "indexing": "loading",    # Progress wave - indexing
        "running": "blocks",      # Filling block - executing
        "generating": "wave",     # Vertical wave - generating
        "writing": "arrows",      # Spinning arrows - writing
    }

    def __init__(self, label: str, phase_type: str = "thinking", **kwargs) -> None:
        super().__init__(**kwargs)
        self.phase_label = label
        self.phase_type = phase_type.lower()
        self._frame = 0
        self._spinning = True
        spinner_key = self._PHASE_CONFIG.get(self.phase_type, "thinking")
        self._spinner = _SPINNERS[spinner_key]

    def on_mount(self) -> None:
        if self._spinning:
            # Adjust tick rate based on spinner complexity
            tick_rate = 0.15 if len(self._spinner) > 6 else 0.2
            self.set_interval(tick_rate, self._tick)

    def freeze(self) -> None:
        self._spinning = False
        self.refresh()

    def _tick(self) -> None:
        if self._spinning:
            self._frame += 1
            self.refresh()

    def render(self) -> Text:
        text = Text()
        if self._spinning:
            color = _t("secondary")
            text.append(
                f" {self._spinner[self._frame % len(self._spinner)]} ",
                style=f"bold {color}",
            )
            text.append(self.phase_label, style=f"bold {color}")
        else:
            text.append(" ◇ ", style=f"dim {_t('muted')}")
            text.append(self.phase_label, style=f"dim {_t('muted')}")
        return text


# ── Inline tool rows ─────────────────────────────────────────────────────────

_DIFF_PREFIX = {"add": "+", "del": "-", "hunk": "  ", "ctx": "  ", "ellipsis": "  "}


def _diff_tint_bg(kind: str) -> str | None:
    """Background wash for a diff kind, from the active theme preset."""
    theme = get_theme()
    return {
        "add": theme.diff_add_bg,
        "del": theme.diff_del_bg,
        "hunk": theme.diff_hunk_bg,
    }.get(kind)


def _diff_flat_style(kind: str) -> str:
    """Fallback flat line color for a diff kind, from the active theme."""
    theme = get_theme()
    return {
        "add": f"#{theme.success}",
        "del": f"#{theme.error}",
        "hunk": f"#{theme.meta}",
        "ellipsis": f"#{theme.faint}",
    }.get(kind, f"dim #{theme.muted}")


#: Token colors for syntax-highlighted diffs (pygments Token types).
#: A deliberately small, calm palette — full monokai-per-token would be noise.
_PYGMENTS_STYLES = {
    Comment: "italic #6a7a8a",
    Keyword: "bold #d2a8ff",
    Keyword.Type: "#6cb6ff",
    String: "#7ee787",
    Number: "#ffd700",
    Name.Function: "bold #6cb6ff",
    Name.Class: "bold #ffd700",
    Name.Decorator: "#ff9f43",
    Name.Builtin: "#6cb6ff",
    Name.Exception: "bold #f47067",
    Operator: "#ff9f43",
    Generic.Deleted: "#f47067",
    Generic.Inserted: "#4ac26b",
}


@lru_cache(maxsize=32)
def _lexer_for(path: str | None):
    """Pygments lexer for a file path, or None when unknown."""
    if not path:
        return None
    try:
        return get_lexer_for_filename(path)
    except ClassNotFound:
        return None


def _highlight_diff_line(line: str, kind: str, path: str | None) -> Text:
    """One diff line: syntax-highlight tokens + add/del tint underneath.

    Tokens are colored by pygments; the add/del coloring becomes a subtle
    background wash so removed/added code is still readable *as code*.
    Falls back to the flat red/green/… line style when no lexer matches or
    pygments errors for any reason.
    """
    prefix = _DIFF_PREFIX.get(kind, "  ")
    if len(line) > 140:
        line = line[:137] + "…"
    flat_style = _diff_flat_style(kind)
    lexer = _lexer_for(path)
    if lexer is None:
        return Text(f"   {prefix} {line}", style=flat_style)
    tint_bg = _diff_tint_bg(kind)
    tint = Style(bgcolor=tint_bg) if tint_bg else None
    try:
        result = Text()
        result.append(f"   {prefix} ", style="dim")
        for tok_type, tok_value in lexer.get_tokens(line):
            if not tok_value or tok_value == "\n":
                continue
            color = _PYGMENTS_STYLES.get(tok_type)
            if color is not None:
                style = Style.parse(color)
            else:
                style = Style.parse(flat_style) if isinstance(flat_style, str) else flat_style
            if tint is not None:
                style = tint + style
            result.append(tok_value, style)
        return result
    except Exception:
        return Text(f"   {prefix} {line}", style=flat_style)


class ToolRow(Static):
    """
    One compact inline row per tool call with color-coded categories:

        ⠋ run_command (command='pytest')        ← in-flight (animated)
        ✓ read_file  (path='a.py')   12ms
        ✗ edit_file  (path='b.py')   3ms
          └ ERROR: merge conflict               ← auto-expanded on failure
        ✓ edit_file  📝 main.py  +12 -4   8ms   ← diffable tools (click to
          ▾ 14 lines                            ←   expand the inline diff)
            @@ -1,3 +1,5 @@
            - old line
            + new line
    """

    # Tool category → semantic color key in the active Theme preset.
    _TOOL_CATEGORIES = {
        # Search/analysis tools
        "search_code": "secondary",
        "find_symbol": "secondary",
        "find_definition": "secondary",
        "grep": "secondary",
        "get_context": "secondary",
        "get_dependencies": "secondary",
        "find_references": "secondary",
        # Read tools
        "read_file": "warning",
        "list_dir": "warning",
        # Write/edit tools
        "write_file": "success",
        "edit_file": "success",
        "delete_file": "success",
        # Command tools
        "run_command": "meta",
        # Git tools
        "git": "accent",
        # Memory tools
        "memory": "error",
        # Test tools
        "test": "secondary",
    }

    def __init__(
        self,
        name: str,
        args_str: str = "",
        *,
        verbose: bool = True,
        spinner_type: str = "running",
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
        self._spinner_type = spinner_type
        self._spinner = _SPINNERS[spinner_type]
        # Diff state — filled by the app when the tool touched a file.
        self.diff_path: str | None = None
        self.snapshot: str | None = None
        self.snapshot_path: str | None = None
        self._diff_lines: list[tuple[str, str]] = []
        self._diff_added = 0
        self._diff_removed = 0
        self.expanded = False

    def _get_tool_color(self) -> str:
        """Get the theme color key for this tool's category."""
        return _t(self._TOOL_CATEGORIES.get(self.tool_name, "text"))

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        if self._spinning:
            tick_rate = 0.1 if len(self._spinner) > 6 else 0.15
            self.set_interval(tick_rate, self._tick)

    def start_spinner(self, spinner_type: str = "running") -> None:
        """Start spinning with a specific spinner type."""
        self._spinner_type = spinner_type
        self._spinner = _SPINNERS[spinner_type]
        self._spinning = True
        if self.is_mounted:
            tick_rate = 0.1 if len(self._spinner) > 6 else 0.15
            self.set_interval(tick_rate, self._tick)

    def finish(self, success: bool, duration_ms: float, output: str = "") -> None:
        self._spinning = False
        self.success = success
        self.duration_ms = duration_ms
        self.output = output
        self.refresh()

    def set_snapshot(self, path: str, content: str) -> None:
        """Record the file content before the tool ran (for the diff)."""
        self.snapshot_path = path
        self.snapshot = content

    def set_diff(self, path: str, lines: list[tuple[str, str]], added: int, removed: int) -> None:
        self.diff_path = path
        self._diff_lines = lines
        self._diff_added = added
        self._diff_removed = removed
        self.refresh()

    def _tick(self) -> None:
        if self._spinning:
            self._frame = (self._frame + 1) % len(self._spinner)
            self.refresh()

    # ── Interaction ──────────────────────────────────────────────────────────

    def on_click(self, event) -> None:
        if self.diff_path and self._diff_lines:
            self.expanded = not self.expanded
            self.refresh()

    def render(self) -> Text:
        text = Text()
        tool_color = self._get_tool_color()
        
        if self._spinning:
            text.append(f" {self._spinner[self._frame]} ", style=f"bold {tool_color}")
            text.append(self.tool_name, style=f"bold {tool_color}")
            if self.verbose and self.args_str:
                text.append(f"  {self.args_str}", style=f"dim {_t('muted')}")
            return text

        icon = "✓" if self.success else "✗"
        icon_style = f"bold {_t('success')}" if self.success else f"bold {_t('error')}"
        text.append(f" {icon} ", style=icon_style)
        text.append(self.tool_name, style=f"bold {tool_color}")

        if self.diff_path and self.success:
            # Code-gen summary: 📝 path  +N -M
            text.append(f"  📝 {self.diff_path}", style=f"bold {_t('meta')}")
            if self._diff_added or self._diff_removed:
                text.append(f"  +{self._diff_added}", style=f"bold {_t('success')}")
                text.append(f" -{self._diff_removed}", style=f"bold {_t('error')}")
            elif self.verbose and self.args_str:
                text.append(f"  {self.args_str}", style=f"dim {_t('muted')}")
        elif self.verbose and self.args_str:
            text.append(f"  {self.args_str}", style=f"dim {_t('muted')}")

        if self.duration_ms is not None:
            # Color-coded duration bar
            if self.duration_ms < 100:
                dur_color = _t("success")  # Fast - green
            elif self.duration_ms < 500:
                dur_color = _t("warning")  # Medium - yellow
            else:
                dur_color = _t("error")  # Slow - red
            text.append(f"  {self.duration_ms:.0f}ms", style=f"dim {dur_color}")

        if not self.success and self.output:
            preview = self.output.strip().splitlines()
            first = preview[0][:90] if preview else ""
            if first:
                text.append(f"\n   └ {first}", style=f"dim {_t('error')}")

        if self.expanded and self._diff_lines:
            text.append(self._render_diff())
        return text

    def _render_diff(self) -> Text:
        """Expandable inline diff with per-token syntax highlighting."""
        d = Text("\n")
        d.append(f"   ▾ {len(self._diff_lines)} lines", style=f"dim {_t('muted')}")
        for kind, line in self._diff_lines:
            d.append_text(Text("\n"))
            d.append_text(_highlight_diff_line(line, kind, self.diff_path))
        return d


# ── Collapsible info rows (memory / search / repo / debug / plan) ─────────────

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
        trusted: bool = False,
        **kwargs,
    ) -> None:
        """``trusted=True`` renders title/body as Rich markup verbatim.

        Callers that compose Rich markup themselves (search results,
        observability tables…) opt in; anything derived from model/tool/user
        output goes through ``escape`` by default.
        """
        super().__init__(**kwargs)
        self._title = title
        self._body = body
        self._prefix = prefix
        self.trusted = trusted
        self.expanded = False

    def _label(self) -> str:
        title = self._title if self.trusted else escape(self._title)
        return f" {self._prefix} {title}"

    def _content(self) -> str:
        body = self._body or "[dim](empty)[/]"
        return body if self.trusted else escape(body)

    def compose(self) -> ComposeResult:
        yield Static(
            self._label(),
            id="info-row-toggle",
            markup=True,
        )
        with Vertical(id="info-row-body"):
            yield Static(
                self._content(),
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
            self.query_one("#info-row-toggle", Static).update(self._label())
        except Exception:
            pass

    def set_body(self, body: str) -> None:
        self._body = body
        try:
            self.query_one(".info-row-content", Static).update(self._content())
        except Exception:
            pass

    def toggle(self) -> None:
        self.expanded = not self.expanded
        body = self.query_one("#info-row-body", Vertical)
        label = f" ▾ {self._title}" if self.expanded else self._label()
        if not self.trusted and self.expanded:
            label = f" ▾ {escape(self._title)}"
        self.query_one("#info-row-toggle", Static).update(label)
        body.display = self.expanded


# ── Attachment chips ─────────────────────────────────────────────────────────

class AttachmentChip(Static):
    """A removable chip for an attached file, shown above the input pill."""

    class Removed(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, path: str, *, warning: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.warning = warning

    def render(self) -> Text:
        name = Path(self.path).name or self.path
        icon = "img" if is_image(self.path) else "file"
        t = Text()
        t.append(f" {icon}:", style=f"dim {_t('accent')}")
        t.append(name, style=f"bold {_t('text')}")
        if self.warning:
            t.append(" !", style=f"bold {_t('warning')}")
        t.append("  ×", style=f"dim {_t('error')}")
        return t

    def on_click(self, event) -> None:
        self.post_message(self.Removed(self.path))


# ── Loader pill (in place of the input while a request is in flight) ─────────

_PHASE_LABELS = {
    "planning": "Planning",
    "thinking": "Thinking",
    "running": "Running",
    "generating": "Generating",
    "done": "Done",
}


class LoaderPill(Widget):
    """Premium rounded pill with dynamic animations and phase-specific spinners.

    Shows the live phase label with an animated glyph that changes based on
    the current phase, plus a sleek stop button.
    """

    class StopRequested(Message):
        pass

    # Phase configurations for the loader pill — color resolved live from
    # the active theme (accent for working phases, success for done).
    _LOADER_PHASES = {
        "planning":   ("◐", "accent", "Planning…"),
        "thinking":   ("◉", "accent", "Thinking…"),
        "searching":  ("⠋", "accent", "Searching…"),
        "indexing":   ("▰▱▱▱▱", "accent", "Indexing…"),
        "running":    ("█", "accent", "Running…"),
        "generating": ("⎽", "accent", "Generating…"),
        "writing":    ("→", "accent", "Writing…"),
        "done":       ("✓", "success", "Done"),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phase = "thinking"
        self._frame = 0
        self._full_spinner = _SPINNERS["thinking"]

    def compose(self) -> ComposeResult:
        yield Static("◉", id="loader-icon")
        yield Static("Thinking…", id="loader-label")
        yield Static(" esc ", id="loader-stop")

    def on_mount(self) -> None:
        self.set_interval(0.15, self._tick)

    def set_phase(self, phase: str) -> None:
        """Update the loader with phase-specific styling."""
        self._phase = phase.lower() if phase.lower() in self._LOADER_PHASES else "thinking"
        icon, color, label = self._LOADER_PHASES[self._phase]
        try:
            # Update label and get the widget to apply color
            label_widget = self.query_one("#loader-label", Static)
            label_widget.update(label)
            
            # Update icon widget
            icon_widget = self.query_one("#loader-icon", Static)
            icon_widget.update(icon)
        except Exception:
            pass

    def _tick(self) -> None:
        if self._phase == "done":
            return
        self._frame += 1
        try:
            # Use full animated spinner for main phases
            icon = self._full_spinner[self._frame % len(self._full_spinner)]
            self.query_one("#loader-icon", Static).update(icon)
        except Exception:
            pass

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "loader-stop":
            self.post_message(self.StopRequested())


# ── Inline status line (thin, above the input) ───────────────────────────────

_STATE_GLYPHS = {
    "idle":     ("○", "faint"),
    "active":   ("●", "success"),
    "thinking": ("◉", "accent"),
    "running":  ("◉", "accent"),
    "done":     ("●", "success"),
    "error":    ("●", "error"),
}


class InlineStatus(Static):
    """
    Premium system status line with comprehensive metrics and feature indicators:

        ● ACTIVE  session 8f2c7b · gemini-pro · 12 tools · 5 iter · 2.4k tok · 01:23
        [◉ MEM] [◉ RET] [◎ RAG] [◉ MCP]  Features: Memory, Retrieval, RAG, MCP
    """

    # Feature status indicators → theme color keys
    _FEATURES = {
        "memory":    ("mem",  "accent"),
        "retrieval": ("ret",  "accent"),
        "rag":       ("rag",  "accent"),
        "mcp":       ("mcp",  "accent"),
        "index":     ("idx",  "accent"),
        "sandbox":   ("sbx",  "accent"),
    }

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
        # Feature status tracking
        self._feature_status = {
            "memory": True,
            "retrieval": False,
            "rag": False,
            "mcp": False,
            "index": False,
            "sandbox": True,
        }
        # Premium spinners for active states
        self._state_spinners = _SPINNERS["pulse"]
        # Session metrics (kept across incremental update_stats calls)
        self._memory_hits = 0
        self._retrieval_hits = 0
        self._cost_estimate = 0.0

    def set_feature_status(self, feature: str, active: bool) -> None:
        """Update feature active status."""
        if feature in self._feature_status:
            self._feature_status[feature] = active
            self._refresh()

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.2, self._tick)

    def _tick(self) -> None:
        if self._state in ("thinking", "running", "active"):
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
        memory_hits: int | None = None,
        retrieval_hits: int | None = None,
        cost_estimate: float | None = None,
    ) -> None:
        """Enhanced stats with more metrics."""
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
        # Store additional metrics — None leaves the previous value intact so
        # incremental updates (state-only, tokens-only …) never wipe stats.
        if memory_hits is not None:
            self._memory_hits = memory_hits
        if retrieval_hits is not None:
            self._retrieval_hits = retrieval_hits
        if cost_estimate is not None:
            self._cost_estimate = cost_estimate
        self._refresh()

    def _elapsed_text(self) -> str:
        ms = self._elapsed_ms
        if self._started_at is not None:
            ms += (time.time() - self._started_at) * 1000
        total = int(ms // 1000)
        return f"{total // 60}:{total % 60:02d}"

    def _render_features(self) -> Text:
        """Render compact feature pills."""
        feat_text = Text()
        feat_text.append("  ", style="dim")

        for feat, (label, color_key) in self._FEATURES.items():
            active = self._feature_status[feat]
            if active:
                feat_text.append(f" {label} ", style=f"bold {_t(color_key)}")
            else:
                feat_text.append(f" {label} ", style=f"dim #{get_theme().border}")
        return feat_text

    def _refresh(self) -> None:
        glyph, color_key = _STATE_GLYPHS.get(self._state, ("○", "faint"))
        color = _t(color_key)
        if self._state in ("thinking", "running"):
            glyph = self._state_spinners[self._frame % len(self._state_spinners)]

        text = Text()

        # Glyph + state label
        text.append(f" {glyph} ", style=f"bold {color}")
        text.append(self._state.upper(), style=f"bold {color}")

        # Session / model
        text.append(f"  {self._session[:8]}", style=f"dim #{get_theme().faint}")
        text.append(f"  {self._model[:18]}", style=f"dim {_t('muted')}")

        # Tool / iter counts — only show when non-zero
        if self._tool_calls:
            text.append(f"  ⚙ {self._tool_calls}", style=f"dim {_t('muted')}")
        if self._iterations:
            text.append(f"  ↻ {self._iterations}", style=f"dim {_t('muted')}")

        # Token count + context meter
        text.append_text(self._render_token_bar())

        # Memory / retrieval hits — proof the subsystems are alive
        memory_hits = getattr(self, "_memory_hits", 0)
        retrieval_hits = getattr(self, "_retrieval_hits", 0)
        if memory_hits:
            text.append(f"  mem {memory_hits}", style=f"dim {_t('error')}")
        if retrieval_hits:
            text.append(f"  ret {retrieval_hits}", style=f"dim {_t('secondary')}")

        # Optional extras
        if getattr(self, "_cost_estimate", 0) > 0:
            text.append(f"  ${self._cost_estimate:.3f}", style=f"dim {_t('warning')}")

        # Elapsed
        text.append(f"  {self._elapsed_text()}", style=f"dim #{get_theme().faint}")

        # Feature pills — second line
        text.append("\n")
        text.append_text(self._render_features())

        self.update(text)

    def _render_token_bar(self) -> Text:
        """Compact context-usage meter: `▰▰▱▱▱ 2.4k tok 3%`"""
        text = Text()

        # Assume 200k token context window for the meter.
        max_tokens = 200_000
        current = min(self._tokens, max_tokens)
        ratio = current / max_tokens if max_tokens > 0 else 0

        # Choose color based on usage
        if ratio < 0.5:
            bar_color = _t("success")  # Green
        elif ratio < 0.8:
            bar_color = _t("warning")  # Yellow
        else:
            bar_color = _t("error")  # Red

        # Five-segment meter
        filled = int(round(ratio * 5))
        meter = "▰" * filled + "▱" * (5 - filled)

        # Format token count
        if self._tokens >= 1_000_000:
            tok_str = f"{self._tokens / 1_000_000:.1f}M"
        elif self._tokens >= 1_000:
            tok_str = f"{self._tokens / 1_000:.1f}k"
        else:
            tok_str = str(self._tokens)

        text.append(f" · {meter}", style=f"dim {bar_color}")
        text.append(f" {tok_str} tok", style=f"dim {bar_color}")
        if ratio > 0:
            text.append(f" {ratio:.0%}", style=f"dim #{get_theme().faint}")

        return text


# ── The stream panel ─────────────────────────────────────────────────────────

class AgentPanel(Widget):
    """
    The single main panel: conversation stream + status + loader/input.

    Emits 'submit_task' when the user sends input, 'attach_requested' when
    the attach button is clicked, and 'stop_requested' via the loader pill.
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

    class AttachRequested(Message):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stream_widget: MessageWidget | None = None
        self._pending_tools: list[ToolRow] = []
        self._active_phase: PhaseRow | None = None
        self._attachments: list[str] = []
        self.verbose = True

    def on_mount(self) -> None:
        """Configure the multiline editor with the slash-command registry."""
        try:
            self.query_one("#agent-input", CommandInput).set_commands(SLASH_COMMANDS)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-panel"):
            with ScrollableContainer(id="stream"):
                pass
            yield InlineStatus(id="status-line")
            yield LoaderPill(id="loader-pill")
            with Vertical(id="agent-input-area"):
                yield Static(id="suggestion-list")
                with Horizontal(id="attach-chips"):
                    pass
                with Horizontal(id="agent-input-bar"):
                    yield Static("+", id="attach-button")
                    yield Static("❯", id="agent-prompt-icon")
                    yield CommandInput(
                        id="agent-input",
                        placeholder="Ask anything about this codebase…  (/ for commands)",
                    )
            yield Static(
                "enter send · /help commands · ctrl+t verbose · ctrl+p provider · esc cancel",
                id="input-hints",
            )

    def on_command_input_submit(self, event: "CommandInput.Submit") -> None:
        text = event.text.strip()
        if text:
            self.post_message(self.SubmitTask(text))

    def on_command_input_suggestions_changed(
        self, event: "CommandInput.SuggestionsChanged"
    ) -> None:
        """Render the live slash-command autocomplete list above the prompt."""
        try:
            widget = self.query_one("#suggestion-list", Static)
        except Exception:
            return
        if not event.matches:
            widget.update("")
            widget.remove_class("has-content")
            return
        lines = []
        accent = _t("accent")
        for name in event.matches[:6]:
            desc = SLASH_COMMANDS.get(name, "")
            lines.append(f"  [bold {accent}]/{name}[/]  [dim]{desc}[/]")
        widget.update("\n".join(lines))
        widget.add_class("has-content")

    # ── Stream helpers ───────────────────────────────────────────────────────

    def set_feature_status(self, feature: str, active: bool) -> None:
        """Set feature status indicator in the status line."""
        status_line = self.query_one("#status-line", InlineStatus)
        status_line.set_feature_status(feature, active)
        
    def update_system_metrics(self, **kwargs) -> None:
        """Update system metrics in status line."""
        status_line = self.query_one("#status-line", InlineStatus)
        status_line.update_stats(**kwargs)

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
        """User text is escaped — it may contain brackets, code, anything."""
        self._append(MessageWidget("user", text))

    def add_assistant_message(self, text: str, *, trusted: bool = False) -> None:
        """Assistant bubble.

        Default (``trusted=False``): text is escaped and rendered as Markdown
        when complete — safe for model output containing brackets.
        ``trusted=True``: text is Rich markup authored by this app (help,
        status tables…) and rendered verbatim — dynamic content inside it
        must be pre-escaped by the caller.
        """
        widget = MessageWidget("assistant", text, trusted=trusted)
        self._append(widget)
        if trusted:
            # Keep it a markup Static — do NOT convert to Markdown, which
            # would strip the tags and reflow the layout.
            widget.update(text)
        else:
            widget.set_markdown(text)

    def add_error(self, text: str, *, trusted: bool = False) -> None:
        self._append(MessageWidget("error", text, trusted=trusted))

    def add_banner(self, text: str) -> None:
        """Render the CLI banner (trusted Rich markup from tracera.logging)."""
        self._append(Static(text, markup=True, classes="banner-block"))

    def add_meta(self, text: str) -> None:
        """Meta lines are authored by this app — trusted Rich markup.

        Dynamic content interpolated into meta strings must be pre-escaped
        by the caller (paths, model names…).
        """
        self._append(MessageWidget("meta", text, trusted=True))

    def add_thinking_disclosure(self, entries: list[tuple[str, str]]) -> None:
        if not entries:
            return
        self._append(ThinkingDisclosure(entries))

    async def type_message(self, text: str) -> None:
        """Type a message into the conversation character-by-character.

        Typed text is plain (trusted=False) — pass escaped/Markup-free strings
        only, or pre-render the styled parts via add_assistant_message.
        """
        import asyncio
        stream = self._stream()
        widget = MessageWidget("assistant", "")
        stream.mount(widget)
        for ch in text:  # set_content escapes untrusted content — pass raw
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
        """Finish the streamed turn — upgrade it to full Markdown rendering.

        The live stream renders escaped plain text (cheap, safe on every
        delta); once the response is complete the same widget re-renders as
        rich.markdown.Markdown: headers, lists, tables, highlighted code
        fences. Bracketed text like ``data[0]`` survives both phases.
        """
        if self._stream_widget is not None:
            if full_text is not None:
                self._stream_widget.set_markdown(full_text)
            self._stream_widget = None
        elif full_text:
            self.add_assistant_message(full_text)

    # ── Phase markers ────────────────────────────────────────────────────────

    def add_phase(self, label: str) -> PhaseRow:
        """Stream a phase marker (◇ Planning / ⠋ Thinking …), freezing the last."""
        if self._active_phase is not None:
            self._active_phase.freeze()
        row = PhaseRow(label)
        self._active_phase = row
        self._append(row)
        return row

    def freeze_phase(self) -> None:
        if self._active_phase is not None:
            self._active_phase.freeze()
            self._active_phase = None

    # ── Inline tool rows ─────────────────────────────────────────────────────

    def tool_start(self, name: str, detail: str = "") -> ToolRow:
        """Open an in-flight tool row with an animated spinner."""
        row = ToolRow(name, detail, verbose=self.verbose)
        row.start_spinner()
        self._pending_tools.append(row)
        self._append(row)
        return row

    def tool_end(
        self,
        name: str,
        success: bool,
        duration_ms: float = 0.0,
        output: str = "",
    ) -> ToolRow:
        """Finalize the most recent pending row for *name*; returns the row."""
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
        return row

    def finalize_pending(self, reason: str = "cancelled") -> None:
        """Mark in-flight tool rows as failed (used when the task is cancelled)."""
        for row in self._pending_tools:
            row.finish(False, 0.0, reason)
        self._pending_tools.clear()
        self.freeze_phase()

    # ── Collapsible info rows (memory / repo / debug / plan) ─────────────────

    def add_info_row(
        self,
        title: str,
        body: str = "",
        *,
        prefix: str = "→",
        trusted: bool = False,
    ) -> CollapsibleRow:
        """Collapsible row — pass ``trusted=True`` for app-authored markup."""
        row = CollapsibleRow(title, body, prefix=prefix, trusted=trusted)
        self._append(row)
        return row

    # ── Attachments ──────────────────────────────────────────────────────────

    def _chips(self) -> Horizontal:
        return self.query_one("#attach-chips", Horizontal)

    @property
    def attachments(self) -> list[str]:
        return list(self._attachments)

    def add_attachment(self, path: str, *, warning: bool = False) -> bool:
        """Add an attachment chip; returns False if it is already attached."""
        if path in self._attachments:
            return False
        self._attachments.append(path)
        self._chips().display = True
        self._chips().mount(AttachmentChip(path, warning=warning))
        return True

    def remove_attachment(self, path: str) -> None:
        for chip in list(self.query("AttachmentChip")):
            if chip.path == path:
                # remove() is deferred to the message pump — hide it now so
                # the chip disappears immediately.
                chip.display = False
                chip.remove()
        if path in self._attachments:
            self._attachments.remove(path)
        if not self._attachments:
            self._chips().display = False

    def clear_attachments(self) -> None:
        for chip in list(self.query("AttachmentChip")):
            chip.display = False
            chip.remove()
        self._attachments.clear()
        self._chips().display = False

    def on_attachment_chip_removed(self, event: AttachmentChip.Removed) -> None:
        self.remove_attachment(event.path)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        stream = self._stream()
        for child in list(stream.children):
            child.remove()
        self._stream_widget = None
        self._pending_tools.clear()
        self._active_phase = None
        self.clear_attachments()