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
from pathlib import Path

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Input
from textual.containers import Horizontal, Vertical, ScrollableContainer
from rich.text import Text

from tracera.tui.diffutil import is_image

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


# ── Phase marker rows (full agent-loop visualization) ────────────────────────

class PhaseRow(Static):
    """A premium phase marker with dynamic spinner selection based on phase:

        ◐ Planning     ← active (animated circle)
        ◇ Planning     ← superseded by the next phase (dim)
    """

    # Map phases to appropriate spinners - all consistent blue theme
    _PHASE_CONFIG = {
        "planning": ("thinking", "#6cb6ff"),  # Circle rotation - planning
        "thinking": ("pulse", "#6cb6ff"),     # Pulsing - thinking
        "searching": ("running", "#6cb6ff"),  # Classic - searching
        "indexing": ("loading", "#6cb6ff"),   # Progress wave - indexing
        "running": ("blocks", "#6cb6ff"),     # Filling block - executing
        "generating": ("wave", "#6cb6ff"),    # Vertical wave - generating
        "writing": ("arrows", "#6cb6ff"),     # Spinning arrows - writing
    }

    def __init__(self, label: str, phase_type: str = "thinking", **kwargs) -> None:
        super().__init__(**kwargs)
        self.phase_label = label
        self.phase_type = phase_type.lower()
        self._frame = 0
        self._spinning = True
        # Get appropriate spinner and color for this phase
        spinner_key, self._color = self._PHASE_CONFIG.get(self.phase_type, ("thinking", "#6cb6ff"))
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
            text.append(
                f" {self._spinner[self._frame % len(self._spinner)]} ",
                style=f"bold {self._color}",
            )
            text.append(self.phase_label, style=f"bold {self._color}")
        else:
            text.append(" ◇ ", style="dim #9a9aa3")
            text.append(self.phase_label, style="dim #9a9aa3")
        return text


# ── Inline tool rows ─────────────────────────────────────────────────────────

_DIFF_STYLES = {
    "add": "#4ac26b",
    "del": "#f47067",
    "ctx": "#9a9aa3",
    "hunk": "#d2a8ff",
    "ellipsis": "#55555e",
}
_DIFF_PREFIX = {"add": "+", "del": "-", "hunk": "  ", "ctx": "  ", "ellipsis": "  "}


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

    # Tool category colors for timeline visualization
    _TOOL_CATEGORIES = {
        # Search/analysis tools - blue
        "search_code": "#6cb6ff",
        "find_symbol": "#6cb6ff",
        "find_definition": "#6cb6ff",
        "grep": "#6cb6ff",
        "get_context": "#6cb6ff",
        "get_dependencies": "#6cb6ff",
        "find_references": "#6cb6ff",
        # Read tools - yellow
        "read_file": "#ffd700",
        "list_dir": "#ffd700",
        # Write/edit tools - green
        "write_file": "#4ac26b",
        "edit_file": "#4ac26b",
        "delete_file": "#4ac26b",
        # Command tools - purple
        "run_command": "#d2a8ff",
        # Git tools - orange
        "git": "#ff9f43",
        # Memory tools - pink
        "memory": "#ff6b6b",
        # Test tools - cyan
        "test": "#00d2d3",
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
        """Get color based on tool category."""
        return self._TOOL_CATEGORIES.get(self.tool_name, "#dcdcf5")

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

    # ── Rendering ────────────────────────────────────────────────────────────

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

    def render(self) -> Text:
        text = Text()
        tool_color = self._get_tool_color()
        
        if self._spinning:
            text.append(f" {self._spinner[self._frame]} ", style=f"bold {tool_color}")
            text.append(self.tool_name, style=f"bold {tool_color}")
            if self.verbose and self.args_str:
                text.append(f"  {self.args_str}", style="dim #8a8a96")
            return text

        icon = "✓" if self.success else "✗"
        icon_style = "bold #4ac26b" if self.success else "bold #f47067"
        text.append(f" {icon} ", style=icon_style)
        text.append(self.tool_name, style=f"bold {tool_color}")

        if self.diff_path and self.success:
            # Code-gen summary: 📝 path  +N -M
            text.append(f"  📝 {self.diff_path}", style="bold #d2a8ff")
            if self._diff_added or self._diff_removed:
                text.append(f"  +{self._diff_added}", style="bold #4ac26b")
                text.append(f" -{self._diff_removed}", style="bold #f47067")
            elif self.verbose and self.args_str:
                text.append(f"  {self.args_str}", style="dim #8a8a96")
        elif self.verbose and self.args_str:
            text.append(f"  {self.args_str}", style="dim #8a8a96")

        if self.duration_ms is not None:
            # Color-coded duration bar
            if self.duration_ms < 100:
                dur_color = "#4ac26b"  # Fast - green
            elif self.duration_ms < 500:
                dur_color = "#ffd700"  # Medium - yellow
            else:
                dur_color = "#f47067"  # Slow - red
            text.append(f"  {self.duration_ms:.0f}ms", style=f"dim {dur_color}")

        if not self.success and self.output:
            preview = self.output.strip().splitlines()
            first = preview[0][:90] if preview else ""
            if first:
                text.append(f"\n   └ {first}", style="dim #f47067")

        if self.expanded and self._diff_lines:
            text.append(self._render_diff())
        return text

    def _render_diff(self) -> Text:
        d = Text("\n")
        d.append(f"   ▾ {len(self._diff_lines)} lines", style="dim #9a9aa3")
        for kind, line in self._diff_lines:
            style = _DIFF_STYLES.get(kind, "dim #9a9aa3")
            prefix = _DIFF_PREFIX.get(kind, "  ")
            if len(line) > 140:
                line = line[:137] + "…"
            d.append(f"\n   {prefix} {line}", style=style)
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
        icon = "🖼" if is_image(self.path) else "📄"
        t = Text()
        t.append(f" {icon} ", style="#6cb6ff")
        t.append(name, style="bold #dcdcf5")
        if self.warning:
            t.append(" [!]", style="bold #d4a72c")
        t.append("  ✕", style="dim #f47067")
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

    # Phase configurations for the loader pill - consistent blue theme
    _LOADER_PHASES = {
        "planning": ("◐", "#6cb6ff", "Planning task..."),
        "thinking": ("◉", "#6cb6ff", "Processing..."),
        "searching": ("⠋", "#6cb6ff", "Searching codebase..."),
        "indexing": ("▰▱▱▱▱", "#6cb6ff", "Building index..."),
        "running": ("█", "#6cb6ff", "Executing..."),
        "generating": ("⎽", "#6cb6ff", "Generating response..."),
        "writing": ("→", "#6cb6ff", "Writing changes..."),
        "done": ("✓", "#6cb6ff", "Complete!"),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phase = "thinking"
        self._frame = 0
        self._full_spinner = _SPINNERS["pulse"]

    def compose(self) -> ComposeResult:
        yield Static("◉", id="loader-icon")
        yield Static("Processing...", id="loader-label")
        yield Static(" ● ", id="loader-stop")

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
    "idle": ("●", "#9a9aa3"),
    "active": ("●", "#4ac26b"),
    "thinking": ("⠋", "#d4a72c"),
    "running": ("⠋", "#6cb6ff"),
    "done": ("●", "#4ac26b"),
    "error": ("●", "#f47067"),
}


class InlineStatus(Static):
    """
    Premium system status line with comprehensive metrics and feature indicators:

        ● ACTIVE  session 8f2c7b · gemini-pro · 12 tools · 5 iter · 2.4k tok · 01:23
        [◉ MEM] [◉ RET] [◎ RAG] [◉ MCP]  Features: Memory, Retrieval, RAG, MCP
    """

    # Feature status indicators - all consistent blue theme
    _FEATURES = {
        "memory": ("MEMORY", "#6cb6ff"),  # Memory layer
        "retrieval": ("RETR", "#6cb6ff"),  # Code retrieval
        "rag": ("RAG", "#6cb6ff"),        # RAG active
        "mcp": ("MCP", "#6cb6ff"),        # MCP server
        "index": ("IDX", "#6cb6ff"),      # Code index
        "sandbox": ("SBX", "#6cb6ff"),    # Sandbox
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
        # Store additional metrics
        self._memory_hits = memory_hits or 0
        self._retrieval_hits = retrieval_hits or 0
        self._cost_estimate = cost_estimate or 0.0
        self._refresh()

    def _elapsed_text(self) -> str:
        ms = self._elapsed_ms
        if self._started_at is not None:
            ms += (time.time() - self._started_at) * 1000
        total = int(ms // 1000)
        return f"{total // 60}:{total % 60:02d}"

    def _render_features(self) -> Text:
        """Render feature status indicators."""
        feat_text = Text()
        feat_text.append("  [", style="dim #55555e")
        for feat, (label, color) in self._FEATURES.items():
            active = self._feature_status[feat]
            icon = "◉" if active else "◎"
            feat_text.append(f"{icon} {label}", style=f"{'bold' if active else 'dim'} {color}")
            if feat != list(self._FEATURES.keys())[-1]:
                feat_text.append(" ", style="dim #55555e")
        feat_text.append("]", style="dim #55555e")
        return feat_text

    def _refresh(self) -> None:
        glyph, color = _STATE_GLYPHS.get(self._state, ("●", "#9a9aa3"))
        if self._state in ("thinking", "running"):
            glyph = self._state_spinners[self._frame % len(self._state_spinners)]
        
        text = Text()
        # Main status
        text.append(f" {glyph} ", style=f"bold {color}")
        text.append(self._state.upper(), style=f"bold {color}")
        
        # Core metrics
        text.append(f"   session {self._session[:8]}", style="dim #9a9aa3")
        text.append(f" · {self._model[:14]}", style="dim #9a9aa3")
        text.append(f" · {self._tool_calls} tools", style="dim #9a9aa3")
        text.append(f" · {self._iterations} iter", style="dim #9a9aa3")
        
        # Token progress bar
        text.append_text(self._render_token_bar())
        
        # Enhanced metrics
        if hasattr(self, '_memory_hits') and self._memory_hits > 0:
            text.append(f" · mem:{self._memory_hits}", style="dim #d2a8ff")
        if hasattr(self, '_retrieval_hits') and self._retrieval_hits > 0:
            text.append(f" · ret:{self._retrieval_hits}", style="dim #4ac26b")
        if hasattr(self, '_cost_estimate') and self._cost_estimate > 0:
            text.append(f" · ${self._cost_estimate:.3f}", style="dim #ffd700")
        
        # Elapsed time
        text.append(f" · {self._elapsed_text()}", style="dim #d2a8ff")
        
        # Feature status indicators on new line
        text.append("\n")
        text.append_text(self._render_features())
        
        self.update(text)

    def _render_token_bar(self) -> Text:
        """Render a compact token usage progress bar."""
        text = Text()
        
        # Assume 100k token context limit for visualization
        max_tokens = 100_000
        current = min(self._tokens, max_tokens)
        ratio = current / max_tokens if max_tokens > 0 else 0
        
        # Choose color based on usage
        if ratio < 0.5:
            bar_color = "#4ac26b"  # Green
        elif ratio < 0.8:
            bar_color = "#ffd700"  # Yellow
        else:
            bar_color = "#f47067"  # Red
        
        # Format token count
        if self._tokens >= 1_000_000:
            tok_str = f"{self._tokens / 1_000_000:.1f}M"
        elif self._tokens >= 1_000:
            tok_str = f"{self._tokens / 1_000:.1f}k"
        else:
            tok_str = str(self._tokens)
        
        text.append(f" · {tok_str} tok", style=f"dim {bar_color}")
        
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

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-panel"):
            with ScrollableContainer(id="stream"):
                pass
            yield InlineStatus(id="status-line")
            yield LoaderPill(id="loader-pill")
            with Vertical(id="agent-input-area"):
                with Horizontal(id="attach-chips"):
                    pass
                with Horizontal(id="agent-input-bar"):
                    yield Static("＋", id="attach-button")
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
        self._append(MessageWidget("user", text))

    def add_assistant_message(self, text: str) -> None:
        self._append(MessageWidget("assistant", text))

    def add_error(self, text: str) -> None:
        self._append(MessageWidget("error", text))

    def add_banner(self, text: str) -> None:
        """Render the CLI banner at the top of the app's first frame."""
        self._append(Static(text, markup=True, classes="banner-block"))

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
    ) -> CollapsibleRow:
        row = CollapsibleRow(title, body, prefix=prefix)
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