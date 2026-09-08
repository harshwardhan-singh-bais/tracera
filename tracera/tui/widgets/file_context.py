"""
TRACERA File Context Panel — shows recently accessed/modified files.

Displays a compact list of files the agent has worked on, with
syntax highlighting indicators and click-to-preview.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical

from rich.text import Text


# File extension to language mapping for syntax indicators
_LANG_COLORS = {
    ".py": ("Py", "#3572A5"),
    ".js": ("Js", "#f1e05a"),
    ".ts": ("Ts", "#3178c6"),
    ".tsx": ("Tx", "#3178c6"),
    ".jsx": ("Jx", "#f1e05a"),
    ".rs": ("Rs", "#dea584"),
    ".go": ("Go", "#00ADD8"),
    ".java": ("Jv", "#b07219"),
    ".c": ("C ", "#555555"),
    ".cpp": ("C+", "#f34b7d"),
    ".h": ("H ", "#555555"),
    ".rb": ("Rb", "#701516"),
    ".php": ("Ph", "#4F5D95"),
    ".swift": ("Sw", "#F05138"),
    ".kt": ("Kt", "#A97BFF"),
    ".css": ("Cs", "#563d7c"),
    ".html": ("Ht", "#e34c26"),
    ".json": ("Js", "#292929"),
    ".yaml": ("Ym", "#cb171e"),
    ".yml": ("Ym", "#cb171e"),
    ".toml": ("Tm", "#9c4221"),
    ".md": ("Md", "#083fa1"),
    ".sql": ("Sq", "#e38c00"),
    ".sh": ("Sh", "#89e051"),
    ".bash": ("Sh", "#89e051"),
    ".zsh": ("Sh", "#89e051"),
}


class FileEntry:
    """Represents a file in the context panel."""

    def __init__(self, path: str, action: str = "read", line_count: int = 0) -> None:
        self.path = path
        self.action = action  # read, edit, write, search
        self.line_count = line_count
        self.access_count = 1

    @property
    def name(self) -> str:
        return Path(self.path).name or self.path

    @property
    def extension(self) -> str:
        return Path(self.path).suffix.lower()

    @property
    def lang_info(self) -> tuple[str, str]:
        return _LANG_COLORS.get(self.extension, ("??", "#9a9aa3"))

    @property
    def action_icon(self) -> str:
        icons = {
            "read": "📖",
            "edit": "✏️",
            "write": "📝",
            "search": "🔍",
        }
        return icons.get(self.action, "📄")


class FileContextPanel(Widget):
    """
    Compact panel showing recently accessed files.

    ┌─ FILES ─────────────────────────────────────┐
    │  📝 main.py        Py   +12 -4    (3x)      │
    │  📖 auth/middleware.py  Py          (1x)     │
    │  🔍 utils/helpers.py   Py          (2x)     │
    └─────────────────────────────────────────────┘
    """

    class FileClicked(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    MAX_FILES = 8

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._files: deque[FileEntry] = deque(maxlen=self.MAX_FILES)
        self._expanded: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="file-context-container"):
            yield Static(self._render_header(), id="file-context-header")
            yield Static(self._render_files(), id="file-context-list")

    def _render_header(self) -> Text:
        text = Text()
        text.append(" 📁 ", style="bold #6cb6ff")
        text.append("FILES", style="bold #6cb6ff")
        count = len(self._files)
        if count > 0:
            text.append(f"  ({count})", style="dim #9a9aa3")
        text.append("  ", style="dim")
        text.append("─" * 30, style="dim #3a3a4a")
        return text

    def _render_files(self) -> Text:
        text = Text()
        if not self._files:
            text.append("  No files accessed yet", style="dim #55555e")
            return text

        for entry in self._files:
            lang_label, lang_color = entry.lang_info
            action_icon = entry.action_icon

            text.append(f"\n  {action_icon} ", style="dim")
            text.append(entry.name[:24], style="bold #e0e0ff")

            text.append(f"  {lang_label}", style=f"bold {lang_color}")

            if entry.line_count > 0:
                text.append(f"  {entry.line_count}L", style="dim #9a9aa3")

            if entry.access_count > 1:
                text.append(f"  ({entry.access_count}x)", style="dim #6cb6ff")

            if self._expanded == entry.path:
                text.append(f"\n    └─ {entry.path}", style="dim #55555e")

        return text

    def add_file(self, path: str, action: str = "read", line_count: int = 0) -> None:
        """Add or update a file in the context panel."""
        for entry in self._files:
            if entry.path == path:
                entry.access_count += 1
                entry.action = action
                if line_count > 0:
                    entry.line_count = line_count
                self._refresh()
                return

        self._files.appendleft(FileEntry(path, action, line_count))
        self._refresh()

    def remove_file(self, path: str) -> None:
        self._files = deque(
            (e for e in self._files if e.path != path),
            maxlen=self.MAX_FILES,
        )
        self._refresh()

    def clear(self) -> None:
        self._files.clear()
        self._expanded = None
        self._refresh()

    def toggle_expand(self, path: str) -> None:
        if self._expanded == path:
            self._expanded = None
        else:
            self._expanded = path
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.query_one("#file-context-list", Static).update(self._render_files())
        except Exception:
            pass

    def on_click(self, event) -> None:
        """Handle click on file entries."""
        try:
            widget = event.widget
            if hasattr(widget, "id") and widget.id == "file-context-list":
                self.post_message(self.FileClicked(""))
        except Exception:
            pass
