"""
TRACERA TUI Theme System — swappable accent presets.

Three presets (Claude orange, Crush magenta, Nord blue) drive the TUI's
colors through Textual's CSS-variable mechanism: ``TraceraTUI.get_css_variables()``
injects ``$accent``/``$success``/… into the stylesheet, and every widget that
renders Rich ``Text`` inline reads the active preset from :func:`get_theme`
so the two layers stay in sync.

Persistence: the chosen preset name is stored in ``<data-dir>/tui_theme.json``
and restored on the next launch. ``/theme`` cycles through presets live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.text import Text


@dataclass(frozen=True)
class Theme:
    """One color preset. All values are hex colors (no ``#`` needed here)."""

    name: str
    label: str
    accent: str       # brand / prompt / active states
    secondary: str    # info rows, search results
    success: str
    error: str
    warning: str
    meta: str         # purple metadata
    muted: str        # dim text
    faint: str        # barely-there borders/hints
    text: str
    surface: str      # subtle hover background
    border: str
    scrollbar: str
    diff_add_bg: str = "#12291a"
    diff_del_bg: str = "#2d1214"
    diff_hunk_bg: str = "#1a1230"

    def css_variables(self) -> dict[str, str]:
        """Textual CSS variables derived from this preset."""
        return {
            "t-accent": f"#{self.accent}",
            "t-secondary": f"#{self.secondary}",
            "t-success": f"#{self.success}",
            "t-error": f"#{self.error}",
            "t-warning": f"#{self.warning}",
            "t-meta": f"#{self.meta}",
            "t-muted": f"#{self.muted}",
            "t-faint": f"#{self.faint}",
            "t-text": f"#{self.text}",
            "t-surface": f"#{self.surface}",
            "t-border": f"#{self.border}",
            "t-scroll": f"#{self.scrollbar}",
        }


#: ── Presets ──────────────────────────────────────────────────────────────────
#: claude  — the original warm Claude Code orange (default)
#: crush   — Crush-style hot magenta/pink
#: nord    — cool nordic blue

THEMES: dict[str, Theme] = {
    "claude": Theme(
        name="claude",
        label="Claude (warm orange)",
        accent="da8548",
        secondary="6cb6ff",
        success="4ac26b",
        error="f47067",
        warning="ffd700",
        meta="d2a8ff",
        muted="6a6a80",
        faint="3a3a4a",
        text="ebebf0",
        surface="1e1e2a",
        border="2d2d3d",
        scrollbar="da8548",
    ),
    "crush": Theme(
        name="crush",
        label="Crush (hot magenta)",
        accent="ff6b9d",
        secondary="7c9dff",
        success="5fd7a0",
        error="ff5f7e",
        warning="ffcc66",
        meta="c792ea",
        muted="7a7a8c",
        faint="3d3d50",
        text="f2f2f7",
        surface="221826",
        border="3a2a40",
        scrollbar="ff6b9d",
        diff_add_bg="#122a20",
        diff_del_bg="#2d1018",
        diff_hunk_bg="#1e1030",
    ),
    "nord": Theme(
        name="nord",
        label="Nord (cool blue)",
        accent="88c0d0",
        secondary="81a1c1",
        success="a3be8c",
        error="bf616a",
        warning="ebcb8b",
        meta="b48ead",
        muted="616e7c",
        faint="3b4252",
        text="eceff4",
        surface="2e3440",
        border="3b4252",
        scrollbar="88c0d0",
        diff_add_bg="#22303c",
        diff_del_bg="#33232a",
        diff_hunk_bg="#2e3440",
    ),
}

DEFAULT_THEME = "claude"

#: Module-level active preset (per-process; the App sets it on mount).
_active_theme: Theme = THEMES[DEFAULT_THEME]


def get_theme() -> Theme:
    """The active preset — used by widgets that render Rich Text inline."""
    return _active_theme


def set_theme(name: str) -> Theme:
    """Switch the active preset (process-wide). Returns the new preset."""
    global _active_theme
    _active_theme = THEMES.get(name, THEMES[DEFAULT_THEME])
    return _active_theme


def next_theme() -> Theme:
    """Cycle to the next preset in declaration order."""
    names = list(THEMES)
    current = _active_theme.name
    idx = names.index(current) if current in names else -1
    return set_theme(names[(idx + 1) % len(names)])


def theme_label() -> Text:
    """A one-line Rich Text summary: `◆ theme: Crush (hot magenta)`."""
    t = Text()
    t.append(" ◆ ", style=f"bold #{_active_theme.accent}")
    t.append("theme: ", style="dim")
    t.append(_active_theme.label, style=f"bold #{_active_theme.accent}")
    return t


# ── Persistence ──────────────────────────────────────────────────────────────

def theme_store_path(data_dir: Path) -> Path:
    return Path(data_dir) / "tui_theme.json"


def load_saved_theme(data_dir: Path) -> str:
    """Return the persisted preset name (or the default). Never raises."""
    try:
        path = theme_store_path(data_dir)
        if path.exists():
            name = json.loads(path.read_text(encoding="utf-8")).get("theme")
            if name in THEMES:
                return str(name)
    except Exception:
        pass
    return DEFAULT_THEME


def save_theme(data_dir: Path, name: str) -> bool:
    """Persist the preset name; returns True on success."""
    if name not in THEMES:
        return False
    try:
        path = theme_store_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": name}), encoding="utf-8")
        return True
    except Exception:
        return False


__all__ = [
    "Theme",
    "THEMES",
    "DEFAULT_THEME",
    "get_theme",
    "set_theme",
    "next_theme",
    "theme_label",
    "load_saved_theme",
    "save_theme",
]
