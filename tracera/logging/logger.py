"""
TRACERA Rich-powered structured logger.

Features:
- Coloured console output via Rich
- Optional file logging (plain text)
- Module-level log() shortcut
- Tracera-styled log prefix with neon colours
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# ── TRACERA colour theme ──────────────────────────────────────────────────────

_THEME = Theme(
    {
        "tracera.debug": "dim cyan",
        "tracera.info": "bold bright_cyan",
        "tracera.warning": "bold yellow",
        "tracera.error": "bold red",
        "tracera.critical": "bold white on red",
        "tracera.tool": "bold magenta",
        "tracera.agent": "bold bright_green",
        "tracera.llm": "bold bright_blue",
        "tracera.memory": "bold orchid",
        "tracera.plan": "bold gold1",
        "tracera.success": "bold bright_green",
    }
)

_console = Console(theme=_THEME, stderr=True)
_file_console: Console | None = None


# ── Root logger setup ─────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    show_path: bool = False,
) -> None:
    """
    Configure TRACERA logging.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_file: Optional path to write plain-text logs.
        show_path: Show source file/line in Rich output.
    """
    global _file_console

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Rich handler for console
    rich_handler = RichHandler(
        console=_console,
        show_path=show_path,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(numeric_level)

    handlers: list[logging.Handler] = [rich_handler]

    # File handler (plain text)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a TRACERA module."""
    return logging.getLogger(f"tracera.{name}")


# ── Rich console helpers ──────────────────────────────────────────────────────

def get_console() -> Console:
    """Return the shared Rich console."""
    return _console


# ── Pixel-wordmark banner ───────────────────────────────────────────────────

#: Lowercase "tracera" in a chunky 5-row pixel font (4 columns per glyph).
#: No border, no box — the word IS the hero.
_PIXEL_FONT: dict[str, list[str]] = {
    "t": ["████", "  █ ", "  █ ", "  █ ", "  ██"],
    "r": ["████", "█   ", "█   ", "█   ", "█   "],
    "a": ["████", "   █", "████", "█  █", "████"],
    "c": ["████", "█   ", "█   ", "█   ", "████"],
    "e": ["████", "█  █", "████", "█   ", "████"],
}
_PIXEL_WORD = "tracera"
_PIXEL_ROWS: list[str] = [
    "".join(_PIXEL_FONT[ch][row] + (" " if i < len(_PIXEL_WORD) - 1 else "")
           for i, ch in enumerate(_PIXEL_WORD))
    for row in range(5)
]

#: Left-to-right gradient applied per letter (accent blue → prompt purple).
_WORD_COLORS = [
    "#6cb6ff", "#7db9ff", "#8fc0ff", "#a3c9ff", "#b8ccff", "#c5b0ff", "#d2a8ff",
]
_GHOST = "#232330"  # unlit pixels during materialization
_SWEEP = "#ffffff"   # scanline highlight


def _word_frame_markup(lit: set[tuple[int, int]] | None = None,
                       sweep_col: int | None = None) -> str:
    """Rich-markup frame of the pixel word.

    ``lit=None`` → every pixel on (final state). Otherwise only pixels in the
    set are lit and the rest render as dim ghost pixels (loading effect).
    ``sweep_col`` highlights one pixel column in white (scanline sweep).
    """
    lines: list[str] = []
    for r, row in enumerate(_PIXEL_ROWS):
        parts: list[str] = ["  "]  # small left margin
        col = 0
        last = len(_PIXEL_WORD) - 1
        for i, ch in enumerate(_PIXEL_WORD):
            color = _WORD_COLORS[i % len(_WORD_COLORS)]
            for pc in _PIXEL_FONT[ch][r]:
                if pc == " ":
                    parts.append(" ")
                else:
                    on = lit is None or (r, col) in lit
                    if sweep_col is not None and col == sweep_col and on:
                        parts.append(f"[bold {_SWEEP}]█[/]")
                    elif on:
                        parts.append(f"[{color}]█[/]")
                    else:
                        parts.append(f"[{_GHOST}]{pc}[/]")
                col += 1
            if i < last:
                parts.append(" ")  # inter-letter gap
        lines.append("".join(parts))
    return "\n".join(lines)


def _word_pixel_positions() -> list[tuple[int, int]]:
    """All (row, col) pixel coordinates in the word, row-major."""
    positions: list[tuple[int, int]] = []
    for r, row in enumerate(_PIXEL_ROWS):
        for c, ch in enumerate(row):
            if ch == "█":
                positions.append((r, c))
    return positions


def _materialize_frames(steps: int = 12) -> list[str]:
    """Progressive pixel-materialization frames (deterministic seed)."""
    import random
    rng = random.Random(0x7ACEA)  # fixed → same assemble order every run
    order = _word_pixel_positions()
    rng.shuffle(order)
    frames: list[str] = []
    per_step = max(1, len(order) // steps)
    lit: set[tuple[int, int]] = set()
    for i in range(0, len(order), per_step):
        lit.update(order[i:i + per_step])
        frames.append(_word_frame_markup(lit=lit))
    frames.append(_word_frame_markup())  # all pixels on
    return frames


def _sweep_frames(bands: int = 7) -> list[str]:
    """Bright scanline sweeping across the finished wordmark."""
    word_width = len(_PIXEL_ROWS[0])
    frames: list[str] = []
    for k in range(bands):
        col = int(k * (word_width - 1) / max(1, bands - 1))
        frames.append(_word_frame_markup(sweep_col=col))
    return frames


def _tagline_block() -> str:
    """Tagline + capability strips under the wordmark (no border)."""
    return (
        "\n"
        "[bold #d2a8ff]  ◆ Agentic Code Intelligence & Autonomous Coding Engine[/]\n"
        "[dim #6cb6ff]  v2.0[/]  [dim #9a9aa3]│[/]  [bold #4ac26b]●[/] [bold #e0e0ff]40+ Tools[/]  "
        "[dim #9a9aa3]│[/]  [bold #d2a8ff]◆[/] [bold #e0e0ff]Memory[/]  [dim #9a9aa3]│[/]  "
        "[bold #6cb6ff]◉[/] [bold #e0e0ff]RAG[/]  [dim #9a9aa3]│[/]  [dim #55555e]Ready[/]\n"
    )


def banner_text() -> str:
    """The final static banner frame — pixel "tracera" wordmark, no border.

    This is what the TUI reproduces as its first frame; the animated
    materialization lives in :func:`animate_banner`.
    """
    return "\n" + _word_frame_markup() + "\n" + _tagline_block()


def animate_banner(console: Console | None = None, *, duration: float = 1.1) -> None:
    """Play the pixel loader: scattered pixels assemble into the wordmark,
    a scanline sweeps across, then the frame settles.

    Uses Rich ``Live`` with ``transient=True`` so the animation erases itself
    and only the final static banner (printed by the caller) remains.
    Non-TTY consoles skip straight to the static frame.
    """
    from rich.live import Live
    from rich.text import Text
    import time

    console = console or _console
    if not console.is_terminal:
        return
    try:
        mat_frames = _materialize_frames()
        sweep = _sweep_frames()
        mat_delay = max(0.02, (duration * 0.7) / max(1, len(mat_frames)))
        sweep_delay = max(0.02, (duration * 0.3) / max(1, len(sweep)))
        with Live(
            Text.from_markup(_word_frame_markup(lit=set())),
            console=console,
            refresh_per_second=30,
            transient=True,
        ) as live:
            for frame in mat_frames:
                live.update(Text.from_markup(frame))
                time.sleep(mat_delay)
            for frame in sweep:
                live.update(Text.from_markup(frame))
                time.sleep(sweep_delay)
    except Exception:
        # Animation is cosmetic — any failure just means no animation.
        return


def print_banner() -> str:
    """Play the pixel loader, then print the final banner; returns its text."""
    animate_banner()
    _console.print(banner_text(), highlight=False)
    return banner_text()


# ── Styled log helpers (used by non-logging code for visual output) ───────────

def log_tool(name: str, args: dict[str, Any]) -> None:
    _console.print(f"[tracera.tool]⚙  {name}[/] {_fmt_args(args)}")


def log_agent(message: str) -> None:
    _console.print(f"[tracera.agent]◈  AGENT[/] {message}")


def log_llm(provider: str, model: str, tokens: int | None = None) -> None:
    tok = f" · {tokens:,} tok" if tokens else ""
    _console.print(f"[tracera.llm]◉  LLM[/] [dim]{provider}/{model}{tok}[/]")


def log_memory(action: str, key: str) -> None:
    _console.print(f"[tracera.memory]◈  MEM[/] [dim]{action}[/] {key}")


def log_plan(step: int, total: int, text: str) -> None:
    _console.print(f"[tracera.plan]◈  PLAN[/] [{step}/{total}] {text}")


def log_success(message: str) -> None:
    _console.print(f"[tracera.success]✓  {message}[/]")


def log_error_panel(title: str, message: str) -> None:
    from rich.panel import Panel
    _console.print(
        Panel(message, title=f"[bold red]{title}[/]", border_style="red")
    )


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in list(args.items())[:3]:
        s = repr(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"[dim]{k}[/]={s}")
    return " ".join(parts)