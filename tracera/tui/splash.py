"""
TRACERA Splash Screen — "decrypt/materialize" boot animation.

The wordmark resolves from random block-glyph noise, character-by-character,
in a soft left-to-right diagonal wipe: dim grey noise cells lock into the
accent gradient (``#6cb6ff`` → ``#d2a8ff``), with a few flickers just behind
the resolve front so the wipe doesn't look mechanical. A subtitle appears
beneath the finished word, then the main TUI takes over.

Skips (straight to the resolved frame) when:
  - ``TRACERA_NO_ANIMATION`` is set, or
  - ``NO_COLOR`` is set, or
  - the app is running headless (``run_test`` / CI / piped output) — in that
    case the screen pops immediately so automated tests are never delayed.

Any keypress or click skips immediately. All timing lives in
:class:`SplashConfig` so the effect is tweakable in one place.
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Static

from tracera.tui.theme import get_theme

#: Glyphs a not-yet-resolved cell cycles through while "encrypted".
_NOISE_GLYPHS = "░▒▓█▚▞▟▙#%&8@"

#: Reduced-motion / non-interactive opt-outs.
_SKIP_ENV_VARS = ("TRACERA_NO_ANIMATION", "NO_COLOR")


class SplashConfig:
    """Tuning knobs for the decrypt effect (seconds / columns)."""

    def __init__(
        self,
        *,
        duration: float = 1.5,        # total resolve time
        tick_rate: float = 1 / 30,    # ~30 fps
        front_width: float = 7.0,     # soft wipe front, in columns
        flicker_chance: float = 0.18, # per-cell flicker chance behind the front
        subtitle_hold: float = 0.55,  # pause after the subtitle appears
    ) -> None:
        self.duration = duration
        self.tick_rate = tick_rate
        self.front_width = front_width
        self.flicker_chance = flicker_chance
        self.subtitle_hold = subtitle_hold


def _is_headless(app) -> bool:
    driver = getattr(app, "_driver", None)
    return bool(driver is not None and getattr(driver, "is_headless", False))


def animation_enabled(app) -> bool:
    """False when env/headless says: don't animate, just show the word."""
    if any(os.environ.get(var) for var in _SKIP_ENV_VARS):
        return False
    return not _is_headless(app)


def _word_grid() -> tuple[list[list[str]], int]:
    """The wordmark as a square cell grid, plus its widest row width."""
    from tracera.logging.logger import _PIXEL_ROWS

    grid = [list(row) for row in _PIXEL_ROWS]
    width = max(len(row) for row in grid)
    for row in grid:
        row.extend([" "] * (width - len(row)))
    return grid, width


class SplashScreen(Screen):
    """Full-screen decrypt/materialize splash; pops itself when done."""

    DEFAULT_CSS = """
    SplashScreen {
        background: transparent;
    }
    #splash-body {
        width: 1fr;
        height: 1fr;
        align: center top;
        padding-top: 4;
    }
    #splash-word {
        width: auto;
        height: auto;
    }
    #splash-subtitle {
        width: auto;
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        config: SplashConfig | None = None,
        on_complete: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = config or SplashConfig()
        self.on_complete = on_complete
        self._grid, self._width = _word_grid()
        self._rows = len(self._grid)
        self._rng = random.Random()
        # Per-cell resolve deadline (seconds since mount): the left edge
        # resolves first; jitter keeps the wipe front soft, not rigid.
        start = time.monotonic()
        jitter_scale = self.cfg.front_width / max(1.0, self._width / 3)
        self._resolve_at: list[list[float]] = [
            [
                start
                + max(0.0, (c / max(1, self._width - 1)) * self.cfg.duration
                      + self._rng.uniform(-0.35, 0.35) * jitter_scale)
                if self._grid[r][c] != " " else 0.0
                for c in range(self._width)
            ]
            for r in range(self._rows)
        ]
        self._current: list[list[str]] = [
            [self._rng.choice(_NOISE_GLYPHS) if ch != " " else " " for ch in row]
            for row in self._grid
        ]
        self._timer: Timer | None = None
        self._finished = False

    # ── Layout ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-body"):
            yield Static("", id="splash-word")
            yield Static("", id="splash-subtitle")

    def on_mount(self) -> None:
        if animation_enabled(self.app):
            self._render_frame()
            self._timer = self.set_interval(self.cfg.tick_rate, self._tick)
        else:
            self._finish()

    # ── Animation ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """One animation frame: resolve, flicker, cycle noise, render."""
        if self._finished:
            return
        now = time.monotonic()
        near_front = self.cfg.front_width / max(1, self._width) * self.cfg.duration
        all_done = True
        for r in range(self._rows):
            for c in range(self._width):
                if self._grid[r][c] == " ":
                    continue
                resolve_at = self._resolve_at[r][c]
                if now >= resolve_at:
                    self._current[r][c] = self._grid[r][c]
                else:
                    all_done = False
                    if resolve_at - now < near_front and self._rng.random() < self.cfg.flicker_chance:
                        # Flicker: briefly show the real glyph before locking.
                        self._current[r][c] = self._grid[r][c]
                    else:
                        self._current[r][c] = self._rng.choice(_NOISE_GLYPHS)
        self._render_frame()
        if all_done:
            self._finish()

    def _render_frame(self) -> None:
        """Paint the grid: gradient for resolved cells, dim grey for noise."""
        theme = get_theme()
        word = self.query_one("#splash-word", Static)
        resolved_color = [f"#{theme.secondary}", f"#{theme.meta}"]
        noise_color = f"#{theme.faint}"
        lines: list[str] = []
        for r in range(self._rows):
            # Top rows lean blue, bottom rows lean purple.
            color = resolved_color[0] if r / max(1, self._rows - 1) < 0.5 else resolved_color[1]
            parts: list[str] = []
            for c in range(self._width):
                ch = self._current[r][c]
                if ch == " ":
                    parts.append(" ")
                elif ch == self._grid[r][c]:
                    parts.append(f"[bold {color}]{ch}[/]")
                else:
                    parts.append(f"[{noise_color}]{ch}[/]")
            lines.append("".join(parts))
        word.update(Text.from_markup("\n".join(lines)))

    def _finish(self) -> None:
        """Stop animating, show the resolved word + subtitle, then pop."""
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._current = [row[:] for row in self._grid]
        self._render_frame()
        self._show_subtitle()
        if _is_headless(self.app):
            self._pop()  # never delay automated tests
        else:
            self.set_timer(self.cfg.subtitle_hold, self._pop)

    def _show_subtitle(self) -> None:
        theme = get_theme()
        self.query_one("#splash-subtitle", Static).update(
            Text.from_markup(
                f"  [bold #{theme.meta}]◆ Agentic Code Intelligence & "
                f"Autonomous Coding Engine[/]  [dim #{theme.faint}]·[/]  "
                f"[dim #{theme.muted}]v2.0[/]\n"
                f"  [dim #{theme.faint}]press any key to continue[/]"
            )
        )

    def _pop(self) -> None:
        callback = self.on_complete
        self.on_complete = None
        if self.app.screen is self:
            self.app.pop_screen()
        if callback is not None:
            callback()

    # ── Skip on any input ────────────────────────────────────────────────────

    def on_key(self, event: events.Key) -> None:
        event.stop()
        event.prevent_default()
        self._finish()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self._finish()
