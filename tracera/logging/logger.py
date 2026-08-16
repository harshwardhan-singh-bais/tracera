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


def banner_text() -> str:
    """The TRACERA ASCII art banner, as Rich markup."""
    return (
        "\n"
        "[bold bright_cyan]"
        " ████████╗██████╗  █████╗  ██████╗███████╗██████╗  █████╗ \n"
        " ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗\n"
        "    ██║   ██████╔╝███████║██║     █████╗  ██████╔╝███████║\n"
        "    ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗██╔══██║\n"
        "    ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║██║  ██║\n"
        "    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝[/]\n"
        "[dim]   Agentic Code Intelligence & Autonomous Coding Engine[/]\n"
    )


def print_banner() -> str:
    """Print the TRACERA ASCII art banner to the console; returns the text."""
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
