"""
TRACERA CLI Entry Point.

Commands:
  tracera              → Open the interactive TUI
  tracera ask          → Single-shot agent query
  tracera index        → Index the workspace (future)
  tracera search       → Search code (future)
  tracera status       → Show system status
  tracera memory       → Manage persistent memory
  tracera tui          → Open TUI explicitly
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="tracera",
    help="TRACERA — Agentic Code Intelligence & Autonomous Coding Engine",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)

console = Console()


def _get_settings():
    from tracera.config import get_settings
    return get_settings()


def _setup() -> None:
    """Common initialisation: logging, dirs."""
    settings = _get_settings()
    settings.ensure_dirs()
    from tracera.logging import setup_logging
    setup_logging(
        level=settings.tracera_log_level,
        log_file=settings.tracera_log_file,
    )


def _build_agent(settings=None, workspace_path: Path | None = None):
    """Build a ReActAgent from current settings."""
    if settings is None:
        settings = _get_settings()

    from tracera.providers import create_provider
    from tracera.tools.registry import create_default_registry
    from tracera.workspace.sandbox import WorkspaceSandbox
    from tracera.agent.react_loop import ReActAgent

    ws_path = workspace_path or settings.tracera_workspace
    workspace = WorkspaceSandbox(
        ws_path,
        max_file_size=settings.tracera_indexing_max_file_size,
    )
    registry = create_default_registry(workspace)
    provider = create_provider(settings=settings)
    agent = ReActAgent(
        provider=provider,
        registry=registry,
        max_iterations=settings.tracera_max_iterations,
        max_tool_calls=settings.tracera_max_tool_calls,
        model=settings.tracera_default_model,
        temperature=settings.tracera_default_temperature,
        max_tokens=settings.tracera_default_max_tokens,
    )
    return agent, workspace, provider


# ── Default command: open TUI ─────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root directory."),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="LLM provider (openai/anthropic/gemini/ollama)."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model ID."),
    ] = None,
) -> None:
    """
    [bold cyan]TRACERA[/] — Agentic Code Intelligence & Autonomous Coding Engine.

    Run without arguments to open the interactive TUI.
    """
    if ctx.invoked_subcommand is not None:
        return

    # Open TUI by default
    _setup()
    settings = _get_settings()

    if provider:
        settings.tracera_default_provider = provider
    if model:
        settings.tracera_default_model = model

    workspace_path = (workspace or settings.tracera_workspace).resolve()

    from tracera.logging import print_banner
    print_banner()

    try:
        agent, ws, prov = _build_agent(settings, workspace_path)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        console.print("[dim]Check your .env file and API keys.[/]")
        raise typer.Exit(1)

    from tracera.agent.memory import AgentMemory
    from tracera.tui.app import TraceraTUI

    memory = AgentMemory(settings.memory_dir)
    tui = TraceraTUI(agent=agent, memory=memory, workspace_path=workspace_path)
    tui.run()


# ── ask ───────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    task: Annotated[str, typer.Argument(help="Task or question for the agent.")],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root."),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Stream the response."),
    ] = True,
) -> None:
    """Ask the agent a question or give it a task (non-interactive)."""
    _setup()
    settings = _get_settings()
    if provider:
        settings.tracera_default_provider = provider
    if model:
        settings.tracera_default_model = model

    workspace_path = (workspace or settings.tracera_workspace).resolve()

    from tracera.logging import print_banner
    print_banner()

    try:
        agent, ws, prov = _build_agent(settings, workspace_path)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)

    from tracera.agent.react_loop import AgentEventType

    async def _run():
        console.print(f"\n[bold cyan]Task:[/] {task}\n")
        async for event in await agent.run(task):
            match event.type:
                case AgentEventType.THINKING:
                    console.print(
                        f"[dim]◌  Thinking (iteration {event.iteration + 1})…[/]"
                    )
                case AgentEventType.TOOL_START:
                    console.print(
                        f"[green]⚙  {event.tool_name}[/]  "
                        + str(list((event.tool_args or {}).keys()))
                    )
                case AgentEventType.TOOL_END:
                    status = "✓" if event.tool_success else "✗"
                    color = "green" if event.tool_success else "red"
                    console.print(f"[{color}]{status}[/] {event.tool_name}")
                case AgentEventType.RESPONSE_COMPLETE:
                    from rich.markdown import Markdown
                    console.print("\n")
                    console.print(Panel(
                        Markdown(event.text or ""),
                        title="[bold cyan]TRACERA[/]",
                        border_style="cyan",
                    ))
                case AgentEventType.ERROR:
                    console.print(f"[bold red]Error:[/] {event.text}")

    asyncio.run(_run())


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
) -> None:
    """Show TRACERA system status."""
    _setup()
    settings = _get_settings()
    workspace_path = (workspace or settings.tracera_workspace).resolve()

    table = Table(title="TRACERA Status", border_style="cyan", show_header=True)
    table.add_column("Property", style="cyan bold")
    table.add_column("Value", style="white")

    table.add_row("Profile", settings.tracera_profile)
    table.add_row("Workspace", str(workspace_path))
    table.add_row("Data Directory", str(settings.tracera_data_dir))
    table.add_row("Default Provider", settings.tracera_default_provider)
    table.add_row("Default Model", settings.tracera_default_model)
    table.add_row("Max Iterations", str(settings.tracera_max_iterations))
    table.add_row("Max Tool Calls", str(settings.tracera_max_tool_calls))

    # Check API keys
    providers = []
    if settings.google_api_key:
        providers.append("gemini")
    if settings.groq_api_key:
        providers.append("groq")
    providers.append("ollama (local)")

    table.add_row("Available Providers", ", ".join(providers))

    # Memory stats
    from tracera.agent.memory import AgentMemory
    memory = AgentMemory(settings.memory_dir)
    table.add_row("Memory Entries", str(memory.count))

    console.print(table)


# ── memory ────────────────────────────────────────────────────────────────────

memory_app = typer.Typer(name="memory", help="Manage persistent agent memory.")
app.add_typer(memory_app)


@memory_app.command("list")
def memory_list(
    category: Annotated[
        Optional[str],
        typer.Option("--category", "-c", help="Filter by category."),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option("--query", "-q", help="Search query."),
    ] = None,
) -> None:
    """List memory entries."""
    _setup()
    settings = _get_settings()
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(settings.memory_dir)

    if query:
        entries = memory.retrieve(query, k=20)
    elif category:
        try:
            cat = MemoryCategory(category)
            entries = memory.get_by_category(cat)
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/]")
            raise typer.Exit(1)
    else:
        entries = list(memory._entries.values())

    if not entries:
        console.print("[dim]No memory entries found.[/]")
        return

    table = Table(border_style="purple", show_header=True)
    table.add_column("ID", style="dim", width=8)
    table.add_column("Category", style="bold magenta", width=16)
    table.add_column("Content", style="white")
    table.add_column("Importance", justify="right", width=10)

    for e in entries[:50]:
        table.add_row(
            e.id[:7],
            e.category.value,
            e.content[:80],
            f"{e.importance:.1f}",
        )

    console.print(table)
    console.print(f"[dim]Total: {memory.count} entries[/]")


@memory_app.command("add")
def memory_add(
    content: Annotated[str, typer.Argument(help="Memory content.")],
    category: Annotated[
        str,
        typer.Option("--category", "-c"),
    ] = "project_fact",
) -> None:
    """Add a memory entry."""
    _setup()
    settings = _get_settings()
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(settings.memory_dir)
    try:
        cat = MemoryCategory(category)
    except ValueError:
        console.print(f"[red]Unknown category: {category}[/]")
        raise typer.Exit(1)

    entry = memory.add(content, cat, source="cli")
    console.print(f"[green]✓[/] Added memory entry [cyan]{entry.id[:7]}[/]")


@memory_app.command("clear")
def memory_clear(
    confirm: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation."),
    ] = False,
) -> None:
    """Clear all memory entries."""
    _setup()
    settings = _get_settings()
    from tracera.agent.memory import AgentMemory

    memory = AgentMemory(settings.memory_dir)
    if not confirm:
        typer.confirm(f"Delete all {memory.count} memory entries?", abort=True)

    memory._entries.clear()
    memory._save()
    console.print("[green]✓[/] Memory cleared.")


# ── tui ───────────────────────────────────────────────────────────────────────

@app.command()
def tui(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m"),
    ] = None,
) -> None:
    """Open the interactive TUI (same as running `tracera` with no arguments)."""
    # Delegate to main with no subcommand
    console.print("[dim]Opening TUI...[/]")
    ctx = typer.Context(app)
    main(ctx=ctx, workspace=workspace, provider=provider, model=model)


if __name__ == "__main__":
    app()
