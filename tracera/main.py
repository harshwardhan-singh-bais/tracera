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


def _build_agent(settings=None, workspace_path: Path | None = None, retrieval_pipeline=None):
    """Build a ReActAgent from current settings.

    If retrieval_pipeline is provided, the agent is extended with code-search
    tools (Phase 28) and a retrieval-aware system prompt (Phase 31).
    """
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

    # Phase 28: extend registry with code-search tools if index is available
    if retrieval_pipeline is not None:
        from tracera.tools.registry import extend_registry_with_retrieval
        _, symbol_retriever, expander, *_ = retrieval_pipeline
        extend_registry_with_retrieval(registry, symbol_retriever, expander)

    provider = create_provider(settings=settings)

    # Phase 31: repository-aware system prompt
    system_prompt = (
        "You are TRACERA, an autonomous code intelligence agent running in a developer's terminal.\n"
        "You have access to a fully indexed codebase. When answering questions about the code or\n"
        "making changes, ALWAYS use `search_code` or `find_symbol` first to locate the relevant\n"
        "symbols before falling back to `grep` or `list_dir`. This ensures you understand the\n"
        "architecture before editing. Think step-by-step, use tools systematically, and always\n"
        "verify your changes are correct before responding."
        if retrieval_pipeline is not None
        else (
            "You are TRACERA, an autonomous code intelligence agent running in a developer's terminal.\n"
            "Think step-by-step, use tools systematically, and verify your changes."
        )
    )

    agent = ReActAgent(
        provider=provider,
        registry=registry,
        max_iterations=settings.tracera_max_iterations,
        max_tool_calls=settings.tracera_max_tool_calls,
        model=settings.tracera_default_model,
        temperature=settings.tracera_default_temperature,
        max_tokens=settings.tracera_default_max_tokens,
        system_prompt=system_prompt,
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

    # Load retrieval pipeline if index exists (Phase 31: repository-aware agent)
    retrieval_pipeline = None
    index_manifest = settings.index_dir / "index_manifest.json"
    if index_manifest.exists():
        try:
            retrieval_pipeline = _build_retrieval_pipeline(settings, workspace_path)
            console.print("[dim green]✓ Code index loaded — retrieval-aware mode active[/]")
        except Exception as e:
            console.print(f"[dim yellow]⚠ Index load failed ({e}) — basic mode[/]")

    try:
        agent, ws, prov = _build_agent(settings, workspace_path, retrieval_pipeline)
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

    # Load retrieval pipeline if index exists (Phase 31)
    retrieval_pipeline = None
    index_manifest = settings.index_dir / "index_manifest.json"
    if index_manifest.exists():
        try:
            retrieval_pipeline = _build_retrieval_pipeline(settings, workspace_path)
        except Exception:
            pass

    try:
        agent, ws, prov = _build_agent(settings, workspace_path, retrieval_pipeline)
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

    # Index status
    index_manifest = settings.index_dir / "index_manifest.json"
    index_status = "[bold green]indexed[/]" if index_manifest.exists() else "[dim yellow]not indexed (run: tracera index)[/]"
    table.add_row("Code Index", index_status)


    # Memory stats
    from tracera.agent.memory import AgentMemory
    memory = AgentMemory(settings.memory_dir)
    table.add_row("Memory Entries", str(memory.count))


    from tracera.providers import list_available_providers
    providers_info = list_available_providers(settings)

    provider_table = Table(title="Provider Status (ranked by quality)", border_style="cyan", show_header=True)
    provider_table.add_column("#", style="dim", width=3)
    provider_table.add_column("Provider", style="bold cyan", width=12)
    provider_table.add_column("Status", width=10)
    provider_table.add_column("Key Env Var", style="dim", width=22)
    provider_table.add_column("Default Model", style="dim")

    for p in providers_info:
        status_icon = "[bold green]OK[/]" if p["available"] else "[dim red]--[/]"
        active_marker = " [active]" if p["name"] == settings.tracera_default_provider else ""
        provider_table.add_row(
            str(p["rank"]),
            p["name"] + active_marker,
            status_icon,
            p["key_env"],
            p["model"],
        )

    console.print(table)
    console.print(provider_table)





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


# ── Pipeline factory ──────────────────────────────────────────────────────────

def _build_retrieval_pipeline(settings=None, workspace_path: Path | None = None):
    """
    Build the full Phase 16-24 retrieval pipeline.

    Returns:
        (incremental_indexer, symbol_retriever, context_expander, reranker,
         context_engine, compressor, embedder, vector_store, bm25_index)
    """
    if settings is None:
        settings = _get_settings()

    ws_path = workspace_path or settings.tracera_workspace
    index_dir = settings.index_dir
    embed_cache = index_dir / "embed_cache"

    from tracera.retrieval.bm25 import BM25Index
    from tracera.retrieval.embedder import EmbeddingPipeline
    from tracera.retrieval.vector_store import VectorStore
    from tracera.retrieval.dense import DenseRetriever
    from tracera.retrieval.hybrid import HybridRetriever
    from tracera.retrieval.symbol_retrieval import SymbolAwareRetriever
    from tracera.retrieval.context_expander import ContextExpander
    from tracera.retrieval.reranker import CrossEncoderReranker
    from tracera.retrieval.incremental import IncrementalIndexer
    from tracera.agent.context_engine import ContextAssemblyEngine
    from tracera.agent.compressor import ContextCompressor

    # Phase 17: Embedder
    embedder = EmbeddingPipeline(
        model_name=settings.tracera_embedding_model,
        device=settings.tracera_embedding_device,
        cache_dir=embed_cache,
    )

    # Phase 16: BM25 — load from disk if exists, else empty
    bm25 = BM25Index()
    bm25_path = index_dir / "bm25.json"
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)

    # Phase 18: LanceDB vector store
    vector_store = VectorStore(
        uri=settings.lancedb_uri,
        dimension=embedder.dimension if bm25.doc_count == 0 else 384,
    )

    # Phase 19: Dense retriever
    dense = DenseRetriever(embedder, vector_store)

    # Phase 20: Hybrid retriever (BM25 + Dense via RRF)
    hybrid = HybridRetriever(bm25, dense, bm25_weight=0.5, dense_weight=0.5)

    # Phase 21: Symbol-aware retriever
    symbol_retriever = SymbolAwareRetriever(hybrid)

    # Phase 22: Context expander
    expander = ContextExpander(bm25, vector_store)

    # Phase 23: Cross-encoder reranker
    reranker = CrossEncoderReranker(top_n=5)

    # Phase 24: Incremental indexer (owns the full pipeline)
    indexer = IncrementalIndexer(
        workspace_root=ws_path,
        bm25_index=bm25,
        embedder=embedder,
        vector_store=vector_store,
        index_dir=index_dir,
        max_file_size=settings.tracera_indexing_max_file_size,
    )

    # Phase 29: Context engine
    context_engine = ContextAssemblyEngine(max_tokens=settings.tracera_max_context_tokens)

    # Phase 30: Compressor
    compressor = ContextCompressor(target_tokens=15_000)

    return indexer, symbol_retriever, expander, reranker, context_engine, compressor, embedder, vector_store, bm25


# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root."),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Force full re-index instead of incremental."),
    ] = False,
) -> None:
    """
    Index the workspace codebase for intelligent retrieval.

    Runs the full Phase 16-24 pipeline:
      scan → parse (tree-sitter) → extract symbols → chunk →
      BM25 index + embed (local model) → LanceDB vector store
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    console.print(f"\n[bold cyan]TRACERA Index[/] — Scanning [cyan]{ws_path}[/]")
    if rebuild:
        console.print("[yellow]⚠  Full rebuild requested — re-indexing everything.[/]")

    try:
        indexer, *_ = _build_retrieval_pipeline(settings, ws_path)
    except Exception as e:
        console.print(f"[bold red]Pipeline init failed:[/] {e}")
        raise typer.Exit(1)

    with console.status("[bold green]Indexing...[/]") as status:
        try:
            stats = indexer.run(full_rebuild=rebuild)
        except Exception as e:
            console.print(f"[bold red]Indexing failed:[/] {e}")
            raise typer.Exit(1)

    table = Table(title="Index Stats", border_style="green", show_header=True)
    table.add_column("Metric", style="cyan bold")
    table.add_column("Value", style="white")
    table.add_row("Files scanned", str(stats["total_scanned"]))
    table.add_row("New files indexed", str(stats["new"]))
    table.add_row("Modified files re-indexed", str(stats["modified"]))
    table.add_row("Deleted files removed", str(stats["deleted"]))
    table.add_row("Files unchanged (skipped)", str(stats["skipped"]))
    table.add_row("Total chunks indexed", str(stats["chunks_indexed"]))
    console.print(table)
    console.print("[bold green]✓[/] Index complete.")


# ── search ────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
    k: Annotated[int, typer.Option("--k", "-k", help="Number of results.")] = 5,
    language: Annotated[
        Optional[str],
        typer.Option("--lang", "-l", help="Filter by language (python/js/ts/...)."),
    ] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Apply cross-encoder reranking."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show retrieval scores."),
    ] = False,
) -> None:
    """
    Search the indexed codebase using Hybrid BM25 + Dense retrieval.

    Phases used: 16 (BM25) + 17-18 (Embed+Vector) + 19-21 (Dense+Hybrid+Symbol)
    Optionally applies Phase 23 (cross-encoder reranking).
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    console.print(f"\n[bold cyan]Search:[/] [white]{query}[/]")

    try:
        _, symbol_retriever, expander, reranker, context_engine, *_ = \
            _build_retrieval_pipeline(settings, ws_path)
    except Exception as e:
        console.print(f"[bold red]Pipeline init failed:[/] {e}")
        raise typer.Exit(1)

    with console.status("[bold green]Searching...[/]"):
        try:
            results = symbol_retriever.search(query, k=k * 2, language=language)
            # Phase 22: Context expansion
            results = expander.expand(results, max_additional=3)
            # Phase 23: Optional reranking
            if rerank:
                results = reranker.rerank(query, results, k=k)
            else:
                results = results[:k]
        except Exception as e:
            console.print(f"[bold red]Search failed:[/] {e}")
            raise typer.Exit(1)

    if not results:
        console.print(
            "[yellow]No results. Run [bold]tracera index[/] first to build the index.[/]"
        )
        return

    from rich.syntax import Syntax
    from rich.panel import Panel as RPanel

    for i, r in enumerate(results, 1):
        symbol = r.get("symbol") or "—"
        sym_type = r.get("symbol_type") or ""
        fp = r.get("file_path") or "unknown"
        start = r.get("start_line", "?")
        end = r.get("end_line", "?")
        content = r.get("content", "")
        lang = r.get("language") or "text"
        reason = r.get("_expansion_reason", "")

        score_info = ""
        if debug:
            rrf = r.get("_rrf_score", 0)
            final = r.get("_final_score", 0)
            rerank_s = r.get("_rerank_score", "—")
            score_info = f" [dim](rrf={rrf:.3f} final={final:.3f} rerank={rerank_s})[/]"

        title = f"[{i}] [bold cyan]{symbol}[/] ({sym_type})  [dim]{fp} L{start}-{end}[/]"
        if reason:
            title += f"  [italic yellow]{reason}[/]"
        title += score_info

        syntax = Syntax(content[:800], lang, theme="monokai", line_numbers=False)
        console.print(RPanel(syntax, title=title, border_style="cyan"))


# ── review ────────────────────────────────────────────────────────────────────

@app.command()
def review(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p"),
    ] = None,
    summary: Annotated[
        Optional[str],
        typer.Option("--summary", "-s", help="Brief description of what was implemented."),
    ] = None,
) -> None:
    """
    Run an autonomous self-review of current git changes (Phase 37).

    Gets a git diff of your uncommitted changes, retrieves related code context,
    and asks the LLM to critique the implementation for bugs and design issues.
    """
    _setup()
    settings = _get_settings()
    if provider:
        settings.tracera_default_provider = provider
    ws_path = (workspace or settings.tracera_workspace).resolve()

    console.print("\n[bold cyan]TRACERA Self-Review[/] (Phase 37)\n")

    try:
        agent, ws, prov = _build_agent(settings, ws_path)
        _, symbol_retriever, *_ = _build_retrieval_pipeline(settings, ws_path)
    except Exception as e:
        console.print(f"[bold red]Init failed:[/] {e}")
        raise typer.Exit(1)

    from tracera.agent.autonomous import SelfReviewer

    reviewer = SelfReviewer(ws_path, symbol_retriever)

    async def _run():
        with console.status("[bold green]Running self-review...[/]"):
            result = await reviewer.review(prov, implementation_summary=summary or "")
        console.print(Panel(result, title="[bold cyan]Self-Review Report[/]", border_style="cyan"))

    asyncio.run(_run())


if __name__ == "__main__":
    app()

