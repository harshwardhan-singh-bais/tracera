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
    # Phase 2: workspace lifecycle — create the `.tracera/` data dirs
    from tracera.workspace.lifecycle import WorkspaceLifecycle
    WorkspaceLifecycle(settings.tracera_data_dir).initialise()


def _build_agent(settings=None, workspace_path: Path | None = None, retrieval_pipeline=None):
    """Build a ReActAgent from current settings.

    If retrieval_pipeline is provided, the agent is extended with code-search
    tools (Phase 28) and a retrieval-aware system prompt (Phase 31).
    """
    if settings is None:
        settings = _get_settings()

    from tracera.tools.registry import create_default_registry
    from tracera.workspace.sandbox import WorkspaceSandbox
    from tracera.agent.react_loop import ReActAgent

    ws_path = workspace_path or settings.tracera_workspace
    workspace = WorkspaceSandbox(
        ws_path,
        max_file_size=settings.tracera_indexing_max_file_size,
    )
    registry = create_default_registry(workspace)

    # Phase 28: extend registry with code-search tools if index is available.
    # Phases 29/30: pass the context engine + compressor so retrieval tool
    # output is assembled + compressed before it reaches the LLM.
    if retrieval_pipeline is not None:
        from tracera.tools.registry import extend_registry_with_retrieval
        symbol_retriever = retrieval_pipeline[1]
        expander = retrieval_pipeline[2]
        graph_retriever = retrieval_pipeline[-1]
        context_engine = retrieval_pipeline[4]
        compressor = retrieval_pipeline[5]
        extend_registry_with_retrieval(
            registry, symbol_retriever, expander, graph_retriever,
            context_engine=context_engine,
            compressor=compressor,
        )

    # Provider with automatic failover across all configured APIs
    provider = _build_provider(settings)

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

    # Phase 10 → 8: give the agent access to persistent memory
    from tracera.agent.memory import AgentMemory
    memory = AgentMemory(settings.memory_dir)

    # Phase 9: let the agent plan every task up front (todo tracking + replan)
    from tracera.agent.planner import TaskDecomposer
    decomposer = TaskDecomposer(provider)

    agent = ReActAgent(
        provider=provider,
        registry=registry,
        max_iterations=settings.tracera_max_iterations,
        max_tool_calls=settings.tracera_max_tool_calls,
        model=settings.tracera_default_model,
        temperature=settings.tracera_default_temperature,
        max_tokens=settings.tracera_default_max_tokens,
        system_prompt=system_prompt,
        memory_provider=lambda: memory.build_context(""),
        # Phase 10: the agent writes decisions/errors back to persistent memory
        memory_writer=lambda kind, content: _write_memory(memory, kind, content),
        # Phase 9: decompose tasks into tracked todo lists
        decomposer=decomposer,
        context_budget_tokens=settings.tracera_context_budget_tokens,
    )
    return agent, workspace, provider


def _write_memory(memory, kind: str, content: str) -> None:
    """
    Phase 10 — persist agent outcomes into memory.

    kind: 'decision' → PAST_DECISION · 'error' → ERROR_PATTERN
    """
    from tracera.agent.memory import MemoryCategory
    try:
        text = content.strip()
        if not text:
            return
        if kind == "error":
            memory.add(
                text[:300],
                MemoryCategory.ERROR_PATTERN,
                source="agent",
                importance=0.5,
            )
        else:
            memory.add(
                text[:300],
                MemoryCategory.PAST_DECISION,
                source="agent",
                importance=0.4,
            )
    except Exception:
        pass


def _build_provider(settings=None):
    """
    Build the LLM provider for the agent.

    When multiple provider keys are configured, returns a FailoverProvider
    that automatically falls back to the next available API when a call
    fails (rate limit, auth, overload) — Groq → OpenAI → Gemini → Ollama ….
    """
    if settings is None:
        settings = _get_settings()

    from tracera.providers import (
        _PROVIDER_MODELS,
        create_provider,
        list_available_providers,
    )
    from tracera.providers.failover import FailoverProvider

    ranked = list_available_providers(settings)
    providers = []
    for i, info in enumerate(ranked):
        if not info["available"]:
            continue
        # First provider honours the user's default model; fallbacks use
        # each provider's recommended model (a Groq model on OpenAI would
        # be invalid).
        model = "" if i == 0 else _PROVIDER_MODELS.get(info["name"], "")
        try:
            providers.append(create_provider(name=info["name"], model=model, settings=settings))
        except Exception as e:
            console.print(f"[dim yellow]⚠ Provider {info['name']} unavailable ({e})[/]")

    if not providers:
        # No keys at all — let create_provider raise the proper error
        return create_provider(settings=settings)
    if len(providers) == 1:
        return providers[0]
    return FailoverProvider(providers)



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
    tui = TraceraTUI(
        agent=agent,
        memory=memory,
        workspace_path=workspace_path,
        retrieval_pipeline=retrieval_pipeline,
    )
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
         context_engine, compressor, embedder, vector_store, bm25_index,
         graph_retriever)
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
    from tracera.graph.symbol_graph import SymbolGraph
    from tracera.graph.graph_retrieval import GraphRetriever
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

    # Phase 18: LanceDB vector store.
    # Reuse the dimension of an existing table when present (avoids loading the
    # embedding model just to learn the dimension); otherwise derive it from the
    # configured embedding model.
    vector_store = VectorStore(uri=settings.lancedb_uri)
    dimension = vector_store.existing_dimension() or embedder.dimension
    vector_store = VectorStore(uri=settings.lancedb_uri, dimension=dimension)

    # Phase 25-26: Symbol relationship graph + dependency-aware retrieval
    graph = SymbolGraph()
    graph_path = index_dir / "symbol_graph.json"
    if graph_path.exists():
        try:
            graph = SymbolGraph.load(graph_path)
        except Exception as e:
            console.print(f"[dim yellow]⚠ Symbol graph load failed ({e}) — starting empty[/]")
    graph_retriever = GraphRetriever(graph, bm25)

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
        symbol_graph=graph,
    )

    # Phase 29: Context engine
    context_engine = ContextAssemblyEngine(max_tokens=settings.tracera_max_context_tokens)

    # Phase 30: Compressor
    compressor = ContextCompressor(target_tokens=15_000)

    return (
        indexer,
        symbol_retriever,
        expander,
        reranker,
        context_engine,
        compressor,
        embedder,
        vector_store,
        bm25,
        graph_retriever,
    )


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
        pipeline = _build_retrieval_pipeline(settings, ws_path)
        symbol_retriever, expander, reranker, graph_retriever = (
            pipeline[1], pipeline[2], pipeline[3], pipeline[-1]
        )
    except Exception as e:
        console.print(f"[bold red]Pipeline init failed:[/] {e}")
        raise typer.Exit(1)

    with console.status("[bold green]Searching...[/]"):
        try:
            results = symbol_retriever.search(query, k=k * 2, language=language)
            # Phase 22: Context expansion
            results = expander.expand(results, max_additional=3)
            # Phase 26: dependency-aware graph expansion (when a graph exists)
            results = graph_retriever.expand_with_graph(
                results, max_depth=1, max_total=k * 2
            )
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


# ── fix ──────────────────────────────────────────────────────────────────────

@app.command()
def fix(
    task: Annotated[str, typer.Argument(help="Coding task to implement or fix.")],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p"),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", "-i", help="Max fix-loop iterations."),
    ] = 5,
) -> None:
    """
    Run the autonomous fix loop (Phases 35-36 + 38).

    For each failing test, retrieves the relevant code (Phase 35), drives the
    agent to edit it (Phase 36: plan → edit → test → retry), and wraps the whole
    run with a pre/post regression check (Phase 38).
    """
    _setup()
    settings = _get_settings()
    if provider:
        settings.tracera_default_provider = provider
    ws_path = (workspace or settings.tracera_workspace).resolve()

    console.print("\n[bold cyan]TRACERA Autonomous Fix Loop[/] (Phases 35-36, 38)\n")

    # Retrieval pipeline is optional: the loop still works on basic tools alone.
    pipeline = None
    try:
        pipeline = _build_retrieval_pipeline(settings, ws_path)
    except Exception as e:
        console.print(f"[dim yellow]⚠ Retrieval pipeline unavailable ({e}) — basic tools only[/]")

    try:
        agent, ws, prov = _build_agent(settings, ws_path, pipeline)
    except Exception as e:
        console.print(f"[bold red]Init failed:[/] {e}")
        raise typer.Exit(1)

    from tracera.tools.test_runner import TestRunner
    from tracera.agent.autonomous import AutonomousFixLoop, RetrievalDebugger, RegressionProtector
    from tracera.agent.context_engine import ContextAssemblyEngine
    from tracera.agent.compressor import ContextCompressor
    from tracera.agent.planner import TaskDecomposer

    symbol_retriever = pipeline[1] if pipeline else None
    context_engine = pipeline[4] if pipeline else ContextAssemblyEngine()
    compressor = pipeline[5] if pipeline else ContextCompressor()

    test_runner = TestRunner(ws_path)
    debugger = RetrievalDebugger(symbol_retriever, context_engine, compressor=compressor)
    protector = RegressionProtector(ws_path, test_runner)
    # Phase 9: plan the task up front and replan after failed attempts
    decomposer = TaskDecomposer(prov)
    fix_loop = AutonomousFixLoop(
        ws_path, test_runner, debugger,
        max_iterations=max_iterations, decomposer=decomposer,
    )

    console.print("[dim]Taking pre-task regression baseline…[/]")
    baseline = protector.snapshot_before()
    console.print(f"[dim]Baseline: {baseline.summary}[/]\n")

    async def _run():
        return await fix_loop.run(task, prov, agent)

    result = asyncio.run(_run())

    attempts_table = Table(title="Fix Loop Attempts", border_style="cyan", show_header=True)
    attempts_table.add_column("#", style="dim", width=3)
    attempts_table.add_column("Test Run", style="white")
    attempts_table.add_column("Result", width=8)
    attempts_table.add_column("Diagnosis", style="dim", width=45)
    for a in result.attempts:
        attempts_table.add_row(
            str(a.iteration),
            a.test_report.summary,
            "✅ PASS" if a.success else "❌ FAIL",
            (a.patch_description or "")[:45],
        )
    console.print(attempts_table)

    if result.final_success:
        console.print(
            f"\n[bold green]✓ All tests passing after {result.total_iterations} attempt(s).[/]"
        )
    else:
        console.print(
            f"\n[bold yellow]Fix loop exhausted after {result.total_iterations} attempt(s) — "
            "tests still failing.[/]"
        )

    # Phase 38: regression protection
    console.print("\n[dim]Verifying no regressions…[/]")
    report = protector.verify_after()
    reg_table = Table(title="Regression Report (Phase 38)", border_style="yellow", show_header=True)
    reg_table.add_column("Metric", style="cyan bold")
    reg_table.add_column("Value", style="white")
    reg_table.add_row("Tests passing before", str(report["pre_passed"]))
    reg_table.add_row("Tests passing after", str(report["post_passed"]))
    reg_table.add_row(
        "Regressions",
        f"[bold red]{report['regressions']}[/]"
        if report["regressions"]
        else "[bold green]0[/]",
    )
    reg_table.add_row("Changed files", ", ".join(report["changed_files"][:5]) or "—")
    reg_table.add_row(
        "Overall",
        "[bold green]PASS[/]" if report["overall_success"] else "[bold red]FAIL[/]",
    )
    console.print(reg_table)


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


# ── delegate ─────────────────────────────────────────────────────────────────

@app.command()
def delegate(
    task: Annotated[str, typer.Argument(help="Task to delegate to the sub-agent fleet.")],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w"),
    ] = None,
    parallel: Annotated[
        bool,
        typer.Option("--parallel", help="Run independent steps concurrently."),
    ] = False,
) -> None:
    """
    Delegate a task to specialized sub-agents (Phases 42-44).

    The main agent's task is decomposed into steps, each assigned to a
    role (Researcher / Coder / Tester / Reviewer / Debugger), then the
    results are aggregated into a report.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    console.print(f"\n[bold cyan]TRACERA Delegation[/] — {task[:80]}\n")

    try:
        agent, workspace_sandbox, prov = _build_agent(settings, ws_path)
        registry = agent.registry
    except Exception as e:
        console.print(f"[bold red]Init failed:[/] {e}")
        raise typer.Exit(1)

    from tracera.agent.orchestrator import TaskOrchestrator
    from tracera.agent.subagents import build_sub_agent_fleet

    fleet = build_sub_agent_fleet(
        prov, registry,
        model=settings.tracera_default_model,
        max_iterations=settings.tracera_max_iterations,
        max_tool_calls=settings.tracera_max_tool_calls,
    )
    orchestrator = TaskOrchestrator(fleet, parallel=parallel)

    async def _run():
        async for event in orchestrator.delegate(task):
            etype = event["type"]
            if etype == "plan_ready":
                console.print(event["plan"].to_markdown())
                console.print()
            elif etype == "agent_start":
                step = event["step"]
                console.print(f"[bold cyan]>> {step.role.value.title()}[/] {step.task[:70]}")
            elif etype == "agent_end":
                step, result = event["step"], event["result"]
                icon = "[green]OK[/]" if result.status.value == "success" else "[red]FAIL[/]"
                console.print(
                    f"  {icon} {result.label} — {result.latency_ms:.0f}ms · "
                    f"{result.iterations} iter · {result.tool_calls} tools"
                )
                if result.output:
                    console.print(Panel(
                        result.output[:800],
                        title=f"[bold]{result.label} result[/]",
                        border_style="dim",
                    ))
                if result.error:
                    console.print(f"  [red]error:[/] {result.error[:300]}")
            elif etype == "report":
                report = event["report"]
                console.print("\n[bold]Aggregated report[/]")
                console.print(Panel(
                    report.to_markdown(),
                    title="[bold cyan]Delegation Report[/]",
                    border_style="cyan",
                ))

    asyncio.run(_run())


# ── eval ─────────────────────────────────────────────────────────────────────

eval_app = typer.Typer(
    name="eval",
    help="Evaluation & benchmarking (Phases 45-50).",
)
app.add_typer(eval_app)


@eval_app.command("dataset")
def eval_dataset(
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the dataset JSON.")] = Path(".tracera/eval/dataset.json"),
) -> None:
    """
    Phase 45 — write the example retrieval-evaluation dataset to disk.

    Edit the generated JSON to match your codebase (real file paths / symbols
    as ground truth), then run `tracera eval retrieval`.
    """
    from tracera.evaluation.dataset import example_dataset
    dataset = example_dataset()
    path = dataset.save(output)
    console.print(
        f"[bold green]OK[/] Wrote {len(dataset)} benchmark queries to [cyan]{path}[/]\n\n"
        "Edit the ground-truth files/symbols to match your codebase before "
        "running the benchmark."
    )


@eval_app.command("retrieval")
def eval_retrieval(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to the evaluation dataset JSON."),
    ] = Path(".tracera/eval/dataset.json"),
    workspace: Annotated[Optional[Path], typer.Option("--workspace", "-w")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(".tracera/eval/retrieval_report.md"),
    include: Annotated[Optional[str], typer.Option("--include", help="Comma-separated strategy names.")] = None,
) -> None:
    """
    Phases 46-48 — run the retrieval benchmark.

    Compares grep / BM25 / dense / hybrid / hybrid+reranker on the dataset
    and reports Recall@k, MRR, nDCG@k, latency, and context size.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    if not dataset_path.exists():
        console.print(
            f"[bold red]Dataset not found:[/] {dataset_path}\n"
            "Run `tracera eval dataset` to create the example dataset first."
        )
        raise typer.Exit(1)

    from tracera.evaluation.dataset import EvaluationDataset
    from tracera.evaluation.retrieval_benchmark import RetrievalBenchmark
    from tracera.evaluation.strategies import build_doc_resolver, build_strategies

    dataset = EvaluationDataset.load(dataset_path)

    pipeline = _build_retrieval_pipeline(settings, ws_path)
    _, _, _, reranker, _, _, embedder, vector_store, bm25, _ = pipeline

    from tracera.retrieval.dense import DenseRetriever
    from tracera.retrieval.hybrid import HybridRetriever
    dense = DenseRetriever(embedder, vector_store)
    hybrid = HybridRetriever(bm25, dense)

    include_list = [s.strip() for s in include.split(",")] if include else None
    strategies = build_strategies(
        workspace=ws_path,
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        reranker=reranker,
        resolve_doc=build_doc_resolver(vector_store),
        include=include_list,
    )
    if not strategies:
        console.print("[bold red]No strategies could be built — is the code index present?[/]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Retrieval benchmark[/] — {len(dataset)} queries, {len(strategies)} strategies\n")
    report = RetrievalBenchmark(dataset, strategies).run()
    console.print(report.to_markdown())
    report.save(output)
    console.print(f"\n[bold green]OK[/] Report saved to [cyan]{output}[/]")


@eval_app.command("agent")
def eval_agent(
    tasks: Annotated[
        Optional[str],
        typer.Option("--tasks", help="Comma-separated coding tasks to run."),
    ] = None,
    workspace: Annotated[Optional[Path], typer.Option("--workspace", "-w")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(".tracera/eval/agent_report.md"),
) -> None:
    """
    Phase 49 — run the end-to-end agent benchmark.

    Each task is run through the agent; metrics: success, tests passed,
    iterations, tool calls, retrieval calls, tokens, latency, cost.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    default_tasks = [
        "Where is JWT authentication implemented? Summarize the flow.",
        "Find the function that handles database retries and explain it.",
    ]
    task_list = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

    from tracera.evaluation.agent_benchmark import AgentBenchmark

    agent, _, prov = _build_agent(settings, ws_path)

    async def runner(task: str) -> dict:
        outcome = {"tool_names": set()}
        async for event in await agent.run(task):
            if event.type.value == "response_complete":
                outcome["output"] = event.text or ""
                outcome["success"] = True
                outcome["iterations"] = event.metadata.get("iterations", 0)
            elif event.type.value == "tool_end":
                outcome["tool_names"].add(event.tool_name or "")
            elif event.type.value == "error":
                outcome["error"] = event.text
        return outcome

    bench = AgentBenchmark(runner, tasks=task_list, name="cli-agent")
    report = asyncio.run(bench.run())
    console.print(report.to_markdown())
    report_path = output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    console.print(f"\n[bold green]OK[/] Report saved to [cyan]{report_path}[/]")


@eval_app.command("ablation")
def eval_ablation(
    tasks: Annotated[
        Optional[str],
        typer.Option("--tasks", help="Comma-separated coding tasks to run per arm."),
    ] = None,
    workspace: Annotated[Optional[Path], typer.Option("--workspace", "-w")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(".tracera/eval/ablation_report.md"),
) -> None:
    """
    Phase 50 — run the ablation study.

    Compares Agent / +BM25 / +Dense / +Hybrid / +Hybrid+Reranker /
    +Hybrid+Graph on the same tasks to show which components matter.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    default_tasks = [
        "Find where authentication is handled and explain the flow.",
        "Locate the retry logic for database calls.",
    ]
    task_list = [t.strip() for t in tasks.split(",")] if tasks else default_tasks

    from tracera.evaluation.ablation import AblationFramework, AblationConfig

    # Build an agent runner per ablation config, then benchmark it.
    async def build_agent(config: AblationConfig):
        agent, _, prov = _build_agent(settings, ws_path)
        enabled = config
        tool_filter = None
        if not (enabled.bm25 or enabled.dense or enabled.hybrid):
            tool_filter = lambda name: not name.startswith(("search_", "find_", "get_"))

        async def runner(task: str) -> dict:
            outcome = {"tool_names": set()}
            async for event in await agent.run(task):
                if event.type.value == "response_complete":
                    outcome["output"] = event.text or ""
                    outcome["success"] = True
                    outcome["iterations"] = event.metadata.get("iterations", 0)
                elif event.type.value == "tool_end":
                    outcome["tool_names"].add(event.tool_name or "")
                elif event.type.value == "error":
                    outcome["error"] = event.text
            return outcome

        return runner

    framework = AblationFramework(task_list, build_agent, name="cli-ablation")
    report = asyncio.run(framework.run())
    console.print(report.to_markdown())
    report.save(output)
    console.print(f"\n[bold green]OK[/] Report saved to [cyan]{output}[/]")


# ── mcp ──────────────────────────────────────────────────────────────────────

mcp_app = typer.Typer(
    name="mcp",
    help="MCP server & client integration (Phases 39-41).",
)
app.add_typer(mcp_app)


@mcp_app.command("serve")
def mcp_serve(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root."),
    ] = None,
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="stdio | sse | streamable-http"),
    ] = "stdio",
    check: Annotated[
        bool,
        typer.Option("--check", help="List exposed tools and exit (no server)."),
    ] = False,
) -> None:
    """
    Run the TRACERA MCP server (Phase 39).

    Exposes the existing capabilities — search_code, find_symbol,
    find_references, get_context, get_dependencies, run_tests,
    inspect_repository — over the Model Context Protocol.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    from tracera.mcp.server import EXPOSED_TOOLS, TraceraMCPServer
    server = TraceraMCPServer(settings, ws_path)

    if check:
        tools = asyncio.run(server.mcp.list_tools())
        table = Table(
            title=f"MCP Server Tools ({len(tools)})",
            border_style="cyan",
            show_header=True,
        )
        table.add_column("Tool", style="bold cyan")
        table.add_column("Description", style="white")
        for t in tools:
            table.add_row(
                t.name,
                (t.description or "").replace("\n", " ")[:80],
            )
        console.print(table)
        return

    console.print(f"[dim]TRACERA MCP server — transport={transport} workspace={ws_path}[/]")
    console.print(f"[dim]Tools: {', '.join(EXPOSED_TOOLS)}[/]")
    server.mcp.run(transport=transport)


@mcp_app.command("connect")
def mcp_connect(
    config: Annotated[
        Path,
        typer.Argument(help="JSON file with MCP server declarations."),
    ],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root for native tools."),
    ] = None,
) -> None:
    """
    Connect to external MCP servers (Phases 40-41).

    Loads server declarations from a JSON file, connects over the MCP
    protocol, and merges every remote tool into the unified ToolRegistry
    alongside the native tools.
    """
    _setup()
    settings = _get_settings()
    ws_path = (workspace or settings.tracera_workspace).resolve()

    from tracera.mcp.manager import MCPManager
    from tracera.tools.registry import create_default_registry
    from tracera.workspace.sandbox import WorkspaceSandbox

    try:
        manager = MCPManager.from_file(config)
    except Exception as e:
        console.print(f"[bold red]Failed to load config:[/] {e}")
        raise typer.Exit(1)

    registry = create_default_registry(WorkspaceSandbox(ws_path))
    native_names = list(registry.names)

    async def _run() -> int:
        merged = await manager.connect_all()
        for name, tools in merged.items():
            table = Table(
                title=f"Server: {name} — {len(tools)} tools",
                border_style="cyan",
                show_header=True,
            )
            table.add_column("Tool", style="bold cyan")
            table.add_column("Description", style="white")
            for t in tools:
                table.add_row(
                    t["name"],
                    (t.get("description") or "").replace("\n", " ")[:70],
                )
            console.print(table)
        added = await manager.register(merged, registry)
        await manager.disconnect_all()
        return added

    try:
        added = asyncio.run(_run())
    except Exception as e:
        console.print(f"[bold red]Connection failed:[/] {e}")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Unified Tool Registry:[/] {len(native_names)} native + "
        f"{added} MCP tools"
    )
    unified = Table(show_header=True, border_style="green")
    unified.add_column("#", style="dim", width=3)
    unified.add_column("Tool", style="bold cyan")
    unified.add_column("Origin", style="dim")
    for i, name in enumerate(registry.names, 1):
        origin = "mcp" if name not in native_names else "native"
        unified.add_row(str(i), name, origin)
    console.print(unified)


if __name__ == "__main__":
    app()

