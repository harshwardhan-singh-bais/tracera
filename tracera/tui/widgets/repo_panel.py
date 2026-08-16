"""
Phase 58 — Repository inspection UI.

A sidebar widget that renders an overview of the repository:

    - top-level structure (files / dirs)
    - indexed symbols (classes, functions, methods) from the symbol graph
    - symbol dependencies
    - retrieval results for the current query
    - recent agent actions (tool calls)

Data is gathered from the existing sandbox / symbol graph / settings — no
new indexing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer


class RepoPanel(Widget):
    """Repository inspection panel (Phase 58)."""

    DEFAULT_CSS = """
    RepoPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="repo-panel-root"):
            yield Static(" REPOSITORY ", id="repo-title", classes="panel-title")
            with ScrollableContainer(id="repo-content"):
                yield Static("[dim]Run /inspect to load the repository view.[/]", id="repo-display", markup=True)

    # ── Rendering ────────────────────────────────────────────────────────────

    def _update(self, content: str) -> None:
        try:
            self.query_one("#repo-display", Static).update(content)
        except Exception:
            pass

    async def show_repository(self, workspace_path: Path | str, settings=None) -> None:
        """Render files, languages, git state and index status."""
        root = Path(workspace_path)
        lines: list[str] = [f"[bold cyan]Repository:[/] {root}\n"]

        # ── Top-level structure ──────────────────────────────────────────────
        try:
            from tracera.workspace.sandbox import WorkspaceSandbox
            sandbox = WorkspaceSandbox(root)
            entries = await sandbox.list_directory(".", max_depth=1)
            dirs: list[str] = []
            files: list[str] = []
            for e in entries:
                if len(e.relative.parts) == 1:
                    (dirs if e.is_dir else files).append(str(e.relative))
            lines.append("[bold]Structure[/]")
            if dirs:
                lines.append("  [dim]dirs:[/] " + ", ".join(sorted(dirs)[:20]))
            if files:
                lines.append("  [dim]files:[/] " + ", ".join(sorted(files)[:20]))
        except Exception as e:
            lines.append(f"[dim]Structure unavailable: {e}[/]")
        lines.append("")

        # ── Languages ────────────────────────────────────────────────────────
        try:
            from tracera.indexer.scanner import RepositoryScanner
            scanner = RepositoryScanner(root)
            langs: dict[str, int] = {}
            total = 0
            for meta in scanner.scan():
                total += 1
                lang = meta.language or "other"
                langs[lang] = langs.get(lang, 0) + 1
            top = sorted(langs.items(), key=lambda kv: -kv[1])[:8]
            lines.append(f"[bold]Source files:[/] {total}")
            if top:
                lines.append("  [dim]" + " · ".join(f"{l} {c}" for l, c in top) + "[/]")
        except Exception:
            pass
        lines.append("")

        # ── Git state ────────────────────────────────────────────────────────
        try:
            from tracera.git.operations import GitRepo
            repo = GitRepo(root)
            status = repo.status()
            lines.append(
                f"[bold]Git:[/] branch `{status.branch}` — "
                f"{'dirty' if status.is_dirty else 'clean'}"
            )
            recent = repo.log(max_count=2)
            for c in recent:
                lines.append(f"  [dim]• {c.hexsha[:7]} {c.summary[:60]}[/]")
        except Exception:
            lines.append("[dim]Git: not a repository[/]")
        lines.append("")

        # ── Index status ─────────────────────────────────────────────────────
        if settings is not None:
            manifest = settings.index_dir / "index_manifest.json"
            lines.append(
                "[bold]Code index:[/] "
                + ("[green]indexed[/]" if manifest.exists() else "[yellow]not indexed[/]")
            )
            lines.append("")

        self._update("\n".join(lines))

    def show_symbols(self, settings=None, *, limit: int = 40) -> None:
        """Render indexed symbols (classes / functions / methods) by file."""
        lines: list[str] = ["[bold cyan]Symbols[/]\n"]
        if settings is None:
            self._update("\n".join(lines) + "[dim]No settings — symbols unavailable.[/]")
            return
        graph_path = settings.index_dir / "symbol_graph.json"
        if not graph_path.exists():
            self._update(
                "\n".join(lines)
                + "[dim]No symbol graph — run `tracera index`.[/]"
            )
            return
        try:
            from tracera.graph.symbol_graph import SymbolGraph
            graph = SymbolGraph.load(graph_path)
            by_file: dict[str, list[tuple[str, str]]] = {}
            for node in graph.nodes:
                path = node.get("file_path") or "?"
                name = node.get("name") or "?"
                kind = node.get("symbol_type") or "symbol"
                by_file.setdefault(path, []).append((name, kind))
            shown = 0
            for path in sorted(by_file)[:10]:
                lines.append(f"[bold]{path}[/]")
                for name, kind in by_file[path][:8]:
                    if shown >= limit:
                        break
                    lines.append(f"  [dim]{kind:8s}[/] {name}")
                    shown += 1
        except Exception as e:
            lines.append(f"[dim]Symbol graph load failed: {e}[/]")
        self._update("\n".join(lines))

    def show_dependencies(self, symbol: str, settings=None, *, limit: int = 25) -> None:
        """Render the dependency chain of a symbol from the graph."""
        lines: list[str] = [f"[bold cyan]Dependencies: {symbol}[/]\n"]
        if settings is None:
            self._update("\n".join(lines) + "[dim]No settings available.[/]")
            return
        graph_path = settings.index_dir / "symbol_graph.json"
        if not graph_path.exists():
            self._update("\n".join(lines) + "[dim]No symbol graph available.[/]")
            return
        try:
            from tracera.graph.symbol_graph import SymbolGraph
            graph = SymbolGraph.load(graph_path)
            neighbors = graph.neighbors_of(symbol)
            if not neighbors:
                self._update("\n".join(lines) + "[dim]No dependencies found.[/]")
                return
            for i, n in enumerate(neighbors[:limit]):
                lines.append(f"  [dim]•[/] {n}")
            if len(neighbors) > limit:
                lines.append(f"  [dim]… {len(neighbors) - limit} more[/]")
        except Exception as e:
            lines.append(f"[dim]Failed: {e}[/]")
        self._update("\n".join(lines))

    def show_retrieval(self, query: str, hits: list[dict[str, Any]], *, limit: int = 15) -> None:
        """Render retrieval results for a query."""
        lines: list[str] = [f"[bold cyan]Retrieval:[/] {query}\n"]
        if not hits:
            lines.append("[dim]No results.[/]")
            self._update("\n".join(lines))
            return
        for hit in hits[:limit]:
            path = hit.get("file_path") or hit.get("id") or "?"
            symbol = hit.get("symbol") or ""
            score = hit.get("_relevance_score") or hit.get("_rrf_score") or ""
            label = f"[bold]{path}[/]"
            if symbol:
                label += f" [dim]({symbol})[/]"
            if score:
                label += f" [dim]· {float(score):.3f}[/]"
            lines.append(label)
            content = (hit.get("content") or "").strip().splitlines()
            if content:
                lines.append("  [dim]" + content[0][:90] + "[/]")
        if len(hits) > limit:
            lines.append(f"[dim]… {len(hits) - limit} more[/]")
        self._update("\n".join(lines))

    def show_actions(self, actions: list[tuple[str, str, bool]]) -> None:
        """Render recent agent actions: (tool_name, detail, success)."""
        lines: list[str] = ["[bold cyan]Agent actions[/]\n"]
        if not actions:
            lines.append("[dim]No actions recorded this session.[/]")
        for name, detail, ok in actions[-12:]:
            icon = "[green]✓[/]" if ok else "[red]✗[/]"
            lines.append(f"  {icon} [bold]{name}[/] {detail[:70]}")
        self._update("\n".join(lines))

    def clear(self) -> None:
        self._update("[dim]Repository view cleared.[/]")
