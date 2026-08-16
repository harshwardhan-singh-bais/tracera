"""
Phase 59 — Retrieval debugging mode.

Shows, for one query, what every retrieval strategy returned, so the
pipeline's behaviour is inspectable:

    Query: "Where is authentication handled?"
      BM25:
        1. auth/middleware.py  (score 8.42)
      Dense:
        ...
      Hybrid:
        ...
      Reranker:
        ...
      Final context:
        ...
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer


class DebugPanel(Widget):
    """Retrieval debugging panel (Phase 59)."""

    DEFAULT_CSS = """
    DebugPanel {
        height: 1fr;
        layout: vertical;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="debug-panel-root"):
            yield Static(" RETRIEVAL DEBUG ", id="debug-title", classes="panel-title")
            with ScrollableContainer(id="debug-content"):
                yield Static(
                    "[dim]Run /debug <query> to compare retrieval strategies.[/]",
                    id="debug-display",
                    markup=True,
                )

    # ── Rendering ────────────────────────────────────────────────────────────

    def _update(self, content: str) -> None:
        try:
            self.query_one("#debug-display", Static).update(content)
        except Exception:
            pass

    def show_query(
        self,
        query: str,
        strategy_results: dict[str, list[dict[str, Any]]],
        *,
        final_context: str | None = None,
        limit: int = 5,
    ) -> None:
        """
        Render the per-strategy results for *query*.

        Args:
            query: the debugged query.
            strategy_results: {strategy_name: [hit dicts]}.
            final_context: optional assembled/compressed context block.
        """
        lines: list[str] = [f"[bold cyan]Query:[/] {query}\n"]
        if not strategy_results:
            lines.append("[dim]No strategies available — is the code index built?[/]")
            self._update("\n".join(lines))
            return

        for strategy, hits in strategy_results.items():
            lines.append(f"[bold]{strategy.upper()}[/]")
            if not hits:
                lines.append("  [dim]— no results —[/]")
            for i, hit in enumerate(hits[:limit], 1):
                path = hit.get("file_path") or hit.get("id") or "?"
                score = hit.get("_relevance_score") or hit.get("_rrf_score") or hit.get("_rerank_score") or ""
                symbol = hit.get("symbol") or ""
                line = f"  {i}. [bold]{path}[/]"
                if symbol:
                    line += f" [dim]({symbol})[/]"
                if score:
                    line += f"  [dim]· {float(score):.3f}[/]"
                lines.append(line)
                content = (hit.get("content") or "").strip().splitlines()
                if content:
                    lines.append("     [dim]" + content[0][:80] + "[/]")
            if len(hits) > limit:
                lines.append(f"  [dim]… {len(hits) - limit} more[/]")
            lines.append("")

        if final_context:
            lines.append("[bold]FINAL CONTEXT[/]")
            lines.append(final_context[:1200])
        self._update("\n".join(lines))

    def show_error(self, message: str) -> None:
        self._update(f"[red]Debug error:[/] {message}")

    def clear(self) -> None:
        self._update("[dim]Debug view cleared.[/]")
