"""
Phase 39 — TRACERA MCP Server.

Exposes TRACERA's code-intelligence capabilities over the Model Context
Protocol so external agents (Claude Desktop, Cursor, other MCP clients)
can consume them.

This layer is a pure *adapter*: every MCP tool delegates to the existing
implementation — nothing is re-implemented here.

    Existing Tracera function  →  MCP wrapper  →  MCP tool
    search_code(query)             search_code     search_code
    find_symbol(name)              find_symbol     find_symbol
    find_references(symbol)        find_references find_references
    get_context(symbol)            get_context     get_context
    get_dependencies(symbol)       get_dependencies get_dependencies
    run_tests(...)                 run_tests       run_tests
    (scanner + git + tests)        inspect_repository inspect_repository

Run it standalone with:  tracera mcp serve
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from tracera.config.settings import Settings, get_settings
from tracera.logging import get_logger

log = get_logger("mcp.server")

# The 7 capabilities exposed over MCP (Phase 39).
EXPOSED_TOOLS = [
    "search_code",
    "find_symbol",
    "find_references",
    "get_context",
    "get_dependencies",
    "run_tests",
    "inspect_repository",
]

SERVER_INSTRUCTIONS = (
    "TRACERA — Agentic Code Intelligence & Autonomous Coding Engine.\n"
    "Tools expose code search, symbol lookup, dependency analysis, test "
    "execution, and repository inspection for the workspace this server "
    "was started in."
)


class TraceraMCPServer:
    """
    Builds a FastMCP server whose tools adapt TRACERA's existing capabilities.

    The retrieval pipeline is constructed lazily (and only when a code index
    exists) so the server starts fast and never downloads models at boot.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        workspace_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ws_path = (workspace_path or self._settings.tracera_workspace).resolve()
        self._pipeline: Any = None
        self._pipeline_error: str | None = None
        self._retrieval_tools: dict[str, Any] = {}

        self._mcp = FastMCP(
            "tracera",
            instructions=SERVER_INSTRUCTIONS,
        )
        self._register_all()

    # ── Construction ──────────────────────────────────────────────────────────

    @property
    def mcp(self) -> FastMCP:
        """The underlying FastMCP instance (for run/list_tools/call_tool)."""
        return self._mcp

    @property
    def workspace_path(self) -> Path:
        return self._ws_path

    def _register_all(self) -> None:
        self._mcp.add_tool(self.search_code, name="search_code")
        self._mcp.add_tool(self.find_symbol, name="find_symbol")
        self._mcp.add_tool(self.find_references, name="find_references")
        self._mcp.add_tool(self.get_context, name="get_context")
        self._mcp.add_tool(self.get_dependencies, name="get_dependencies")
        self._mcp.add_tool(self.run_tests, name="run_tests")
        self._mcp.add_tool(self.inspect_repository, name="inspect_repository")

    # ── Lazy pipeline (Phases 16-26) ─────────────────────────────────────────

    def _index_available(self) -> bool:
        return (self._settings.index_dir / "index_manifest.json").exists()

    def _pipeline_once(self) -> tuple[Any, str | None]:
        """Build the retrieval pipeline once, lazily. Returns (pipeline, error)."""
        if self._pipeline is None and self._pipeline_error is None:
            if not self._index_available():
                self._pipeline_error = (
                    "No code index found for this workspace. "
                    "Run `tracera index` first to enable code search."
                )
            else:
                try:
                    # Single source of truth for the Phase 16-26 pipeline factory.
                    from tracera.main import _build_retrieval_pipeline
                    self._pipeline = _build_retrieval_pipeline(
                        self._settings, self._ws_path
                    )
                    log.info("Retrieval pipeline loaded for MCP server")
                except Exception as e:
                    log.exception("Retrieval pipeline failed to load")
                    self._pipeline_error = f"Code index unavailable: {e}"
        return self._pipeline, self._pipeline_error

    def _retrieval_tool(self, name: str) -> Any | None:
        """Lazily instantiate (and cache) the Phase 27 tool adapters."""
        if name in self._retrieval_tools:
            return self._retrieval_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.code_search import (
            FindReferencesTool,
            FindSymbolTool,
            GetContextTool,
            GetDependenciesTool,
            SearchCodeTool,
        )

        symbol_retriever, expander, _, _, context_engine, compressor, *_ = pipeline
        graph_retriever = pipeline[-1]
        graph = graph_retriever.graph

        factory: dict[str, Any] = {
            "search_code": lambda: SearchCodeTool(
                symbol_retriever, compressor, context_engine
            ),
            "find_symbol": lambda: FindSymbolTool(symbol_retriever),
            "find_references": lambda: FindReferencesTool(graph),
            "get_context": lambda: GetContextTool(
                symbol_retriever, expander, graph_retriever,
                compressor=compressor, context_engine=context_engine,
            ),
            "get_dependencies": lambda: GetDependenciesTool(graph),
        }
        if name in factory:
            self._retrieval_tools[name] = factory[name]()
        return self._retrieval_tools.get(name)

    async def _run_retrieval(self, tool_name: str, **kwargs: Any) -> str:
        tool = self._retrieval_tool(tool_name)
        if tool is None:
            _, err = self._pipeline_once()
            return f"ERROR: {err or 'retrieval unavailable'}"
        result = await tool.execute(**kwargs)
        if not result.success:
            return f"ERROR: {result.error}"
        return result.output

    # ── MCP tools ─────────────────────────────────────────────────────────────

    async def search_code(self, query: str, k: int = 5, language: str | None = None) -> str:
        """Search the indexed codebase (hybrid BM25 + dense retrieval).

        Args:
            query: The search query (e.g. "authentication middleware").
            k: Number of results to return (default 5).
            language: Optional language filter (python, javascript, ...).
        """
        return await self._run_retrieval("search_code", query=query, k=k, language=language)

    async def find_symbol(self, name: str, symbol_type: str = "any") -> str:
        """Find the definition of a specific class, function, or method by name.

        Args:
            name: The exact or partial symbol name.
            symbol_type: One of class, function, method, any (default any).
        """
        return await self._run_retrieval(
            "find_symbol", name=name, symbol_type=symbol_type
        )

    async def find_references(self, symbol: str) -> str:
        """Find everywhere a symbol is referenced or called (graph-backed).

        Args:
            symbol: Name of the symbol to find references for.
        """
        return await self._run_retrieval("find_references", symbol=symbol)

    async def get_context(self, symbol: str) -> str:
        """Get full context for a symbol: definition, parent, related code.

        Args:
            symbol: The symbol name to get context for.
        """
        return await self._run_retrieval("get_context", symbol=symbol)

    async def get_dependencies(self, symbol: str) -> str:
        """Get the dependency chain of a symbol via graph traversal.

        Args:
            symbol: Name of the symbol to inspect.
        """
        return await self._run_retrieval("get_dependencies", symbol=symbol)

    async def run_tests(
        self,
        framework: str | None = None,
        test_paths: list[str] | None = None,
    ) -> str:
        """Run the project's test suite and return a structured report.

        Args:
            framework: Optional override (pytest, unittest, npm, cargo).
            test_paths: Optional specific test files/dirs to run.
        """
        from tracera.tools.test_runner import TestRunner

        # Run tests with the interpreter that is serving this MCP connection,
        # not whatever `python` happens to resolve to on PATH.
        import sys
        runner = TestRunner(self._ws_path, python=sys.executable)
        report = await asyncio.to_thread(runner.run, framework=framework, test_paths=test_paths)

        lines = [report.summary, ""]
        for f in report.failures[:20]:
            location = f"{f.file_path}:{f.line_number}" if f.file_path else f.test_name
            lines.append(f"- {location}: {f.error_type}: {f.error_message[:200]}")
        if not report.failures and not report.success and report.raw_output:
            lines.append(report.raw_output[:1500])
        return "\n".join(lines).strip() or "No tests detected."

    async def inspect_repository(self, path: str | None = None) -> str:
        """Return an overview of the repository: structure, languages, git state.

        Args:
            path: Optional path to inspect (defaults to the server workspace).
        """
        root = Path(path).resolve() if path else self._ws_path
        if not root.exists() or not root.is_dir():
            return f"ERROR: Not a directory: {root}"

        lines: list[str] = [f"## Repository: {root}", ""]

        # ── Structure (Phase 2 sandbox listing) ─────────────────────────────
        try:
            from tracera.workspace.sandbox import WorkspaceSandbox
            sandbox = WorkspaceSandbox(root)
            entries = await sandbox.list_directory(".", max_depth=2)
            top_dirs: list[str] = []
            top_files: list[str] = []
            for e in entries:
                parts = e.relative.parts
                if len(parts) == 1:
                    (top_dirs if e.is_dir else top_files).append(str(e.relative))
            lines.append("**Top-level:**")
            if top_dirs:
                lines.append("  dirs:  " + ", ".join(sorted(top_dirs)[:30]))
            if top_files:
                lines.append("  files: " + ", ".join(sorted(top_files)[:30]))
            lines.append("")
        except Exception as e:
            lines.append(f"**Structure:** unavailable ({e})")
            lines.append("")

        # ── Languages (Phase 11 scanner) ─────────────────────────────────────
        try:
            from tracera.indexer.scanner import RepositoryScanner
            scanner = RepositoryScanner(root)
            lang_counts: dict[str, int] = {}
            total = 0
            for meta in scanner.scan():
                total += 1
                lang = meta.language or "other"
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            lines.append(f"**Source files:** {total}")
            if lang_counts:
                top_langs = sorted(lang_counts.items(), key=lambda kv: -kv[1])[:10]
                lines.append("  languages: " + ", ".join(f"{l} ({c})" for l, c in top_langs))
            lines.append("")
        except Exception as e:
            lines.append(f"**Languages:** unavailable ({e})")
            lines.append("")

        # ── Git state (Phase 3) ──────────────────────────────────────────────
        try:
            from tracera.git.operations import GitRepo
            repo = GitRepo(root)
            status = repo.status()
            lines.append(f"**Git:** branch `{status.branch}` — {'dirty' if status.is_dirty else 'clean'}")
            counts = []
            if status.staged:
                counts.append(f"{len(status.staged)} staged")
            if status.unstaged:
                counts.append(f"{len(status.unstaged)} modified")
            if status.untracked:
                counts.append(f"{len(status.untracked)} untracked")
            if counts:
                lines.append("  changes: " + ", ".join(counts))
            recent = repo.log(max_count=3)
            if recent:
                lines.append("  recent commits:")
                for c in recent:
                    lines.append(f"    - {c.hexsha[:7]} {c.summary[:70]}")
        except Exception:
            lines.append("**Git:** not a git repository")
        lines.append("")

        # ── Tests (Phase 32) ─────────────────────────────────────────────────
        try:
            from tracera.tools.test_runner import TestDiscovery
            fw = TestDiscovery(root).detect_framework()
            lines.append(f"**Tests:** framework={fw}")
        except Exception:
            lines.append("**Tests:** unknown")
        lines.append("")

        # ── Index status ─────────────────────────────────────────────────────
        index_manifest = self._settings.index_dir / "index_manifest.json"
        lines.append(
            "**Code index:** " + ("indexed (retrieval tools active)" if index_manifest.exists() else "not indexed (run `tracera index`)")
        )

        return "\n".join(lines)


def build_mcp_server(
    settings: Settings | None = None,
    workspace_path: Path | None = None,
) -> FastMCP:
    """Build a FastMCP instance exposing TRACERA's 7 capabilities."""
    return TraceraMCPServer(settings=settings, workspace_path=workspace_path).mcp


def main(transport: str = "stdio") -> None:
    """Entry point for `tracera mcp serve`."""
    server = TraceraMCPServer()
    server.mcp.run(transport=transport)


if __name__ == "__main__":
    main()
