"""
TRACERA MCP Server — Comprehensive Code Intelligence + Memory Exposure.

Exposes TRACERA's full code-intelligence and memory capabilities over the
Model Context Protocol so external agents (Claude Desktop, Cursor, Gemini,
other MCP clients) can consume them.

Architecture:
    External Agent  →  MCP Protocol  →  This Server  →  TRACERA Engine
                                                   ↓
                                        ┌─────────────────────┐
                                        │ Code Intelligence   │
                                        │ Memory Engine       │
                                        │ Context Assembly    │
                                        │ Safety Analysis     │
                                        └─────────────────────┘

The server is a thin adapter: every MCP tool delegates to the existing
implementation — nothing is re-implemented here.

Run it standalone with:  tracera mcp serve

Tool categories:
    CODE_INTELLIGENCE  — search, symbols, references, dependencies, AST, graph
    CONTEXT            — task assembly, ranked context, repo map, session stats
    MEMORY             — recall, remember, forget, sessions, knowledge graph
    SAFETY             — edit/delete safety, refactoring, risk, provenance
    REPOSITORY         — tests, repository inspection, git state
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from tracera.config.settings import Settings, get_settings
from tracera.logging import get_logger

log = get_logger("mcp.server")

# ── Tool catalog ──────────────────────────────────────────────────────────────

CODE_INTELLIGENCE_TOOLS = [
    "search_code",
    "find_symbol",
    "find_references",
    "get_context",
    "get_dependencies",
    "find_importers",
    "get_blast_radius",
    "get_call_hierarchy",
    "find_dead_code",
    "get_changed_symbols",
    "get_hotspots",
    "search_ast",
    "get_class_hierarchy",
    "get_dependency_cycles",
    "get_coupling_metrics",
    "get_endpoint_impact",
]

CONTEXT_TOOLS = [
    "assemble_task_context",
    "get_ranked_context",
    "plan_turn",
    "get_session_stats",
    "get_repo_map",
]

MEMORY_TOOLS = [
    "recall_memory",
    "remember_memory",
    "forget_memory",
    "list_sessions",
    "search_memory",
    "get_memory_graph",
]

SAFETY_TOOLS = [
    "check_edit_safe",
    "check_delete_safe",
    "plan_refactoring",
    "get_pr_risk_profile",
    "get_symbol_provenance",
    "audit_agent_config",
]

REPOSITORY_TOOLS = [
    "run_tests",
    "inspect_repository",
]

ALL_MCP_TOOLS = (
    CODE_INTELLIGENCE_TOOLS
    + CONTEXT_TOOLS
    + MEMORY_TOOLS
    + SAFETY_TOOLS
    + REPOSITORY_TOOLS
)

SERVER_INSTRUCTIONS = (
    "TRACERA — MCP-native Code Intelligence & Persistent Agent Memory.\n\n"
    "This server exposes TRACERA's full code-intelligence and memory "
    "capabilities over MCP. Any MCP-compatible agent (Claude Desktop, "
    "Cursor, Gemini, etc.) can connect and gain:\n\n"
    "CODE INTELLIGENCE:\n"
    "  search_code, find_symbol, find_references, get_context,\n"
    "  get_dependencies, find_importers, get_blast_radius,\n"
    "  get_call_hierarchy, find_dead_code, get_changed_symbols,\n"
    "  get_hotspots, search_ast, get_class_hierarchy,\n"
    "  get_dependency_cycles, get_coupling_metrics, get_endpoint_impact\n\n"
    "CONTEXT:\n"
    "  assemble_task_context, get_ranked_context, plan_turn,\n"
    "  get_session_stats, get_repo_map\n\n"
    "MEMORY:\n"
    "  recall_memory, remember_memory, forget_memory, list_sessions,\n"
    "  search_memory, get_memory_graph\n\n"
    "SAFETY:\n"
    "  check_edit_safe, check_delete_safe, plan_refactoring,\n"
    "  get_pr_risk_profile, get_symbol_provenance, audit_agent_config\n\n"
    "REPOSITORY:\n"
    "  run_tests, inspect_repository\n\n"
    "All tools delegate to TRACERA's underlying engine — the same engine "
    "used by TRACERA's own CLI agent. No duplicate implementations."
)


class TraceraMCPServer:
    """
    Builds a FastMCP server whose tools adapt TRACERA's full capabilities.

    The retrieval pipeline is constructed lazily (and only when a code index
    exists) so the server starts fast and never downloads models at boot.

    Memory components are also lazy — they're only needed when memory tools
    are called.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        workspace_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ws_path = (workspace_path or self._settings.tracera_workspace).resolve()
        # Lazy-loaded components
        self._pipeline: Any = None
        self._pipeline_error: str | None = None
        self._retrieval_tools: dict[str, Any] = {}
        self._ast_tools: dict[str, Any] = {}
        self._refactor_tools: dict[str, Any] = {}
        self._session_tools: dict[str, Any] = {}
        self._provenance_tools: dict[str, Any] = {}
        self._memory_tools: dict[str, Any] = {}
        # Memory components (lazy)
        self._enhanced_memory: Any = None
        self._session_manager: Any = None
        self._triple_store: Any = None
        self._context_recall: Any = None
        self._legacy_memory: Any = None

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
        """Register all MCP tools across all categories."""
        # Code Intelligence
        self._mcp.add_tool(self.search_code, name="search_code")
        self._mcp.add_tool(self.find_symbol, name="find_symbol")
        self._mcp.add_tool(self.find_references, name="find_references")
        self._mcp.add_tool(self.get_context, name="get_context")
        self._mcp.add_tool(self.get_dependencies, name="get_dependencies")
        self._mcp.add_tool(self.find_importers, name="find_importers")
        self._mcp.add_tool(self.get_blast_radius, name="get_blast_radius")
        self._mcp.add_tool(self.get_call_hierarchy, name="get_call_hierarchy")
        self._mcp.add_tool(self.find_dead_code, name="find_dead_code")
        self._mcp.add_tool(self.get_changed_symbols, name="get_changed_symbols")
        self._mcp.add_tool(self.get_hotspots, name="get_hotspots")
        self._mcp.add_tool(self.search_ast, name="search_ast")
        self._mcp.add_tool(self.get_class_hierarchy, name="get_class_hierarchy")
        self._mcp.add_tool(self.get_dependency_cycles, name="get_dependency_cycles")
        self._mcp.add_tool(self.get_coupling_metrics, name="get_coupling_metrics")
        self._mcp.add_tool(self.get_endpoint_impact, name="get_endpoint_impact")
        # Context
        self._mcp.add_tool(self.assemble_task_context, name="assemble_task_context")
        self._mcp.add_tool(self.get_ranked_context, name="get_ranked_context")
        self._mcp.add_tool(self.plan_turn, name="plan_turn")
        self._mcp.add_tool(self.get_session_stats, name="get_session_stats")
        self._mcp.add_tool(self.get_repo_map, name="get_repo_map")
        # Memory
        self._mcp.add_tool(self.recall_memory, name="recall_memory")
        self._mcp.add_tool(self.remember_memory, name="remember_memory")
        self._mcp.add_tool(self.forget_memory, name="forget_memory")
        self._mcp.add_tool(self.list_sessions, name="list_sessions")
        self._mcp.add_tool(self.search_memory, name="search_memory")
        self._mcp.add_tool(self.get_memory_graph, name="get_memory_graph")
        # Safety
        self._mcp.add_tool(self.check_edit_safe, name="check_edit_safe")
        self._mcp.add_tool(self.check_delete_safe, name="check_delete_safe")
        self._mcp.add_tool(self.plan_refactoring, name="plan_refactoring")
        self._mcp.add_tool(self.get_pr_risk_profile, name="get_pr_risk_profile")
        self._mcp.add_tool(self.get_symbol_provenance, name="get_symbol_provenance")
        self._mcp.add_tool(self.audit_agent_config, name="audit_agent_config")
        # Repository
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
                    from tracera.main import _build_retrieval_pipeline
                    self._pipeline = _build_retrieval_pipeline(
                        self._settings, self._ws_path
                    )
                    log.info("Retrieval pipeline loaded for MCP server")
                except Exception as e:
                    log.exception("Retrieval pipeline failed to load")
                    self._pipeline_error = f"Code index unavailable: {e}"
        return self._pipeline, self._pipeline_error

    def _ensure_pipeline(self) -> str | None:
        """Ensure pipeline is loaded. Returns error message or None."""
        _, err = self._pipeline_once()
        return err

    # ── Lazy memory components ────────────────────────────────────────────────

    def _ensure_memory(self) -> str | None:
        """Lazily initialize memory components. Returns error or None."""
        if self._context_recall is not None:
            return None
        try:
            from tracera.agent.memory import AgentMemory
            from tracera.memory.session import SessionManager
            from tracera.memory.recall import ContextRecall, EnhancedMemoryStore
            from tracera.memory.triples import TripleStore

            memory_dir = self._settings.memory_dir

            self._legacy_memory = AgentMemory(memory_dir)
            self._session_manager = SessionManager(memory_dir)
            self._enhanced_memory = EnhancedMemoryStore(memory_dir)
            self._triple_store = TripleStore()

            triples_path = memory_dir / "memory_triples.json"
            if triples_path.exists():
                try:
                    self._triple_store = TripleStore.load(triples_path)
                except Exception:
                    pass

            self._context_recall = ContextRecall(
                memory_store=self._enhanced_memory,
                session_manager=self._session_manager,
                triple_store=self._triple_store,
                legacy_memory=self._legacy_memory,
            )
            log.info("Memory components loaded for MCP server")
            return None
        except Exception as e:
            log.warning("Memory components failed to load: %s", e)
            return f"Memory unavailable: {e}"

    # ── Tool instance caching ─────────────────────────────────────────────────

    def _get_retrieval_tool(self, name: str) -> Any | None:
        """Get or create a retrieval tool instance."""
        if name in self._retrieval_tools:
            return self._retrieval_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.code_search import (
            FindReferencesTool, FindSymbolTool, GetContextTool,
            GetDependenciesTool, SearchCodeTool,
        )

        symbol_retriever, expander, _, _, context_engine, compressor, *_ = pipeline
        graph_retriever = pipeline[-1]
        graph = graph_retriever.graph

        factory: dict[str, Any] = {
            "search_code": lambda: SearchCodeTool(symbol_retriever, compressor, context_engine, context_recall),
            "find_symbol": lambda: FindSymbolTool(symbol_retriever, context_recall=context_recall),
            "find_references": lambda: FindReferencesTool(graph),
            "get_context": lambda: GetContextTool(
                symbol_retriever, expander, graph_retriever,
                compressor=compressor, context_engine=context_engine, context_recall=context_recall,
            ),
            "get_dependencies": lambda: GetDependenciesTool(graph),
        }
        if name in factory:
            self._retrieval_tools[name] = factory[name]()
        return self._retrieval_tools.get(name)

    def _get_ast_tool(self, name: str) -> Any | None:
        """Get or create an AST tool instance."""
        if name in self._ast_tools:
            return self._ast_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.ast_tools import (
            FindImportersTool, GetBlastRadiusTool, GetCallHierarchyTool,
            FindDeadCodeTool, GetChangedSymbolsTool, GetHotspotsTool,
            SearchAstTool, GetClassHierarchyTool,
        )

        factory: dict[str, Any] = {
            "find_importers": lambda: FindImportersTool(pipeline),
            "get_blast_radius": lambda: GetBlastRadiusTool(pipeline),
            "get_call_hierarchy": lambda: GetCallHierarchyTool(pipeline),
            "find_dead_code": lambda: FindDeadCodeTool(pipeline),
            "get_changed_symbols": lambda: GetChangedSymbolsTool(None, pipeline),
            "get_hotspots": lambda: GetHotspotsTool(None, pipeline),
            "search_ast": lambda: SearchAstTool(None),
            "get_class_hierarchy": lambda: GetClassHierarchyTool(pipeline),
        }
        if name in factory:
            self._ast_tools[name] = factory[name]()
        return self._ast_tools.get(name)

    def _get_refactor_tool(self, name: str) -> Any | None:
        """Get or create a refactor tool instance."""
        if name in self._refactor_tools:
            return self._refactor_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.refactor_tools import (
            PlanRefactoringTool, CheckEditSafeTool,
            CheckDeleteSafeTool, GetPrRiskProfileTool,
        )

        factory: dict[str, Any] = {
            "plan_refactoring": lambda: PlanRefactoringTool(pipeline, None),
            "check_edit_safe": lambda: CheckEditSafeTool(pipeline),
            "check_delete_safe": lambda: CheckDeleteSafeTool(pipeline),
            "get_pr_risk_profile": lambda: GetPrRiskProfileTool(None, pipeline),
        }
        if name in factory:
            self._refactor_tools[name] = factory[name]()
        return self._refactor_tools.get(name)

    def _get_session_tool(self, name: str) -> Any | None:
        """Get or create a session/context tool instance."""
        if name in self._session_tools:
            return self._session_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.session_tools import (
            AssembleTaskContextTool, PlanTurnTool,
            GetRankedContextTool, GetSessionStatsTool, GetRepoMapTool,
        )

        factory: dict[str, Any] = {
            "assemble_task_context": lambda: AssembleTaskContextTool(pipeline),
            "get_ranked_context": lambda: GetRankedContextTool(pipeline),
            "plan_turn": lambda: PlanTurnTool(pipeline),
            "get_session_stats": lambda: GetSessionStatsTool(None),
            "get_repo_map": lambda: GetRepoMapTool(pipeline, None),
        }
        if name in factory:
            self._session_tools[name] = factory[name]()
        return self._session_tools.get(name)

    def _get_provenance_tool(self, name: str) -> Any | None:
        """Get or create a provenance tool instance."""
        if name in self._provenance_tools:
            return self._provenance_tools[name]

        pipeline, err = self._pipeline_once()
        if err or pipeline is None:
            return None

        from tracera.tools.provenance_tools import (
            GetSymbolProvenanceTool, AuditAgentConfigTool,
            GetEndpointImpactTool, GetDependencyCyclesTool,
            GetCouplingMetricsTool,
        )

        factory: dict[str, Any] = {
            "get_symbol_provenance": lambda: GetSymbolProvenanceTool(pipeline, None),
            "audit_agent_config": lambda: AuditAgentConfigTool(None, pipeline),
            "get_endpoint_impact": lambda: GetEndpointImpactTool(pipeline),
            "get_dependency_cycles": lambda: GetDependencyCyclesTool(pipeline),
            "get_coupling_metrics": lambda: GetCouplingMetricsTool(pipeline),
        }
        if name in factory:
            self._provenance_tools[name] = factory[name]()
        return self._provenance_tools.get(name)

    # ── Generic tool runner ───────────────────────────────────────────────────

    async def _run_tool(self, tool_getter, tool_name: str, **kwargs: Any) -> str:
        """Generic runner: get tool, execute, format result."""
        tool = tool_getter(tool_name)
        if tool is None:
            _, err = self._pipeline_once()
            return f"ERROR: {err or 'Tool unavailable — run `tracera index` first.'}"
        result = await tool.execute(**kwargs)
        if not result.success:
            return f"ERROR: {result.error}"
        return result.output

    # ══════════════════════════════════════════════════════════════════════════
    # CODE INTELLIGENCE TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    async def search_code(self, query: str, k: int = 5, language: str | None = None) -> str:
        """Search the indexed codebase (hybrid BM25 + dense retrieval).

        Args:
            query: The search query (e.g. "authentication middleware").
            k: Number of results to return (default 5).
            language: Optional language filter (python, javascript, ...).
        """
        return await self._run_tool(self._get_retrieval_tool, "search_code", query=query, k=k, language=language)

    async def find_symbol(self, name: str, symbol_type: str = "any") -> str:
        """Find the definition of a specific class, function, or method by name.

        Args:
            name: The exact or partial symbol name.
            symbol_type: One of class, function, method, any (default any).
        """
        return await self._run_tool(self._get_retrieval_tool, "find_symbol", name=name, symbol_type=symbol_type)

    async def find_references(self, symbol: str) -> str:
        """Find everywhere a symbol is referenced or called (graph-backed).

        Args:
            symbol: Name of the symbol to find references for.
        """
        return await self._run_tool(self._get_retrieval_tool, "find_references", symbol=symbol)

    async def get_context(self, symbol: str) -> str:
        """Get full context for a symbol: definition, parent, related code.

        Args:
            symbol: The symbol name to get context for.
        """
        return await self._run_tool(self._get_retrieval_tool, "get_context", symbol=symbol)

    async def get_dependencies(self, symbol: str) -> str:
        """Get the dependency chain of a symbol via graph traversal.

        Args:
            symbol: Name of the symbol to inspect.
        """
        return await self._run_tool(self._get_retrieval_tool, "get_dependencies", symbol=symbol)

    async def find_importers(self, path: str, max_results: int = 20) -> str:
        """Find all files/symbols that import a given file or module.

        Args:
            path: File path to find importers for (e.g. 'src/auth.py').
            max_results: Maximum results to return (default 20).
        """
        return await self._run_tool(self._get_ast_tool, "find_importers", path=path, max_results=max_results)

    async def get_blast_radius(self, symbol: str, max_depth: int = 4) -> str:
        """Compute blast radius — what breaks if a symbol changes.

        Args:
            symbol: Symbol name to compute blast radius for.
            max_depth: Maximum traversal depth (default 4).
        """
        return await self._run_tool(self._get_ast_tool, "get_blast_radius", symbol=symbol, max_depth=max_depth)

    async def get_call_hierarchy(self, symbol: str, direction: str = "both", max_depth: int = 3) -> str:
        """Trace callers and callees N levels deep through the call graph.

        Args:
            symbol: Symbol to trace.
            direction: "callers", "callees", or "both" (default both).
            max_depth: Maximum depth (default 3).
        """
        return await self._run_tool(
            self._get_ast_tool, "get_call_hierarchy",
            symbol=symbol, direction=direction, max_depth=max_depth,
        )

    async def find_dead_code(self) -> str:
        """Find symbols unreachable from entry points (main, app, cli, test files)."""
        return await self._run_tool(self._get_ast_tool, "find_dead_code")

    async def get_changed_symbols(self) -> str:
        """Map git diff to affected symbols — shows exactly what changed."""
        return await self._run_tool(self._get_ast_tool, "get_changed_symbols")

    async def get_hotspots(self, top_n: int = 15) -> str:
        """Find risky code by complexity × churn (high complexity + frequent changes).

        Args:
            top_n: Number of hotspots to return (default 15).
        """
        return await self._run_tool(self._get_ast_tool, "get_hotspots", top_n=top_n)

    async def search_ast(self, query: str, preset: str | None = None, language: str | None = None) -> str:
        """Cross-language AST pattern matching (anti-patterns, structural queries).

        Args:
            query: Pattern to search for (e.g. 'call:*.unwrap()', 'string:/password/i').
            preset: Preset detector (empty_catch, bare_except, hardcoded_secret, eval_exec, todo_fixme, magic_number).
            language: Language filter (python, javascript, typescript, etc.).
        """
        kwargs: dict[str, Any] = {"query": query}
        if preset:
            kwargs["preset"] = preset
        if language:
            kwargs["language"] = language
        return await self._run_tool(self._get_ast_tool, "search_ast", **kwargs)

    async def get_class_hierarchy(self, class_name: str) -> str:
        """Traverse inheritance: base classes, subclasses, and methods.

        Args:
            class_name: Name of the class to inspect.
        """
        return await self._run_tool(self._get_ast_tool, "get_class_hierarchy", class_name=class_name)

    async def get_dependency_cycles(self) -> str:
        """Detect circular import chains using NetworkX cycle detection."""
        return await self._run_tool(self._get_provenance_tool, "get_dependency_cycles")

    async def get_coupling_metrics(self, path: str | None = None) -> str:
        """Per-module coupling metrics: afferent (Ca), efferent (Ce), instability ratio.

        Args:
            path: Optional specific module path (default: all modules).
        """
        kwargs: dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        return await self._run_tool(self._get_provenance_tool, "get_coupling_metrics", **kwargs)

    async def get_endpoint_impact(self, endpoint: str) -> str:
        """What breaks if you change an HTTP endpoint or handler.

        Args:
            endpoint: HTTP endpoint path or handler symbol name.
        """
        return await self._run_tool(self._get_provenance_tool, "get_endpoint_impact", endpoint=endpoint)

    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXT TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    async def assemble_task_context(self, task: str, max_tokens: int = 8000) -> str:
        """One-call task orchestration: classify intent, extract anchors, get context.

        Args:
            task: Natural language task description.
            max_tokens: Token budget for the context capsule (default 8000).
        """
        return await self._run_tool(
            self._get_session_tool, "assemble_task_context",
            task=task, max_tokens=max_tokens,
        )

    async def get_ranked_context(self, query: str, max_tokens: int = 8000) -> str:
        """Token-budgeted context pack ranked by relevance.

        Args:
            query: What to get context for.
            max_tokens: Token budget (default 8000).
        """
        return await self._run_tool(
            self._get_session_tool, "get_ranked_context",
            query=query, max_tokens=max_tokens,
        )

    async def plan_turn(self, query: str) -> str:
        """Confidence-guided routing before first read — probes the index.

        Args:
            query: The user's query to analyze.
        """
        return await self._run_tool(self._get_session_tool, "plan_turn", query=query)

    async def get_session_stats(self) -> str:
        """Token usage, file reads, estimated cost, and tool usage breakdown."""
        return await self._run_tool(self._get_session_tool, "get_session_stats")

    async def get_repo_map(self) -> str:
        """Cold-start orientation: PageRank-ranked repository overview."""
        return await self._run_tool(self._get_session_tool, "get_repo_map")

    # ══════════════════════════════════════════════════════════════════════════
    # MEMORY TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    async def recall_memory(self, query: str, k: int = 10) -> str:
        """Search across all memory sources for relevant context.

        Args:
            query: What to search for in memory.
            k: Number of results to return (default 10).
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            context = self._context_recall.recall(
                query, k=k, max_chars=8000,
                include_sessions=True, include_triples=True, include_legacy=True,
            )
            return context or "No relevant memories found."
        except Exception as e:
            return f"ERROR: Memory recall failed: {e}"

    async def remember_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.7,
    ) -> str:
        """Store a piece of information in persistent memory.

        Args:
            content: The memory content to store.
            memory_type: One of fact, rule, preference, relationship, skill, event.
            importance: How important (0.0-1.0, default 0.7).
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            from tracera.memory.taxonomy import (
                create_fact, create_rule, create_preference,
                create_relationship, create_skill, create_event,
            )
            factories = {
                "fact": create_fact, "rule": create_rule,
                "preference": create_preference, "relationship": create_relationship,
                "skill": create_skill, "event": create_event,
            }
            factory = factories.get(memory_type, create_fact)
            memory = factory(content, importance=importance)
            self._enhanced_memory.add(memory)
            return f"Memory stored ({memory_type}): {content[:100]}"
        except Exception as e:
            return f"ERROR: Failed to store memory: {e}"

    async def forget_memory(self, memory_id: str = "", content_match: str = "") -> str:
        """Delete a memory by ID or content match.

        Args:
            memory_id: The ID of the memory to delete.
            content_match: A content fragment to search for and delete.
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            if memory_id:
                deleted = self._enhanced_memory.delete(memory_id)
                return f"Memory {memory_id[:8]} {'deleted' if deleted else 'not found'}."
            if content_match:
                results = self._enhanced_memory.recall(content_match, k=5)
                count = 0
                for mem in results:
                    if content_match.lower() in mem.content.lower():
                        self._enhanced_memory.delete(mem.id)
                        count += 1
                return f"Deleted {count} memories matching '{content_match[:50]}'."
            return "ERROR: Provide either memory_id or content_match."
        except Exception as e:
            return f"ERROR: Failed to delete memory: {e}"

    async def list_sessions(self, k: int = 10) -> str:
        """List recent coding sessions with outcomes and details.

        Args:
            k: Number of sessions to show (default 10).
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            sessions = self._session_manager.sessions[:k]
            if not sessions:
                return "No past sessions found."
            lines = ["## Recent Sessions\n"]
            for i, session in enumerate(sessions, 1):
                duration = ""
                if session.duration_seconds:
                    mins = int(session.duration_seconds / 60)
                    duration = f" ({mins}m)" if mins > 0 else f" ({int(session.duration_seconds)}s)"
                icon = {"success": "✅", "failure": "❌", "partial": "⚠️"}.get(session.outcome, "📋")
                files = f", {len(session.files_touched)} files" if session.files_touched else ""
                lines.append(f"{i}. {icon} [{session.outcome}] {session.task[:70]}{duration}{files}")
                if session.summary:
                    lines.append(f"   Summary: {session.summary[:120]}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: Failed to list sessions: {e}"

    async def search_memory(self, query: str, k: int = 10, memory_type: str | None = None) -> str:
        """Search the enhanced memory store with TF-IDF ranking.

        Args:
            query: What to search for.
            k: Number of results (default 10).
            memory_type: Optional filter (fact, rule, preference, etc.).
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            from tracera.memory.taxonomy import MemoryType
            mt = None
            if memory_type:
                try:
                    mt = MemoryType(memory_type)
                except ValueError:
                    return f"ERROR: Unknown memory type '{memory_type}'. Valid: fact, rule, preference, relationship, skill, event"
            results = self._enhanced_memory.recall(query, k=k, memory_type=mt)
            if not results:
                return "No matching memories found."
            lines = [f"## Memory Search: '{query}'\n"]
            for mem in results:
                icon = {
                    "fact": "📌", "rule": "📏", "relationship": "🔗",
                    "skill": "🛠️", "preference": "⭐", "event": "📋",
                }.get(mem.memory_type.value, "•")
                conf = f" ({mem.confidence:.0%})" if mem.confidence < 0.9 else ""
                lines.append(f"- {icon} [{mem.memory_type.value}] {mem.content}{conf}")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: Memory search failed: {e}"

    async def get_memory_graph(self, concept: str = "", max_depth: int = 2) -> str:
        """Get the knowledge graph of semantic relationships.

        Args:
            concept: Optional concept to focus on (empty = full graph summary).
            max_depth: Traversal depth (default 2).
        """
        err = self._ensure_memory()
        if err:
            return f"ERROR: {err}"
        try:
            if concept:
                triples = self._triple_store.get_neighbors(concept, max_depth=max_depth)
                if not triples:
                    return f"No relationships found for '{concept}'."
                lines = [f"## Knowledge Graph: '{concept}'\n"]
                for t in triples:
                    lines.append(f"- {t.subject} → {t.predicate} → {t.object}")
                return "\n".join(lines)
            else:
                # Summary
                stats = self._enhanced_memory.stats()
                triple_count = self._triple_store.triple_count
                lines = [
                    "## Knowledge Graph Summary\n",
                    f"**Memories:** {stats['total']} total",
                ]
                if stats.get("by_type"):
                    for mt, count in stats["by_type"].items():
                        lines.append(f"  - {mt}: {count}")
                lines.append(f"\n**Triples:** {triple_count}")
                if triple_count > 0:
                    lines.append("\n**Recent relationships:**")
                    for t in self._triple_store.all_triples[:10]:
                        lines.append(f"- {t.subject} → {t.predicate} → {t.object}")
                return "\n".join(lines)
        except Exception as e:
            return f"ERROR: Memory graph failed: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # SAFETY TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    async def check_edit_safe(self, symbol: str) -> str:
        """Preflight check before modifying a symbol — scores risk 0.0-1.0.

        Args:
            symbol: Symbol to check edit safety for.
        """
        return await self._run_tool(self._get_refactor_tool, "check_edit_safe", symbol=symbol)

    async def check_delete_safe(self, symbol: str) -> str:
        """Preflight check before deleting a symbol — checks callers and entry points.

        Args:
            symbol: Symbol to check delete safety for.
        """
        return await self._run_tool(self._get_refactor_tool, "check_delete_safe", symbol=symbol)

    async def plan_refactoring(self, operation: str, symbol: str, target: str = "") -> str:
        """Generate edit-ready refactoring instructions.

        Args:
            operation: Type of refactoring (rename, move, extract, change_signature).
            symbol: Symbol name to refactor.
            target: New name / destination / extracted name / parameter mapping.
        """
        return await self._run_tool(
            self._get_refactor_tool, "plan_refactoring",
            operation=operation, symbol=symbol, target=target,
        )

    async def get_pr_risk_profile(self) -> str:
        """Composite risk score for uncommitted changes or a branch."""
        return await self._run_tool(self._get_refactor_tool, "get_pr_risk_profile")

    async def get_symbol_provenance(self, symbol: str) -> str:
        """Git archaeology — trace every commit that touched a symbol.

        Args:
            symbol: Symbol name to trace provenance for.
        """
        return await self._run_tool(self._get_provenance_tool, "get_symbol_provenance", symbol=symbol)

    async def audit_agent_config(self) -> str:
        """Scan agent config files (CLAUDE.md, .cursorrules) for token waste."""
        return await self._run_tool(self._get_provenance_tool, "audit_agent_config")

    # ══════════════════════════════════════════════════════════════════════════
    # REPOSITORY TOOLS
    # ══════════════════════════════════════════════════════════════════════════

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

        # Structure
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

        # Languages
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

        # Git state
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

        # Tests
        try:
            from tracera.tools.test_runner import TestDiscovery
            fw = TestDiscovery(root).detect_framework()
            lines.append(f"**Tests:** framework={fw}")
        except Exception:
            lines.append("**Tests:** unknown")
        lines.append("")

        # Index status
        index_manifest = self._settings.index_dir / "index_manifest.json"
        lines.append(
            "**Code index:** " + (
                "indexed (retrieval tools active)" if index_manifest.exists()
                else "not indexed (run `tracera index`)"
            )
        )

        # Memory status
        err = self._ensure_memory()
        if err is None:
            mem_count = self._enhanced_memory.count
            triple_count = self._triple_store.triple_count
            session_count = len(self._session_manager.sessions)
            lines.append(f"**Memory:** {mem_count} memories, {triple_count} triples, {session_count} sessions")
        else:
            lines.append("**Memory:** not initialized")

        return "\n".join(lines)


def build_mcp_server(
    settings: Settings | None = None,
    workspace_path: Path | None = None,
) -> FastMCP:
    """Build a FastMCP instance exposing TRACERA's full capabilities."""
    return TraceraMCPServer(settings=settings, workspace_path=workspace_path).mcp


def main(transport: str = "stdio") -> None:
    """Entry point for `tracera mcp serve`."""
    server = TraceraMCPServer()
    server.mcp.run(transport=transport)


if __name__ == "__main__":
    main()
