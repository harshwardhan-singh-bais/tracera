"""
TRACERA Tool Registry — Phase 6.

Central registry for all tools.
Handles registration, discovery, and execution.
"""

from __future__ import annotations

from typing import Any

from tracera.errors import ToolNotFoundError
from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("tools.registry")


class ToolRegistry:
    """
    Central registry mapping tool names to Tool instances.
    
    The registry is the single source of truth for what the agent can do.
    It also converts tools to LLM-ready schemas.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites if the same name exists."""
        if tool.name in self._tools:
            log.debug("Overwriting tool: %s", tool.name)
        self._tools[tool.name] = tool
        log.debug("Registered tool: %s", tool.name)

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Tool:
        """Return a tool by name, raising ToolNotFoundError if missing."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self):
        """Return all tools as ToolSchema objects for LLM submission."""
        from tracera.providers.base import ToolSchema
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(
        self, name: str, tool_call_id: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """
        Look up and execute a tool by name.
        
        Validates the tool exists, then delegates to safe_execute().
        """
        tool = self.get(name)
        log.debug("Executing tool %s with %s", name, list(arguments.keys()))
        result = await tool.safe_execute(tool_call_id, arguments)
        if not result.success:
            log.warning("Tool %s failed: %s", name, result.error)
        return result

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


def create_default_registry(workspace=None) -> ToolRegistry:
    """
    Create a ToolRegistry pre-loaded with all default coding tools.
    
    Args:
        workspace: WorkspaceSandbox instance. If None, uses current directory.
    """
    from tracera.tools.read_file import ReadFileTool
    from tracera.tools.write_file import WriteFileTool
    from tracera.tools.edit_file import EditFileTool
    from tracera.tools.list_dir import ListDirTool
    from tracera.tools.grep import GrepTool
    from tracera.tools.run_command import RunCommandTool
    from tracera.tools.git_tool import GitTool
    from tracera.workspace.sandbox import WorkspaceSandbox
    from pathlib import Path

    if workspace is None:
        workspace = WorkspaceSandbox(Path(".").resolve())

    registry = ToolRegistry()
    registry.register_many([
        ReadFileTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        ListDirTool(workspace),
        GrepTool(workspace),
        RunCommandTool(workspace),
        GitTool(workspace),
    ])
    return registry


# ── Tool Profiles / Tiers (Step 23) ────────────────────────────────────────────

TOOL_PROFILES: dict[str, list[str]] = {
    "core": [
        "search_code",
        "find_symbol",
        "find_definition",
        "get_context",
        "get_dependencies",
        "get_file_outline",
        "get_repo_map",
        "assemble_code_context",
    ],
    "standard": [
        "search_code",
        "find_symbol",
        "search_symbols",
        "find_definition",
        "get_symbol_source",
        "get_context",
        "get_dependencies",
        "get_file_outline",
        "get_repo_map",
        "assemble_code_context",
        "find_references",
        "get_call_hierarchy",
        "get_class_hierarchy",
        "get_blast_radius",
        "get_changed_symbols",
        "get_index_freshness",
        # Read-only analysis tools advertised by /features — every alias in
        # SLASH_TOOLS must resolve to a registered tool (see test_slash_commands).
        "find_dead_code",
        "get_hotspots",
        "calculate_pagerank",
        "plan_refactoring",
        "get_code_provenance",
        "assess_change_risk",
        "structural_search",
        "get_session_stats",
        "plan_code_task",
        "find_implementations",
    ],
    "advanced": [
        "search_code",
        "find_symbol",
        "find_definition",
        "get_context",
        "get_dependencies",
        "get_file_outline",
        "get_repo_map",
        "assemble_code_context",
        "find_references",
        "get_call_hierarchy",
        "get_class_hierarchy",
        "get_dependencies",
        "get_blast_radius",
        "get_changed_symbols",
        "get_index_freshness",
        "find_dead_code",
        "get_hotspots",
        "calculate_pagerank",
        "plan_refactoring",
        "get_code_provenance",
        "assess_change_risk",
        "structural_search",
        "get_session_stats",
        "plan_code_task",
        "find_implementations",
        "search_symbols",
        "get_symbol_source",
    ]
}


def get_tools_for_profile(profile: str) -> list[str]:
    """Get the list of tool names that should be registered for a given profile."""
    return TOOL_PROFILES.get(profile, TOOL_PROFILES["standard"])


def filter_tools_by_profile(tools: list[Tool], profile: str) -> list[Tool]:
    """Filter a list of tools to only include those allowed by the specified profile."""
    allowed = set(get_tools_for_profile(profile))
    return [tool for tool in tools if tool.name in allowed]


def extend_registry_with_retrieval(
    registry: ToolRegistry,
    retriever: Any,
    expander: Any,
    graph_retriever: Any | None = None,
    context_engine: Any | None = None,
    compressor: Any | None = None,
    context_recall: Any | None = None,
    tool_profile: str = "standard",
    retrieval_pipeline: Any | None = None,
    workspace: Any | None = None,
) -> ToolRegistry:
    """
    Phase 21-41 — Extend an existing registry with full code-intelligence tools.
    
    Implements tool tiering/profiles (Step 23) and registers all code intelligence
    tools based on the selected profile. Core tools are always available, standard
    adds common analysis tools, advanced includes all analytical capabilities.

    Args:
        registry: An existing ToolRegistry (from create_default_registry).
        retriever: A SymbolAwareRetriever instance.
        expander: A ContextExpander instance.
        graph_retriever: A GraphRetriever instance (optional).
        context_engine: A ContextAssemblyEngine instance (optional).
        compressor: A ContextCompressor instance (optional).
        context_recall: Memory recall integration (optional).
        tool_profile: Tool tier profile (core/standard/advanced)

    Returns:
        The same registry, extended with retrieval tools.
    """
    from tracera.tools.code_search import (
        SearchCodeTool,
        FindSymbolTool,
        FindDefinitionTool,
        GetContextTool,
    )
    from tracera.tools.ast_tools import (
        # Core tools (always available)
        GetFileOutlineTool,
        GetRepoMapTool,
        AssembleCodeContextTool,
        GetDependenciesTool,
        # Standard tools
        FindReferencesTool,
        GetCallHierarchyTool,
        GetClassHierarchyTool,
        GetBlastRadiusTool,
        GetChangedSymbolsTool,
        GetIndexFreshnessTool,
        # Advanced tools (only if profile allows)
        FindDeadCodeTool,
        GetHotspotsTool,
        CalculatePageRankTool,
        PlanRefactoringTool,
        GetCodeProvenanceTool,
        AssessChangeRiskTool,
        StructuralSearchTool,
        GetSessionStatsTool,
        PlanCodeTaskTool,
        FindImplementationsTool,
        SearchSymbolsTool,
        GetSymbolSourceTool,
    )

    # ── Pipeline-tuple plumbing ─────────────────────────────────────────────
    # The ast_tools family all accept ``retrieval_pipeline`` (the 10-tuple built
    # by ``_build_retrieval_pipeline``) and extract the SymbolGraph themselves
    # via ``_get_graph``. Passing anything else (GraphRetriever, bare retriever,
    # compressor…) made every one of those tools fail at runtime with
    # AttributeError ('GraphRetriever' object has no attribute '_g', etc.).
    #
    # When this function is called without the full pipeline (older call sites
    # pass individual components), we still reconstruct a minimal tuple shape:
    #   [indexer, retriever, expander, _, context_engine, compressor, _, _, _, graph_retriever]
    if retrieval_pipeline is not None:
        pipeline = retrieval_pipeline
    else:
        pipeline = (None, retriever, expander, None, context_engine, compressor, None, None, None, graph_retriever)

    # Create all code intelligence tools
    all_ci_tools = [
        # Core (from code_search.py - these are the primary agent-facing tools)
        SearchCodeTool(retriever, compressor=compressor, context_engine=context_engine, context_recall=context_recall),
        FindSymbolTool(retriever, context_recall=context_recall),
        FindDefinitionTool(retriever, compressor=compressor, context_recall=context_recall),
        GetContextTool(retriever, expander, graph_retriever, compressor=compressor, context_engine=context_engine, context_recall=context_recall),
        # NOTE: the pipeline-backed GetDependenciesTool from ast_tools is used
        # here, NOT the code_search variant — the latter wraps a bare SymbolGraph
        # and crashes with AttributeError when handed the 10-tuple.
        GetDependenciesTool(pipeline),
        GetFileOutlineTool(graph_retriever.graph if graph_retriever is not None and hasattr(graph_retriever, "graph") else graph_retriever),
        GetRepoMapTool(graph_retriever.graph if graph_retriever is not None and hasattr(graph_retriever, "graph") else graph_retriever),
        AssembleCodeContextTool(retrieval_pipeline=pipeline),
        # Standard
        FindReferencesTool(pipeline),
        GetCallHierarchyTool(pipeline),
        GetClassHierarchyTool(pipeline),
        GetBlastRadiusTool(pipeline),
        GetChangedSymbolsTool(retrieval_pipeline=pipeline),
        GetIndexFreshnessTool(pipeline),
        # Advanced
        FindDeadCodeTool(pipeline),
        GetHotspotsTool(workspace, retrieval_pipeline=pipeline),
        CalculatePageRankTool(pipeline),
        PlanRefactoringTool(pipeline),
        GetCodeProvenanceTool(pipeline),
        AssessChangeRiskTool(pipeline),
        StructuralSearchTool(pipeline),
        GetSessionStatsTool(retrieval_pipeline=pipeline),
        PlanCodeTaskTool(pipeline),
        FindImplementationsTool(pipeline),
        SearchSymbolsTool(pipeline),
        GetSymbolSourceTool(pipeline),
    ]

    # Filter out None values and apply profile filtering
    ci_tools = [t for t in all_ci_tools if t is not None]
    filtered_tools = filter_tools_by_profile(ci_tools, tool_profile)
    
    # Register all filtered tools
    registry.register_many(filtered_tools)
    log.info(f"Registered {len(filtered_tools)} code intelligence tools for profile '{tool_profile}'")
    
    return registry


def extend_registry_with_ast_tools(
    registry: ToolRegistry,
    retrieval_pipeline=None,
    workspace=None,
    session_manager=None,
) -> ToolRegistry:
    """
    Register jCodeMunch-inspired structural analysis and intelligence tools.

    Adds unique AST-level tools that complement the code-search tools:
    - AST structural queries (find_importers)
    - Refactoring and safety preflight (check_edit_safe, check_delete_safe, plan_refactoring, get_pr_risk_profile)
    - Code provenance and git history (get_symbol_provenance)
    - Risk assessment (assess_change_risk)
    - Module coupling and dependency cycle detection (get_dependency_cycles, get_coupling_metrics, get_endpoint_impact)
    - Structural pattern matching (structural_search)
    - PageRank importance (calculate_pagerank)
    """
    from tracera.tools.ast_tools import (
        FindImportersTool,
        CalculatePageRankTool,
        PlanRefactoringTool,
        GetCodeProvenanceTool,
        AssessChangeRiskTool,
        StructuralSearchTool,
    )
    from tracera.tools.refactor_tools import (
        PlanRefactoringTool as RefactorPlanTool,
        CheckEditSafeTool,
        CheckDeleteSafeTool,
        GetPrRiskProfileTool,
    )
    from tracera.tools.provenance_tools import (
        GetSymbolProvenanceTool,
        GetDependencyCyclesTool,
        GetCouplingMetricsTool,
        GetEndpointImpactTool,
        AuditAgentConfigTool,
    )

    tools: list[Tool] = [
        # Unique structural analysis (not in code_search)
        FindImportersTool(retrieval_pipeline),
        # Refactoring and safety preflight
        CheckEditSafeTool(retrieval_pipeline),
        CheckDeleteSafeTool(retrieval_pipeline),
        GetPrRiskProfileTool(None, retrieval_pipeline),
        # Code provenance
        GetSymbolProvenanceTool(retrieval_pipeline, workspace),
        GetDependencyCyclesTool(retrieval_pipeline),
        GetCouplingMetricsTool(retrieval_pipeline),
        GetEndpointImpactTool(retrieval_pipeline),
        AuditAgentConfigTool(None, retrieval_pipeline),
        # Risk assessment
        AssessChangeRiskTool(retrieval_pipeline, workspace),
        # Structural pattern matching
        StructuralSearchTool(retrieval_pipeline),
        # PageRank importance
        CalculatePageRankTool(retrieval_pipeline),
    ]

    # Wire up the pipeline reference for tools that need it
    for tool in tools:
        if hasattr(tool, '_pipeline') and tool._pipeline is None:
            tool._pipeline = retrieval_pipeline

    registry.register_many(tools)
    log.info(
        "Registry extended with AST/intelligence tools: %s",
        ", ".join(t.name for t in tools),
    )
    return registry