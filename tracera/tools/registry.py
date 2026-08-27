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


def extend_registry_with_retrieval(
    registry: ToolRegistry,
    retriever,
    expander,
    graph_retriever=None,
    context_engine=None,
    compressor=None,
) -> ToolRegistry:
    """
    Phase 27/28 — Extend an existing registry with code-intelligence tools.

    This makes the agent retrieval-aware: it can now call search_code,
    find_symbol, find_definition, and get_context as native tools alongside
    read_file/grep. When a symbol graph is available (Phases 25-26), the
    graph-backed tools find_references and get_dependencies are registered
    too, and get_context includes graph neighbours.

    Phases 29/30: when context_engine/compressor are provided, retrieval
    tool output is assembled + compressed into LLM-ready context blocks
    instead of raw chunk dumps.

    Args:
        registry: An existing ToolRegistry (from create_default_registry).
        retriever: A SymbolAwareRetriever instance.
        expander: A ContextExpander instance.
        graph_retriever: A GraphRetriever instance (optional).
        context_engine: A ContextAssemblyEngine instance (optional).
        compressor: A ContextCompressor instance (optional).

    Returns:
        The same registry, extended with retrieval tools.
    """
    from tracera.tools.code_search import (
        SearchCodeTool,
        FindSymbolTool,
        FindDefinitionTool,
        GetContextTool,
        FindReferencesTool,
        GetDependenciesTool,
    )

    tools: list[Tool] = [
        SearchCodeTool(retriever, compressor, context_engine),
        FindSymbolTool(retriever),
        FindDefinitionTool(retriever, compressor),
    ]

    if graph_retriever is not None:
        graph = graph_retriever.graph
        tools.append(GetContextTool(
            retriever, expander, graph_retriever,
            compressor=compressor, context_engine=context_engine,
        ))
        tools.append(FindReferencesTool(graph))
        tools.append(GetDependenciesTool(graph))
    else:
        tools.append(GetContextTool(
            retriever, expander,
            compressor=compressor, context_engine=context_engine,
        ))

    registry.register_many(tools)
    log.info(
        "Registry extended with retrieval tools: %s",
        ", ".join(t.name for t in tools),
    )
    return registry


def extend_registry_with_ast_tools(
    registry: ToolRegistry,
    retrieval_pipeline=None,
    workspace=None,
    session_manager=None,
) -> ToolRegistry:
    """
    Register jCodeMunch-inspired structural analysis and intelligence tools.

    Adds 20+ tools covering:
    - AST structural queries (find_importers, get_blast_radius, etc.)
    - Refactoring and safety preflight (plan_refactoring, check_edit_safe, etc.)
    - Session economics and context assembly
    - Symbol provenance and agent config auditing
    - Module coupling and dependency cycle detection
    """
    from tracera.tools.ast_tools import (
        FindImportersTool, GetBlastRadiusTool, GetCallHierarchyTool,
        FindDeadCodeTool, GetChangedSymbolsTool, GetHotspotsTool,
        SearchAstTool, GetClassHierarchyTool,
    )
    from tracera.tools.refactor_tools import (
        PlanRefactoringTool, CheckEditSafeTool,
        CheckDeleteSafeTool, GetPrRiskProfileTool,
    )
    from tracera.tools.session_tools import (
        AssembleTaskContextTool, PlanTurnTool,
        GetRankedContextTool, GetSessionStatsTool, GetRepoMapTool,
    )
    from tracera.tools.provenance_tools import (
        GetSymbolProvenanceTool, AuditAgentConfigTool,
        GetEndpointImpactTool, GetDependencyCyclesTool,
        GetCouplingMetricsTool,
    )

    tools: list[Tool] = [
        # Structural analysis
        FindImportersTool(retrieval_pipeline),
        GetBlastRadiusTool(retrieval_pipeline),
        GetCallHierarchyTool(retrieval_pipeline),
        FindDeadCodeTool(retrieval_pipeline),
        GetChangedSymbolsTool(workspace, retrieval_pipeline),
        GetHotspotsTool(workspace, retrieval_pipeline),
        SearchAstTool(workspace),
        GetClassHierarchyTool(retrieval_pipeline),
        # Refactoring & safety
        PlanRefactoringTool(retrieval_pipeline, workspace),
        CheckEditSafeTool(retrieval_pipeline),
        CheckDeleteSafeTool(retrieval_pipeline),
        GetPrRiskProfileTool(workspace, retrieval_pipeline),
        # Session & context
        AssembleTaskContextTool(retrieval_pipeline),
        PlanTurnTool(retrieval_pipeline),
        GetRankedContextTool(retrieval_pipeline),
        GetSessionStatsTool(session_manager),
        GetRepoMapTool(retrieval_pipeline, workspace),
        # Provenance & config
        GetSymbolProvenanceTool(retrieval_pipeline, workspace),
        AuditAgentConfigTool(workspace, retrieval_pipeline),
        GetEndpointImpactTool(retrieval_pipeline),
        GetDependencyCyclesTool(retrieval_pipeline),
        GetCouplingMetricsTool(retrieval_pipeline),
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

