"""Tools package."""
from tracera.tools.base import Tool, ToolResult
from tracera.tools.registry import ToolRegistry, create_default_registry
from tracera.tools.read_file import ReadFileTool
from tracera.tools.write_file import WriteFileTool
from tracera.tools.edit_file import EditFileTool
from tracera.tools.list_dir import ListDirTool
from tracera.tools.grep import GrepTool
from tracera.tools.run_command import RunCommandTool

# jCodeMunch-inspired structural analysis tools
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

__all__ = [
    "Tool", "ToolResult", "ToolRegistry", "create_default_registry",
    "ReadFileTool", "WriteFileTool", "EditFileTool",
    "ListDirTool", "GrepTool", "RunCommandTool",
    # AST & structural
    "FindImportersTool", "GetBlastRadiusTool", "GetCallHierarchyTool",
    "FindDeadCodeTool", "GetChangedSymbolsTool", "GetHotspotsTool",
    "SearchAstTool", "GetClassHierarchyTool",
    # Refactoring & safety
    "PlanRefactoringTool", "CheckEditSafeTool",
    "CheckDeleteSafeTool", "GetPrRiskProfileTool",
    # Session & context
    "AssembleTaskContextTool", "PlanTurnTool",
    "GetRankedContextTool", "GetSessionStatsTool", "GetRepoMapTool",
    # Provenance & config
    "GetSymbolProvenanceTool", "AuditAgentConfigTool",
    "GetEndpointImpactTool", "GetDependencyCyclesTool",
    "GetCouplingMetricsTool",
]
