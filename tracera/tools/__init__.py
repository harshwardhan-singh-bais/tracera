"""Tools package."""
from tracera.tools.base import Tool, ToolResult
from tracera.tools.registry import ToolRegistry, create_default_registry
from tracera.tools.read_file import ReadFileTool
from tracera.tools.write_file import WriteFileTool
from tracera.tools.edit_file import EditFileTool
from tracera.tools.list_dir import ListDirTool
from tracera.tools.grep import GrepTool
from tracera.tools.run_command import RunCommandTool

# Code intelligence tools (Steps 21-41)
from tracera.tools.ast_tools import (
    GetBlastRadiusTool, GetCallHierarchyTool,
    FindDeadCodeTool, GetChangedSymbolsTool, GetHotspotsTool,
    FindReferencesTool, FindImplementationsTool,
    SearchSymbolsTool, GetSymbolSourceTool, GetFileOutlineTool,
    GetRepoMapTool, AssembleCodeContextTool,
    GetDependenciesTool, GetIndexFreshnessTool,
    CalculatePageRankTool, PlanRefactoringTool,
    GetCodeProvenanceTool, AssessChangeRiskTool,
    StructuralSearchTool, GetSessionStatsTool,
    PlanCodeTaskTool,
)

__all__ = [
    "Tool", "ToolResult", "ToolRegistry", "create_default_registry",
    "ReadFileTool", "WriteFileTool", "EditFileTool",
    "ListDirTool", "GrepTool", "RunCommandTool",
    # Code intelligence
    "GetBlastRadiusTool", "GetCallHierarchyTool",
    "FindDeadCodeTool", "GetChangedSymbolsTool", "GetHotspotsTool",
    "FindReferencesTool", "FindImplementationsTool",
    "SearchSymbolsTool", "GetSymbolSourceTool", "GetFileOutlineTool",
    "GetRepoMapTool", "AssembleCodeContextTool",
    "GetDependenciesTool", "GetIndexFreshnessTool",
    "CalculatePageRankTool", "PlanRefactoringTool",
    "GetCodeProvenanceTool", "AssessChangeRiskTool",
    "StructuralSearchTool", "GetSessionStatsTool",
    "PlanCodeTaskTool",
]