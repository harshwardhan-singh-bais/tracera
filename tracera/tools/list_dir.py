"""list_dir tool — Phase 7."""

from __future__ import annotations
from typing import Any
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox


class ListDirTool(Tool):
    """List files and directories in the workspace."""

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List files and subdirectories in a workspace directory. "
            "Use this to understand the project structure before reading or editing files."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace root. Default: '.'",
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "How many levels deep to list (default: 2, max: 5).",
                    "default": 2,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.'). Default: false.",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        path: str = ".",
        max_depth: int = 2,
        include_hidden: bool = False,
        **_: Any,
    ) -> ToolResult:
        try:
            max_depth = min(max_depth, 5)
            entries = await self._ws.list_directory(
                path, include_hidden=include_hidden, max_depth=max_depth
            )

            lines = []
            for entry in entries:
                indent = "  " * (len(entry.relative.parts) - 1)
                if entry.is_dir:
                    lines.append(f"{indent}📁 {entry.relative.name}/")
                else:
                    size = _human_size(entry.size)
                    lines.append(f"{indent}📄 {entry.relative.name} ({size})")

            output = "\n".join(lines) or "(empty directory)"
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=f"Directory: {path}\n\n{output}",
                path=path,
                count=len(entries),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), path=path)


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size //= 1024
    return f"{size}GB"
