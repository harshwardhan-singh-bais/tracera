"""write_file tool — Phase 7."""

from __future__ import annotations
from typing import Any
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox


class WriteFileTool(Tool):
    """Write or create a file in the workspace."""

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file in the workspace, creating it if it does not exist. "
            "Overwrites the entire file. For partial edits, prefer edit_file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file.",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8).",
                    "default": "utf-8",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        **_: Any,
    ) -> ToolResult:
        try:
            exists_before = self._ws.exists(path)
            resolved = await self._ws.write_text(path, content, encoding=encoding)
            action = "updated" if exists_before else "created"
            lines = content.count("\n") + 1
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=f"File {action}: {path} ({lines} lines, {len(content)} chars)",
                path=path,
                action=action,
                lines=lines,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), path=path)
