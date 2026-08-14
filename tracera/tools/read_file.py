"""read_file tool — Phase 7."""

from __future__ import annotations
from typing import Any
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox


class ReadFileTool(Tool):
    """Read the contents of a file in the workspace."""

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the text contents of a file within the workspace. "
            "Returns the full file text. Use this to understand existing code before editing."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-indexed, inclusive). Optional.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (1-indexed, inclusive). Optional.",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            content = await self._ws.read_text(path)
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                sl = max(1, start_line or 1) - 1
                el = min(total_lines, end_line or total_lines)
                lines = lines[sl:el]
                content = "".join(lines)
                note = f"\n[Lines {sl+1}–{el} of {total_lines}]"
            else:
                note = f"\n[{total_lines} lines]"

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=content + note,
                path=path,
                lines=total_lines,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), path=path)
