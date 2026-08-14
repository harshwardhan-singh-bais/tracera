"""grep tool — Phase 7."""

from __future__ import annotations
from typing import Any
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox


class GrepTool(Tool):
    """Search for a pattern in workspace files."""

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search for a regex pattern in workspace files. "
            "Returns matching lines with file name and line number. "
            "Use this to find where specific code, functions, or strings are defined."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Default: '.' (entire workspace).",
                    "default": ".",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search. Default: false.",
                    "default": False,
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to files with these extensions, e.g. ['.py', '.ts'].",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                    "default": 50,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        case_insensitive: bool = False,
        file_extensions: list[str] | None = None,
        max_results: int = 50,
        **_: Any,
    ) -> ToolResult:
        try:
            results = await self._ws.grep(
                pattern,
                path,
                case_insensitive=case_insensitive,
                include_extensions=file_extensions,
                max_results=max_results,
            )

            if not results:
                return ToolResult.ok(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"No matches for pattern: {pattern!r}",
                    pattern=pattern,
                    count=0,
                )

            lines = []
            for r in results:
                lines.append(f"{r['file']}:{r['line']}: {r['content']}")

            output = "\n".join(lines)
            if len(results) >= max_results:
                output += f"\n\n[Showing first {max_results} matches]"

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                pattern=pattern,
                count=len(results),
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), pattern=pattern)
