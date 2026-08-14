"""edit_file tool — Phase 7."""

from __future__ import annotations
from typing import Any
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox


class EditFileTool(Tool):
    """
    Surgically edit a file by replacing specific text.
    Safer than write_file for targeted modifications.
    """

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit an existing file by replacing a specific block of text with new text. "
            "The old_text must exactly match text in the file (including whitespace). "
            "Prefer this over write_file for surgical edits to avoid overwriting unrelated code."
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
                "old_text": {
                    "type": "string",
                    "description": (
                        "The exact text to replace. Must uniquely identify the target location. "
                        "Include enough surrounding context to be unique."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum number of replacements to make (default: 1).",
                    "default": 1,
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int = 1,
        **_: Any,
    ) -> ToolResult:
        try:
            n = await self._ws.edit_text(path, old_text, new_text, count=count)
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=f"Replaced {n} occurrence(s) in {path}",
                path=path,
                replacements=n,
            )
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), path=path)
