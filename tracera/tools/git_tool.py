"""
git tool — wires Phase 3 (Git integration) into the agent's toolbelt.

Read-only git operations only: status, diff, log, branch, file_history.
"""

from __future__ import annotations

from typing import Any

from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox
from tracera.git.operations import GitRepo, detect_git_repo
from tracera.logging import get_logger

log = get_logger("tools.git")


class GitTool(Tool):
    """Safe, read-only git operations on the workspace repository."""

    name = "git"
    description = (
        "Run read-only git operations on the workspace repository: "
        "status (working tree state), diff (uncommitted changes), log (commit history), "
        "branch (branches), or file_history (history of one file). "
        "Use this to understand what has changed before editing or verifying work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "diff", "log", "branch", "file_history"],
                "description": "Which git operation to run. Default: status.",
                "default": "status",
            },
            "path": {
                "type": "string",
                "description": "File path (relative) — used by diff and file_history.",
            },
            "max_count": {
                "type": "integer",
                "description": "Max commits for log/file_history (default: 10).",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, workspace: WorkspaceSandbox) -> None:
        self._ws = workspace

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.parameters

    async def execute(
        self,
        operation: str = "status",
        path: str | None = None,
        max_count: int = 10,
    ) -> ToolResult:
        try:
            repo = detect_git_repo(self._ws.root)
            if repo is None:
                return ToolResult.fail(
                    self.name, "",
                    f"Not a git repository: {self._ws.root}",
                    operation=operation,
                )

            op = (operation or "status").lower()
            output = self._run_operation(repo, op, path, max_count)
            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                operation=op,
            )
        except Exception as e:
            log.error("git tool failed (%s): %s", operation, e)
            return ToolResult.fail(self.name, "", str(e), operation=operation)

    def _run_operation(
        self, repo: GitRepo, op: str, path: str | None, max_count: int
    ) -> str:
        if op == "status":
            status = repo.status()
            lines = [f"Branch: {status.branch}"]
            lines.append(f"Dirty: {'yes' if status.is_dirty else 'no'}")
            if status.staged:
                lines.append("\nStaged:")
                lines.extend(f"  + {p}" for p in status.staged[:50])
            if status.unstaged:
                lines.append("\nModified (unstaged):")
                lines.extend(f"  M {p}" for p in status.unstaged[:50])
            if status.untracked:
                lines.append("\nUntracked:")
                lines.extend(f"  ? {p}" for p in status.untracked[:50])
            return "\n".join(lines)

        if op == "diff":
            diff = repo.diff(path=path) if path else repo.diff()
            if not diff.diff_text.strip():
                return "No changes."
            return (
                f"{diff.files_changed} file(s) changed, "
                f"+{diff.insertions} / -{diff.deletions}\n\n"
                + diff.diff_text[:4000]
            )

        if op == "log":
            commits = repo.log(max_count=max_count)
            if not commits:
                return "No commits."
            return "\n".join(str(c) for c in commits)

        if op == "branch":
            branches = repo.branches()
            if not branches:
                return "No branches."
            return "\n".join(branches)

        if op == "file_history":
            if not path:
                return "file_history requires a 'path' argument."
            commits = repo.file_history(path, max_count=max_count)
            if not commits:
                return f"No history for {path}."
            return "\n".join(str(c) for c in commits)

        return f"Unknown git operation: {op}"
