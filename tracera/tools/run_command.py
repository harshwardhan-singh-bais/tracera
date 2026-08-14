"""run_command tool — Phase 7."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

from tracera.errors import CommandNotAllowedError
from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult
from tracera.workspace.sandbox import WorkspaceSandbox

log = get_logger("tools.run_command")


class RunCommandTool(Tool):
    """
    Execute a shell command inside the workspace directory.
    
    Only pre-approved commands are allowed. All commands run with the
    workspace root as the working directory.
    """

    def __init__(
        self,
        workspace: WorkspaceSandbox,
        *,
        allowed_commands: list[str] | None = None,
        timeout: int = 30,
    ) -> None:
        self._ws = workspace
        self._timeout = timeout
        if allowed_commands is None:
            from tracera.config import get_settings
            try:
                allowed_commands = get_settings().allowed_commands
            except Exception:
                allowed_commands = ["git", "python", "python3", "pytest", "npm", "node",
                                     "cargo", "make", "ruff", "mypy"]
        self._allowed = set(allowed_commands)

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command in the workspace directory. "
            "Only approved commands are allowed (git, python, pytest, npm, cargo, etc.). "
            "Use for running tests, linting, building, or git operations."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The full command to run, e.g. 'pytest tests/ -v' or 'git status'.",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Subdirectory to run in (relative to workspace root). Default: workspace root.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default: configured value.",
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables to set.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["command"],
        }

    def _check_allowed(self, command: str) -> None:
        """Raise if the command's executable is not in the allowed list."""
        parts = shlex.split(command)
        if not parts:
            raise ValueError("Empty command")
        executable = Path(parts[0]).name.lower()
        # Strip extensions on Windows (e.g. python.exe → python)
        executable = executable.replace(".exe", "").replace(".cmd", "")
        if executable not in self._allowed:
            raise CommandNotAllowedError(executable)

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            self._check_allowed(command)
        except CommandNotAllowedError as e:
            return ToolResult.fail(self.name, "", str(e), command=command)
        except Exception as e:
            return ToolResult.fail(self.name, "", str(e), command=command)

        # Resolve working directory
        if working_dir:
            try:
                cwd = str(self._ws.resolve(working_dir))
            except Exception as e:
                return ToolResult.fail(self.name, "", str(e), command=command)
        else:
            cwd = str(self._ws.root)

        effective_timeout = timeout or self._timeout

        # Build environment
        cmd_env = dict(os.environ)
        if env:
            cmd_env.update(env)

        log.debug("Running: %s (cwd=%s, timeout=%ds)", command, cwd, effective_timeout)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=cmd_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult.fail(
                    self.name,
                    "",
                    f"Command timed out after {effective_timeout}s: {command}",
                    command=command,
                    returncode=-1,
                )

            stdout_str = stdout.decode("utf-8", errors="replace").rstrip()
            stderr_str = stderr.decode("utf-8", errors="replace").rstrip()
            returncode = proc.returncode or 0

            parts = []
            if stdout_str:
                parts.append(stdout_str)
            if stderr_str:
                parts.append(f"[stderr]\n{stderr_str}")

            output = "\n".join(parts) if parts else "(no output)"
            output += f"\n\n[Exit code: {returncode}]"

            if returncode != 0:
                return ToolResult.fail(
                    self.name,
                    "",
                    output,
                    command=command,
                    returncode=returncode,
                )

            return ToolResult.ok(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                command=command,
                returncode=returncode,
            )

        except Exception as e:
            return ToolResult.fail(self.name, "", f"Failed to run command: {e}", command=command)
