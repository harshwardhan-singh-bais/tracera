"""
TRACERA exception hierarchy.

All TRACERA exceptions inherit from TracerError so callers can catch them
with a single except clause when desired.
"""

from __future__ import annotations


class TracerError(Exception):
    """Base class for all TRACERA exceptions."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{super().__str__()} — {self.detail}"
        return super().__str__()


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigError(TracerError):
    """Raised when configuration is invalid or missing."""


class MissingAPIKeyError(ConfigError):
    """Raised when a required provider API key is not set."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(
            f"Missing API key for provider '{provider}'",
            detail=f"Set the {env_var} environment variable.",
        )
        self.provider = provider
        self.env_var = env_var


# ── Workspace / Filesystem ────────────────────────────────────────────────────

class WorkspaceError(TracerError):
    """Raised for workspace filesystem errors."""


class PathTraversalError(WorkspaceError):
    """Raised when a path escapes the workspace sandbox."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Path traversal denied: '{path}'",
            detail="Paths must remain within the workspace root.",
        )
        self.path = path


class FileNotFoundInWorkspaceError(WorkspaceError):
    """Raised when a requested file does not exist inside the workspace."""

    def __init__(self, path: str) -> None:
        super().__init__(f"File not found in workspace: '{path}'")
        self.path = path


class FileSizeLimitError(WorkspaceError):
    """Raised when a file exceeds the configured size limit."""

    def __init__(self, path: str, size: int, limit: int) -> None:
        super().__init__(
            f"File too large: '{path}' ({size:,} bytes)",
            detail=f"Limit is {limit:,} bytes.",
        )
        self.path = path
        self.size = size
        self.limit = limit


# ── Git ───────────────────────────────────────────────────────────────────────

class GitError(TracerError):
    """Raised for git operation failures."""


class NotAGitRepositoryError(GitError):
    """Raised when the workspace is not inside a git repository."""


# ── LLM Providers ─────────────────────────────────────────────────────────────

class ProviderError(TracerError):
    """Raised for LLM provider communication failures."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"LLM provider not found: '{name}'")
        self.name = name


class ProviderRateLimitError(ProviderError):
    """Raised when the LLM provider returns a rate limit error."""


class ProviderAuthError(ProviderError):
    """Raised when the LLM provider rejects credentials."""


class ProviderContextLengthError(ProviderError):
    """Raised when the input exceeds the provider's context limit."""


# ── Tools ─────────────────────────────────────────────────────────────────────

class ToolError(TracerError):
    """Raised for tool execution failures."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Tool not found: '{name}'")
        self.name = name


class ToolValidationError(ToolError):
    """Raised when tool input arguments fail schema validation."""


class CommandExecutionError(ToolError):
    """Raised when a shell command exits with a non-zero code."""

    def __init__(self, command: str, returncode: int, stderr: str) -> None:
        super().__init__(
            f"Command failed (exit {returncode}): {command}",
            detail=stderr[:500] if stderr else None,
        )
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class CommandNotAllowedError(ToolError):
    """Raised when a shell command is not in the allowed list."""

    def __init__(self, command: str) -> None:
        super().__init__(
            f"Command not allowed: '{command}'",
            detail="Add it to TRACERA_ALLOWED_SHELL_COMMANDS.",
        )
        self.command = command


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(TracerError):
    """Raised for agent loop failures."""


class MaxIterationsError(AgentError):
    """Raised when the agent exceeds the maximum iteration limit."""

    def __init__(self, iterations: int) -> None:
        super().__init__(
            f"Agent exceeded maximum iterations ({iterations})",
            detail="Increase TRACERA_MAX_ITERATIONS or simplify the task.",
        )
        self.iterations = iterations


class MaxToolCallsError(AgentError):
    """Raised when the agent exceeds the maximum tool call limit."""


class PlanningError(AgentError):
    """Raised when the planner cannot decompose a task."""


# ── Memory ────────────────────────────────────────────────────────────────────

class MemoryError(TracerError):
    """Raised for persistent memory failures."""
