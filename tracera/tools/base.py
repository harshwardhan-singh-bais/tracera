"""
TRACERA Tool Abstraction — Phase 6.

Defines the Tool ABC and ToolResult that all coding tools implement.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    The result of executing a tool.
    
    Separates structured output (for the agent) from display text (for the UI).
    """
    tool_name: str
    tool_call_id: str
    output: str               # text sent back to LLM as tool_result
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @classmethod
    def ok(
        cls,
        tool_name: str,
        tool_call_id: str,
        output: str,
        **metadata: Any,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=output,
            success=True,
            metadata=metadata,
        )

    @classmethod
    def fail(
        cls,
        tool_name: str,
        tool_call_id: str,
        error: str,
        **metadata: Any,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=f"ERROR: {error}",
            success=False,
            error=error,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        status = "ok" if self.success else "fail"
        return f"<ToolResult {self.tool_name} [{status}] {self.output[:40]!r}>"


class Tool(abc.ABC):
    """
    Abstract base class for all TRACERA tools.
    
    Each tool must define:
    - name: unique identifier used by the LLM
    - description: plain English description of what the tool does
    - parameters_schema: JSON Schema object for the tool's input
    - execute(): the implementation
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool name (snake_case, e.g. 'read_file')."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Description shown to the LLM."""

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """
        JSON Schema for the tool's input parameters.
        Must be a valid JSON Schema 'object' with 'properties'.
        """

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Must not raise exceptions — return ToolResult.fail() on error.
        All expensive I/O should be async.
        """

    def to_schema(self) -> "from tracera.providers.base import ToolSchema; ToolSchema":  # type: ignore[return-value]
        from tracera.providers.base import ToolSchema
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    async def safe_execute(
        self, tool_call_id: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """
        Validate arguments and execute the tool.
        Catches all exceptions and wraps them in ToolResult.fail().
        """
        t0 = time.perf_counter()
        try:
            result = await self.execute(**arguments)
        except Exception as e:
            result = ToolResult.fail(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                error=str(e),
            )
        result.tool_call_id = tool_call_id
        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    def __repr__(self) -> str:
        return f"<Tool name={self.name}>"
