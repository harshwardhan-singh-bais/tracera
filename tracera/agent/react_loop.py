"""
TRACERA Core ReAct Agent Loop — Phase 8.

Implements the Reason → Act → Observe cycle with:
- Configurable iteration limits
- Tool error handling and retries
- Streaming events for the TUI
- Graceful termination
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from tracera.conversation.state import ConversationState, MessageType
from tracera.errors import AgentError, MaxIterationsError, MaxToolCallsError
from tracera.logging import get_logger, log_agent, log_tool
from tracera.providers.base import LLMMessage, LLMProvider, ToolCallRequest, ToolSchema
from tracera.tools.registry import ToolRegistry

log = get_logger("agent.react")


# ── Agent events (for TUI streaming) ─────────────────────────────────────────

class AgentEventType(str, Enum):
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    RESPONSE_DELTA = "response_delta"
    RESPONSE_COMPLETE = "response_complete"
    PLAN_UPDATE = "plan_update"
    MEMORY_UPDATE = "memory_update"
    ERROR = "error"
    DONE = "done"


@dataclass
class AgentEvent:
    """A streaming event emitted by the agent loop."""
    type: AgentEventType
    iteration: int = 0
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: str | None = None
    tool_success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are TRACERA — an expert AI coding assistant with deep knowledge of software engineering.

You have access to tools that let you read, write, and execute code in the user's workspace.

## Guidelines
- Think carefully before acting. Understand the codebase structure before making changes.
- Use read_file and list_dir to explore before editing.
- Prefer edit_file over write_file for targeted changes.
- Use grep to find relevant code locations.
- Run tests after making changes to verify correctness.
- Be precise and explain your reasoning.
- If you encounter an error, analyze it and retry with a fix.

## Tool Use
Always use tools when you need to interact with the filesystem or run commands.
Do not assume file contents — read them first.

When you have completed the task, provide a clear summary of what was done.
"""


# ── ReAct Agent ───────────────────────────────────────────────────────────────

class ReActAgent:
    """
    Core ReAct (Reason + Act) agent loop.
    
    Flow:
        User message
            ↓
        LLM (with tools) → Tool calls OR final response
            ↓                     ↓
        Execute tools       Return response
            ↓
        Observations → LLM
            ↓
        Repeat until done or limit reached
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        max_iterations: int = 50,
        max_tool_calls: int = 200,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        system_prompt: str | None = None,
        retry_on_tool_error: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self.retry_on_tool_error = retry_on_tool_error
        self.max_retries = max_retries
        self._tool_call_count = 0

    async def run(
        self,
        task: str,
        *,
        conversation: ConversationState | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Run the agent on *task*, yielding AgentEvents.
        This is an async generator — iterate it to drive the loop.
        
        Args:
            task: The user's request.
            conversation: Existing conversation state. Creates new if None.
        """
        return self._run_loop(task, conversation=conversation)

    async def _run_loop(
        self,
        task: str,
        conversation: ConversationState | None,
    ) -> AsyncIterator[AgentEvent]:
        if conversation is None:
            conversation = ConversationState(system_prompt=self.system_prompt)
        else:
            # Ensure system prompt is set
            if not any(m.type == MessageType.SYSTEM for m in conversation.messages):
                conversation.add_system(self.system_prompt)

        conversation.add_user(task)
        self._tool_call_count = 0

        log_agent(f"Starting task: {task[:80]}")
        tools = self.registry.schemas()

        for iteration in range(self.max_iterations):
            yield AgentEvent(
                type=AgentEventType.THINKING,
                iteration=iteration,
                text=f"Iteration {iteration + 1}/{self.max_iterations}",
            )

            # ── LLM call ──────────────────────────────────────────────────────
            try:
                response = await self.provider.complete(
                    conversation.llm_messages(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools if tools else None,
                )
            except Exception as e:
                error_msg = f"LLM call failed: {e}"
                conversation.add_error(error_msg)
                yield AgentEvent(
                    type=AgentEventType.ERROR,
                    iteration=iteration,
                    text=error_msg,
                )
                break

            # Record LLM usage
            conversation.record_llm_usage(
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                latency_ms=response.latency_ms,
            )

            # ── Tool calls ────────────────────────────────────────────────────
            if response.has_tool_calls:
                tool_calls = response.tool_calls or []
                conversation.add_tool_calls(tool_calls, iteration=iteration)

                for tool_call in tool_calls:
                    self._tool_call_count += 1
                    if self._tool_call_count > self.max_tool_calls:
                        raise MaxToolCallsError(self.max_tool_calls)

                    yield AgentEvent(
                        type=AgentEventType.TOOL_START,
                        iteration=iteration,
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                    )
                    log_tool(tool_call.name, tool_call.arguments)

                    # Execute with retries
                    result = await self._execute_with_retry(tool_call)

                    conversation.add_tool_result(
                        tool_call.id,
                        tool_call.name,
                        result.output,
                        success=result.success,
                        duration_ms=result.duration_ms,
                    )

                    yield AgentEvent(
                        type=AgentEventType.TOOL_END,
                        iteration=iteration,
                        tool_name=tool_call.name,
                        tool_output=result.output,
                        tool_success=result.success,
                        metadata={"duration_ms": result.duration_ms},
                    )

                # Continue loop — go back to LLM with observations
                continue

            # ── Final response ────────────────────────────────────────────────
            final_text = response.content or ""
            conversation.add_assistant(final_text, iteration=iteration)

            yield AgentEvent(
                type=AgentEventType.RESPONSE_COMPLETE,
                iteration=iteration,
                text=final_text,
                metadata={
                    "iterations": iteration + 1,
                    "tool_calls": self._tool_call_count,
                    "total_tokens": conversation.stats.total_tokens,
                    "total_latency_ms": conversation.stats.total_latency_ms,
                },
            )
            yield AgentEvent(type=AgentEventType.DONE, iteration=iteration)
            return

        # Hit max iterations
        err = MaxIterationsError(self.max_iterations)
        conversation.add_error(str(err))
        yield AgentEvent(
            type=AgentEventType.ERROR,
            iteration=self.max_iterations,
            text=str(err),
        )
        yield AgentEvent(type=AgentEventType.DONE, iteration=self.max_iterations)

    async def _execute_with_retry(self, tool_call: ToolCallRequest):
        """Execute a tool call with retry on error."""
        from tracera.tools.base import ToolResult

        last_result = None
        for attempt in range(self.max_retries):
            result = await self.registry.execute(
                tool_call.name, tool_call.id, tool_call.arguments
            )
            if result.success or not self.retry_on_tool_error:
                return result
            last_result = result
            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # backoff
                log.debug("Retrying tool %s (attempt %d)", tool_call.name, attempt + 2)

        return last_result  # type: ignore[return-value]

    async def ask(
        self,
        task: str,
        *,
        conversation: ConversationState | None = None,
    ) -> str:
        """
        Convenience method — run the agent and return the final text response.
        Collects all events and returns the response content.
        """
        final_text = ""
        async for event in await self.run(task, conversation=conversation):
            if event.type == AgentEventType.RESPONSE_COMPLETE:
                final_text = event.text or ""
        return final_text
