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
from typing import Any, AsyncIterator, Callable

from tracera.conversation.state import ConversationMessage, ConversationState, MessageType
from tracera.errors import AgentError, MaxIterationsError, MaxToolCallsError
from tracera.logging import get_logger, log_agent, log_tool
from tracera.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    ToolSchema,
)
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
        memory_provider: Callable[[], str] | None = None,
        # Phase 10: called with (kind, content) when the agent learns something
        # worth persisting — 'decision' on completion, 'error' on failure.
        memory_writer: Callable[[str, str], None] | None = None,
        # Phase 9: optional TaskDecomposer. When set, the loop decomposes the
        # task up front, emits PLAN_UPDATE events, and marks items done as
        # the work progresses.
        decomposer: Any | None = None,
        streaming: bool = True,
        context_budget_tokens: int = 12_000,
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
        # Phase 10 → 8: injects persistent memory context into the conversation
        self.memory_provider = memory_provider
        # Phase 10: writes agent outcomes (decisions/errors) back to memory
        self.memory_writer = memory_writer
        # Phase 9: planning — decompose the task and track todo progress.
        self.decomposer = decomposer
        self._active_plan: Any | None = None
        self._plan_item_index = 0
        # Phase 4/8: stream the LLM response and emit RESPONSE_DELTA events.
        # Falls back to a plain complete() call when the provider has no real
        # streaming support.
        self.streaming = streaming
        # Context budget: the conversation is compacted before each LLM call
        # so low-TPM providers (Groq free tier etc.) don't get 413s.
        self.context_budget_tokens = context_budget_tokens
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

        # Phase 10: inject persistent memory into the conversation once per session
        if self.memory_provider is not None:
            has_memory = any(
                m.type == MessageType.SYSTEM and m.metadata.get("memory")
                for m in conversation.messages
            )
            if not has_memory:
                memory_ctx = self.memory_provider()
                if memory_ctx:
                    msg = ConversationMessage.system(memory_ctx)
                    msg.metadata["memory"] = True
                    conversation.add(msg)

        conversation.add_user(task)
        self._tool_call_count = 0
        self._plan_item_index = 0

        # Phase 9: decompose the task into a plan up front (best-effort —
        # falls back to a single-step plan if decomposition fails).
        if self.decomposer is not None:
            try:
                self._active_plan = await self.decomposer.decompose(task)
                log.info(
                    "Plan ready: %d steps for %r",
                    len(self._active_plan.items), task[:60],
                )
            except Exception as e:
                log.warning("Task decomposition failed: %s", e)
                self._active_plan = None
            if self._active_plan is not None:
                yield AgentEvent(
                    type=AgentEventType.PLAN_UPDATE,
                    iteration=0,
                    text=self._active_plan.to_markdown(),
                    metadata={
                        "plan": self._active_plan.to_dict(),
                        "progress": self._active_plan.progress,
                    },
                )

        log_agent(f"Starting task: {task[:80]}")
        tools = self.registry.schemas()

        terminated_by_error = False

        for iteration in range(self.max_iterations):
            yield AgentEvent(
                type=AgentEventType.THINKING,
                iteration=iteration,
                text=f"Iteration {iteration + 1}/{self.max_iterations}",
            )

            # ── LLM call ──────────────────────────────────────────────────────
            # Keep the conversation within the token budget so low-TPM
            # providers don't reject it with a 413 (request too large).
            conversation.compact_history(self.context_budget_tokens)
            try:
                if self.streaming:
                    response, deltas = await self._stream_response(
                        conversation.llm_messages(),
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=tools,
                    )
                    # Phase 8: emit token-by-token response deltas for live UIs
                    for delta in deltas:
                        yield AgentEvent(
                            type=AgentEventType.RESPONSE_DELTA,
                            iteration=iteration,
                            text=delta,
                        )
                else:
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
                # Phase 10: remember recurring provider/LLM failures
                if self.memory_writer is not None:
                    self.memory_writer("error", error_msg)
                    yield AgentEvent(
                        type=AgentEventType.MEMORY_UPDATE,
                        iteration=iteration,
                        text="error pattern saved",
                    )
                yield AgentEvent(
                    type=AgentEventType.ERROR,
                    iteration=iteration,
                    text=error_msg,
                )
                terminated_by_error = True
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

                    # Phase 9: advance the plan — mark the next pending item
                    # in_progress so todo state reflects the actual work.
                    if self._active_plan is not None:
                        items = [
                            i for i in self._active_plan.items
                            if i.status.value == "pending"
                        ]
                        if items:
                            items[0].start()
                            yield AgentEvent(
                                type=AgentEventType.PLAN_UPDATE,
                                iteration=iteration,
                                text=self._active_plan.to_markdown(),
                                metadata={
                                    "plan": self._active_plan.to_dict(),
                                    "progress": self._active_plan.progress,
                                },
                            )

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

            # Phase 9: mark the remaining plan items done and report progress
            if self._active_plan is not None:
                for item in self._active_plan.items:
                    if item.status.value in ("pending", "in_progress"):
                        item.complete()
                yield AgentEvent(
                    type=AgentEventType.PLAN_UPDATE,
                    iteration=iteration,
                    text=self._active_plan.to_markdown(),
                    metadata={
                        "plan": self._active_plan.to_dict(),
                        "progress": self._active_plan.progress,
                    },
                )

            # Phase 10: persist the completed decision to memory
            if self.memory_writer is not None and final_text:
                self.memory_writer("decision", final_text)
                yield AgentEvent(
                    type=AgentEventType.MEMORY_UPDATE,
                    iteration=iteration,
                    text="decision saved",
                )

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

        if terminated_by_error:
            # Already reported the real failure above — don't also claim the
            # loop "exceeded max iterations", that's misleading.
            yield AgentEvent(
                type=AgentEventType.DONE,
                iteration=self.max_iterations,
                metadata={"terminated_by_error": True},
            )
            return

        # Hit max iterations (genuinely ran out of iterations)
        err = MaxIterationsError(self.max_iterations)
        conversation.add_error(str(err))
        # Phase 10: persist the error pattern to memory
        if self.memory_writer is not None:
            self.memory_writer("error", str(err))
            yield AgentEvent(
                type=AgentEventType.MEMORY_UPDATE,
                iteration=self.max_iterations,
                text="error pattern saved",
            )
        yield AgentEvent(
            type=AgentEventType.ERROR,
            iteration=self.max_iterations,
            text=str(err),
        )
        yield AgentEvent(type=AgentEventType.DONE, iteration=self.max_iterations)

    async def _stream_response(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int,
        tools: list[ToolSchema],
    ) -> tuple[LLMResponse, list[str]]:
        """
        Stream an LLM response, collecting text deltas and tool calls.

        Returns ``(response, text_deltas)``. If the provider's stream yields
        nothing usable (no real streaming support) or raises, falls back to
        a plain ``complete()`` call so the loop always makes progress.
        """
        import time as _time

        from tracera.providers.base import TokenUsage

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        usage = TokenUsage()
        t0 = _time.perf_counter()

        try:
            async for ev in self.provider.stream(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
            ):
                if ev.type == "text_delta" and ev.text:
                    text_parts.append(ev.text)
                elif ev.type == "tool_call_complete" and ev.tool_call is not None:
                    tool_calls.append(ev.tool_call)
                elif ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
        except Exception as e:
            # Streaming failed → fall back to a plain call
            log.debug("Streaming failed (%s), falling back to complete()", e)
            response = await self.provider.complete(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
            )
            if response.content:
                return response, [response.content]
            return response, []

        # Nothing usable streamed (provider without real streaming)
        if not text_parts and not tool_calls:
            response = await self.provider.complete(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
            )
            if response.content:
                return response, [response.content]
            return response, []

        latency_ms = (_time.perf_counter() - t0) * 1000
        response = LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            usage=usage,
            model=model or self.provider.default_model,
            finish_reason="stop",
            latency_ms=latency_ms,
        )
        return response, text_parts

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
