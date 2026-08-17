"""
TRACERA OpenAI Provider.

Supports OpenAI, Azure OpenAI, Groq, Together, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from tracera.errors import (
    MissingAPIKeyError,
    ProviderAuthError,
    ProviderContextLengthError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderError,
)


def _classify_error(e: Exception) -> ProviderError:
    """
    Map an openai-SDK exception to the right TRACERA error subclass.

    Permanent failures (bad key 401, payment required 402, forbidden 403,
    model/endpoint not found 404) raise ProviderUnavailableError so failover
    knows retrying is pointless. Rate limits (429) stay transient.
    """
    import openai

    if isinstance(e, openai.AuthenticationError):
        return ProviderAuthError(f"OpenAI authentication failed: {e}")
    if isinstance(e, openai.RateLimitError):
        return ProviderRateLimitError(f"OpenAI rate limit: {e}")
    if isinstance(e, openai.APIStatusError):
        status = getattr(e, "status_code", 0)
        detail = str(e)
        if status == 401:
            return ProviderAuthError(f"OpenAI authentication failed: {e}")
        if status in (402, 403, 404):
            return ProviderUnavailableError(f"Provider unavailable (HTTP {status}): {e}")
        if status == 400 and "context" in detail.lower():
            return ProviderContextLengthError(f"Context too long: {e}")
        return ProviderError(f"OpenAI request failed (HTTP {status}): {e}")
    if isinstance(e, openai.APIConnectionError):
        return ProviderError(f"OpenAI connection error: {e}")
    return ProviderError(f"OpenAI request failed: {e}")
from tracera.logging import get_logger
from tracera.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    Role,
    StreamEvent,
    TokenUsage,
    ToolCallRequest,
    ToolSchema,
)

log = get_logger("providers.openai")


class OpenAIProvider(LLMProvider):
    """
    Provider adapter for OpenAI and OpenAI-compatible APIs.

    Supports:
    - OpenAI (api.openai.com)
    - Azure OpenAI (custom base_url)
    - Groq (api.groq.com)
    - Together AI (api.together.xyz)
    - Ollama (via openai-compatible mode)
    - Local vLLM / LM Studio
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        organization: str | None = None,
        default_model: str = "gpt-4o",
        provider_name: str = "openai",
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ProviderError(
                "openai package not installed",
                detail="Run: pip install openai",
            )

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._default_model = default_model
        self._provider_name = provider_name
        log.debug("OpenAIProvider initialised: %s @ %s", provider_name, base_url)

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    # ── Message conversion ────────────────────────────────────────────────────

    def _messages_to_openai(
        self, messages: list[LLMMessage], system: str | None
    ) -> list[dict]:
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                d: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    d["content"] = msg.content
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": __import__("json").dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif msg.role == Role.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                })
        return result

    def _parse_tool_calls(self, raw_calls: Any) -> list[ToolCallRequest]:
        import json
        result = []
        for tc in raw_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result.append(ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))
        return result

    # ── Complete ──────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        import openai

        model_id = model or self._default_model
        oai_messages = self._messages_to_openai(messages, system)

        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = [t.to_openai_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _classify_error(e)

        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        usage_obj = response.usage

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = self._parse_tool_calls(choice.message.tool_calls)

        usage = TokenUsage(
            prompt_tokens=usage_obj.prompt_tokens if usage_obj else 0,
            completion_tokens=usage_obj.completion_tokens if usage_obj else 0,
            total_tokens=usage_obj.total_tokens if usage_obj else 0,
        )

        log.debug(
            "OpenAI %s: %d tok in %.0fms",
            model_id, usage.total_tokens, latency_ms,
        )

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency_ms,
        )

    # ── Stream ────────────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        import json
        import openai

        model_id = model or self._default_model
        oai_messages = self._messages_to_openai(messages, system)

        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = [t.to_openai_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        # Track tool call assembly
        pending_tool_calls: dict[int, dict] = {}

        try:
            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        # Usage chunk
                        if chunk.usage:
                            yield StreamEvent(
                                type="usage",
                                usage=TokenUsage(
                                    prompt_tokens=chunk.usage.prompt_tokens,
                                    completion_tokens=chunk.usage.completion_tokens,
                                    total_tokens=chunk.usage.total_tokens,
                                ),
                            )
                        continue

                    delta = choice.delta

                    # Text delta
                    if delta.content:
                        yield StreamEvent(type="text_delta", text=delta.content)

                    # Tool call delta
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in pending_tool_calls:
                                pending_tool_calls[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tc_delta.function.name if tc_delta.function else "",
                                    "args_str": "",
                                }
                            if tc_delta.function and tc_delta.function.arguments:
                                pending_tool_calls[idx]["args_str"] += (
                                    tc_delta.function.arguments
                                )
                            if tc_delta.id:
                                pending_tool_calls[idx]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                pending_tool_calls[idx]["name"] = tc_delta.function.name

                    if choice.finish_reason in ("tool_calls", "stop"):
                        # Emit completed tool calls
                        for idx in sorted(pending_tool_calls):
                            tc_data = pending_tool_calls[idx]
                            try:
                                args = json.loads(tc_data["args_str"] or "{}")
                            except Exception:
                                args = {}
                            yield StreamEvent(
                                type="tool_call_complete",
                                tool_call=ToolCallRequest(
                                    id=tc_data["id"],
                                    name=tc_data["name"],
                                    arguments=args,
                                ),
                            )
                        pending_tool_calls.clear()

        except Exception as e:
            raise _classify_error(e)

        yield StreamEvent(type="done")

    async def count_tokens(
        self, messages: list[LLMMessage], *, model: str | None = None
    ) -> int:
        """Use tiktoken if available, otherwise fall back to heuristic."""
        try:
            import tiktoken
            model_id = model or self._default_model
            try:
                enc = tiktoken.encoding_for_model(model_id)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            total = 0
            for m in messages:
                total += 4  # message overhead
                total += len(enc.encode(m.content or ""))
            return total
        except ImportError:
            return await super().count_tokens(messages, model=model)
