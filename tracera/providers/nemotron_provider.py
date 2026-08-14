"""
TRACERA Nemotron Provider.

Thin subclass of OpenAIProvider that injects the special `extra_body`
required by NVIDIA's Nemotron reasoning models.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from tracera.providers.base import (
    LLMMessage,
    LLMResponse,
    StreamEvent,
    ToolSchema,
)
from tracera.providers.openai_provider import OpenAIProvider
from tracera.logging import get_logger

log = get_logger("providers.nemotron")

_NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
_NEMOTRON_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class NemotronProvider(OpenAIProvider):
    """
    Provider adapter for NVIDIA Nemotron reasoning models.

    Uses the OpenAI SDK against NVIDIA's inference endpoint, but injects
    `extra_body` with `reasoning_budget` to enable chain-of-thought.
    """

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = _NEMOTRON_DEFAULT_MODEL,
        reasoning_budget: int = 16384,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=_NEMOTRON_BASE_URL,
            default_model=default_model,
            provider_name="nemotron",
        )
        self._reasoning_budget = reasoning_budget
        log.debug("NemotronProvider initialised: model=%s budget=%d", default_model, reasoning_budget)

    def _extra_body(self) -> dict:
        return {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": self._reasoning_budget,
        }

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        import openai
        import json
        import time

        from tracera.errors import ProviderAuthError, ProviderRateLimitError, ProviderError
        from tracera.providers.base import TokenUsage

        model_id = model or self._default_model
        oai_messages = self._messages_to_openai(messages, system)

        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(),
        )
        if tools:
            kwargs["tools"] = [t.to_openai_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"Nemotron authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderRateLimitError(f"Nemotron rate limit: {e}")
        except Exception as e:
            raise ProviderError(f"Nemotron request failed: {e}")

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

        from tracera.providers.base import LLMResponse
        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency_ms,
        )
