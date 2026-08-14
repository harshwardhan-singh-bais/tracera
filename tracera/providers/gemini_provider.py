"""
TRACERA Google Gemini Provider.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from tracera.errors import ProviderAuthError, ProviderError, ProviderRateLimitError
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

log = get_logger("providers.gemini")


class GeminiProvider(LLMProvider):
    """Provider adapter for Google Gemini models."""

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "gemini-2.5-flash",
    ) -> None:
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ProviderError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        self._genai.configure(api_key=api_key)
        self._default_model = default_model
        log.debug("GeminiProvider initialised: %s", default_model)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _to_gemini_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[dict]]:
        """Convert to Gemini history format. Returns (system_instruction, history)."""
        system_parts: list[str] = []
        history: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content or "")
            elif msg.role == Role.USER:
                history.append({"role": "user", "parts": [msg.content or ""]})
            elif msg.role == Role.ASSISTANT:
                history.append({"role": "model", "parts": [msg.content or ""]})

        return "\n\n".join(system_parts) or None, history

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
        model_id = model or self._default_model
        auto_system, history = self._to_gemini_messages(messages)
        effective_system = system or auto_system

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model_kwargs: dict[str, Any] = {"model_name": model_id}
        if effective_system:
            model_kwargs["system_instruction"] = effective_system

        try:
            import asyncio
            gemini_model = self._genai.GenerativeModel(
                **model_kwargs,
                generation_config=gen_config,
            )

            # Extract last user message and use history for the rest
            if history and history[-1]["role"] == "user":
                last_message = history[-1]["parts"][0]
                chat_history = history[:-1]
            else:
                last_message = ""
                chat_history = history

            chat = gemini_model.start_chat(history=chat_history)
            t0 = time.perf_counter()
            response = await asyncio.to_thread(chat.send_message, last_message)
            latency_ms = (time.perf_counter() - t0) * 1000

            text = response.text or None
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
            )

            return LLMResponse(
                content=text,
                tool_calls=None,
                usage=usage,
                model=model_id,
                finish_reason="stop",
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise ProviderError(f"Gemini request failed: {e}")

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
        # Gemini streaming via async iterator
        import asyncio

        model_id = model or self._default_model
        auto_system, history = self._to_gemini_messages(messages)
        effective_system = system or auto_system

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        model_kwargs: dict[str, Any] = {"model_name": model_id}
        if effective_system:
            model_kwargs["system_instruction"] = effective_system

        try:
            gemini_model = self._genai.GenerativeModel(
                **model_kwargs,
                generation_config=gen_config,
            )
            if history and history[-1]["role"] == "user":
                last_message = history[-1]["parts"][0]
                chat_history = history[:-1]
            else:
                last_message = ""
                chat_history = history

            chat = gemini_model.start_chat(history=chat_history)
            response_stream = await asyncio.to_thread(
                chat.send_message, last_message, stream=True
            )
            for chunk in response_stream:
                if chunk.text:
                    yield StreamEvent(type="text_delta", text=chunk.text)

        except Exception as e:
            raise ProviderError(f"Gemini stream failed: {e}")

        yield StreamEvent(type="done")
