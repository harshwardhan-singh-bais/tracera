"""
TRACERA Providers package.

Provider factory and registry.
"""

from __future__ import annotations

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
from tracera.providers.openai_provider import OpenAIProvider
from tracera.providers.gemini_provider import GeminiProvider
from tracera.providers.ollama_provider import OllamaProvider
from tracera.errors import MissingAPIKeyError, ProviderNotFoundError


def create_provider(
    name: str | None = None,
    model: str | None = None,
    *,
    settings=None,
) -> LLMProvider:
    """
    Factory function — create a provider from settings.

    Args:
        name: Provider name (openai, anthropic, gemini, ollama, groq, mistral, together).
        model: Model ID override.
        settings: Settings instance (fetched from config if None).
    """
    if settings is None:
        from tracera.config import get_settings
        settings = get_settings()

    provider_name = name or settings.tracera_default_provider
    match provider_name.lower():
        case "gemini" | "google":
            api_key = settings.google_api_key
            if not api_key:
                raise MissingAPIKeyError("gemini", "GOOGLE_API_KEY")
            return GeminiProvider(api_key=api_key, default_model=model_id)

        case "ollama":
            return OllamaProvider(
                base_url=f"{settings.ollama_base_url}/v1",
                default_model=model_id,
            )

        case "groq":
            api_key = settings.groq_api_key
            if not api_key:
                raise MissingAPIKeyError("groq", "GROQ_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                default_model=model_id,
                provider_name="groq",
            )

        case "together":
            api_key = settings.together_api_key
            if not api_key:
                raise MissingAPIKeyError("together", "TOGETHER_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
                default_model=model_id,
                provider_name="together",
            )

        case "mistral":
            api_key = settings.mistral_api_key
            if not api_key:
                raise MissingAPIKeyError("mistral", "MISTRAL_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                default_model=model_id,
                provider_name="mistral",
            )

        case _:
            raise ProviderNotFoundError(provider_name)


__all__ = [
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "Role",
    "StreamEvent",
    "TokenUsage",
    "ToolCallRequest",
    "ToolSchema",
    "OpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "create_provider",
]
