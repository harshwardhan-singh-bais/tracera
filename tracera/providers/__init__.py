"""
TRACERA Providers package.

Provider factory and registry — supports 8 providers all via OpenAI SDK.

Fallback order (when auto-selecting):
  1. Groq        — fastest, strongest default, great for agent loops
  2. Cerebras    — ultra-fast + strong long-context
  3. NVIDIA NIM  — very strong reasoning + large context
  4. SambaNova   — large-context + high-throughput
  5. Mistral     — strong general fallback, large context
  6. OpenRouter  — meta-fallback (access to many models/providers)
  7. Ollama      — local unlimited, limited by local GPU/model
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
from tracera.providers.nemotron_provider import NemotronProvider
from tracera.providers.ollama_provider import OllamaProvider
from tracera.errors import MissingAPIKeyError, ProviderNotFoundError
from tracera.logging import get_logger

log = get_logger("providers")


# ── Ranked fallback order ─────────────────────────────────────────────────────
# Used when tracera_default_provider = "auto"
# Each entry: (provider_name, settings_attr, base_url, default_model)
_FALLBACK_ORDER = [
    # NOTE: default models below were verified against live accounts in 2026;
    # some classic IDs (llama-3.3-70b-versatile, meta-llama/llama-3.1-405b-
    # instruct, nvidia/llama-3.1-nemotron-70b-instruct) are no longer served.
    ("groq",       "groq_api_key",       "https://api.groq.com/openai/v1",                          "openai/gpt-oss-120b"),
    ("openai",     "openai_api_key",     "https://api.openai.com/v1",                              "gpt-4o"),
    ("cerebras",   "cerebras_api_key",   "https://api.cerebras.ai/v1",                              "gpt-oss-120b"),
    ("nvidia",     "nvidia_api_key",     "https://integrate.api.nvidia.com/v1",                     "meta/llama-3.1-8b-instruct"),
    ("sambanova",  "sambanova_api_key",  "https://api.sambanova.ai/v1",                             "Meta-Llama-3.1-70B-Instruct"),
    ("mistral",    "mistral_api_key",    "https://api.mistral.ai/v1",                               "mistral-large-latest"),
    ("openrouter", "openrouter_api_key", "https://openrouter.ai/api/v1",                            "meta-llama/llama-3.3-70b-instruct"),
    ("anthropic",  "anthropic_api_key",  "https://api.anthropic.com/v1",                             "claude-3-5-sonnet-latest"),
    ("gemini",     "google_api_key",     "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro"),
    ("ollama",     None,                 None,                                                       "llama3.2"),
]

# Per-provider recommended models (used when TRACERA_DEFAULT_MODEL is not set per-provider)
_PROVIDER_MODELS: dict[str, str] = {
    "openai":     "gpt-4o",
    "anthropic":  "claude-3-5-sonnet-latest",
    "gemini":     "gemini-2.5-pro",
    "groq":       "openai/gpt-oss-120b",
    "cerebras":   "gpt-oss-120b",
    "nvidia":     "meta/llama-3.1-8b-instruct",
    "nemotron":   "nvidia/nemotron-3-ultra-550b-a55b",
    "sambanova":  "Meta-Llama-3.1-70B-Instruct",
    "mistral":    "mistral-large-latest",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "together":   "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "ollama":     "llama3.2",
}


def create_provider(
    name: str | None = None,
    model: str | None = None,
    *,
    settings=None,
) -> LLMProvider:
    """
    Factory — create a provider from settings or explicit arguments.

    If name is "auto" or TRACERA_DEFAULT_PROVIDER=auto, automatically selects
    the best available provider based on which API keys are present in .env,
    following the ranked fallback order.

    Supported providers:
        openai      — OpenAI (gpt-4o, etc.)
        gemini      — Google Gemini via OpenAI-compatible endpoint
        groq        — Groq (llama-3.3-70b-versatile, mixtral-8x7b, etc.)
        cerebras    — Cerebras (llama-3.3-70b, etc.)
        nvidia      — NVIDIA NIM (nvidia/llama-3.1-nemotron-70b-instruct, etc.)
        sambanova   — SambaNova Cloud (Meta-Llama-3.1-70B-Instruct, etc.)
        mistral     — Mistral AI (mistral-large-latest, codestral-latest, etc.)
        nemotron    — NVIDIA Nemotron (reasoning with chain-of-thought budget)
        openrouter  — OpenRouter (meta-llama/llama-3.1-405b-instruct, etc.)
        together    — Together AI (open-source models)
        ollama      — Local Ollama (any model installed locally)
        auto        — Automatically select based on ranked fallback order
    """
    if settings is None:
        from tracera.config import get_settings
        settings = get_settings()

    provider_name = (name or settings.tracera_default_provider).lower()
    # Use provider-appropriate default model if none specified
    model_id = model or settings.tracera_default_model or _PROVIDER_MODELS.get(provider_name, "")

    # Auto-select: pick best available provider by ranked fallback order
    if provider_name == "auto":
        return _auto_select_provider(settings, model)

    return _create_named_provider(provider_name, model_id, settings)


def _auto_select_provider(settings, model: str | None = None) -> LLMProvider:
    """
    Walk the ranked fallback order and return the first provider that has
    a valid API key in the current settings (.env).
    """
    for pname, key_attr, base_url, default_model in _FALLBACK_ORDER:
        # Ollama has no key requirement
        if key_attr is None:
            log.info("Auto-selected provider: ollama (local, no key required)")
            from tracera.config import get_settings
            s = settings
            return OllamaProvider(
                base_url=f"{s.ollama_base_url}/v1",
                default_model=model or default_model,
            )

        api_key = getattr(settings, key_attr, None)
        if api_key:
            log.info("Auto-selected provider: %s (key present)", pname)
            model_id = model or _PROVIDER_MODELS.get(pname, default_model)
            return _create_named_provider(pname, model_id, settings)

    raise RuntimeError(
        "No provider API keys found. Add at least one key to .env "
        "(GROQ_API_KEY, CEREBRAS_API_KEY, etc.) or run Ollama locally."
    )


def _create_named_provider(provider_name: str, model_id: str, settings) -> LLMProvider:
    """Create a specific named provider. Raises if key missing."""
    match provider_name:
        # ── 0. OpenAI ────────────────────────────────────────────────────────
        case "openai":
            api_key = settings.openai_api_key
            if not api_key:
                raise MissingAPIKeyError("openai", "OPENAI_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                default_model=model_id or _PROVIDER_MODELS["openai"],
                provider_name="openai",
            )

        # ── 0b. Anthropic (via OpenAI-SDK compatibility layer) ────────────────
        case "anthropic":
            api_key = settings.anthropic_api_key
            if not api_key:
                raise MissingAPIKeyError("anthropic", "ANTHROPIC_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.anthropic.com/v1",
                default_model=model_id or _PROVIDER_MODELS["anthropic"],
                provider_name="anthropic",
            )

        # ── 0c. Gemini (via OpenAI-compatible endpoint) ───────────────────────
        case "gemini":
            api_key = settings.google_api_key
            if not api_key:
                raise MissingAPIKeyError("gemini", "GOOGLE_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                default_model=model_id or _PROVIDER_MODELS["gemini"],
                provider_name="gemini",
            )

        # ── 1. Groq ──────────────────────────────────────────────────────────
        case "groq":
            api_key = settings.groq_api_key
            if not api_key:
                raise MissingAPIKeyError("groq", "GROQ_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                default_model=model_id or _PROVIDER_MODELS["groq"],
                provider_name="groq",
            )

        # ── 2. Cerebras ───────────────────────────────────────────────────────
        case "cerebras":
            api_key = settings.cerebras_api_key
            if not api_key:
                raise MissingAPIKeyError("cerebras", "CEREBRAS_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.cerebras.ai/v1",
                default_model=model_id or _PROVIDER_MODELS["cerebras"],
                provider_name="cerebras",
            )

        # ── 3. NVIDIA NIM ─────────────────────────────────────────────────────
        case "nvidia":
            api_key = settings.nvidia_api_key
            if not api_key:
                raise MissingAPIKeyError("nvidia", "NVIDIA_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
                default_model=model_id or _PROVIDER_MODELS["nvidia"],
                provider_name="nvidia",
            )

        # ── 4. Nemotron (NVIDIA reasoning) ────────────────────────────────────
        case "nemotron":
            api_key = settings.nemotron_api_key
            if not api_key:
                raise MissingAPIKeyError("nemotron", "NEMOTRON_API_KEY")
            return NemotronProvider(
                api_key=api_key,
                default_model=model_id or _PROVIDER_MODELS["nemotron"],
            )

        # ── 5. SambaNova ──────────────────────────────────────────────────────
        case "sambanova":
            api_key = settings.sambanova_api_key
            if not api_key:
                raise MissingAPIKeyError("sambanova", "SAMBANOVA_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.sambanova.ai/v1",
                default_model=model_id or _PROVIDER_MODELS["sambanova"],
                provider_name="sambanova",
            )

        # ── 6. Mistral ────────────────────────────────────────────────────────
        case "mistral":
            api_key = settings.mistral_api_key
            if not api_key:
                raise MissingAPIKeyError("mistral", "MISTRAL_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                default_model=model_id or _PROVIDER_MODELS["mistral"],
                provider_name="mistral",
            )

        # ── 7. OpenRouter ─────────────────────────────────────────────────────
        case "openrouter":
            api_key = settings.openrouter_api_key
            if not api_key:
                raise MissingAPIKeyError("openrouter", "OPENROUTER_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_model=model_id or _PROVIDER_MODELS["openrouter"],
                provider_name="openrouter",
            )

        # ── 8. Together ───────────────────────────────────────────────────────
        case "together":
            api_key = settings.together_api_key
            if not api_key:
                raise MissingAPIKeyError("together", "TOGETHER_API_KEY")
            return OpenAIProvider(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
                default_model=model_id or _PROVIDER_MODELS["together"],
                provider_name="together",
            )

        # ── 9. Ollama (local) ─────────────────────────────────────────────────
        case "ollama":
            return OllamaProvider(
                base_url=f"{settings.ollama_base_url}/v1",
                default_model=model_id or _PROVIDER_MODELS["ollama"],
            )

        case _:
            raise ProviderNotFoundError(provider_name)


def list_available_providers(settings=None) -> list[dict]:
    """
    Return a list of all providers and whether their key is present.
    Ordered by fallback priority (best first).

    Returns:
        List of dicts: {name, available, key_env, model}
    """
    if settings is None:
        from tracera.config import get_settings
        settings = get_settings()

    result = []
    for rank, (pname, key_attr, _, default_model) in enumerate(_FALLBACK_ORDER, 1):
        available = (
            key_attr is None  # Ollama needs no key
            or bool(getattr(settings, key_attr, None))
        )
        result.append({
            "rank": rank,
            "name": pname,
            "available": available,
            "key_env": key_attr.upper() if key_attr else "none",
            "model": _PROVIDER_MODELS.get(pname, default_model),
        })
    return result


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
    "NemotronProvider",
    "OllamaProvider",
    "create_provider",
    "list_available_providers",
    "_PROVIDER_MODELS",
    "_FALLBACK_ORDER",
]
