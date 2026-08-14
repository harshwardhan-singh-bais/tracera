"""
TRACERA Ollama Provider.

Connects to a local Ollama instance via the OpenAI-compatible endpoint.
"""

from __future__ import annotations

from tracera.logging import get_logger
from tracera.providers.base import ToolSchema
from tracera.providers.openai_provider import OpenAIProvider

log = get_logger("providers.ollama")


class OllamaProvider(OpenAIProvider):
    """
    Provider adapter for local Ollama models.

    Uses Ollama's OpenAI-compatible API endpoint.
    Requires Ollama to be running: https://ollama.ai
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        default_model: str = "qwen2.5-coder",
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=base_url,
            default_model=default_model,
            provider_name="ollama",
        )
        log.debug("OllamaProvider initialised: %s @ %s", default_model, base_url)

    @property
    def name(self) -> str:
        return "ollama"

    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    self._client.base_url.copy_with(path="/api/tags")  # type: ignore
                )
                return resp.status_code == 200
        except Exception:
            return False
