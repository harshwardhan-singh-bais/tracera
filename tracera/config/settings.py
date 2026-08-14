"""
TRACERA configuration system.

Uses Pydantic Settings v2 for type-safe, env-driven configuration.
Supports multiple profiles: local | development | production | evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


Profile = Literal["local", "development", "production", "evaluation"]


class Settings(BaseSettings):
    """
    TRACERA global configuration.
    All values can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",          # prefixed fields use explicit names below
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime ──────────────────────────────────────────────────────────────

    tracera_profile: Profile = Field("development", alias="TRACERA_PROFILE")
    tracera_workspace: Path = Field(Path("."), alias="TRACERA_WORKSPACE")
    tracera_log_level: str = Field("INFO", alias="TRACERA_LOG_LEVEL")
    tracera_log_file: Path | None = Field(None, alias="TRACERA_LOG_FILE")
    tracera_data_dir: Path = Field(Path(".tracera"), alias="TRACERA_DATA_DIR")

    # ── Default Model ─────────────────────────────────────────────────────────

    tracera_default_provider: str = Field("gemini", alias="TRACERA_DEFAULT_PROVIDER")
    tracera_default_model: str = Field("gemini-2.5-pro", alias="TRACERA_DEFAULT_MODEL")
    tracera_default_temperature: float = Field(0.2, alias="TRACERA_DEFAULT_TEMPERATURE")
    tracera_default_max_tokens: int = Field(8192, alias="TRACERA_DEFAULT_MAX_TOKENS")

    # ── Safety Limits ─────────────────────────────────────────────────────────

    tracera_max_iterations: int = Field(50, alias="TRACERA_MAX_ITERATIONS")
    tracera_max_tool_calls: int = Field(200, alias="TRACERA_MAX_TOOL_CALLS")
    tracera_max_context_tokens: int = Field(128_000, alias="TRACERA_MAX_CONTEXT_TOKENS")
    tracera_command_timeout: int = Field(30, alias="TRACERA_COMMAND_TIMEOUT")
    tracera_indexing_max_file_size: int = Field(
        2 * 1024 * 1024, alias="TRACERA_INDEXING_MAX_FILE_SIZE"
    )

    # ── Security ──────────────────────────────────────────────────────────────

    tracera_secret_key: str = Field("dev-secret", alias="TRACERA_SECRET_KEY")
    tracera_allowed_shell_commands: str = Field(
        "git,python,python3,pytest,npm,node,cargo,make,ruff,mypy",
        alias="TRACERA_ALLOWED_SHELL_COMMANDS",
    )
    tracera_require_confirmation_for: str = Field(
        "delete", alias="TRACERA_REQUIRE_CONFIRMATION_FOR"
    )



    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    google_cloud_project: str | None = Field(None, alias="GOOGLE_CLOUD_PROJECT")

    groq_api_key: str | None = Field(None, alias="GROQ_API_KEY")
    together_api_key: str | None = Field(None, alias="TOGETHER_API_KEY")
    mistral_api_key: str | None = Field(None, alias="MISTRAL_API_KEY")
    cohere_api_key: str | None = Field(None, alias="COHERE_API_KEY")

    ollama_base_url: str = Field(
        "http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    # ── Embedding ─────────────────────────────────────────────────────────────

    tracera_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        alias="TRACERA_EMBEDDING_MODEL",
    )
    tracera_embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        "cpu", alias="TRACERA_EMBEDDING_DEVICE"
    )
    huggingface_token: str | None = Field(None, alias="HUGGINGFACE_TOKEN")

    # ── Vector DB ─────────────────────────────────────────────────────────────

    lancedb_uri: str = Field(".tracera/index/lancedb", alias="LANCEDB_URI")

    # ── Observability ─────────────────────────────────────────────────────────

    langfuse_public_key: str | None = Field(None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        "https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # ── Derived properties ────────────────────────────────────────────────────

    @field_validator("tracera_workspace", mode="before")
    @classmethod
    def resolve_workspace(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @field_validator("tracera_data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        return Path(v)

    @field_validator("tracera_log_level", mode="before")
    @classmethod
    def upper_log_level(cls, v: str) -> str:
        return str(v).upper()

    @model_validator(mode="after")
    def resolve_data_dir_relative(self) -> "Settings":
        """Make data_dir absolute relative to workspace if not absolute."""
        if not self.tracera_data_dir.is_absolute():
            object.__setattr__(
                self,
                "tracera_data_dir",
                self.tracera_workspace / self.tracera_data_dir,
            )
        return self

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def allowed_commands(self) -> list[str]:
        return [c.strip() for c in self.tracera_allowed_shell_commands.split(",")]

    @property
    def confirmation_required_tools(self) -> list[str]:
        return [t.strip() for t in self.tracera_require_confirmation_for.split(",")]

    @property
    def memory_dir(self) -> Path:
        return self.tracera_data_dir / "memory"

    @property
    def logs_dir(self) -> Path:
        return self.tracera_data_dir / "logs"

    @property
    def index_dir(self) -> Path:
        return self.tracera_data_dir / "index"

    @property
    def is_production(self) -> bool:
        return self.tracera_profile == "production"

    @property
    def is_evaluation(self) -> bool:
        return self.tracera_profile == "evaluation"

    def ensure_dirs(self) -> None:
        """Create all required TRACERA data directories."""
        for directory in (
            self.tracera_data_dir,
            self.memory_dir,
            self.logs_dir,
            self.index_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def provider_api_key(self, provider: str) -> str | None:
        """Return the API key for a given provider name."""
        mapping: dict[str, str | None] = {
            "gemini": self.google_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "together": self.together_api_key,
            "mistral": self.mistral_api_key,
            "cohere": self.cohere_api_key,
            "ollama": None,  # local, no key needed
        }
        return mapping.get(provider.lower())


# ── Singleton pattern ─────────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton, loading from .env on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (useful in tests)."""
    global _settings
    _settings = None
