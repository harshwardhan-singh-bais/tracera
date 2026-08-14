"""
TRACERA configuration profiles.

Each profile provides sensible defaults that can be overridden by .env vars.
"""

from __future__ import annotations

from typing import Any


_PROFILES: dict[str, dict[str, Any]] = {
    "local": {
        "tracera_log_level": "DEBUG",
        "tracera_max_iterations": 100,
        "tracera_max_tool_calls": 500,
        "tracera_command_timeout": 60,
        "tracera_default_provider": "ollama",
        "tracera_default_model": "qwen2.5-coder",
        "tracera_default_temperature": 0.3,
    },
    "development": {
        "tracera_log_level": "INFO",
        "tracera_max_iterations": 50,
        "tracera_max_tool_calls": 200,
        "tracera_command_timeout": 30,
        "tracera_default_provider": "openai",
        "tracera_default_model": "gpt-4o",
        "tracera_default_temperature": 0.2,
    },
    "production": {
        "tracera_log_level": "WARNING",
        "tracera_max_iterations": 30,
        "tracera_max_tool_calls": 100,
        "tracera_command_timeout": 15,
        "tracera_default_provider": "anthropic",
        "tracera_default_model": "claude-opus-4-5",
        "tracera_default_temperature": 0.1,
    },
    "evaluation": {
        "tracera_log_level": "INFO",
        "tracera_max_iterations": 75,
        "tracera_max_tool_calls": 300,
        "tracera_command_timeout": 60,
        "tracera_default_provider": "openai",
        "tracera_default_model": "gpt-4o",
        "tracera_default_temperature": 0.0,   # deterministic for benchmarks
    },
}


def get_profile_defaults(profile: str) -> dict[str, Any]:
    """Return default settings dict for the given profile name."""
    return dict(_PROFILES.get(profile, _PROFILES["development"]))
