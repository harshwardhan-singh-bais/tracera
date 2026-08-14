"""Tests for TRACERA configuration system."""

import os
import pytest
from pathlib import Path


def test_settings_defaults(tmp_path, monkeypatch):
    """Settings load with sensible defaults."""
    monkeypatch.chdir(tmp_path)
    # Isolate from any TRACERA_* env vars the developer may have set
    for var in list(os.environ):
        if var.startswith("TRACERA_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    from tracera.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()  # type: ignore[call-arg]

    assert s.tracera_profile == "development"
    assert s.tracera_max_iterations == 50
    assert s.tracera_default_provider == "groq"
    assert s.tracera_log_level == "INFO"


def test_settings_env_override(tmp_path, monkeypatch):
    """Env vars override defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACERA_MAX_ITERATIONS", "99")
    monkeypatch.setenv("TRACERA_LOG_LEVEL", "debug")
    monkeypatch.setenv("TRACERA_DEFAULT_PROVIDER", "ollama")

    from tracera.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()  # type: ignore[call-arg]

    assert s.tracera_max_iterations == 99
    assert s.tracera_log_level == "DEBUG"  # uppercased
    assert s.tracera_default_provider == "ollama"


def test_settings_allowed_commands():
    """allowed_commands property splits correctly."""
    from tracera.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()  # type: ignore[call-arg]
    cmds = s.allowed_commands
    assert "git" in cmds
    assert "python" in cmds


def test_settings_ensure_dirs(tmp_path, monkeypatch):
    """ensure_dirs creates all TRACERA data directories."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACERA_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TRACERA_DATA_DIR", str(tmp_path / ".tracera"))

    from tracera.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()  # type: ignore[call-arg]
    s.ensure_dirs()

    assert s.memory_dir.exists()
    assert s.logs_dir.exists()
    assert s.index_dir.exists()


def test_profile_defaults():
    """Profile defaults return expected values."""
    from tracera.config.profiles import get_profile_defaults

    dev = get_profile_defaults("development")
    assert dev["tracera_log_level"] == "INFO"
    assert dev["tracera_default_provider"] in ("openai", "groq")

    local = get_profile_defaults("local")
    assert local["tracera_log_level"] == "DEBUG"
    assert local["tracera_default_provider"] == "ollama"

    prod = get_profile_defaults("production")
    assert prod["tracera_log_level"] == "WARNING"


def test_provider_api_key_mapping(monkeypatch):
    """provider_api_key returns correct keys."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")

    from tracera.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()  # type: ignore[call-arg]

    assert s.provider_api_key("groq") == "gsk-test"
    assert s.provider_api_key("mistral") == "mist-test"
    assert s.provider_api_key("ollama") is None
