"""
MCP host catalog — where each MCP client stores its server config, and how.

Used by ``tracera mcp install`` to write a ready-to-use TRACERA server entry
into the config file of whichever MCP host the user runs (Claude Desktop,
Claude Code, Cursor, VS Code, Codex CLI, Windsurf, Gemini CLI, Cline, …).

Every entry produces the same logical server::

    command: <python or uv>
    args:    run --directory <repo> tracera mcp serve   (uv)  |  -m tracera.main mcp serve
    env:     PYTHONUTF8=1   (guarantees a clean UTF-8 stdio JSON-RPC stream)

JSON shapes differ per host, hence the small ``inject`` strategies below:
    - ``mcpServers``  – top-level {"mcpServers": {...}}}  (Claude Desktop,
                        Claude Code, Codex, Windsurf, Cline, Gemini CLI)
    - ``.vscode/mcp.json`` – {"servers": {...}} with a "type" field
    - ``.cursor/mcp.json`` – {"mcpServers": {...}} (same shape, project path)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Host:
    key: str
    name: str
    scope: str  # "user" or "project"
    config_path: Callable[[], Path]
    #: "mcpServers" (top-level key) or "servers" (vs code style)
    container_key: str = "mcpServers"
    #: extra key merged into the server entry (e.g. {"type": "stdio"} for VS Code)
    entry_extras: dict[str, Any] = field(default_factory=dict)
    #: VS Code mcp.json lives in the project, not next to the host config
    project_relative: str | None = None


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


def _claude_desktop_path() -> Path:
    if sys.platform == "darwin":
        return _home("Library", "Application Support", "Claude", "claude_desktop_config.json")
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return _home(".config", "Claude", "claude_desktop_config.json")


def _claude_code_path() -> Path:
    return _home(".claude.json")


def _codex_path() -> Path:
    return _home(".codex", "config.toml")


def _cursor_path() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "Cursor" / "mcp.json"
    return _home(".cursor", "mcp.json")


def _windsurf_path() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "Windsurf" / "mcp_config.json"
    return _home(".codeium", "windsurf", "mcp_config.json")


def _cline_path() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return (
            Path(appdata)
            / "Code"
            / "User"
            / "globalStorage"
            / "saoudrizwan.claude-dev"
            / "settings"
            / "cline_mcp_settings.json"
        )
    return _home(
        ".config",
        "Code",
        "User",
        "globalStorage",
        "saoudrizwan.claude-dev",
        "settings",
        "cline_mcp_settings.json",
    )


def _vscode_mcp_json() -> Path:
    return Path(".vscode") / "mcp.json"


def _gemini_path() -> Path:
    return _home(".gemini", "settings.json")


def _antigravity_path() -> Path:
    """Google Antigravity — Gemini-agent IDE; user-level MCP config."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "Antigravity" / "mcp_config.json"
    if sys.platform == "darwin":
        return _home("Library", "Application Support", "Antigravity", "mcp_config.json")
    return _home(".antigravity", "mcp_config.json")


HOSTS: dict[str, Host] = {
    "claude-desktop": Host("claude-desktop", "Claude Desktop", "user", _claude_desktop_path),
    "claude-code": Host("claude-code", "Claude Code", "user", _claude_code_path),
    "codex": Host(
        "codex",
        "Codex CLI",
        "user",
        _codex_path,
    ),
    "cursor": Host("cursor", "Cursor", "project", _cursor_path),
    "windsurf": Host("windsurf", "Windsurf", "user", _windsurf_path),
    "vscode": Host(
        "vscode",
        "VS Code (Copilot)",
        "project",
        _vscode_mcp_json,
        container_key="servers",
        entry_extras={"type": "stdio"},
        project_relative=".vscode/mcp.json",
    ),
    "cline": Host("cline", "Cline (VS Code)", "user", _cline_path),
    "gemini-cli": Host("gemini-cli", "Gemini CLI", "user", _gemini_path),
    "antigravity": Host("antigravity", "Google Antigravity", "user", _antigravity_path),
}


def stdio_server_entry(repo_root: Path) -> dict[str, Any]:
    """
    Build the MCP server entry for TRACERA over stdio.

    Uses ``<python> -m tracera.main mcp serve`` with PYTHONUTF8=1 — the
    layout every host accepts, pointing at *this* checkout so edits apply
    without reinstalling.
    """
    return {
        "command": sys.executable,
        "args": [
            "-X",
            "utf8",
            "-m",
            "tracera.main",
            "mcp",
            "serve",
            "--workspace",
            str(repo_root),
        ],
        "env": {"PYTHONUTF8": "1"},
        "cwd": str(repo_root),
    }


def http_server_entry(url: str) -> dict[str, Any]:
    """Server entry for a remotely running `tracera mcp serve -t sse/http`."""
    return {"url": url, "transport": "http"}


__all__ = [
    "Host",
    "HOSTS",
    "stdio_server_entry",
    "http_server_entry",
]
