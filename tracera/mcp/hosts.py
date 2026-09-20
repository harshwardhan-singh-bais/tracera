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
    #: reshape the canonical stdio entry into whatever this host expects.
    #: Default (None) writes it verbatim.
    entry_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def build_entry(self, repo_root: Path) -> dict[str, Any]:
        """Canonical stdio entry, shaped for this host."""
        entry = dict(stdio_server_entry(repo_root))
        entry.update(self.entry_extras)
        if self.entry_transform is not None:
            entry = self.entry_transform(entry)
        return entry


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


def _vscode_globalstorage(ext_id: str, filename: str) -> Path:
    """Shared helper for VS Code extension globalStorage configs."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        base = Path(appdata) / "Code" / "User" / "globalStorage"
    elif sys.platform == "darwin":
        base = _home("Library", "Application Support", "Code", "User", "globalStorage")
    else:
        base = _home(".config", "Code", "User", "globalStorage")
    return base / ext_id / "settings" / filename


def _roo_path() -> Path:
    return _vscode_globalstorage(
        "rooveterinaryinc.roo-cline", "cline_mcp_settings.json"
    )


def _kilo_path() -> Path:
    return _vscode_globalstorage("kilocode.kilo-code", "mcp_settings.json")


def _zed_path() -> Path:
    """Zed uses `context_servers`, not `mcpServers`."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "Zed" / "settings.json"
    if sys.platform == "darwin":
        return _home("Library", "Application Support", "Zed", "settings.json")
    return _home(".config", "zed", "settings.json")


def _continue_path() -> Path:
    return _home(".continue", "config.json")


def _amazonq_path() -> Path:
    return _home(".aws", "amazonq", "mcp.json")


def _opencode_path() -> Path:
    """OpenCode stores MCP servers under a top-level `mcp` key."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(_home("AppData", "Roaming")))
        return Path(appdata) / "opencode" / "opencode.json"
    return _home(".config", "opencode", "opencode.json")


def _qwen_path() -> Path:
    return _home(".qwen", "settings.json")


def _kiro_path() -> Path:
    return _home(".kiro", "settings", "mcp.json")


def _claude_code_project() -> Path:
    return Path(".mcp.json")


def _cursor_project() -> Path:
    return Path(".cursor") / "mcp.json"


def _zed_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Zed keeps MCP servers under `context_servers` and wants a flat command.

    Shape follows Zed's documented ``context_servers`` entry; if the server does
    not appear, check Zed's current docs — this schema has changed across
    releases.
    """
    return {
        "source": "custom",
        "command": entry["command"],
        "args": list(entry.get("args", [])),
        "env": dict(entry.get("env", {})),
    }


def _argv_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    OpenCode wants one argv array plus an explicit transport `type`.

    Shape follows OpenCode's documented ``mcp`` schema (``type: "local"``).
    """
    return {
        "type": "local",
        "command": [entry["command"], *entry.get("args", [])],
        "enabled": True,
        "environment": dict(entry.get("env", {})),
    }


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
    # ── project-scoped variants ───────────────────────────────────────────
    # Preferred where a host supports them: the wiring travels with the repo
    # instead of depending on the user's home-directory layout.
    "claude-code-project": Host(
        "claude-code-project",
        "Claude Code (project .mcp.json)",
        "project",
        _claude_code_project,
        project_relative=".mcp.json",
    ),
    "cursor-project": Host(
        "cursor-project",
        "Cursor (project .cursor/mcp.json)",
        "project",
        _cursor_project,
        project_relative=".cursor/mcp.json",
    ),
    # ── other VS Code-family agents ───────────────────────────────────────
    "roo": Host("roo", "Roo Code (VS Code)", "user", _roo_path),
    "kilo": Host("kilo", "Kilo Code (VS Code)", "user", _kilo_path),
    # ── hosts with a non-standard entry shape ─────────────────────────────
    "zed": Host(
        "zed",
        "Zed",
        "user",
        _zed_path,
        container_key="context_servers",
        entry_transform=_zed_entry,
    ),
    "opencode": Host(
        "opencode",
        "OpenCode",
        "user",
        _opencode_path,
        container_key="mcp",
        entry_transform=_argv_entry,
    ),
    # ── other JSON-config hosts ───────────────────────────────────────────
    "continue": Host("continue", "Continue", "user", _continue_path),
    "amazonq": Host("amazonq", "Amazon Q Developer", "user", _amazonq_path),
    "qwen": Host("qwen", "Qwen Code", "user", _qwen_path),
    "kiro": Host("kiro", "Kiro", "user", _kiro_path),
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
