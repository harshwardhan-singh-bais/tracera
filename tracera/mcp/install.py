"""
Config writer for ``tracera mcp install``.

Writes/updates the TRACERA server entry in a host's JSON config:

- preserves every existing key and other MCP servers;
- merges (not replaces) an existing ``tracera`` entry;
- writes atomically (temp file + rename) with a ``.bak`` sidecar;
- reports exactly what changed.

Codex CLI uses TOML, so it gets a dedicated writer with the same guarantees.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracera.mcp.hosts import Host, stdio_server_entry


@dataclass
class InstallResult:
    host: str
    path: Path
    status: str  # "installed" | "updated" | "unchanged" | "created"
    detail: str


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _entry_matches(existing: Any, entry: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    return existing.get("command") == entry.get("command") and existing.get("args") == entry.get(
        "args"
    )


def install_into_host(host: Host, repo_root: Path, *, force: bool = False) -> InstallResult:
    """Write the TRACERA stdio entry into ``host``'s JSON config."""
    entry = stdio_server_entry(repo_root)
    entry.update(host.entry_extras)

    path = host.config_path()
    if host.project_relative is not None:
        # project-scoped config (VS Code) → relative to the workspace
        path = repo_root / host.project_relative

    if not path.exists():
        _atomic_write_json(path, {host.container_key: {"tracera": entry}})
        return InstallResult(
            host.key, path, "created", f"created {path.name} with the tracera server"
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config root is not a JSON object")
    except (OSError, ValueError) as e:
        return InstallResult(host.key, path, "skipped", f"cannot parse {path.name}: {e}")

    container = data.setdefault(host.container_key, {})
    if not isinstance(container, dict):
        return InstallResult(
            host.key,
            path,
            "skipped",
            f"'{host.container_key}' is not an object; refusing to overwrite",
        )

    if "tracera" in container and _entry_matches(container["tracera"], entry) and not force:
        return InstallResult(host.key, path, "unchanged", "already installed and up to date")

    had_previous = "tracera" in container
    container["tracera"] = entry
    _atomic_write_json(path, data)
    return InstallResult(
        host.key,
        path,
        "updated" if had_previous else "installed",
        f"{'updated existing' if had_previous else 'added'} 'tracera' server in {path}",
    )


# ── Codex CLI (TOML) ──────────────────────────────────────────────────────────


def _toml_server_block(repo_root: Path) -> str:
    py = str(sys.executable).replace("\\", "\\\\")
    root = str(repo_root).replace("\\", "\\\\")
    return (
        f"[mcp_servers.tracera]\n"
        f'command = "{py}"\n'
        f'args = ["-X", "utf8", "-m", "tracera.main", "mcp", "serve", '
        f'"--workspace", "{root}"]\n'
        f"\n"
        f"[mcp_servers.tracera.env]\n"
        f'PYTHONUTF8 = "1"\n'
    )


def install_codex(repo_root: Path, *, force: bool = False) -> InstallResult:
    """Write the TRACERA MCP block into ~/.codex/config.toml."""
    host = "codex"
    from tracera.mcp.hosts import HOSTS

    path = HOSTS["codex"].config_path()
    block = _toml_server_block(repo_root)

    if path.exists():
        content = path.read_text(encoding="utf-8")
        if "[mcp_servers.tracera]" in content and not force:
            return InstallResult(
                host, path, "unchanged", "codex config already has the tracera block"
            )
        if "[mcp_servers.tracera]" in content:
            # replace the existing block
            lines = content.splitlines(keepends=True)
            out, skipping = [], False
            for line in lines:
                if line.strip() == "[mcp_servers.tracera]":
                    skipping = True
                    out.append(block)
                    continue
                if skipping and line.strip().startswith("["):
                    skipping = False
                if not skipping:
                    out.append(line)
            path.with_suffix(".toml.bak").write_text(content, encoding="utf-8")
            path.write_text("".join(out), encoding="utf-8")
            return InstallResult(host, path, "updated", f"updated the tracera block in {path}")
        path.with_suffix(".toml.bak").write_text(content, encoding="utf-8")
        path.write_text(content.rstrip() + "\n\n" + block, encoding="utf-8")
        return InstallResult(host, path, "installed", f"appended the tracera block to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")
    return InstallResult(host, path, "created", f"created {path} with the tracera block")


__all__ = ["InstallResult", "install_into_host", "install_codex"]
