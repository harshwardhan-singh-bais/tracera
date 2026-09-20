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
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracera.mcp.hosts import Host


@dataclass
class InstallResult:
    host: str
    path: Path
    status: str  # "installed" | "updated" | "unchanged" | "created"
    detail: str


# ── JSONC tolerance ──────────────────────────────────────────────────────────

#: Hosts whose config file is JSON-with-comments rather than strict JSON.
#: Zed ships `settings.json` with a commented header explaining the format,
#: so a plain `json.loads` fails on a *default* install.
_JSONC_HOSTS = frozenset({"zed"})


def _strip_jsonc(text: str) -> str:
    """
    Remove ``//`` and ``/* */`` comments and trailing commas.

    String-aware: a ``//`` inside a value (``"https://…"``) must survive, so we
    track quoting and escapes rather than regexing the text blindly.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False

    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    # A removed comment can leave a dangling comma before the closing brace.
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_config(path: Path, *, jsonc: bool = False) -> tuple[dict[str, Any] | None, bool, str]:
    """
    Parse a config file.

    Returns ``(data, had_comments, error)``. ``data`` is None when the file
    cannot be parsed. ``had_comments`` is True when comments had to be
    stripped — the caller reports that, because writing the file back as
    plain JSON drops them (the ``.bak`` keeps the original).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, False, f"cannot read {path.name}: {e}"

    had_comments = jsonc and ("//" in raw or "/*" in raw)
    text = _strip_jsonc(raw) if jsonc else raw

    try:
        data = json.loads(text)
    except ValueError as e:
        return None, had_comments, f"cannot parse {path.name}: {e}"

    if not isinstance(data, dict):
        return None, had_comments, f"{path.name} root is not a JSON object"
    return data, had_comments, ""


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _entry_matches(existing: Any, entry: dict[str, Any]) -> bool:
    """
    True when the on-disk entry already equals what we would write.

    Compares the whole mapping rather than just ``command``/``args`` so it stays
    correct for hosts whose entry shape differs (Zed nests under ``command``,
    OpenCode uses a single argv array) — those have no top-level ``args`` key.
    """
    if not isinstance(existing, dict):
        return False
    return existing == entry


def install_into_host(host: Host, repo_root: Path, *, force: bool = False) -> InstallResult:
    """Write the TRACERA stdio entry into ``host``'s JSON config."""
    entry = host.build_entry(repo_root)

    path = host.config_path()
    if host.project_relative is not None:
        # project-scoped config (VS Code) → relative to the workspace
        path = repo_root / host.project_relative

    if not path.exists():
        _atomic_write_json(path, {host.container_key: {"tracera": entry}})
        return InstallResult(
            host.key, path, "created", f"created {path.name} with the tracera server"
        )

    data, had_comments, error = _load_config(path, jsonc=host.key in _JSONC_HOSTS)
    if data is None:
        return InstallResult(host.key, path, "skipped", error)

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
    detail = f"{'updated existing' if had_previous else 'added'} 'tracera' server in {path}"
    if had_comments:
        detail += " (comments normalised; original kept in the .bak)"
    return InstallResult(
        host.key,
        path,
        "updated" if had_previous else "installed",
        detail,
    )


def host_config_path(host: Host, repo_root: Path) -> Path:
    """Where ``host``'s config lives for this workspace (project hosts included)."""
    if host.project_relative is not None:
        return repo_root / host.project_relative
    return host.config_path()


def inspect_host(host: Host, repo_root: Path) -> tuple[str, str]:
    """
    Read-only check of whether TRACERA is registered in ``host``.

    Returns ``(status, detail)`` with status one of ``installed`` (entry present
    and current), ``outdated`` (present but differs from what we would write),
    ``missing``, or ``unreadable``. Used by ``tracera integrate doctor`` and by
    ``apply`` to avoid rewriting a config that is already correct.
    """
    path = host_config_path(host, repo_root)

    if host.key == "codex":
        if not path.exists():
            return "missing", f"{path} does not exist"
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return "unreadable", f"cannot read {path}: {e}"
        if "[mcp_servers.tracera]" not in content:
            return "missing", "no [mcp_servers.tracera] block"
        if _toml_server_block(repo_root).strip() in content:
            return "installed", str(path)
        return "outdated", "block present but points elsewhere"

    if not path.exists():
        return "missing", f"{path} does not exist"
    data, _had_comments, error = _load_config(path, jsonc=host.key in _JSONC_HOSTS)
    if data is None:
        return "unreadable", error
    container = data.get(host.container_key)
    if not isinstance(container, dict) or "tracera" not in container:
        return "missing", f"no 'tracera' entry under '{host.container_key}'"
    if _entry_matches(container["tracera"], host.build_entry(repo_root)):
        return "installed", str(path)
    return "outdated", "entry present but differs from the current layout"


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


__all__ = [
    "InstallResult",
    "host_config_path",
    "inspect_host",
    "install_codex",
    "install_into_host",
]
