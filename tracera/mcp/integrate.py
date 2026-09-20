"""
Integration engine — the thing ``tracera integrate`` actually runs.

Two independent wires make a harness "integrated":

1. **The MCP server** — so the agent can *call* TRACERA
   (:mod:`tracera.mcp.install`, host shapes from :mod:`tracera.mcp.hosts`).
2. **The agent brief** — so the agent *knows to*. Written into whichever
   instruction file the harness reads (:mod:`tracera.mcp.agent_brief`, targets
   from :mod:`tracera.mcp.harness`).

A harness needs both. Wiring only the first produces a tool the model never
calls; wiring only the second produces advice to call a tool that is not
there. This module runs both, reports per-target status, and never writes
outside a managed block or over an existing config without a ``.bak``.

Everything is idempotent: running ``apply`` twice reports ``unchanged`` the
second time and touches nothing. ``dry_run`` computes the same report without
writing, so ``--dry-run`` is a genuine preview rather than a promise.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tracera.mcp import agent_brief
from tracera.mcp.harness import HARNESSES, Harness, InstructionTarget
from tracera.mcp.hosts import HOSTS
from tracera.mcp.install import install_codex, install_into_host, inspect_host

#: Statuses that mean "this run changed something on disk".
_WRITE_STATUSES = frozenset({"installed", "updated", "created"})


@dataclass
class Action:
    """One thing we did (or would do) to one file or one config."""

    harness: str
    harness_name: str
    kind: str  # "brief" | "mcp"
    target: str  # display path, or host key for MCP
    path: Path | None
    status: str
    detail: str = ""

    @property
    def changed(self) -> bool:
        return self.status in _WRITE_STATUSES

    def to_dict(self) -> dict[str, object]:
        return {
            "harness": self.harness,
            "harness_name": self.harness_name,
            "kind": self.kind,
            "target": self.target,
            "path": str(self.path) if self.path else None,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class IntegrationReport:
    workspace: Path
    dry_run: bool
    actions: list[Action] = field(default_factory=list)

    @property
    def changed(self) -> list[Action]:
        return [a for a in self.actions if a.changed]

    @property
    def errors(self) -> list[Action]:
        return [a for a in self.actions if a.status == "error"]

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.actions:
            out[a.status] = out.get(a.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace": str(self.workspace),
            "dry_run": self.dry_run,
            "summary": self.by_status(),
            "actions": [a.to_dict() for a in self.actions],
        }


# ── instruction-file writes ──────────────────────────────────────────────────


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write ``text`` atomically, keeping a ``.bak`` of the previous contents.

    A half-written instruction file is worse than no instruction file: the
    harness would read a truncated brief. Write to a sibling temp file and
    ``os.replace`` so readers only ever see the old or the new content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_brief(
    harness: Harness,
    target: InstructionTarget,
    repo_root: Path,
    body: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Action:
    """Splice the managed brief into one instruction file."""
    path = target.path(repo_root)

    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else None
    except OSError as e:
        return Action(
            harness.key, harness.name, "brief", target.display, path, "error",
            f"cannot read: {e}",
        )

    try:
        new_text, status = agent_brief.splice(existing, body, target.fmt)
    except agent_brief.MalformedBlockError as e:
        return Action(
            harness.key, harness.name, "brief", target.display, path, "skipped", str(e)
        )

    if status == "unchanged" and force:
        status = "updated"  # caller asked for a rewrite; content is the same

    if dry_run:
        return Action(
            harness.key,
            harness.name,
            "brief",
            target.display,
            path,
            f"would-{status}" if status != "unchanged" else "unchanged",
            "already current" if status == "unchanged" else f"would {status}",
        )

    if status == "unchanged":
        return Action(
            harness.key, harness.name, "brief", target.display, path, "unchanged",
            "already current",
        )

    try:
        _atomic_write_text(path, new_text)
    except OSError as e:
        return Action(
            harness.key, harness.name, "brief", target.display, path, "error",
            f"cannot write: {e}",
        )

    detail = {
        "installed": "appended a managed block",
        "updated": "refreshed the managed block",
    }.get(status, status)
    return Action(harness.key, harness.name, "brief", target.display, path, status, detail)


# ── the main entry point ─────────────────────────────────────────────────────


def apply(
    keys: list[str] | None = None,
    *,
    repo_root: Path,
    instructions: bool = True,
    mcp: bool = True,
    scope: str | None = "project",
    detected_only: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> IntegrationReport:
    """
    Wire TRACERA into ``keys`` (or every harness).

    ``scope`` defaults to ``"project"`` so a run never edits files under the
    user's home directory unless asked — pass ``scope="user"`` (or ``None``
    for both) to include user-scoped instruction files. MCP config paths are
    whatever the host itself uses; several hosts have no project-level option,
    so those are always written to the home directory regardless of ``scope``.
    """
    report = IntegrationReport(workspace=repo_root, dry_run=dry_run)

    selected: list[Harness]
    if keys:
        selected = [HARNESSES[k.lower()] for k in keys if k.lower() in HARNESSES]
    else:
        selected = [
            h for h in HARNESSES.values() if not detected_only or h.detected(repo_root)
        ]

    # ── instruction files ────────────────────────────────────────────────
    if instructions:
        # AGENTS.md is shared by a dozen harnesses, so resolve the body per
        # *file* rather than per harness: if any harness sharing the file can
        # speak MCP, the tool-aware brief is the right one to publish there —
        # the repo will have the server wired for it. Deciding this by catalog
        # order instead would make the output depend on dict iteration.
        by_path: dict[Path, tuple[Harness, InstructionTarget, bool]] = {}
        for harness in selected:
            for target in harness.targets(scope=scope):
                path = target.path(repo_root)
                previous = by_path.get(path)
                if previous is None:
                    by_path[path] = (harness, target, harness.mcp_supported)
                else:
                    owner, chosen, mcp_seen = previous
                    by_path[path] = (
                        owner,
                        chosen,
                        mcp_seen or harness.mcp_supported,
                    )

        for harness, target, mcp_aware in by_path.values():
            body = agent_brief.brief_body(mcp=mcp_aware, workspace=str(repo_root))
            report.actions.append(
                write_brief(
                    harness, target, repo_root, body, dry_run=dry_run, force=force
                )
            )

    # ── MCP registration ─────────────────────────────────────────────────
    if mcp:
        for harness in selected:
            if harness.mcp_host is None:
                continue
            host = HOSTS[harness.mcp_host]
            if dry_run:
                status, detail = inspect_host(host, repo_root)
                report.actions.append(
                    Action(
                        harness.key,
                        harness.name,
                        "mcp",
                        harness.mcp_host,
                        None,
                        "would-install" if status != "installed" else "installed",
                        detail,
                    )
                )
                continue
            try:
                if host.key == "codex":
                    r = install_codex(repo_root, force=force)
                else:
                    r = install_into_host(host, repo_root, force=force)
                report.actions.append(
                    Action(
                        harness.key,
                        harness.name,
                        "mcp",
                        harness.mcp_host,
                        r.path,
                        r.status,
                        r.detail,
                    )
                )
            except Exception as e:  # noqa: BLE001 — report, never abort the run
                report.actions.append(
                    Action(
                        harness.key,
                        harness.name,
                        "mcp",
                        harness.mcp_host,
                        None,
                        "error",
                        str(e),
                    )
                )

    return report


# ── doctor ───────────────────────────────────────────────────────────────────


def doctor(repo_root: Path, keys: list[str] | None = None) -> list[dict[str, object]]:
    """
    Read-only health check per harness.

    Reports whether the harness looks present, whether the brief is on disk
    and current, and whether the MCP server is registered. Nothing is written.
    """
    rows: list[dict[str, object]] = []
    selected = (
        [HARNESSES[k.lower()] for k in keys if k.lower() in HARNESSES]
        if keys
        else list(HARNESSES.values())
    )

    for harness in selected:
        brief_states: list[dict[str, str]] = []
        for target in harness.targets(scope="project"):
            path = target.path(repo_root)
            if not path.exists():
                state = "missing"
            else:
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    state = "unreadable"
                else:
                    state = (
                        "current"
                        if agent_brief.has_managed_block(text, target.fmt)
                        else "no-block"
                    )
            brief_states.append({"target": target.display, "state": state})

        mcp_state = "n/a"
        if harness.mcp_host is not None:
            mcp_state, _ = inspect_host(HOSTS[harness.mcp_host], repo_root)

        rows.append(
            {
                "key": harness.key,
                "name": harness.name,
                "detected": harness.detected(repo_root),
                "mcp_host": harness.mcp_host,
                "mcp": mcp_state,
                "briefs": brief_states,
            }
        )
    return rows


def summary_line(report: IntegrationReport) -> str:
    """One-line human summary, used by the CLI and tests."""
    counts = report.by_status()
    if not counts:
        return "nothing to do"
    parts = [f"{n} {status}" for status, n in sorted(counts.items())]
    prefix = "would apply: " if report.dry_run else ""
    return prefix + ", ".join(parts)


def to_json(report: IntegrationReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


__all__ = [
    "Action",
    "IntegrationReport",
    "apply",
    "doctor",
    "summary_line",
    "to_json",
    "write_brief",
]
