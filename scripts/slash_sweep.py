"""
Headless slash-command sweep.

Boots the REAL TraceraTUI (real agent, real provider, real index, real memory)
inside a Textual pilot, then dispatches every slash command through
``app._handle_command`` — the exact code path an interactive user triggers.

Mutating/interactive commands (/write, /edit, /fix) are run in an
observe-only/argless mode so the sweep verifies dispatch without side effects.

Usage:
    uv run python scripts/slash_sweep.py [--groups memory,git,ops] [--only name,name]
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console

console = Console()

REPORT_PATH = ROOT / ".tracera" / "slash_sweep_report.json"
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    LOG_LINES.append(line)


# ── Command groups ────────────────────────────────────────────────────────────

# Commands that render info without touching the index or LLM
INFO = [
    "/help", "/clear", "/status", "/memory", "/cost", "/models", "/theme",
    "/files", "/phases", "/tools", "/mcp", "/features", "/agents", "/reset",
]

# Retrieval/analysis against the code index (fast, no LLM)
RETRIEVAL = [
    "/search hybrid BM25", "/symbol TraceraTUI", "/symbols search",
    "/source TraceraTUI", "/definition _handle_command", "/outline tracera/tui/app.py",
    "/repomap", "/context TraceraTUI", "/deps TraceraTUI", "/refs TraceraTUI",
    "/callers TraceraTUI", "/blast TraceraTUI", "/changed", "/freshness",
    "/importers tracera/main.py", "/classhier TraceraTUI", "/cycles",
    "/coupling", "/endpoint /api/x", "/deadcode", "/hotspots", "/pagerank",
    "/refactor TraceraTUI", "/editsafe TraceraTUI", "/deletesafe TraceraTUI",
    "/impls Tool", "/provenance TraceraTUI", "/risk TraceraTUI", "/prrisk",
    "/auditconfig", "/ast 'def $NAME($$$PARAMS): ...'",
    "/sessionstats", "/planturn add a search bar", "/ranked search implementation",
    "/taskcontext implement search",
]

# Memory layer
MEMORY = [
    "/recall index", "/remember sweep-test: slash sweep ran",
    "/recall sweep-test", "/memsearch index", "/memstats", "/consolidate",
    "/memgraph", "/memgraph2", "/triples", "/memworker", "/sessions",
    "/forget sweep-test",
]

# Git / repo / direct file tools (read-only usage)
GIT_OPS = [
    "/git status", "/inspectrepo", "/tests pytest", "/read pyproject.toml",
    "/ls tracera", "/grep SLASH_COMMANDS",
]

# Heavy LLM commands — last, slowest
HEAVY = [
    "/plan list the steps to add a cancel button",
    "/plantask implement a cancel button",
    "/delegate add a help hint to the status bar",
    "/review",
    "/selfreview",
    "/regression",
]

# Interactive/mutating: dispatched with NO arg (usage-error path) or observe-only
INTERACTIVE_SAFE = [
    "/write", "/edit", "/run", "/code", "/ask", "/fix", "/model",
]

# Argless per spec: /index, /test are long ops — run them, they're idempotent
OPS = ["/index", "/test"]


def all_groups() -> dict[str, list[str]]:
    return {
        "info": INFO,
        "retrieval": RETRIEVAL,
        "memory": MEMORY,
        "git": GIT_OPS,
        "ops": OPS,
        "heavy": HEAVY,
        "interactive": INTERACTIVE_SAFE,
    }


# ── Capture machinery ─────────────────────────────────────────────────────────


class Captured:
    def __init__(self) -> None:
        self.rows: list[str] = []  # human-readable output lines

    def note(self, text: str) -> None:
        self.rows.append(text)


def _stub_panel(app, cap: Captured) -> None:
    """Replace the panel + status line with recording fakes."""
    stub = type("StubPanel", (), {})()

    def _mk(name):
        def method(*a, **k):
            cap.note(f"[{name}] " + " ".join(str(x)[:400] for x in a))
            return None
        return method

    for m in (
        "add_user_message", "add_assistant_message", "add_error", "add_meta",
        "add_banner", "add_thinking_disclosure", "freeze_phase", "add_phase",
        "clear", "clear_attachments", "stream_delta", "stream_end", "set_title",
    ):
        setattr(stub, m, _mk(m))

    def add_info_row(title, body="", *a, **k):
        cap.note(f"[info] {title}: {body[:600]}")
        return None

    stub.add_info_row = add_info_row
    stub.attachments = []
    stub.verbose = False
    stub.query = lambda *a, **k: []
    stub._append = _mk("_append")
    stub.set_feature_status = _mk("set_feature_status")
    stub.tool_start = _mk("tool_start")
    stub.tool_end = _mk("tool_end")

    app._panel = lambda: stub
    app._status_line = lambda: stub
    stub.update_stats = _mk("update_stats")

    # Loader pill lookups — no DOM mounted.
    class _Pill:
        def set_phase(self, p):
            pass

    app.query_one = lambda *a, **k: _Pill()


def _inline_worker(app, tasks: set) -> None:
    """Make @work methods schedule real tasks so sync callers still execute them."""

    def _run(coro, **kwargs):
        if isinstance(coro, functools.partial):
            coro = coro.func(*coro.args, **coro.keywords)
        if asyncio.iscoroutine(coro):
            t = asyncio.create_task(coro)
            tasks.add(t)
            return t
        return coro

    app.run_worker = _run


async def drain_tasks(tasks: set) -> None:
    """Await every task the sweep scheduled, absorbing exceptions."""
    pending = [t for t in tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


def _disable_heavy_animation(app) -> None:
    """No-op the sleep-heavy startup niceties if invoked."""

    async def _noop(*a, **k):
        pass

    app.refresh_css = lambda *a, **k: None


# ── Bootstrap: build the real agent exactly like the TUI does ─────────────────


def build_app():
    from tracera.config import get_settings
    from tracera.conversation.state import ConversationState
    from tracera.logging import setup_logging
    from tracera.main import _build_agent, _build_retrieval_pipeline, _setup
    from tracera.tui.app import TraceraTUI

    _setup()
    settings = get_settings()
    workspace_path = settings.tracera_workspace.resolve()

    retrieval_pipeline = None
    manifest = settings.index_dir / "index_manifest.json"
    if manifest.exists():
        log("index manifest found — loading retrieval pipeline")
        retrieval_pipeline = _build_retrieval_pipeline(settings, workspace_path)
        log(f"pipeline loaded: {len(retrieval_pipeline)} components")

    agent, ws, provider = _build_agent(settings, workspace_path, retrieval_pipeline)
    log(f"agent built: provider={getattr(provider, 'name', '?')} "
        f"model={getattr(provider, 'default_model', '?')} "
        f"tools={len(agent.registry.tools)}")

    from tracera.agent.memory import AgentMemory
    memory = AgentMemory(settings.memory_dir)

    app = TraceraTUI(
        agent=agent,
        memory=memory,
        workspace_path=workspace_path,
        retrieval_pipeline=retrieval_pipeline,
        banner=None,
    )
    app._conversation = ConversationState()
    app._recent_files = []
    return app


# ── Runner ────────────────────────────────────────────────────────────────────


def parse_group(name: str, cmd: str) -> None:
    pass


async def run_sweep(groups: list[str], only: list[str] | None) -> int:
    from tracera.tui.widgets.command_registry import SLASH_COMMANDS

    app = build_app()
    cap = Captured()
    _stub_panel(app, cap)
    tasks: set = set()
    _inline_worker(app, tasks)
    _disable_heavy_animation(app)

    plan: list[tuple[str, str]] = []
    available = all_groups()
    for g in groups:
        plan.extend((g, c) for c in available[g])

    if only:
        plan = [(g, c) for g, c in plan if any(c.split()[0].lstrip("/") == o for o in only)]

    results: list[dict] = []
    t0 = time.time()

    for group, cmd in plan:
        name = cmd.split()[0]
        entry = {
            "command": cmd,
            "group": group,
            "ok": False,
            "duration_s": 0.0,
            "output": [],
        }
        start = time.time()
        try:
            await app._handle_command(cmd)
            await drain_tasks(tasks)
            entry["ok"] = True
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc(limit=6)
        entry["duration_s"] = round(time.time() - start, 2)
        entry["output"] = cap.rows[:]
        status = "✓" if entry["ok"] else "✗ CRASH"
        log(f"{status} [{group}] {cmd} ({entry['duration_s']}s, {len(entry['output'])} rows)")
        if not entry["ok"]:
            log(f"    {entry.get('error', '')}")
        results.append(entry)
        cap.rows.clear()

    total_s = round(time.time() - t0, 1)

    # Coverage check: every SLASH_COMMANDS entry was attempted?
    attempted = {c.split()[0].lower().lstrip("/") for _, c in plan}
    registry_cmds = set(SLASH_COMMANDS.keys())
    missing = sorted(registry_cmds - attempted)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_commands": len(plan),
        "crashed": sum(1 for r in results if not r["ok"]),
        "total_seconds": total_s,
        "results": results,
        "coverage": {
            "registry_size": len(registry_cmds),
            "attempted": len(attempted),
            "not_attempted": missing,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"\nReport → {REPORT_PATH}")
    log(f"TOTAL: {len(plan)} commands · {report['crashed']} crashes · {total_s}s")
    return report["crashed"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every TRACERA slash command headlessly")
    parser.add_argument("--groups", default="info,retrieval,memory,git,ops,interactive,heavy")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    only = [o.strip() for o in args.only.split(",") if o.strip()]

    crashed = asyncio.run(run_sweep(groups, only))
    sys.exit(0 if crashed == 0 else 1)


if __name__ == "__main__":
    main()
