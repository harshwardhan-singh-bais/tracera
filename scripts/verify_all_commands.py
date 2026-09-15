"""
Honest slash-command verification sweep.

Boots the REAL TraceraTUI (real agent, real index, real memory) with a
recording harness, dispatches every documented slash command through
``app._handle_command`` — the exact code path an interactive user triggers —
and classifies each result as:

    WORKS        produced substantive output, no crash
    EMPTY        ran without crashing but produced NO output (suspicious)
    USAGE_ERROR  rejected a bad/missing argument with a usage message
    CRASH        raised an exception

It also cross-checks the registry against the docs-site command tables and
reports commands that exist in one but not the other.

Usage:
    uv run python scripts/verify_all_commands.py                 # full sweep
    uv run python scripts/verify_all_commands.py --fast          # skip LLM-heavy commands
    uv run python scripts/verify_all_commands.py --only search,blast
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rich.console  # noqa: E402  (must follow sys.path setup)

console = rich.console.Console()
REPORT_PATH = ROOT / ".tracera" / "verify_commands_report.json"


def log(msg: str) -> None:
    try:
        console.print(msg)
    except Exception:
        print(msg.encode("ascii", "replace").decode(), flush=True)


# ── The catalog under test: every command the docs claim exists ───────────────
# Mirrors docs-site/content/docs/slash-commands.mdx (85 commands).

DOCS_COMMANDS: dict[str, str] = {
    # 1. orientation
    "/index": "ops",
    "/freshness": "retrieval",
    "/dashboard": "info",
    "/status": "info",
    "/files": "info",
    "/inspectrepo": "git",
    "/inspect": "git",
    "/repomap": "retrieval",
    "/outline": "retrieval",
    "/help": "info",
    "/features": "info",
    # 2. search & discovery
    "/search": "retrieval",
    "/symbol": "retrieval",
    "/symbols": "retrieval",
    "/source": "retrieval",
    "/definition": "retrieval",
    "/context": "retrieval",
    "/grep": "git",
    "/ls": "git",
    "/read": "git",
    "/assemble": "retrieval",
    "/debug": "retrieval",
    "/ranked": "retrieval",
    # 3. structural
    "/refs": "retrieval",
    "/callers": "retrieval",
    "/blast": "retrieval",
    "/importers": "retrieval",
    "/deps": "retrieval",
    "/classhier": "retrieval",
    "/impls": "retrieval",
    "/endpoint": "retrieval",
    "/cycles": "retrieval",
    "/coupling": "retrieval",
    "/ast": "retrieval",
    # 4. quality & risk
    "/hotspots": "retrieval",
    "/deadcode": "retrieval",
    "/pagerank": "retrieval",
    "/risk": "retrieval",
    "/prrisk": "retrieval",
    "/provenance": "retrieval",
    "/auditconfig": "retrieval",
    # 5. planning & execution
    "/plan": "heavy",
    "/plantask": "heavy",
    "/planturn": "retrieval",
    "/taskcontext": "retrieval",
    "/code": "interactive",
    "/fix": "interactive",
    "/delegate": "heavy",
    "/run": "interactive",
    "/write": "interactive",
    "/edit": "interactive",
    # 6. review & regression
    "/review": "heavy",
    "/selfreview": "heavy",
    "/regression": "heavy",
    "/test": "ops",
    "/tests": "git",
    "/changed": "retrieval",
    "/git": "git",
    # 7. multi-agent
    "/agents": "info",
    # 8. memory
    "/remember": "memory",
    "/recall": "memory",
    "/memory": "info",
    "/memsearch": "memory",
    "/memgraph": "memory",
    "/memgraph2": "memory",
    "/triples": "memory",
    "/memstats": "memory",
    "/consolidate": "memory",
    "/memworker": "memory",
    "/forget": "memory",
    "/sessions": "memory",
    # 9. refactoring
    "/refactor": "retrieval",
    "/editsafe": "retrieval",
    "/deletesafe": "retrieval",
    # 10. system
    "/cost": "info",
    "/observability": "info",
    "/sessionstats": "retrieval",
    "/model": "interactive",
    "/models": "info",
    "/tools": "info",
    "/tool": "interactive",
    "/mcp": "info",
    "/phases": "info",
    "/theme": "info",
    "/clear": "info",
    "/reset": "info",
}

# Realistic arguments (against this repo's real symbols/files)
ARGS: dict[str, str] = {
    "/search": "hybrid search pipeline",
    "/symbol": "TraceraTUI",
    "/symbols": "search",
    "/source": "TraceraTUI",
    "/definition": "_handle_command",
    "/outline": "tracera/tui/app.py",
    "/context": "TraceraTUI",
    "/grep": "SLASH_COMMANDS",  # sandbox grep walks every file — keep the pattern rare
    "/ls": "tracera",
    "/read": "pyproject.toml",
    "/assemble": "implement search ranking",
    "/debug": "hybrid search pipeline",
    "/ranked": "search implementation",
    "/refs": "TraceraTUI",
    "/callers": "TraceraTUI",
    "/blast": "TraceraTUI",
    "/importers": "tracera/main.py",
    "/deps": "TraceraTUI",
    "/classhier": "TraceraTUI",
    "/impls": "Tool",
    "/endpoint": "/api/health",
    "/ast": "def $NAME($$$PARAMS): ...",
    "/risk": "TraceraTUI",
    "/provenance": "TraceraTUI",
    "/refactor": "TraceraTUI",
    "/editsafe": "TraceraTUI",
    "/deletesafe": "TraceraTUI",
    "/plan": "add a cancel button to the loader",
    "/plantask": "add a cancel button to the loader",
    "/planturn": "add a search bar",
    "/taskcontext": "implement search ranking",
    "/code": "",  # argless on purpose → expect USAGE_ERROR, not an LLM run
    "/fix": "",  # argless → usage error
    "/delegate": "",  # argless → usage error (avoid real sub-agent run)
    "/write": "",  # argless → usage error (no file writes)
    "/edit": "",  # argless → usage error
    "/run": "",  # argless → usage error (no shell exec)
    "/model": "",  # argless → usage error (no provider switch)
    "/tool": "get_repo_map",  # safe read-only tool
    "/tests": "pytest --collect-only -q",
    "/git": "status",
    "/remember": "verify-sweep: honest command verification ran",
    "/recall": "verify-sweep",
    "/memsearch": "verify-sweep",
    "/forget": "verify-sweep",
}

#: commands that are allowed (expected) to end in a usage error when run argless
EXPECT_USAGE_ERROR = {
    "/code",
    "/fix",
    "/delegate",
    "/write",
    "/edit",
    "/run",
    "/model",
}

#: long/LLM commands skipped by --fast
HEAVY = {"heavy", "ops"}


# ── Harness: record everything the command renders ────────────────────────────


class Captured:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def note(self, text: str) -> None:
        if text:
            self.rows.append(str(text))


def _stub_panel(app, cap: Captured) -> None:
    stub = type("StubPanel", (), {})()

    def _mk(name):
        def method(*a, **k):
            cap.note(" ".join(str(x)[:300] for x in a))
            return None

        return method

    for m in (
        "add_user_message",
        "add_assistant_message",
        "add_error",
        "add_meta",
        "add_banner",
        "add_thinking_disclosure",
        "freeze_phase",
        "add_phase",
        "clear",
        "clear_attachments",
        "stream_delta",
        "stream_end",
        "set_title",
    ):
        setattr(stub, m, _mk(m))

    def add_info_row(title, body="", *a, **k):
        cap.note(f"{title} {body[:600]}")
        return None

    stub.add_info_row = add_info_row
    stub.attachments = []
    stub.verbose = False
    stub.query = lambda *a, **k: []
    stub._append = _mk("_append")
    stub.set_feature_status = _mk("feature")
    stub.tool_start = _mk("tool_start")
    stub.tool_end = _mk("tool_end")

    app._panel = lambda: stub
    app._status_line = lambda: stub
    stub.update_stats = _mk("stats")

    class _Pill:
        def set_phase(self, p):
            pass

    app.query_one = lambda *a, **k: _Pill()


def _inline_worker(app, tasks: set) -> None:
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
    pending = [t for t in tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


def build_app():
    from tracera.config import get_settings
    from tracera.conversation.state import ConversationState
    from tracera.main import _build_agent, _build_retrieval_pipeline, _setup
    from tracera.tui.app import TraceraTUI

    _setup()
    settings = get_settings()
    workspace_path = settings.tracera_workspace.resolve()

    retrieval_pipeline = None
    manifest = settings.index_dir / "index_manifest.json"
    if manifest.exists():
        retrieval_pipeline = _build_retrieval_pipeline(settings, workspace_path)

    agent, ws, provider = _build_agent(settings, workspace_path, retrieval_pipeline)

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


# ── Classification ────────────────────────────────────────────────────────────

_SUBSTANTIVE = re.compile(r"[A-Za-z0-9_]{3,}")
_USAGE_HINTS = ("Usage:", "usage:", "provide", "required", "missing argument")


def classify(entry: dict, cap: Captured, expect_usage_error: bool) -> str:
    if not entry["ok"]:
        if expect_usage_error and any(h in entry.get("error", "") for h in _USAGE_HINTS):
            return "USAGE_ERROR"
        return "CRASH"

    text = "\n".join(cap.rows)
    if not _SUBSTANTIVE.search(text):
        return "EMPTY"

    if expect_usage_error and any(h in text for h in _USAGE_HINTS):
        return "USAGE_ERROR"

    return "WORKS"


# ── Registry ↔ docs cross-check ───────────────────────────────────────────────


def cross_check() -> dict:
    from tracera.tui.widgets.command_registry import SLASH_COMMANDS

    registry = {f"/{k}" for k in SLASH_COMMANDS}
    documented = set(DOCS_COMMANDS)
    return {
        "registry_size": len(registry),
        "documented_size": len(documented),
        "in_registry_not_docs": sorted(registry - documented),
        "in_docs_not_registry": sorted(documented - registry),
    }


# ── Runner ────────────────────────────────────────────────────────────────────


async def run_sweep(fast: bool, only: list[str] | None) -> int:
    app = build_app()
    cap = Captured()
    _stub_panel(app, cap)
    tasks: set = set()
    _inline_worker(app, tasks)

    plan: list[tuple[str, str]] = []
    for cmd, group in sorted(DOCS_COMMANDS.items(), key=lambda kv: (kv[1], kv[0])):
        if fast and group in HEAVY:
            continue
        if only and cmd.lstrip("/") not in only:
            continue
        arg = ARGS.get(cmd, "")
        plan.append((group, f"{cmd} {arg}".strip()))

    results: list[dict] = []
    t0 = time.time()

    for group, cmd in plan:
        name = cmd.split()[0]
        entry = {
            "command": cmd,
            "group": group,
            "ok": False,
            "duration_s": 0.0,
            "output_preview": [],
        }
        start = time.time()
        try:
            await app._handle_command(cmd)
            await drain_tasks(tasks)
            entry["ok"] = True
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc(limit=5)
        entry["duration_s"] = round(time.time() - start, 2)
        entry["output_preview"] = [r[:200] for r in cap.rows[:3]]
        entry["classification"] = classify(entry, cap, name in EXPECT_USAGE_ERROR)
        results.append(entry)

        icon = {
            "WORKS": "[green]✓ WORKS[/]",
            "EMPTY": "[yellow]Ø EMPTY[/]",
            "USAGE_ERROR": "[cyan]? USAGE[/]",
            "CRASH": "[red]✗ CRASH[/]",
        }[entry["classification"]]
        log(f"{icon} [{group:<11}] {cmd} ({entry['duration_s']}s)")
        if entry["classification"] == "CRASH":
            log(f"    [red]{entry.get('error', '')}[/]")
        elif entry["classification"] == "EMPTY":
            log("    [yellow]no output rows captured[/]")
        cap.rows.clear()

    total_s = round(time.time() - t0, 1)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    cc = cross_check()

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "fast" if fast else "full",
        "total_commands": len(results),
        "counts": counts,
        "total_seconds": total_s,
        "results": results,
        "registry_vs_docs": cc,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log("")
    log("[bold]SUMMARY[/]")
    log(f"  WORKS        {counts.get('WORKS', 0)}")
    log(f"  EMPTY        {counts.get('EMPTY', 0)}  (ran, produced nothing — investigate)")
    log(f"  USAGE_ERROR  {counts.get('USAGE_ERROR', 0)}  (argless runs; expected)")
    log(f"  CRASH        {counts.get('CRASH', 0)}")
    log(f"  total {len(results)} commands in {total_s}s → {REPORT_PATH}")
    if cc["in_registry_not_docs"]:
        log(f"  [yellow]undocumented in registry: {', '.join(cc['in_registry_not_docs'])}[/]")
    if cc["in_docs_not_registry"]:
        log(f"  [red]documented but not in registry: {', '.join(cc['in_docs_not_registry'])}[/]")

    bad = counts.get("CRASH", 0) + counts.get("EMPTY", 0)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Honestly verify every TRACERA slash command")
    parser.add_argument("--fast", action="store_true", help="skip LLM-heavy and long ops")
    parser.add_argument("--only", default="", help="comma-separated command names")
    args = parser.parse_args()

    only = [o.strip().lstrip("/") for o in args.only.split(",") if o.strip()]
    bad = asyncio.run(run_sweep(args.fast, only or None))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
