"""
Real-runtime slash dispatch test.

Regression guard for the ``TypeError: object Worker can't be used in 'await'
expression`` bug: the slash dispatcher awaits ``self._run_*`` methods, which
used to be ``@work``-decorated. Textual ``@work`` methods return a Worker, not
a coroutine — awaiting one raises TypeError. Every one of the ~55 dispatch
branches crashed the real TUI (e.g. ``/deadcode``) even though a headless
sweep with a faked ``run_worker`` reported them green.

Two guards here:
1. Static: only ``_run_agent_task`` may carry ``@work`` (it must stay a real
   Worker for cancel bookkeeping).
2. Dynamic: every registered slash command is dispatched through a REAL
   mounted app (``app.run_test()`` pilot) and must not raise. LLM-backed
   commands are monkeypatched with fakes so the test is hermetic and fast.
"""

from __future__ import annotations

import ast
import tracera.tui.app as _app_mod
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tracera.tui.app import TraceraTUI
from tracera.tui.widgets.command_registry import SLASH_COMMANDS


# ── Guard 1 (static): @work only on _run_agent_task ──────────────────────────


def test_only_agent_task_keeps_work_decorator():
    app_src = Path(_app_mod.__file__).read_text(encoding="utf-8")
    on_work: set[str] = set()
    for node in ast.walk(ast.parse(app_src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = getattr(target, "attr", None) or getattr(target, "id", "")
                if name == "work":
                    on_work.add(node.name)
    assert on_work == {"_run_agent_task"}, (
        f"@work leaked onto {sorted(on_work - {'_run_agent_task'})} — "
        "the slash dispatcher awaits these methods; @work makes them return "
        "a Worker and 'await' raises TypeError in the real TUI."
    )


# ── Guard 2 (dynamic): every command through the real runtime ────────────────


class _FakeProvider:
    name = "fake"
    default_model = "fake-model"
    supports_vision = False
    has_vision = False


def _make_app(tmp_path):
    from tracera.agent.memory import AgentMemory

    registry = MagicMock()
    registry.has = MagicMock(return_value=True)
    registry.tools = [
        SimpleNamespace(name="read_file"),
        SimpleNamespace(name="run_command"),
    ]
    registry.get = MagicMock(
        return_value=SimpleNamespace(
            name="generic",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "command": {"type": "string"},
                    "task": {"type": "string"},
                    "name": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["query"],
            },
        )
    )
    registry.execute = AsyncMock(
        return_value=SimpleNamespace(success=True, output="ok", duration_ms=1.0, error=None)
    )

    agent = MagicMock()
    agent.provider = _FakeProvider()
    agent.model = "fake-model"
    agent.registry = registry
    agent.decomposer = None
    agent.max_iterations = 3
    agent.max_tool_calls = 5
    agent._enhanced_memory = None
    agent._triple_store = None
    agent._session_manager = None
    agent._recent_files = []
    agent.run = AsyncMock(return_value=[])

    memory = AgentMemory(tmp_path / "memory")
    return TraceraTUI(
        agent=agent,
        memory=memory,
        workspace_path=tmp_path,
        retrieval_pipeline=None,
        banner=None,
    )


# Plausible dummy args (same shape as test_slash_command_execution).
_ARG_MAP = {
    "model": " mock-model", "search": "test", "debug": "test",
    "plan": "do something", "code": "do something", "ask": "do something",
    "symbol": "Foo", "symbols": "Foo", "source": "Foo", "definition": "Foo",
    "context": "Foo", "deps": "Foo", "outline": "foo.py", "assemble": "do it",
    "refs": "Foo", "callers": "Foo", "blast": "Foo", "classhier": "Foo",
    "endpoint": "/api/test", "impls": "Foo", "provenance": "Foo", "risk": "Foo",
    "refactor": "Foo", "editsafe": "Foo", "deletesafe": "Foo",
    "importers": "foo.py", "git": "status", "tests": "pytest",
    "read": "pyproject.toml", "write": "out.txt", "edit": "pyproject.toml",
    "ls": ".", "grep": "test", "run": "echo hi", "recall": "test",
    "remember": "test memory", "forget": "test", "memsearch": "test",
    "planturn": "test", "ranked": "test", "taskcontext": "do it",
    "delegate": "do something", "tool": "read_file", "plantask": "do it",
    "ast": "def $NAME($$$PARAMS): ...",
}


def _all_commands():
    cmds = []
    for name in sorted(SLASH_COMMANDS):
        arg = _ARG_MAP.get(name, "")
        cmds.append(f"/{name}{arg}" if arg else f"/{name}")
    # /clear and /reset last — they wipe panel/conversation state.
    cmds.remove("/clear")
    cmds.remove("/reset")
    cmds += ["/clear", "/reset"]
    return cmds


class _FakeDecomposer:
    def __init__(self, *a, **k):
        pass

    async def decompose(self, task):
        return SimpleNamespace(items=[], to_markdown=lambda: "fake plan")


class _FakeOrchestrator:
    def __init__(self, fleet, decomposer=None, parallel=False):
        pass

    async def delegate(self, task):
        yield {"type": "plan_ready", "plan": SimpleNamespace(steps=[])}
        yield {
            "type": "agent_end",
            "step": SimpleNamespace(role=SimpleNamespace(value="coder"), task="t"),
            "result": SimpleNamespace(
                status=SimpleNamespace(value="success"), iterations=1, tool_calls=1
            ),
        }
        yield {"type": "report", "report": SimpleNamespace(summary="done")}


class _FakeFixLoop:
    def __init__(self, *a, **k):
        pass

    async def run(self, task, provider, agent):
        return SimpleNamespace(final_success=True, total_iterations=1, attempts=[])


class _FakeReviewer:
    def __init__(self, *a, **k):
        pass

    async def review(self, provider):
        return "LGTM"


class _FakeProtector:
    def __init__(self, *a, **k):
        pass

    def snapshot_before(self):
        return SimpleNamespace(summary="baseline ok")

    def verify_after(self):
        return {
            "overall_success": True, "pre_passed": 1, "post_passed": 1,
            "post_failed": 0, "summary": "ok", "changed_files": [],
        }


async def test_every_slash_command_in_real_textual_runtime(tmp_path, monkeypatch):
    """All SLASH_COMMANDS dispatch through a mounted app without raising.

    This runs inside the real Textual runtime: if any awaited handler is
    @work-decorated, the await raises TypeError right here.
    """
    # Hermetic patches for LLM/backed commands (function-level imports).
    import tracera.agent.autonomous as autonomous_mod
    import tracera.agent.orchestrator as orchestrator_mod
    import tracera.agent.planner as planner_mod
    import tracera.agent.subagents as subagents_mod
    import tracera.main as main_mod

    monkeypatch.setattr(planner_mod, "TaskDecomposer", _FakeDecomposer, raising=False)
    import tracera.tui.app as app_mod
    monkeypatch.setattr(app_mod, "TaskDecomposer", _FakeDecomposer, raising=False)
    monkeypatch.setattr(orchestrator_mod, "TaskOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(subagents_mod, "build_sub_agent_fleet", lambda *a, **k: {})
    monkeypatch.setattr(autonomous_mod, "AutonomousFixLoop", _FakeFixLoop)
    monkeypatch.setattr(autonomous_mod, "SelfReviewer", _FakeReviewer)
    monkeypatch.setattr(autonomous_mod, "RegressionProtector", _FakeProtector)

    def _no_pipeline(*a, **k):
        raise RuntimeError("index rebuild skipped in test")

    monkeypatch.setattr(main_mod, "_build_retrieval_pipeline", _no_pipeline)

    app = _make_app(tmp_path)
    commands = _all_commands()

    failed: list[tuple[str, str]] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        for cmd in commands:
            try:
                await app._handle_command(cmd)
                await pilot.pause()
            except Exception as e:  # noqa: BLE001
                failed.append((cmd, f"{type(e).__name__}: {e}"))

    assert not failed, (
        f"{len(failed)}/{len(commands)} commands crashed in the REAL runtime:\n"
        + "\n".join(f"  {c} -> {e}" for c, e in failed[:15])
    )
