# TRACERA Full Slash-Command Sweep Report

**Date:** 2026-09-12 · **Runner:** `scripts/slash_sweep.py` (headless Textual driver, real agent + Groq + real index)
**Final result: 82/82 commands dispatched with 0 crashes · 331/331 unit tests pass**

## What was executed

| Group | Commands | Result |
|---|---|---|
| info (`/help /clear /status /memory /cost /models /theme /files /phases /tools /mcp /features /agents /reset`) | 14 | all pass |
| retrieval (all code-intelligence aliases: `/search /symbol /symbols /source /definition /outline /repomap /context /deps /refs /callers /blast /changed /freshness /importers /classhier /cycles /coupling /endpoint /deadcode /hotspots /pagerank /refactor /editsafe /deletesafe /impls /provenance /risk /prrisk /auditconfig /ast /sessionstats /planturn /ranked /taskcontext`) | 35 | all pass, real index data |
| memory (`/recall /remember /memsearch /memstats /consolidate /memgraph /memgraph2 /triples /memworker /sessions /forget`) | 11 | all pass incl. store + delete round-trip |
| git/ops (`/git status /inspectrepo /tests pytest /read /ls /grep /index /test`) | 8 | all pass |
| interactive (argless dispatch-safety: `/write /edit /run /code /ask /fix /model`) | 7 | all show correct usage errors |
| heavy LLM (`/plan /plantask /delegate /review /selfreview /regression`) | 6 | all pass with live Groq calls |

Raw evidence: `.tracera/slash_sweep_report.json` (per-command output captured).

## Bugs found and fixed

### 1. CRITICAL — tree-sitter 0.26 segfault killed the entire index pipeline
- **Symptom:** `/index` died silently with SIGSEGV (exit 139) mid-run; `index_manifest.json` contained `files_indexed: 0` from a pytest temp dir. Every index-backed command (`/search`, `/blast`, `/outline`, … ~30 commands) returned "not found in indexed corpus" or "run /index first".
- **Root cause:** `uv.lock` resolved tree-sitter runtime **0.26.0** against language wheels built for an older ABI. `QueryCursor.captures()` returned nodes whose points/bytes were garbage (rows like 35183298347160 in a 2,263-line file), then access-violated during GC or logging. The existing comments in `extractor.py` documented earlier manifestations of the same class of bug.
- **Fix:** `pyproject.toml` now pins `tree-sitter>=0.25.0,<0.26` (language wheels `>=0.23,<0.26`); `uv.lock` updated to 0.25.2. Verified: 5 consecutive parse+query runs clean, extractor returns correct symbols (163 in main.py).

### 2. Registry drift — `/features` advertised tools that were never registered
- **Symptom:** 8 of the ~74 slash aliases failed with "Tool 'x' not available — run /index first": `/deadcode /hotspots /impls /plantask /provenance /refactor /symbols /source`.
- **Root cause:** the `standard` TOOL_PROFILE excluded them; `SearchSymbolsTool`/`GetSymbolSourceTool` were only instantiated by the MCP server, never in the agent/TUI registry.
- **Fix:** added all ten tools to the standard profile and registered `SearchSymbolsTool`/`GetSymbolSourceTool` in `extend_registry_with_retrieval` (`tracera/tools/registry.py`). Registry grew 50 → 58; every `SLASH_TOOLS` alias now resolves (verified programmatically).

### 3. `/memgraph2` crashed (`MemoryGraphTool.execute() missing 1 required positional argument: 'concept'`)
- **Fix:** `concept` is now optional; argless call renders a knowledge-graph overview (triples count + most-connected concepts).

### 4. `/forget <text>` never reached the tool
- **Root cause:** `single_arg_kwargs` mapped the positional to a nonexistent schema key, so `forget_memory` failed with "Provide either memory_id or content_match".
- **Fix:** `_run_forget` in `tracera/tui/app.py` now passes `{"content_match": text}` explicitly. Verified: stores then deletes a memory end-to-end.

### 5. `/symbols` crashed (`'tuple' object has no attribute 'search'`)
- **Root cause:** `SearchSymbolsTool` expects a retriever, was handed the 10-tuple pipeline; also referenced result keys the retriever never returns.
- **Fix:** unwraps `pipeline[1]` when given the tuple; scoring remapped to real keys (`_bm25_score`, `_dense_score`, `symbol`, `start_line`). Verified with ranked results.

### 6. `/plantask <task>` was a dead end
- **Root cause:** handler ignored its argument and always printed usage.
- **Fix:** dispatches to `plan_code_task` (intent + anchors + recommended tool chain). Verified live.

### 7. `/tests pytest` / TestRunner 120 s cap — RESOLVED
- New `TRACERA_TEST_TIMEOUT` setting (default 300 s) in `tracera/config/settings.py`; `TestRunner` reads it when no explicit timeout is passed. Verified: `/tests pytest` now runs the full suite to completion (339/339 passed, 137 s).

### 8. `/hotspots` took ~60 s per call — RESOLVED
- The tool ran `pytest --cov` on every invocation. Coverage is now opt-in (`with_coverage=true`); an existing `coverage.json` is still reused when present. Verified: 61 s → 1.8 s.

### 9. `/delegate` sub-agents could mutate the repo — RESOLVED
- The `git` tool is removed from every sub-agent role set, and `run_command` in sub-agent registries is wrapped with a guard that rejects destructive git mutations (`git commit/push/reset/rebase/checkout/restore/clean/stash/merge/cherry-pick/revert/tag/branch`) with a `PermissionError`. The main agent keeps unrestricted access.

## Found AFTER the sweep: real-TUI dispatch crash (fixed in `ba7b4cd`)

The headless sweep faked Textual's worker layer, which masked two bugs that crashed the **interactive** TUI:

### 10. Every `await self._run_*` dispatch branch raised `TypeError: object Worker can't be used in 'await' expression` (e.g. `/deadcode`)
- **Root cause:** ~55 `_run_*` handlers were decorated `@work`. Textual `@work` methods return a `Worker`, not a coroutine, so awaiting them crashes. The headless driver's `run_worker` shim returned awaitables, hiding it.
- **Fix:** removed `@work` from all leaf handlers (only `_run_agent_task` keeps it, for cancel bookkeeping) and awaited every call site.

### 11. `/memgraph` crashed on mount (`'dict' object has no attribute '_append'`)
- **Root cause:** `MemoryGraphWidget` stored its data in `self._nodes`/`self._edges`, shadowing Textual's internal `Widget._nodes` (a `NodeList` used by the compositor).
- **Fix:** renamed the widget's data attrs (`_mem_nodes`, `_mem_edges`, `_mem_central`, `_mem_type_counts`).

### Regression guard so this can't hide again
- `tests/test_slash_runtime_dispatch.py`: (a) static guard — only `_run_agent_task` may carry `@work`; (b) dynamic guard — every registered slash command is dispatched through a **real mounted app** (`app.run_test()` pilot) with LLM fakes.
- Verified: `scripts/verify_deadcode_real_tui.py` reproduces the user's exact crash path (`/deadcode`, `/hotspots`, `/pagerank`, `/coupling`, `/cycles`) in the real Textual runtime — now clean. Suite: 341/341.

## Follow-up commit

- `00327a5` — sub-agent sandbox, `/hotspots` opt-in coverage, configurable test timeout, regression tests (`tests/test_sweep_fixes.py`, 8 tests).
- `ba16c2c` — reverts the stray `f6b861e "Add styling and accessibility"` commit (12-line `style.css`) left by the delegate run.

## Remaining (accepted / cosmetic)

- `/search` ranking could be tuned further (works, but relevance ordering is basic).
- `uv pip` venv drift resolved by the lock pin; nothing pending.

## Incident during the sweep (worth knowing)

The `/delegate` heavy run lets coder sub-agents execute real shell commands. One of them **committed to the repo** (`f6b861e "Add styling and accessibility"`, adds a 12-line `style.css`) and **gutted `tracera/tui/widgets/agent_panel.py`** (−1,282 lines, deleting the `AgentPanel` class) while dropping a stray `tests/test_styling.py`. Both were recovered/reverted (`git checkout` + delete). **Follow-up recommended:** sub-agents need a read-only or sandboxed workspace guard before `/delegate` is safe to use on a real repo.

## Regression safety

- `tests/test_slash_command_execution.py` + surface tests: 20/20 pass.
- Full suite: **331/331 pass** (also re-verified independently by `/regression`: baseline 334/334 → post 334/334 — includes 3 parametrized variants).
- `/index` rebuilt cleanly after the tree-sitter fix: 186 files, ~220 s.

## Artifacts

- `scripts/slash_sweep.py` — repeatable sweep driver (`uv run python scripts/slash_sweep.py [--groups …] [--only …]`)
- `.tracera/slash_sweep_report.json` — full per-command output
- Modified: `pyproject.toml`, `uv.lock`, `tracera/tools/registry.py`, `tracera/tools/ast_tools.py`, `tracera/tools/memory_tools.py`, `tracera/tui/app.py`
