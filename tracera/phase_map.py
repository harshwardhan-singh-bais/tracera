"""
Phase map — the single source of truth for the TRACERA roadmap.

Each phase entry records what it is and whether it is currently testable:

  - "implemented" — code exists and is exercised by tests / CLI / TUI
  - "excluded"    — not (fully) implemented: phase 41 and phases 60–66
  - "roadmap"     — planned, not implemented: phases 67–72

The TUI's ``/phases`` command renders this map and lets the user tick phases
off as they are verified (progress persists in ``.tracera/phases_progress.json``).
The checklist that goes with it lives in ``tests/PROBLEM_STATEMENT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUS_IMPLEMENTED = "implemented"
STATUS_EXCLUDED = "excluded"
STATUS_ROADMAP = "roadmap"


@dataclass(frozen=True)
class Phase:
    number: int
    title: str
    status: str  # one of STATUS_*


#: Titles for phases 1–59, taken from the README roadmap.
_TITLES: dict[int, str] = {
    1: "Project architecture, config system, Rich logging, Typer CLI",
    2: "Workspace sandbox — path validation, traversal protection",
    3: "Git integration — status, diff, log, branch (+ git agent tool)",
    4: "LLM provider abstraction — OpenAI, Anthropic, Gemini, Groq, Ollama, …",
    5: "Provider-neutral conversation state",
    6: "Tool abstraction + registry with JSON schema",
    7: "Basic coding tools — read, write, edit, grep, run",
    8: "Core ReAct agent loop with streaming",
    9: "Planning system — task decomposition, TODO tracking",
    10: "Persistent agent memory — injected into agent context",
    11: "Repository scanner",
    12: "Tree-sitter parsing",
    13: "Symbol extraction",
    14: "Chunking",
    15: "Index schema",
    16: "BM25 keyword index",
    17: "Local embeddings (sentence-transformers)",
    18: "LanceDB vector index",
    19: "Dense retrieval",
    20: "Hybrid retrieval (BM25 + Dense via RRF)",
    21: "Symbol-aware retrieval",
    22: "Context expansion",
    23: "Cross-encoder reranking",
    24: "Incremental / live indexing",
    25: "Symbol relationship graph",
    26: "Dependency-aware graph retrieval",
    27: "Code-search agent tools (search_code, find_symbol, find_references, get_dependencies, get_context)",
    28: "Registry extension with code-search tools",
    29: "Context assembly engine",
    30: "Context compression",
    31: "Repository-aware agent",
    32: "Test discovery (pytest / npm / cargo)",
    33: "Safe test execution",
    34: "Failure analysis (structured TestFailure)",
    35: "Retrieval-driven debugging",
    36: "Autonomous fix loop",
    37: "Self-review of changes",
    38: "Regression protection",
    39: "MCP server (7 capabilities as MCP tools)",
    40: "MCP client (connect to external MCP servers)",
    41: "MCP manager + unified tool registry",  # excluded — not fully wired
    42: "Sub-agent framework (Researcher / Coder / Tester / Reviewer / Debugger)",
    43: "Task delegation (orchestrator)",
    44: "Result aggregation (conflict detection, shared state)",
    45: "Retrieval evaluation dataset",
    46: "Retrieval metrics (Recall@k, MRR, nDCG@k)",
    47: "Strategy comparison (BM25 vs dense vs hybrid vs reranked)",
    48: "Grep baseline",
    49: "End-to-end agent benchmark",
    50: "Ablation study",
    51: "Prompt-injection detection/neutralization",
    52: "Secret detection & redaction",
    53: "Command safety (blocklist, allowlist, confirmation)",
    54: "MCP security (trust model, permissions, output validation)",
    55: "Resource-limit monitoring (iterations, tool calls, tokens)",
    56: "Single-stream TUI redesign",
    57: "Rich execution display (live phases, inline tool rows)",
    58: "Repository inspection (/inspect, /deps)",
    59: "Retrieval debugging (/debug)",
}

#: Titles for roadmap phases 67–72 (see README "Proven Static Analysis").
_ROADMAP_TITLES: dict[int, str] = {
    67: "Real call-graph resolution engine",
    68: "Data-flow tracking",
    69: "Precise blast-radius from the real graph",
    70: "Behavior-preservation checker for refactors",
    71: "Benchmark against the fakers",
    72: "Language #2",
}

#: Phase 41 (not fully wired) and phases 60–66 (never implemented).
_EXCLUDED: set[int] = {41, *range(60, 67)}


def _build() -> list[Phase]:
    phases: list[Phase] = []
    for number in range(1, 73):
        if number in _EXCLUDED:
            title = (
                "Not implemented" if number >= 60 else _TITLES.get(number, "Not implemented")
            )
            status = STATUS_EXCLUDED
        elif number in _ROADMAP_TITLES:
            title = _ROADMAP_TITLES[number]
            status = STATUS_ROADMAP
        else:
            title = _TITLES[number]
            status = STATUS_IMPLEMENTED
        phases.append(Phase(number=number, title=title, status=status))
    return phases


PHASES: list[Phase] = _build()

_PHASE_BY_NUMBER: dict[int, Phase] = {p.number: p for p in PHASES}


def get_phase(number: int) -> Phase | None:
    return _PHASE_BY_NUMBER.get(number)


def status_of(number: int) -> str:
    phase = get_phase(number)
    return phase.status if phase else STATUS_ROADMAP


def implemented() -> list[Phase]:
    return [p for p in PHASES if p.status == STATUS_IMPLEMENTED]


def excluded() -> list[Phase]:
    return [p for p in PHASES if p.status == STATUS_EXCLUDED]


def roadmap() -> list[Phase]:
    return [p for p in PHASES if p.status == STATUS_ROADMAP]


def counts() -> dict[str, int]:
    """{status → number of phases} for summary lines."""
    result: dict[str, int] = {STATUS_IMPLEMENTED: 0, STATUS_EXCLUDED: 0, STATUS_ROADMAP: 0}
    for phase in PHASES:
        result[phase.status] = result.get(phase.status, 0) + 1
    return result
