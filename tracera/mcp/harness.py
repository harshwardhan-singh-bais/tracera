"""
Agent-harness catalog — which agents exist, how they read instructions, and
how to tell whether they are present.

:mod:`tracera.mcp.hosts` answers *"where does this MCP client keep its server
config"*. That is only half the integration problem. A harness that has
TRACERA registered but no standing instruction will not reach for it — the
model has no reason to prefer ``search_code`` over its own recollection. So
this module adds the missing half:

- :class:`InstructionTarget` — a file the harness reads on every request
  (``AGENTS.md``, ``CLAUDE.md``, ``.cursor/rules/*.mdc``, …), the format it
  expects, and whether it lives in the repo or the user's home.
- :class:`Harness` — one agent/IDE/CLI tool, tying together its MCP host key
  (or ``None`` when it cannot speak MCP) and its instruction targets.
- detection probes, so ``tracera integrate --all`` can target what is
  actually installed instead of writing config for tools the user has never
  opened.

Nothing here writes to disk — :mod:`tracera.mcp.integrate` does that. Keeping
the catalog declarative makes it testable without touching a real home
directory.

Where a harness's config schema has churned across releases, the entry carries
a ``note`` saying so rather than asserting a shape we have not verified.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: The cross-tool standard (https://agents.md). Written first and written for
#: every harness, because a tool that does not recognise its own native file
#: very often still reads AGENTS.md.
AGENTS_MD = "AGENTS.md"


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


# ── path builders ────────────────────────────────────────────────────────────


def _proj(*parts: str) -> Callable[[Path], Path]:
    """Project-relative path, resolved against the repo root."""
    return lambda repo_root: repo_root.joinpath(*parts)


def _user(*parts: str) -> Callable[[Path], Path]:
    """User-scoped path, resolved against the home directory."""
    return lambda repo_root: _home(*parts)


# ── detection probes ─────────────────────────────────────────────────────────


def _proj_has(*parts: str) -> Callable[[Path], bool]:
    return lambda repo_root: repo_root.joinpath(*parts).exists()


def _home_has(*parts: str) -> Callable[[Path], bool]:
    return lambda repo_root: _home(*parts).exists()


def _env_has(*names: str) -> Callable[[Path], bool]:
    """True when any of ``names`` is set — the marker a CLI exports to its children."""
    return lambda repo_root: any(os.environ.get(n) for n in names)


def _any(*probes: Callable[[Path], bool]) -> Callable[[Path], bool]:
    return lambda repo_root: any(p(repo_root) for p in probes)


# ── dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstructionTarget:
    """A file a harness reads for standing instructions."""

    key: str
    #: Shown to the user; ``~`` for home-scoped paths.
    display: str
    fmt: str  # "markdown" | "mdc" | "plaintext"
    resolve: Callable[[Path], Path]
    scope: str = "project"  # "project" | "user"
    note: str = ""

    def path(self, repo_root: Path) -> Path:
        return self.resolve(repo_root)


@dataclass(frozen=True)
class Harness:
    """One agent harness — an IDE, a CLI, or a desktop app."""

    key: str
    name: str
    kind: str  # "cli" | "ide" | "extension" | "desktop"
    #: Key into :data:`tracera.mcp.hosts.HOSTS`, or None when the harness
    #: cannot consume MCP (Aider) and needs the CLI bridge instead.
    mcp_host: str | None
    instructions: tuple[InstructionTarget, ...]
    detect: tuple[Callable[[Path], bool], ...] = field(default_factory=tuple)
    docs: str = ""
    note: str = ""

    @property
    def mcp_supported(self) -> bool:
        return self.mcp_host is not None

    def detected(self, repo_root: Path) -> bool:
        """True when this harness looks present — for ``--all`` targeting."""
        return any(p(repo_root) for p in self.detect)

    def targets(self, *, scope: str | None = None) -> tuple[InstructionTarget, ...]:
        if scope is None:
            return self.instructions
        return tuple(t for t in self.instructions if t.scope == scope)


# ── shared targets ───────────────────────────────────────────────────────────

_AGENTS_MD = InstructionTarget(
    key="agents-md",
    display=AGENTS_MD,
    fmt="markdown",
    resolve=_proj(AGENTS_MD),
)

_CLAUDE_MD = InstructionTarget(
    key="claude-md",
    display="CLAUDE.md",
    fmt="markdown",
    resolve=_proj("CLAUDE.md"),
)

_GEMINI_MD = InstructionTarget(
    key="gemini-md",
    display="GEMINI.md",
    fmt="markdown",
    resolve=_proj("GEMINI.md"),
)


# ── the catalog ──────────────────────────────────────────────────────────────

HARNESSES: dict[str, Harness] = {
    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-desktop": Harness(
        key="claude-desktop",
        name="Claude Desktop",
        kind="desktop",
        mcp_host="claude-desktop",
        # No instruction-file target: Claude Desktop keeps per-project custom
        # instructions inside the app, not in the repo. MCP wiring still works.
        instructions=(),
        detect=(
            _home_has("AppData", "Roaming", "Claude"),
            _home_has("Library", "Application Support", "Claude"),
            _home_has(".config", "Claude"),
        ),
        docs="https://modelcontextprotocol.io/quickstart/user",
        note="MCP-only: custom instructions live in the app, not on disk.",
    ),
    "claude-code": Harness(
        key="claude-code",
        name="Claude Code",
        kind="cli",
        # Project-scoped so the wiring travels with the repo instead of
        # depending on one machine's home directory.
        mcp_host="claude-code-project",
        instructions=(
            _CLAUDE_MD,
            InstructionTarget(
                key="claude-md-user",
                display="~/.claude/CLAUDE.md",
                fmt="markdown",
                resolve=_user(".claude", "CLAUDE.md"),
                scope="user",
            ),
            _AGENTS_MD,
        ),
        detect=(
            _home_has(".claude"),
            _home_has(".claude.json"),
            _proj_has(".claude"),
            _proj_has("CLAUDE.md"),
            _env_has("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"),
        ),
        docs="https://docs.anthropic.com/en/docs/claude-code",
    ),
    # ── OpenAI ────────────────────────────────────────────────────────────
    "codex": Harness(
        key="codex",
        name="Codex CLI",
        kind="cli",
        mcp_host="codex",
        instructions=(_AGENTS_MD,),
        detect=(
            _home_has(".codex"),
            _env_has("CODEX_SANDBOX", "OPENAI_CODEX"),
        ),
        docs="https://github.com/openai/codex",
        note="MCP servers live in ~/.codex/config.toml (TOML, not JSON).",
    ),
    # ── Cursor ────────────────────────────────────────────────────────────
    "cursor": Harness(
        key="cursor",
        name="Cursor",
        kind="ide",
        mcp_host="cursor-project",
        instructions=(
            InstructionTarget(
                key="cursor-mdc",
                display=".cursor/rules/tracera.mdc",
                fmt="mdc",
                resolve=_proj(".cursor", "rules", "tracera.mdc"),
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".cursor"),
            _proj_has(".cursorrules"),
            _home_has(".cursor"),
            _env_has("CURSOR_TRACE_ID", "CURSOR_AGENT"),
        ),
        docs="https://docs.cursor.com/context/rules",
        note=(
            "Cursor reads both .cursor/rules/*.mdc and AGENTS.md. The .mdc file "
            "needs YAML frontmatter; we set alwaysApply: true."
        ),
    ),
    # ── GitHub / Microsoft ────────────────────────────────────────────────
    "vscode-copilot": Harness(
        key="vscode-copilot",
        name="VS Code (Copilot)",
        kind="ide",
        mcp_host="vscode",
        instructions=(
            InstructionTarget(
                key="copilot-instructions",
                display=".github/copilot-instructions.md",
                fmt="markdown",
                resolve=_proj(".github", "copilot-instructions.md"),
            ),
            InstructionTarget(
                key="copilot-scoped",
                display=".github/instructions/tracera.instructions.md",
                fmt="markdown",
                resolve=_proj(".github", "instructions", "tracera.instructions.md"),
                note=(
                    "Copilot reads applyTo-scoped instruction files; this one "
                    "has no applyTo, so it applies repo-wide."
                ),
            ),
        ),
        detect=(
            _proj_has(".github"),
            _proj_has(".vscode"),
            _env_has("VSCODE_PID", "TERM_PROGRAM"),
        ),
        docs="https://code.visualstudio.com/docs/copilot/customization/custom-instructions",
    ),
    "vscode-copilot-chat": Harness(
        key="vscode-copilot-chat",
        name="VS Code (Copilot Chat, user scope)",
        kind="ide",
        mcp_host="vscode",
        instructions=(
            InstructionTarget(
                key="copilot-user",
                display="~/.config/Code/User/prompts/tracera.instructions.md",
                fmt="markdown",
                resolve=_user(".config", "Code", "User", "prompts", "tracera.instructions.md"),
                scope="user",
            ),
        ),
        detect=(),
        docs="https://code.visualstudio.com/docs/copilot/customization/custom-instructions",
        note="User-scope prompts directory; varies by OS (see hosts._vscode_globalstorage).",
    ),
    # ── Cline family (VS Code extensions) ─────────────────────────────────
    "cline": Harness(
        key="cline",
        name="Cline",
        kind="extension",
        mcp_host="cline",
        instructions=(
            InstructionTarget(
                key="clinerules",
                display=".clinerules/tracera.md",
                fmt="markdown",
                resolve=_proj(".clinerules", "tracera.md"),
                note="Cline reads every .md under .clinerules/ (or a .clinerules file).",
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".clinerules"),
            _home_has(".config", "Code", "User", "globalStorage", "saoudrizwan.claude-dev"),
        ),
        docs="https://docs.cline.bot/features/cline-rules",
    ),
    "roo": Harness(
        key="roo",
        name="Roo Code",
        kind="extension",
        mcp_host="roo",
        instructions=(
            InstructionTarget(
                key="roo-rules",
                display=".roo/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".roo", "rules", "tracera.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".roo"),
            _proj_has(".roorules"),
            _home_has(".config", "Code", "User", "globalStorage", "rooveterinaryinc.roo-cline"),
        ),
        docs="https://docs.roocode.com/features/custom-instructions",
    ),
    "kilo": Harness(
        key="kilo",
        name="Kilo Code",
        kind="extension",
        mcp_host="kilo",
        instructions=(
            InstructionTarget(
                key="kilo-rules",
                display=".kilocode/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".kilocode", "rules", "tracera.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".kilocode"),
            _home_has(".config", "Code", "User", "globalStorage", "kilocode.kilo-code"),
        ),
        docs="https://kilocode.ai/docs/features/custom-rules",
    ),
    # ── Windsurf ──────────────────────────────────────────────────────────
    "windsurf": Harness(
        key="windsurf",
        name="Windsurf",
        kind="ide",
        mcp_host="windsurf",
        instructions=(
            InstructionTarget(
                key="windsurf-rules",
                display=".windsurf/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".windsurf", "rules", "tracera.md"),
            ),
            InstructionTarget(
                key="windsurfrules",
                display=".windsurfrules",
                fmt="plaintext",
                resolve=_proj(".windsurfrules"),
                note="Legacy single-file form; still read by older builds.",
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".windsurf"),
            _proj_has(".windsurfrules"),
            _home_has(".codeium"),
        ),
        docs="https://docs.windsurf.com/windsurf/cascade/memories",
    ),
    # ── Google ────────────────────────────────────────────────────────────
    "gemini-cli": Harness(
        key="gemini-cli",
        name="Gemini CLI",
        kind="cli",
        mcp_host="gemini-cli",
        instructions=(_GEMINI_MD, _AGENTS_MD),
        detect=(
            _home_has(".gemini"),
            _env_has("GEMINI_CLI"),
        ),
        docs="https://github.com/google-gemini/gemini-cli",
    ),
    "antigravity": Harness(
        key="antigravity",
        name="Google Antigravity",
        kind="ide",
        mcp_host="antigravity",
        instructions=(_AGENTS_MD,),
        detect=(
            _home_has(".antigravity"),
            _home_has("AppData", "Roaming", "Antigravity"),
            _home_has("Library", "Application Support", "Antigravity"),
        ),
        docs="https://antigravity.google/docs",
        note="Gemini-agent IDE; user-level MCP config under an Antigravity app dir.",
    ),
    # ── Amazon ────────────────────────────────────────────────────────────
    "amazonq": Harness(
        key="amazonq",
        name="Amazon Q Developer",
        kind="cli",
        mcp_host="amazonq",
        instructions=(
            InstructionTarget(
                key="amazonq-rules",
                display=".amazonq/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".amazonq", "rules", "tracera.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(_home_has(".aws", "amazonq"), _proj_has(".amazonq")),
        docs="https://docs.aws.amazon.com/amazonq/",
    ),
    "kiro": Harness(
        key="kiro",
        name="Kiro",
        kind="ide",
        mcp_host="kiro",
        instructions=(
            InstructionTarget(
                key="kiro-steering",
                display=".kiro/steering/tracera.md",
                fmt="markdown",
                resolve=_proj(".kiro", "steering", "tracera.md"),
                note="Kiro calls these 'steering' files; add inclusion: always frontmatter if needed.",
            ),
            _AGENTS_MD,
        ),
        detect=(_home_has(".kiro"), _proj_has(".kiro")),
        docs="https://kiro.dev/docs/steering/",
    ),
    # ── Others ────────────────────────────────────────────────────────────
    "opencode": Harness(
        key="opencode",
        name="OpenCode",
        kind="cli",
        mcp_host="opencode",
        instructions=(_AGENTS_MD,),
        detect=(_home_has(".config", "opencode"), _home_has("AppData", "Roaming", "opencode")),
        docs="https://opencode.ai/docs/",
        note="MCP servers sit under a top-level `mcp` key with a single argv array.",
    ),
    "qwen": Harness(
        key="qwen",
        name="Qwen Code",
        kind="cli",
        mcp_host="qwen",
        instructions=(
            InstructionTarget(
                key="qwen-md",
                display="QWEN.md",
                fmt="markdown",
                resolve=_proj("QWEN.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(_home_has(".qwen"),),
        docs="https://github.com/QwenLM/qwen-code",
    ),
    "continue": Harness(
        key="continue",
        name="Continue",
        kind="extension",
        mcp_host="continue",
        instructions=(
            InstructionTarget(
                key="continue-rules",
                display=".continue/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".continue", "rules", "tracera.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(
            _proj_has(".continue"),
            _home_has(".continue"),
            _proj_has(".continuerules"),
        ),
        docs="https://docs.continue.dev/customize/rules",
    ),
    "zed": Harness(
        key="zed",
        name="Zed",
        kind="ide",
        mcp_host="zed",
        instructions=(_AGENTS_MD,),
        detect=(
            _home_has(".config", "zed"),
            _home_has("AppData", "Roaming", "Zed"),
            _home_has("Library", "Application Support", "Zed"),
        ),
        docs="https://zed.dev/docs/ai/rules",
        note="Zed stores MCP servers under `context_servers`, not `mcpServers`.",
    ),
    "trae": Harness(
        key="trae",
        name="Trae",
        kind="ide",
        mcp_host=None,
        instructions=(
            InstructionTarget(
                key="trae-rules",
                display=".trae/rules/tracera.md",
                fmt="markdown",
                resolve=_proj(".trae", "rules", "tracera.md"),
            ),
            _AGENTS_MD,
        ),
        detect=(_proj_has(".trae"), _home_has(".trae")),
        docs="https://docs.trae.ai/",
        note="Instruction-file wiring only — no verified MCP config path.",
    ),
    # ── No MCP: the CLI bridge is the whole integration ───────────────────
    "aider": Harness(
        key="aider",
        name="Aider",
        kind="cli",
        mcp_host=None,
        instructions=(
            InstructionTarget(
                key="conventions",
                display="CONVENTIONS.md",
                fmt="markdown",
                resolve=_proj("CONVENTIONS.md"),
                note="Aider only reads this when launched with --read CONVENTIONS.md.",
            ),
            _AGENTS_MD,
        ),
        detect=(_proj_has(".aider.conf.yml"), _home_has(".aider.conf.yml"), _home_has(".aider")),
        docs="https://aider.chat/docs/usage/conventions.html",
        note=(
            "Aider has no MCP client. Integrate via the CLI bridge — the brief "
            "tells the model to shell out to `tracera search --json`."
        ),
    ),
}


# ── the universal target ─────────────────────────────────────────────────────


def universal_target() -> InstructionTarget:
    """``AGENTS.md`` — read by the widest set of harnesses."""
    return _AGENTS_MD


def targets_for(
    keys: list[str] | None = None,
    *,
    repo_root: Path,
    detected_only: bool = False,
    scope: str | None = None,
) -> list[tuple[Harness, InstructionTarget]]:
    """
    Resolve ``(harness, target)`` pairs to act on.

    ``keys=None`` selects every harness; ``detected_only`` narrows that to the
    ones whose probes fired. Explicit keys always win over detection — naming a
    harness is a statement of intent, not a guess to be second-guessed.
    """
    selected: list[Harness] = []
    if keys:
        for key in keys:
            h = HARNESSES.get(key.lower())
            if h is not None:
                selected.append(h)
    else:
        selected = [
            h
            for h in HARNESSES.values()
            if not detected_only or h.detected(repo_root)
        ]

    pairs: list[tuple[Harness, InstructionTarget]] = []
    seen: set[Path] = set()
    for h in selected:
        for t in h.targets(scope=scope):
            path = t.path(repo_root)
            if path in seen:
                # Several harnesses share AGENTS.md — write it once.
                continue
            seen.add(path)
            pairs.append((h, t))
    return pairs


def resolve_keys(keys: list[str]) -> tuple[list[str], list[str]]:
    """Split user-supplied keys into ``(known, unknown)``."""
    known, unknown = [], []
    for k in keys:
        (known if k.lower() in HARNESSES else unknown).append(k)
    return known, unknown


def catalog() -> list[dict[str, object]]:
    """Machine-readable catalog, for ``tracera integrate list --json``."""
    return [
        {
            "key": h.key,
            "name": h.name,
            "kind": h.kind,
            "mcp_host": h.mcp_host,
            "mcp_supported": h.mcp_supported,
            "instructions": [
                {
                    "key": t.key,
                    "display": t.display,
                    "format": t.fmt,
                    "scope": t.scope,
                }
                for t in h.instructions
            ],
            "docs": h.docs,
            "note": h.note,
        }
        for h in HARNESSES.values()
    ]


__all__ = [
    "AGENTS_MD",
    "HARNESSES",
    "Harness",
    "InstructionTarget",
    "catalog",
    "resolve_keys",
    "targets_for",
    "universal_target",
]
