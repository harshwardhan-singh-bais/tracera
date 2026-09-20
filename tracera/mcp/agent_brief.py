"""
The agent brief — the standing instruction TRACERA writes into a harness.

A harness that merely has TRACERA *installed* will not use it: the model has
no reason to prefer ``search_code`` over its own guess. The brief is what
supplies that reason. It is written into whichever instruction file the
harness actually reads (``AGENTS.md``, ``CLAUDE.md``, ``.cursor/rules/*.mdc``,
…), so it travels with the repo and survives a fresh clone.

Three concerns live here, kept separate on purpose:

- :func:`brief_body` — *what* to say. Single source of truth; the tool table
  is generated from :mod:`tracera.mcp.server` so it cannot drift from the
  real tool surface.
- :func:`render` — *how* to say it in a given format (markdown / mdc /
  plaintext). Only marker style and frontmatter differ.
- :func:`splice` — *where* it goes inside an existing file. Never clobbers
  user prose: everything between the managed markers is replaced, everything
  outside them is preserved byte-for-byte.
"""

from __future__ import annotations

from typing import Any

#: Bumped whenever the body changes enough that an existing managed block
#: should be rewritten on the next ``tracera integrate`` run.
BRIEF_VERSION = 1

#: Where the managed region starts / ends, per format. Markers are matched as
#: whole lines, so a user's prose that happens to mention the marker inline is
#: not mistaken for one.
_HTML_BEGIN = "<!-- tracera:begin (managed by `tracera integrate`) -->"
_HTML_END = "<!-- tracera:end -->"
_HASH_BEGIN = "# >>> tracera:begin (managed by `tracera integrate`) >>>"
_HASH_END = "# <<< tracera:end <<<"

_MARKERS: dict[str, tuple[str, str]] = {
    "markdown": (_HTML_BEGIN, _HTML_END),
    "mdc": (_HTML_BEGIN, _HTML_END),
    "plaintext": (_HASH_BEGIN, _HASH_END),
}

#: Formats we know how to render.
FORMATS = tuple(_MARKERS)

#: Cursor's rule frontmatter. `alwaysApply: true` is what makes the rule load
#: on every request instead of only when the description matches.
_CURSOR_FRONTMATTER = (
    "---\n"
    "description: TRACERA code intelligence and memory\n"
    "globs:\n"
    "alwaysApply: true\n"
    "---\n"
)

_FRONTMATTER_BY_FORMAT: dict[str, str] = {"mdc": _CURSOR_FRONTMATTER}


# ── the body ─────────────────────────────────────────────────────────────────

#: Grouped tool surface, resolved lazily from the MCP server so the brief can
#: never advertise a tool that was renamed or removed. Importing
#: ``tracera.mcp.server`` pulls in FastMCP, so it is deferred until a caller
#: actually needs the table — ``tracera integrate list`` stays fast.
_TOOL_GROUPS_CACHE: dict[str, list[str]] | None = None


def _tool_groups() -> dict[str, list[str]]:
    global _TOOL_GROUPS_CACHE
    if _TOOL_GROUPS_CACHE is None:
        from tracera.mcp.server import (
            CODE_INTELLIGENCE_TOOLS,
            CONTEXT_TOOLS,
            DIAGNOSTICS_TOOLS,
            MEMORY_TOOLS,
            REPOSITORY_TOOLS,
            SAFETY_TOOLS,
        )

        _TOOL_GROUPS_CACHE = {
            "Code intelligence": list(CODE_INTELLIGENCE_TOOLS),
            "Context": list(CONTEXT_TOOLS),
            "Memory": list(MEMORY_TOOLS),
            "Safety": list(SAFETY_TOOLS),
            "Repository": list(REPOSITORY_TOOLS),
            "Diagnostics": list(DIAGNOSTICS_TOOLS),
        }
    return _TOOL_GROUPS_CACHE


#: The few tools worth naming explicitly, with the moment each one earns its
#: call. A flat dump of 44 names does not teach a model *when* to reach.
_WHEN_MCP: tuple[tuple[str, str], ...] = (
    (
        "get_context",
        "before editing a file you have not read — the definition plus the "
        "callers and dependencies that matter",
    ),
    (
        "search_code",
        "when the user names a concept and you do not know where it lives",
    ),
    (
        "get_blast_radius",
        "before changing a signature — what breaks if this moves",
    ),
    (
        "check_edit_safe",
        "immediately before writing a file, to catch a destructive edit",
    ),
    (
        "recall_memory",
        "at the start of a task, for decisions and conventions already learned",
    ),
    (
        "remember_memory",
        "when the user states a durable preference you would otherwise "
        "re-derive next session",
    ),
    ("run_tests", "before claiming a task is done"),
)

#: The CLI stand-in. Deliberately *not* a 1:1 translation of the MCP list —
#: several tools collapse onto one command, and some (run_tests) have no
#: equivalent at all. Phrased around the moment, not the tool.
_WHEN_CLI: tuple[str, ...] = (
    '`tracera search "<question>" --json` — before editing a file you have not '
    "read, and when the user names a concept and you do not know where it lives",
    '`tracera search "<symbol>" --json` — before changing a signature, to see '
    "everything that references it",
    "`tracera review` — before handing a diff over, to surface risk in the change",
    "`tracera memory list` — at the start of a task, for decisions and "
    "conventions already learned",
    '`tracera memory add "<fact>"` — when the user states a durable preference '
    "you would otherwise re-derive next session",
    "run the project's own test command before claiming a task is done",
)

#: The commands a non-MCP harness can actually shell out to.
_CLI_COMMANDS: tuple[tuple[str, str], ...] = (
    ('tracera search "<query>" --json', "ranked code hits with path:line and scores"),
    ("tracera status --json", "index, memory and provider state"),
    ("tracera index", "build or refresh the code index (run after a big pull)"),
    ("tracera review", "review the current diff for risk before you hand it over"),
    ("tracera memory list", "recall stored decisions and conventions"),
    ('tracera memory add "<fact>"', "store a durable fact"),
    ('tracera ask "<question>"', "one-shot agent query against the repo"),
)


def brief_body(*, mcp: bool = True, workspace: str | None = None) -> str:
    """
    The canonical brief, as markdown.

    ``mcp`` controls whether the tool table is presented as MCP tool names
    (preferred — no shell round-trip) or as the equivalent CLI invocations
    for harnesses that cannot speak MCP.
    """
    lines: list[str] = []
    add = lines.append

    add("## TRACERA — code intelligence & memory")
    add("")
    add(
        "This repository has [TRACERA](https://github.com/tracera/tracera) wired in. "
        "It answers structural questions about the code from a real index — symbols, "
        "call graphs, blast radius, git history — instead of from guesswork."
    )
    add("")
    if mcp:
        add(
            "**Prefer these tools over reading files speculatively.** They are cheap, "
            "they return exact `path:line` locations, and they are the difference "
            "between a plausible answer and a correct one."
        )
    else:
        add(
            "**Prefer the `tracera` CLI over reading files speculatively.** Every "
            "command accepts `--json`, so you can pipe results into your own "
            "reasoning without parsing decorated terminal output."
        )
    add("")

    # ── when to reach for it ──────────────────────────────────────────────
    add("### When to use it")
    add("")
    if mcp:
        for tool, when in _WHEN_MCP:
            add(f"- `{tool}` — {when}.")
    else:
        for item in _WHEN_CLI:
            add(f"- {item}.")
    add("")

    # ── tool surface ──────────────────────────────────────────────────────
    if mcp:
        add("### Available tools")
        add("")
        for group, tools in _tool_groups().items():
            add(f"- **{group}** — {', '.join(f'`{t}`' for t in tools)}")
        add("")
    else:
        # Without MCP there is no honest one-to-one mapping onto the 40 MCP
        # tools, so name the commands that actually exist.
        add("### Available commands")
        add("")
        for cmd, what in _CLI_COMMANDS:
            add(f"- `{cmd}` — {what}")
        add("")
        add("Add `--json` to `search` and `status` for machine-readable output.")
        add("")

    # ── ground rules ──────────────────────────────────────────────────────
    add("### Ground rules")
    add("")
    add(
        "- Cite locations as `path:line`. If you cannot point at one, say so "
        "rather than inventing a symbol."
    )
    add(
        "- If TRACERA reports the index is stale or empty, run "
        "`tracera index` rather than falling back to grep-and-hope."
    )
    add(
        "- Store durable facts with "
        + ("`remember_memory`" if mcp else "`tracera memory add`")
        + " — not in the code as comments."
    )
    add("- Run the project's test command before reporting a task complete.")

    if workspace:
        add("")
        add(f"<!-- workspace: {workspace} -->")

    return "\n".join(lines)


# ── rendering ────────────────────────────────────────────────────────────────


def render(body: str, fmt: str) -> str:
    """
    Wrap ``body`` in the managed markers for ``fmt``.

    The result is the exact string :func:`splice` looks for, so callers should
    always build the block through here rather than by hand.
    """
    if fmt not in _MARKERS:
        raise ValueError(f"unknown instruction format {fmt!r}; expected one of {FORMATS}")
    begin, end = _MARKERS[fmt]
    return f"{begin}\n\n{body.rstrip()}\n\n{end}\n"


def frontmatter(fmt: str) -> str:
    """Leading frontmatter a fresh file of this format requires ('' if none)."""
    return _FRONTMATTER_BY_FORMAT.get(fmt, "")


# ── splicing into an existing file ───────────────────────────────────────────


class MalformedBlockError(ValueError):
    """A begin marker exists with no matching end marker."""


def _find_markers(text: str, fmt: str) -> tuple[int, int] | None:
    """Line indices of (begin, end), or None when neither is present."""
    begin_marker, end_marker = _MARKERS[fmt]
    lines = text.splitlines()
    begin = end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin < 0 and stripped == begin_marker:
            begin = i
        elif begin >= 0 and stripped == end_marker:
            end = i
            break
    if begin < 0:
        return None
    if end < 0:
        raise MalformedBlockError(
            f"found a tracera:begin marker at line {begin + 1} with no matching "
            "tracera:end — refusing to edit; fix or delete the stray marker"
        )
    return begin, end


def has_managed_block(text: str, fmt: str) -> bool:
    """True when the text already carries a well-formed managed block."""
    try:
        return _find_markers(text, fmt) is not None
    except MalformedBlockError:
        return False


def splice(existing: str | None, body: str, fmt: str) -> tuple[str, str]:
    """
    Merge the managed block into ``existing``.

    Returns ``(new_text, status)`` where status is one of:

    ``"installed"``
        no managed block was present — appended (or written whole, if the file
        was empty/absent);
    ``"updated"``
        a managed block was present and its contents changed;
    ``"unchanged"``
        a managed block was present and already byte-identical.

    Text outside the markers is never touched. A begin marker with no matching
    end raises :class:`MalformedBlockError` instead of guessing at the extent
    of the region.
    """
    block = render(body, fmt)

    if existing is None or not existing.strip():
        return frontmatter(fmt) + block, "installed"

    found = _find_markers(existing, fmt)
    if found is None:
        prefix = frontmatter(fmt) if not _has_frontmatter(existing) else ""
        joined = existing.rstrip("\n") + "\n\n" + block
        return prefix + joined, "installed"

    begin, end = found
    lines = existing.splitlines(keepends=True)
    current = "".join(lines[begin : end + 1])
    if current == block:
        return existing, "unchanged"
    new_text = "".join(lines[:begin]) + block + "".join(lines[end + 1 :])
    return new_text, "updated"


def _has_frontmatter(text: str) -> bool:
    return text.lstrip().startswith("---\n")


def strip_block(text: str, fmt: str) -> str:
    """Remove the managed block (used by tests and ``--remove`` style flows)."""
    try:
        found = _find_markers(text, fmt)
    except MalformedBlockError:
        return text
    if found is None:
        return text
    begin, end = found
    lines = text.splitlines(keepends=True)
    return "".join(lines[:begin]) + "".join(lines[end + 1 :])


def block_version(text: str, fmt: str) -> int | None:
    """
    Version of the managed block already on disk, if any.

    Today the markers carry no version, so presence maps to
    :data:`BRIEF_VERSION`; this is the seam for a future ``tracera:begin v2``
    marker without changing callers.
    """
    return BRIEF_VERSION if has_managed_block(text, fmt) else None


def describe() -> dict[str, Any]:
    """Machine-readable summary of the brief, for ``tracera integrate doctor``."""
    groups = _tool_groups()
    return {
        "version": BRIEF_VERSION,
        "formats": list(FORMATS),
        "tool_count": sum(len(v) for v in groups.values()),
        "groups": {k: len(v) for k, v in groups.items()},
    }


__all__ = [
    "BRIEF_VERSION",
    "FORMATS",
    "MalformedBlockError",
    "block_version",
    "brief_body",
    "describe",
    "frontmatter",
    "has_managed_block",
    "render",
    "splice",
    "strip_block",
]
