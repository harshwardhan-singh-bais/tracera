"""
Diff helpers for the TUI — Phase (v3) code-gen summary rows.

Turns the before/after content of a file around a tool call into a compact
unified diff. The TUI snapshots the file when edit_file / write_file starts,
reads it again when the tool ends, and renders a `📝 path +N -M` summary row
with an expandable inline diff (added green, removed red, unchanged dim).
"""

from __future__ import annotations

import difflib

# Files larger than this (bytes) are not diffed — reading them back and
# rendering thousands of lines would stall the stream for little value.
MAX_DIFF_BYTES = 500_000

#: Tools whose effect can be diffed deterministically (path comes from args).
DIFFABLE_TOOLS = ("edit_file", "write_file")

_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff",
}


def is_image(path: str) -> bool:
    """True if *path* looks like an image file (by extension)."""
    return any(path.lower().endswith(s) for s in _IMAGE_SUFFIXES)


def compute_diff(before: str, after: str, path: str) -> tuple[list[tuple[str, str]], int, int]:
    """
    Return ``(lines, added, removed)`` for a unified diff of *before* → *after*.

    ``lines`` is a list of ``(kind, text)`` where kind is one of
    ``"add"`` / ``"del"`` / ``"ctx"`` / ``"hunk"`` so the renderer can color
    each line without parsing diff syntax itself.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    added = 0
    removed = 0

    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    lines: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in before_lines[i1:i2]:
                lines.append(("ctx", line))
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
            for line in before_lines[i1:i2]:
                lines.append(("del", line))
            for line in after_lines[j1:j2]:
                lines.append(("add", line))
        elif tag == "delete":
            removed += i2 - i1
            for line in before_lines[i1:i2]:
                lines.append(("del", line))
        elif tag == "insert":
            added += j2 - j1
            for line in after_lines[j1:j2]:
                lines.append(("add", line))

    if added == 0 and removed == 0:
        return [], 0, 0

    # Trim unchanged context lines around the hunk to keep the diff compact.
    trimmed = _trim_context(lines)
    trimmed.insert(0, ("hunk", f"diff --git a/{path} b/{path}"))
    return trimmed, added, removed


def _trim_context(
    lines: list[tuple[str, str]], max_context: int = 3
) -> list[tuple[str, str]]:
    """Keep at most *max_context* unchanged lines between changed groups."""
    result: list[tuple[str, str]] = []
    ctx_run = 0
    last_was_change = False
    for kind, text in lines:
        if kind == "ctx":
            if ctx_run < max_context:
                result.append((kind, text))
            elif last_was_change:
                result.append(("ellipsis", "⋯"))
                last_was_change = False
            ctx_run += 1
        else:
            result.append((kind, text))
            ctx_run = 0
            last_was_change = True
    return result
