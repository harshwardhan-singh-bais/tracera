"""
Phase 53 — Command safety.

Dangerous-command detection, confirmation policy, and restricted execution
for the shell tool.

    CommandSafety.check(command) → CommandVerdict
        - ALLOW    — safe command, execute freely
        - CONFIRM  — destructive/risky, ask the user first
        - BLOCK    — never execute (rm -rf /, mkfs, ...)

The verdict combines three layers:
    1. a blocklist of destructive command patterns,
    2. the allowlist of approved base commands (settings),
    3. a sandbox check that the working directory stays in the workspace.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tracera.logging import get_logger

log = get_logger("security.command_safety")


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass
class CommandVerdict:
    """Result of a command-safety check."""

    verdict: Verdict
    reason: str
    matched_pattern: str | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "matched_pattern": self.matched_pattern,
        }


class DangerousCommandError(Exception):
    """Raised when a command is blocked outright."""

    def __init__(self, command: str, reason: str) -> None:
        super().__init__(f"Command blocked: {command!r} — {reason}")
        self.command = command
        self.reason = reason


#: Patterns that are never safe to run from an agent.
BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"(^|\s)(rm|rmdir)\s+(-[a-z]*f[a-z]*\s+)?(/\s*$|/\*|\.\s*$|~?\s*$)", "recursive force delete of root/home"),
    (r"(^|\s)mkfs(\s|\.)", "filesystem format"),
    (r"(^|\s)dd\s+.*of=/dev/", "raw device write"),
    (r"(^|\s):\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"(^|\s)shutdown\b", "shutdown"),
    (r"(^|\s)reboot\b", "reboot"),
    (r"(^|\s)halt\b", "halt"),
    (r"(^|\s)chmod\s+(-R\s+)?(777|666|a\+rwx)\b", "world-writable chmod"),
    (r"(^|\s)chown\s+-R\s+[^ ]+\s+/\s*$", "recursive chown of root"),
    (r"(^|\s)git\s+push\s+.*(-f|--force)(\s|$)", "force push"),
    (r"(^|\s)git\s+reset\s+--hard\s+(HEAD|origin/[^\s]+)\s*$", "hard reset"),
    (r"(^|\s)git\s+clean\s+(-f|--force)", "git clean force"),
    (r"(^|\s)dropdb\b", "drop database"),
    (r"(^|\s)pg_dump\s+.*--(no-owner|no-privileges)", "pg dump flag — review"),
]

#: Patterns that need explicit user confirmation before running.
CONFIRM_PATTERNS: list[tuple[str, str]] = [
    (r"(^|\s)(rm|rmdir)\s", "delete"),
    (r"(^|\s)mv\s", "move"),
    (r"(^|\s)(kill|pkill|killall)\b", "kill process"),
    (r"(^|\s)git\s+push\b", "git push"),
    (r"(^|\s)git\s+rebase\b", "git rebase"),
    (r"(^|\s)git\s+checkout\s+(-f|--force)", "git checkout force"),
    (r"(^|\s)(npm|pip|pip3|uv)\s+(uninstall|publish|login)\b", "registry mutation"),
    (r"(^|\s)(python|python3|pip|npm|npx)\s+(-m\s+)?(http\.server|SimpleHTTPServer)\b", "starts a server"),
    (r"(^|\s)curl\s+.*\|\s*(ba)?sh", "pipe to shell"),
    (r"(^|\s)(sudo|su)\s", "privilege escalation"),
    (r"(^|\s)tar\s+.*-C\s+/\s", "extract to root"),
    (r"(^|\s)cp\s+.*/\s*$", "copy into root"),
]


class CommandSafety:
    """Evaluates commands against blocklist, allowlist, and sandbox rules."""

    def __init__(
        self,
        *,
        allowed_commands: list[str] | None = None,
        workspace: Path | None = None,
        require_confirmation: bool = True,
    ) -> None:
        self.allowed_commands = set(allowed_commands or [])
        self.workspace = Path(workspace).resolve() if workspace else None
        self.require_confirmation = require_confirmation
        self._block = [(re.compile(p), reason) for p, reason in BLOCK_PATTERNS]
        self._confirm = [(re.compile(p), reason) for p, reason in CONFIRM_PATTERNS]

    # ── Checks ───────────────────────────────────────────────────────────────

    def check(self, command: str) -> CommandVerdict:
        """Full safety evaluation of *command*."""
        command = (command or "").strip()
        if not command:
            return CommandVerdict(Verdict.BLOCK, "empty command")

        # Layer 1: destructive blocklist.
        for pattern, reason in self._block:
            if pattern.search(command):
                log.warning("BLOCK %r — %s", command, reason)
                return CommandVerdict(Verdict.BLOCK, reason, pattern.pattern)

        # Layer 2: allowlist of approved base commands.
        base = self._base_command(command)
        if self.allowed_commands and base not in self.allowed_commands:
            log.warning("BLOCK %r — command '%s' not in allowlist", command, base)
            return CommandVerdict(
                Verdict.BLOCK,
                f"command '{base}' is not in the approved allowlist",
            )

        # Layer 3: confirmation-required patterns.
        for pattern, reason in self._confirm:
            if pattern.search(command):
                if self.require_confirmation:
                    log.info("CONFIRM %r — %s", command, reason)
                    return CommandVerdict(Verdict.CONFIRM, reason, pattern.pattern)
                return CommandVerdict(Verdict.ALLOW, reason)

        return CommandVerdict(Verdict.ALLOW, "safe command")

    # ── Sandbox ──────────────────────────────────────────────────────────────

    def check_cwd(self, cwd: str | Path | None) -> CommandVerdict:
        """Ensure the working directory stays inside the workspace."""
        if self.workspace is None or cwd is None:
            return CommandVerdict(Verdict.ALLOW, "no sandbox configured")
        resolved = Path(cwd).resolve()
        if self.workspace == resolved or self.workspace in resolved.parents:
            return CommandVerdict(Verdict.ALLOW, "inside workspace")
        return CommandVerdict(
            Verdict.BLOCK,
            f"working directory outside workspace: {resolved}",
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _base_command(command: str) -> str:
        """The first token of the command (strip env-var prefixes)."""
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        # Skip leading VAR=value assignments (e.g. PYTHONPATH=... python x).
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens = tokens[1:]
        return tokens[0] if tokens else ""


def check_command(
    command: str,
    *,
    allowed_commands: list[str] | None = None,
    workspace: Path | None = None,
) -> CommandVerdict:
    """Module-level convenience."""
    return CommandSafety(
        allowed_commands=allowed_commands, workspace=workspace
    ).check(command)
