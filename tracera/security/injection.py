"""
Phase 51 — Prompt-injection defenses.

Code, web pages, and MCP tool outputs can contain instructions aimed at the
LLM ("ignore previous instructions", "you are now a different assistant",
"exfiltrate the API keys", ...). This module detects those patterns and
neutralises the content before it reaches the model.

Pipeline:

    raw content (repo file / retrieved chunk / web / MCP output)
        ↓
    PromptInjectionDetector.scan(text) → list[InjectionFinding]
        ↓
    sanitize_content(text) → content with injected blocks flagged/removed
        ↓
    agent prompt (safer)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from tracera.logging import get_logger

log = get_logger("security.injection")

#: Phrases that signal an instruction aimed at an AI assistant rather than
#: ordinary code/documentation text.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (all |any |the )?(previous|above|prior|earlier) (instructions|prompts|messages|context|directions)", "instruction override"),
    (r"disregard (all |the )?(previous|above|prior|earlier) (instructions|prompts|messages|context)", "instruction override"),
    (r"you (are|must act as|now act as|are now) (a |an )?(different|new) (assistant|agent|AI|model|system)", "role hijack"),
    (r"you are now (a |an )?(assistant|agent|AI|model|system) named", "role hijack"),
    (r"forget (everything|all previous|the system prompt)", "memory wipe"),
    (r"(reveal|show|print|output|send|exfiltrate).{0,40}(system prompt|instructions|api key|secrets|credentials)", "exfiltration request"),
    (r"(system|developer) prompt.{0,30}(reveal|show|leak|expose)", "prompt leak"),
    (r"do not tell (the user|anyone|humans)", "concealment"),
    (r"say you (can't|cannot|did not|didn't) (do|perform|complete) (it|the task)", "false denial"),
    (r"if you (see|find|read) (this|the following)", "conditional trigger"),
    (r"<[^>]*system[^>]*>", "embedded system tag"),
    (r"\*\*important\*\*:?\s*you", "emphasis instruction"),
    (r"reply with (only|just) (yes|no|ok)", "constrained reply"),
]

#: Keep user-visible context: these are findings, not auto-removed.
DEFAULT_SANITIZE_MODE = "flag"  # 'flag' | 'strip'


@dataclass
class InjectionFinding:
    """One detected injection attempt."""

    pattern: str
    kind: str
    snippet: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "kind": self.kind,
            "snippet": self.snippet[:120],
            "start": self.start,
            "end": self.end,
        }


class PromptInjectionDetector:
    """Searches text for prompt-injection patterns."""

    def __init__(
        self,
        patterns: list[tuple[str, str]] | None = None,
        *,
        max_snippet: int = 160,
    ) -> None:
        self._compiled = [
            (re.compile(p, re.IGNORECASE), kind) for p, kind in (patterns or INJECTION_PATTERNS)
        ]
        self.max_snippet = max_snippet

    def scan(self, text: str) -> list[InjectionFinding]:
        """Return all injection findings in *text* (empty list = clean)."""
        if not text:
            return []
        findings: list[InjectionFinding] = []
        for pattern, kind in self._compiled:
            for match in pattern.finditer(text):
                findings.append(
                    InjectionFinding(
                        pattern=pattern.pattern,
                        kind=kind,
                        snippet=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        if findings:
            log.warning("Prompt-injection scan: %d finding(s) in %d chars", len(findings), len(text))
        return findings

    def is_clean(self, text: str) -> bool:
        return not self.scan(text)

    def sanitize(self, text: str, mode: str = DEFAULT_SANITIZE_MODE) -> tuple[str, list[InjectionFinding]]:
        """
        Neutralise injected instructions.

        mode='flag'  → wrap each match in a visible warning marker
                       ([INJECTION DETECTED: ...]).
        mode='strip' → remove the matched text entirely.

        Returns (sanitized_text, findings).
        """
        findings = self.scan(text)
        if not findings:
            return text, findings

        result = text
        if mode == "strip":
            for f in sorted(findings, key=lambda f: f.start, reverse=True):
                result = result[: f.start] + f"[redacted]" + result[f.end :]
        else:
            for f in sorted(findings, key=lambda f: f.start, reverse=True):
                marker = f"[INJECTION DETECTED ({f.kind})]"
                result = result[: f.start] + marker + result[f.end :]
        return result, findings


#: Module-level default detector (lazy).
_detector: PromptInjectionDetector | None = None


def get_detector() -> PromptInjectionDetector:
    global _detector
    if _detector is None:
        _detector = PromptInjectionDetector()
    return _detector


def sanitize_content(
    text: str,
    mode: str = DEFAULT_SANITIZE_MODE,
) -> tuple[str, list[InjectionFinding]]:
    """Module-level convenience: sanitize *text* with the default detector."""
    return get_detector().sanitize(text, mode=mode)


def scan_content(text: str) -> list[InjectionFinding]:
    """Module-level convenience: scan *text* with the default detector."""
    return get_detector().scan(text)


class InjectionPolicy:
    """
    Policy wrapper: decide what to do with content based on its source.

    Sources: 'repository' (files), 'retrieval' (indexed code), 'web',
    'mcp' (external tool output). Different sources get different
    strictness (web and MCP content are untrusted; repository files are
    mostly trusted code but may still contain instructions).
    """

    #: Sources where any injection finding should be stripped outright.
    STRIP_SOURCES = {"web", "mcp"}
    #: Sources where injections are flagged but kept (code may legitimately
    #: contain these phrases in strings, docs, tests).
    FLAG_SOURCES = {"repository", "retrieval"}

    def __init__(
        self,
        detector: PromptInjectionDetector | None = None,
        *,
        strip_sources: Iterable[str] | None = None,
        flag_sources: Iterable[str] | None = None,
    ) -> None:
        self.detector = detector or get_detector()
        self.strip_sources = set(strip_sources or self.STRIP_SOURCES)
        self.flag_sources = set(flag_sources or self.FLAG_SOURCES)

    def process(self, text: str, source: str) -> tuple[str, list[InjectionFinding]]:
        """Sanitize *text* per the policy for *source*."""
        if source in self.strip_sources:
            return self.detector.sanitize(text, mode="strip")
        return self.detector.sanitize(text, mode="flag")
