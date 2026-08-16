"""
Phase 52 — Secret protection.

Detect and redact secrets (API keys, tokens, credentials, .env contents)
before they reach the LLM, logs, MCP outputs, or are echoed back to users.

Usage:

    redact_text("api key is sk-abc123")      → "api key is sk-***REDACTED***"
    scan_text("GITHUB_TOKEN=ghp_xxx")        → [SecretScanResult(...)]
    SecretRedactor().scan_file(".env")       → [SecretScanResult(...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tracera.logging import get_logger

log = get_logger("security.secrets")

REDACTED = "***REDACTED***"

#: (pattern, kind) — anchored to be reasonably specific to avoid mangling
#: normal text. Value capture is greedy but bounded.
SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "github_token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack_token"),
    (re.compile(r"\bxapp-[A-Za-z0-9-]{10,}\b"), "slack_app_token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "google_api_key"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "openai_style_key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "openai_key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "huggingface_token"),
    (re.compile(r"\blin_api_[A-Za-z0-9]{20,}\b"), "linear_api_key"),
    (re.compile(r"\bsntrys_[A-Za-z0-9]{20,}\b"), "sentry_token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"), "jwt_token"),
    (re.compile(r"\b-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private_key"),
    (re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[=:]\s*['\"]?[A-Za-z0-9_\-./+]{8,}['\"]?"), "key_value_pair"),
]

#: File names that are almost certainly secret-bearing.
SECRET_FILENAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials.json", "service-account.json",
    "secrets.yaml", "secrets.yml", "vault", "id_rsa.pub", "known_hosts",
}


@dataclass
class SecretScanResult:
    """A detected secret occurrence."""

    kind: str
    value: str
    start: int
    end: int

    @property
    def preview(self) -> str:
        """First 6 chars + '…' — enough to identify, never enough to leak."""
        if len(self.value) <= 8:
            return "***"
        return self.value[:4] + "…"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "preview": self.preview,
            "start": self.start,
            "end": self.end,
        }


class SecretRedactor:
    """Detects secrets in text/files and produces redacted copies."""

    def __init__(
        self,
        patterns: list[tuple[re.Pattern, str]] | None = None,
    ) -> None:
        self.patterns = patterns or SECRET_PATTERNS

    def scan(self, text: str) -> list[SecretScanResult]:
        """Scan *text* for secrets. Returns findings (possibly empty)."""
        if not text:
            return []
        results: list[SecretScanResult] = []
        for pattern, kind in self.patterns:
            for match in pattern.finditer(text):
                results.append(
                    SecretScanResult(
                        kind=kind,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        if results:
            log.warning("Secret scan: %d secret(s) detected", len(results))
        return results

    def redact(self, text: str) -> tuple[str, list[SecretScanResult]]:
        """Redact all secrets in *text*. Returns (redacted, findings)."""
        findings = self.scan(text)
        if not findings:
            return text, findings
        result = text
        for f in sorted(findings, key=lambda f: f.start, reverse=True):
            result = result[: f.start] + REDACTED + result[f.end :]
        return result, findings

    def scan_file(self, path: str | Path) -> list[SecretScanResult]:
        """Scan a file on disk (binary-safe read)."""
        path = Path(path)
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        return self.scan(text)

    def is_secret_file(self, path: str | Path) -> bool:
        return Path(path).name in SECRET_FILENAMES


#: Module-level default redactor (lazy).
_redactor: SecretRedactor | None = None


def get_redactor() -> SecretRedactor:
    global _redactor
    if _redactor is None:
        _redactor = SecretRedactor()
    return _redactor


def redact_text(text: str) -> tuple[str, list[SecretScanResult]]:
    """Module-level convenience: redact secrets in *text*."""
    return get_redactor().redact(text)


def scan_text(text: str) -> list[SecretScanResult]:
    """Module-level convenience: scan *text* for secrets."""
    return get_redactor().scan(text)
