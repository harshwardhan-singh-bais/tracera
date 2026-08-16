"""
TRACERA Security — Phases 51-55.

    injection.py      — Phase 51: prompt-injection defenses for repository
                        files, retrieved code, web content, MCP outputs.
    secrets.py        — Phase 52: detect & redact API keys, tokens, .env.
    command_safety.py — Phase 53: dangerous command detection, confirmation
                        policy, sandboxed execution.
    mcp_security.py   — Phase 54: MCP server trust model, tool permissions,
                        confirmation for destructive tools, output validation.
    resources.py      — Phase 55: resource-limit monitoring/enforcement.
"""

from tracera.security.injection import (
    InjectionFinding,
    InjectionPolicy,
    PromptInjectionDetector,
    sanitize_content,
)
from tracera.security.secrets import (
    SECRET_PATTERNS,
    SecretRedactor,
    SecretScanResult,
    redact_text,
    scan_text,
)
from tracera.security.command_safety import (
    CommandSafety,
    CommandVerdict,
    DangerousCommandError,
    check_command,
)
from tracera.security.mcp_security import (
    MCPSecurityManager,
    MCPToolPolicy,
    ServerTrust,
    validate_mcp_output,
)
from tracera.security.resources import ResourceMonitor

__all__ = [
    "InjectionFinding",
    "InjectionPolicy",
    "PromptInjectionDetector",
    "sanitize_content",
    "SECRET_PATTERNS",
    "SecretRedactor",
    "SecretScanResult",
    "redact_text",
    "scan_text",
    "CommandSafety",
    "CommandVerdict",
    "DangerousCommandError",
    "check_command",
    "MCPSecurityManager",
    "MCPToolPolicy",
    "ServerTrust",
    "validate_mcp_output",
    "ResourceMonitor",
]
