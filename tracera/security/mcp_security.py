"""
Phase 54 — MCP security.

External MCP servers are third-party code executing in subprocesses — they
need a trust model:

    ServerTrust.TRUSTED     — official / pinned servers, full access
    ServerTrust.UNTRUSTED   — community servers, default-deny-ish policy
    ServerTrust.BLOCKED     — never connect

Tool policies gate what each server may call:

    MCPToolPolicy.ALLOW     — execute freely
    MCPToolPolicy.CONFIRM   — require user confirmation (destructive tools)
    MCPToolPolicy.BLOCK     — never call

Output validation runs every MCP tool result through secret redaction and
prompt-injection detection before it reaches the agent context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tracera.logging import get_logger
from tracera.security.injection import PromptInjectionDetector
from tracera.security.secrets import SecretRedactor

log = get_logger("security.mcp_security")


class ServerTrust(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    BLOCKED = "blocked"


class MCPToolPolicy(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


#: Tool names/substrings that mutate state and should need confirmation
#: unless the server is fully trusted.
DESTRUCTIVE_TOOL_HINTS = (
    "delete", "remove", "rm", "drop", "truncate", "write", "update",
    "create", "edit", "insert", "push", "merge", "publish", "deploy",
    "restart", "stop", "kill", "format", "clear", "overwrite", "rename",
    "move", "execute", "run",
)


@dataclass
class ToolPolicyRules:
    """Per-server tool gating rules."""

    #: Default for tools not otherwise matched.
    default: MCPToolPolicy = MCPToolPolicy.CONFIRM
    #: Tool-name substring → policy.
    rules: dict[str, MCPToolPolicy] = field(default_factory=dict)

    def policy_for(self, tool_name: str) -> MCPToolPolicy:
        lowered = tool_name.lower()
        for fragment, policy in self.rules.items():
            if fragment.lower() in lowered:
                return policy
        # Destructive hints need confirmation by default.
        if any(hint in lowered for hint in DESTRUCTIVE_TOOL_HINTS):
            return MCPToolPolicy.CONFIRM
        return self.default

    def requires_confirmation(self, tool_name: str) -> bool:
        return self.policy_for(tool_name) == MCPToolPolicy.CONFIRM


def default_policy(trust: ServerTrust) -> ToolPolicyRules:
    """Sensible default tool policy for a trust level."""
    if trust == ServerTrust.TRUSTED:
        return ToolPolicyRules(default=MCPToolPolicy.ALLOW)
    if trust == ServerTrust.UNTRUSTED:
        return ToolPolicyRules(default=MCPToolPolicy.CONFIRM)
    return ToolPolicyRules(default=MCPToolPolicy.BLOCK)


#: Registry of servers known to be official / safe-ish by name.
TRUSTED_SERVER_NAMES = {
    "filesystem", "github", "git", "postgres", "postgresql",
    "playwright", "brave-search", "fetch", "memory",
}


@dataclass
class MCPValidationResult:
    """Outcome of validating an MCP tool output."""

    redacted: str
    secrets_found: int
    injections_found: int
    blocked: bool = False
    reason: str | None = None


class MCPSecurityManager:
    """Central gate for external MCP servers: trust, tools, and output."""

    def __init__(
        self,
        *,
        trusted_servers: set[str] | None = None,
        blocked_servers: set[str] | None = None,
        policy: ToolPolicyRules | None = None,
    ) -> None:
        self.trusted_servers = set(trusted_servers or TRUSTED_SERVER_NAMES)
        self.blocked_servers = set(blocked_servers or set())
        self._default_policy = policy
        self._redactor = SecretRedactor()
        self._detector = PromptInjectionDetector()

    # ── Trust model ──────────────────────────────────────────────────────────

    def trust_for(self, server_name: str) -> ServerTrust:
        if server_name in self.blocked_servers:
            return ServerTrust.BLOCKED
        if server_name in self.trusted_servers:
            return ServerTrust.TRUSTED
        return ServerTrust.UNTRUSTED

    def check_connect(self, server_name: str) -> tuple[bool, str | None]:
        """May we connect to this server? Returns (allowed, reason)."""
        trust = self.trust_for(server_name)
        if trust == ServerTrust.BLOCKED:
            return False, f"server '{server_name}' is on the blocklist"
        return True, None

    # ── Tool permissions ─────────────────────────────────────────────────────

    def policy_for_server(self, server_name: str) -> ToolPolicyRules:
        if self._default_policy is not None:
            return self._default_policy
        return default_policy(self.trust_for(server_name))

    def check_tool(
        self, server_name: str, tool_name: str
    ) -> tuple[bool, MCPToolPolicy]:
        """
        May we call *tool_name* on *server_name*?

        Returns (allowed, policy). CONFIRM means "allowed but require user
        confirmation before executing".
        """
        trust = self.trust_for(server_name)
        if trust == ServerTrust.BLOCKED:
            return False, MCPToolPolicy.BLOCK
        policy = self.policy_for_server(server_name).policy_for(tool_name)
        if policy == MCPToolPolicy.BLOCK:
            log.warning("BLOCKED MCP tool call %s:%s", server_name, tool_name)
            return False, policy
        if policy == MCPToolPolicy.CONFIRM:
            log.info("MCP tool %s:%s requires confirmation", server_name, tool_name)
            return True, policy  # allowed with confirmation
        return True, policy

    # ── Output validation ────────────────────────────────────────────────────

    def validate_output(self, text: str) -> MCPValidationResult:
        """
        Sanitize an MCP tool result before it reaches the agent:
        redact secrets, flag/strip prompt-injection attempts.
        """
        redacted, secret_findings = self._redactor.redact(text)
        sanitized, injection_findings = self._detector.sanitize(
            redacted, mode="strip"
        )
        return MCPValidationResult(
            redacted=sanitized,
            secrets_found=len(secret_findings),
            injections_found=len(injection_findings),
        )

    def tool_is_destructive(self, tool_name: str) -> bool:
        return any(hint in tool_name.lower() for hint in DESTRUCTIVE_TOOL_HINTS)


def validate_mcp_output(text: str) -> MCPValidationResult:
    """Module-level convenience."""
    return MCPSecurityManager().validate_output(text)
