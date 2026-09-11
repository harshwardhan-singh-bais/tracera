"""
TRACERA Slash-Command Registry — single source of truth for REPL commands.

The TUI autocomplete (``CommandInput``), the help screen, and the command
palette all read from ``SLASH_COMMANDS`` so a new command only needs to be
added once. The actual handlers live in ``tracera.tui.app``.
"""

from __future__ import annotations

#: command name (without leading "/") → short description shown in
#: autocomplete and /help.
SLASH_COMMANDS: dict[str, str] = {
    "help": "Show all commands and keys",
    "clear": "Clear the conversation",
    "status": "System status (provider, model, tokens, memory)",
    "memory": "Show persistent memory contents",
    "memgraph": "Show the memory knowledge graph",
    "model": "Switch model: /model <id> (or list with /models)",
    "models": "List all provider/model options",
    "plan": "Decompose a task into steps: /plan <task>",
    "code": "Run a coding task: /code <task>",
    "search": "Search the code index (hybrid): /search <query>",
    "debug": "Compare retrieval strategies: /debug <query>",
    "index": "Index the workspace (incremental)",
    "test": "Run the project's test suite",
    "review": "Ask the agent to review current changes",
    "tools": "List available tools",
    "mcp": "Show MCP status & config",
    "cost": "Session token/cost estimate",
    "observability": "Live telemetry (LLM/tool/retrieval/cost)",
    "inspect": "Repository inspection (files, symbols, git)",
    "deps": "Symbol dependency chain: /deps <symbol>",
    "dashboard": "System overview panel",
    "files": "Recently touched files",
    "phases": "Phase map + verified checklist",
    "reset": "Reset conversation state",
}

#: Ordered the way they should appear in autocomplete.
COMMAND_ORDER: list[str] = [
    "help",
    "clear",
    "status",
    "cost",
    "model",
    "models",
    "plan",
    "code",
    "search",
    "debug",
    "index",
    "test",
    "review",
    "tools",
    "mcp",
    "memory",
    "memgraph",
    "inspect",
    "deps",
    "dashboard",
    "files",
    "observability",
    "phases",
    "reset",
]

__all__ = ["SLASH_COMMANDS", "COMMAND_ORDER"]
