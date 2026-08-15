"""
Phase 41 — TRACERA MCP Manager + Unified Tool Registry.

The manager owns multiple MCP server connections and merges their tools
into the same ToolRegistry that holds native tools. After registration the
agent sees one flat tool list:

                    Agent
                      ↓
              Unified Tool Registry
                 ↙          ↘
        Native Tools       MCP Tools
             ↓                 ↓
       Python functions    MCP Client
                              ↓
                    External MCP Server

Server declarations can be loaded from a JSON file, e.g. `.tracera/mcp_servers.json`:

    [
        {"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        {"name": "github", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.logging import get_logger
from tracera.mcp.client import MCPClient

log = get_logger("mcp.manager")


@dataclass
class MCPServerConfig:
    """Declarative description of an external MCP server to connect to."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=str(data["name"]),
            command=str(data["command"]),
            args=list(data.get("args") or []),
            env=dict(data["env"]) if data.get("env") else None,
            cwd=str(data["cwd"]) if data.get("cwd") else None,
        )


class MCPManager:
    """
    Manages a set of external MCP server connections.

    Typical flow:

        manager = MCPManager([MCPServerConfig(...), ...])
        async with manager:
            merged = await manager.connect_all()          # dict[name → tools]
            total = await manager.register_into(registry) # unified registry
    """

    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self._configs = configs or []
        self._clients: dict[str, MCPClient] = {}

    # ── Config loading ───────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path) -> "MCPManager":
        """Load server declarations from a JSON file (see module docstring)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        configs = [MCPServerConfig.from_dict(d) for d in raw]
        return cls(configs)

    def add_server(self, config: MCPServerConfig) -> None:
        self._configs.append(config)

    @property
    def configs(self) -> list[MCPServerConfig]:
        return list(self._configs)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "MCPManager":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect_all()

    async def connect_all(self) -> dict[str, list[dict[str, Any]]]:
        """
        Connect to every configured server and return
        {server_name: [tool definitions]} for the ones that succeeded.
        Failures are logged and skipped — one bad server doesn't kill the rest.
        """
        merged: dict[str, list[dict[str, Any]]] = {}
        for config in self._configs:
            client = MCPClient(
                config.name, config.command, config.args,
                env=config.env, cwd=config.cwd,
            )
            try:
                await client.connect()
                tools = await client.list_tools()
                self._clients[config.name] = client
                merged[config.name] = tools
                log.info(
                    "Connected '%s' — %d tools available",
                    config.name, len(tools),
                )
            except Exception as e:
                log.warning("Failed to connect MCP server '%s': %s", config.name, e)
                await client.disconnect()
        return merged

    async def disconnect_all(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
            except BaseException as e:
                # Teardown must never raise into callers (CancelledError is a
                # BaseException, not an Exception).
                log.warning("Error disconnecting '%s': %s", name, e)
        self._clients.clear()

    # ── Unified registry integration ─────────────────────────────────────────

    async def register(
        self,
        merged: dict[str, list[dict[str, Any]]],
        registry: Any,
    ) -> int:
        """
        Register remote tools (from a prior connect_all() result) into the
        given TRACERA ToolRegistry, side-by-side with native tools.

        Returns the total number of MCP tools registered.
        """
        total = 0
        for name, tool_defs in merged.items():
            client = self._clients.get(name)
            if client is None:
                continue
            native = client.to_native_tools(tool_defs)
            registry.register_many(native)
            total += len(native)
        log.info("Unified registry now has %d MCP tools", total)
        return total

    async def register_into(self, registry: Any) -> int:
        """
        Connect to all servers and register every remote tool into the given
        TRACERA ToolRegistry, side-by-side with native tools.

        Returns the total number of MCP tools registered.
        """
        merged = await self.connect_all()
        try:
            return await self.register(merged, registry)
        finally:
            await self.disconnect_all()

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())
