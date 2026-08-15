"""
Phase 40 — TRACERA MCP Client.

Connects to external MCP servers, lists their tools, and adapts them into
native TRACERA ``Tool`` objects so the agent cannot tell native tools from
remote MCP tools.

    connect() → initialize → tools/list → receive tools → register tools
                                                            ↓
                                                         tools/call

The official MCP SDK (mcp >= 1.2, < 2) speaks the real protocol, so this
client interoperates with any standards-compliant MCP server (filesystem,
GitHub, Postgres, ...).
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tracera.logging import get_logger
from tracera.tools.base import Tool, ToolResult

log = get_logger("mcp.client")

CallToolFn = Callable[[dict[str, Any]], Awaitable[str]]


class MCPTool(Tool):
    """
    A native TRACERA Tool that delegates execution to a remote MCP tool.

    This is the bridge between the two worlds: it looks like any other
    Tool to the agent / ToolRegistry, but its execute() performs an MCP
    tools/call round-trip over the wire.
    """

    name: str
    description: str

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        call_fn: CallToolFn,
        server_name: str = "mcp",
        remote_name: str | None = None,
    ) -> None:
        # ``name`` is the registry-facing name (may be server-prefixed to
        # avoid collisions); ``remote_name`` is the name sent over the wire.
        self._name = name
        self._remote_name = remote_name or name
        self._desc = description
        self._schema = input_schema
        self._call = call_fn
        self._server = server_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema of the remote tool's input (as advertised by the server)."""
        return self._schema

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            text = await self._call(kwargs)
            return ToolResult.ok(
                tool_name=self._name,
                tool_call_id="",
                output=text,
                server=self._server,
            )
        except Exception as e:
            log.warning("MCP tool %s (%s) failed: %s", self._name, self._server, e)
            return ToolResult.fail(self._name, "", str(e), server=self._server)

    def __repr__(self) -> str:
        return f"<MCPTool {self._server}:{self._name}>"


class MCPClient:
    """
    A single connection to an external MCP server.

    Use as an async context manager:

        async with MCPClient("filesystem", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("read_text_file", {"path": "..."})

    The server subprocess is spawned by the official stdio client and torn
    down on exit.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._session: ClientSession | None = None
        self._stack = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    async def connect(self) -> "MCPClient":
        """Spawn the server subprocess and complete MCP initialize."""
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
            cwd=self._cwd,
        )
        stack = __import__("contextlib").AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        self._stack = stack
        log.info("Connected to MCP server '%s' (%s %s)", self.name, self._command, self._args)
        return self

    async def disconnect(self) -> None:
        """Close the connection and terminate the server subprocess."""
        if self._stack is not None:
            try:
                # aclose() can raise CancelledError (a BaseException) when the
                # transport's cancel scope was already torn down — teardown
                # failures must never propagate to callers.
                await self._stack.aclose()
            except BaseException as e:
                log.debug("MCP disconnect for '%s' finished with: %s", self.name, e)
            finally:
                self._stack = None
                self._session = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    # ── Protocol operations ──────────────────────────────────────────────────

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        MCP tools/list. Returns raw tool definitions:
        [{name, description, inputSchema, ...}, ...]
        """
        if self._session is None:
            raise RuntimeError("MCPClient not connected — call connect() first")
        result = await self._session.list_tools()
        return [t.model_dump() for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        MCP tools/call. Returns the text content of the tool result
        (falls back to structured content serialized as JSON).
        """
        if self._session is None:
            raise RuntimeError("MCPClient not connected — call connect() first")
        result = await self._session.call_tool(name, arguments)

        texts: list[str] = []
        for block in result.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                texts.append(block.text)
        if texts:
            return "\n".join(texts)

        if result.structuredContent:
            return json.dumps(result.structuredContent, indent=2, default=str)
        return str(result)

    # ── Integration with the unified ToolRegistry (Phase 41) ─────────────────

    def to_native_tools(
        self,
        tool_defs: list[dict[str, Any]],
        *,
        prefix: bool = True,
    ) -> list[MCPTool]:
        """
        Adapt remote MCP tool definitions into native TRACERA Tool objects.

        The agent receives a flat list of tools — it cannot tell which ones
        are native Python functions and which go over the wire.

        By default remote tools are registered as ``{server}_{tool}``
        (e.g. ``filesystem_read_file``, ``github_search_code``) so multiple
        servers — or a server whose tool names collide with native tools —
        cannot overwrite each other in the unified registry. Set prefix=False
        to register bare remote names instead.
        """
        native: list[MCPTool] = []
        for t in tool_defs:
            remote_name = t.get("name", "")
            if not remote_name:
                continue
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            desc = t.get("description") or f"MCP tool '{remote_name}' on server '{self.name}'"
            registry_name = f"{self.name}_{remote_name}" if prefix else remote_name

            def _make_call(client: "MCPClient", tool_name: str) -> CallToolFn:
                async def _call(arguments: dict[str, Any] | None = None) -> str:
                    return await client.call_tool(tool_name, arguments or {})
                return _call

            native.append(MCPTool(
                name=registry_name,
                description=desc,
                input_schema=schema,
                call_fn=_make_call(self, remote_name),
                server_name=self.name,
                remote_name=remote_name,
            ))
        return native

    async def register_tools(self, registry: Any) -> int:
        """
        Connect-time convenience: list remote tools, adapt them, and register
        them into a TRACERA ToolRegistry (the unified registry, Phase 41).
        Returns the number of tools registered.
        """
        tool_defs = await self.list_tools()
        native = self.to_native_tools(tool_defs)
        registry.register_many(native)
        log.info("Registered %d MCP tools from '%s'", len(native), self.name)
        return len(native)
