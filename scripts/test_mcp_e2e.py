"""End-to-end smoke test: start TRACERA MCP server over stdio and call a tool."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tracera.main", "mcp", "serve"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=60)
            tools = await asyncio.wait_for(session.list_tools(), timeout=30)
            print(f"TOOLS: {len(tools.tools)}")
            names = sorted(t.name for t in tools.tools)
            print("SAMPLE:", ", ".join(names[:6]))

            result = await asyncio.wait_for(
                session.call_tool("get_repo_map", {"max_files": 5}),
                timeout=120,
            )
            text = result.content[0].text if result.content else ""
            print("REPO_MAP_OK:", len(text), "chars")
            print(text[:300])
            print("E2E: PASS")


if __name__ == "__main__":
    asyncio.run(main())
