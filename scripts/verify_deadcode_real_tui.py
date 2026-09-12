"""Direct reproduction of the user's crash: `uv run tracera` → `/deadcode`.

Runs the REAL app in Textual's test runtime (real Worker machinery, real
compositor, real app lifecycle) and dispatches the commands that used to
crash with ``TypeError: object Worker can't be used in 'await' expression``.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from tracera.tui.app import TraceraTUI
from test_slash_runtime_dispatch import _make_app  # reuse the test harness


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tracera-deadcode-check-"))
    app: TraceraTUI = _make_app(tmp)
    async with app.run_test() as pilot:
        await pilot.pause()
        for cmd in ("/deadcode", "/hotspots", "/pagerank", "/coupling", "/cycles"):
            await app._handle_command(cmd)
            await pilot.pause()
        print("OK: /deadcode and 4 sibling commands dispatched in the real TUI runtime without TypeError")


if __name__ == "__main__":
    asyncio.run(main())
