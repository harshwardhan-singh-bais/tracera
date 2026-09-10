"""
TRACERA Observability — Phase 60.

Lightweight, in-process telemetry that records every LLM call, tool call,
retrieval call, agent iteration, and error as it happens, then exposes a
snapshot for the CLI / TUI without pulling in any external APM dependency.

The observer is a process-wide singleton: every :class:`ReActAgent` in the
process writes to it, so a TUI session accumulates a real, complete picture.
"""

from tracera.observability.telemetry import (
    Telemetry,
    get_telemetry,
    reset_telemetry,
)

__all__ = [
    "Telemetry",
    "get_telemetry",
    "reset_telemetry",
]