"""
TRACERA TUI Widgets.

The TUI is a single auto-scrolling stream (Claude Code style) — the only
widgets are the conversation stream panel and the row types it renders.
"""

from tracera.tui.widgets.agent_panel import (
    AgentPanel,
    AttachmentChip,
    CollapsibleRow,
    InlineStatus,
    LoaderPill,
    MessageWidget,
    PhaseRow,
    ToolRow,
    ThinkingDisclosure,
    format_args,
)

__all__ = [
    "AgentPanel",
    "AttachmentChip",
    "CollapsibleRow",
    "InlineStatus",
    "LoaderPill",
    "MessageWidget",
    "PhaseRow",
    "ToolRow",
    "ThinkingDisclosure",
    "format_args",
]
