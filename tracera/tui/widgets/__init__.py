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
from tracera.tui.widgets.dashboard import DashboardWidget
from tracera.tui.widgets.file_context import FileContextPanel
from tracera.tui.widgets.memory_viz import MemoryGraphWidget

__all__ = [
    "AgentPanel",
    "AttachmentChip",
    "CollapsibleRow",
    "DashboardWidget",
    "FileContextPanel",
    "InlineStatus",
    "LoaderPill",
    "MemoryGraphWidget",
    "MessageWidget",
    "PhaseRow",
    "ToolRow",
    "ThinkingDisclosure",
    "format_args",
]
