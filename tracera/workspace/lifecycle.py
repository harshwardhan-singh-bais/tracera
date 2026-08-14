"""
Workspace lifecycle management.

Handles creation of TRACERA data directories and cleanup.
"""

from __future__ import annotations

from pathlib import Path

from tracera.logging import get_logger

log = get_logger("workspace.lifecycle")


class WorkspaceLifecycle:
    """
    Manages the lifecycle of TRACERA's data directory (`.tracera/`).

    Creates required subdirectories on initialisation and provides
    helpers for cleanup and inspection.
    """

    REQUIRED_DIRS = [
        "memory",
        "logs",
        "index/lancedb",
        "index/bm25",
        "cache/embeddings",
        "plans",
    ]

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def initialise(self) -> None:
        """Create all required TRACERA data subdirectories."""
        for subdir in self.REQUIRED_DIRS:
            target = self.data_dir / subdir
            target.mkdir(parents=True, exist_ok=True)
        log.debug("Workspace data directories initialised at %s", self.data_dir)

    def is_initialised(self) -> bool:
        """Return True if the TRACERA data directory exists."""
        return self.data_dir.exists()

    def status(self) -> dict[str, bool]:
        """Return a dict of required dirs and whether they exist."""
        return {
            subdir: (self.data_dir / subdir).exists()
            for subdir in self.REQUIRED_DIRS
        }

    def clean(self, *, confirm: bool = False) -> None:
        """
        Remove all TRACERA index/cache data (not memory or logs).
        Requires *confirm=True* as a safety check.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to clean workspace data.")
        import shutil
        for subdir in ["index", "cache"]:
            target = self.data_dir / subdir
            if target.exists():
                shutil.rmtree(target)
                log.info("Removed %s", target)

    def __repr__(self) -> str:
        return f"<WorkspaceLifecycle data_dir={self.data_dir}>"
