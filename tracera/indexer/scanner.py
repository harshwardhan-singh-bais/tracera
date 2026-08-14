"""
TRACERA Repository Scanner (Phase 11).

Recursively discovers source files, ignores .git and binaries,
respects .gitignore, and collects file metadata.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import pathspec

from tracera.indexer.schema import FileMetadata
from tracera.logging import get_logger

log = get_logger("indexer.scanner")


class RepositoryScanner:
    """Scans a workspace for source code files."""

    def __init__(
        self,
        workspace_root: Path,
        max_file_size: int = 2 * 1024 * 1024,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.max_file_size = max_file_size
        self._gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec:
        """Load .gitignore if present, plus default ignores."""
        patterns = [
            ".git/",
            ".tracera/",
            ".venv/",
            "venv/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.exe",
            "*.bin",
            ".DS_Store",
            "*.jpg", "*.jpeg", "*.png", "*.gif", "*.ico",
            "*.pdf", "*.mp3", "*.mp4", "*.zip", "*.tar.gz",
        ]

        gitignore_path = self.workspace_root / ".gitignore"
        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding="utf-8")
                patterns.extend(content.splitlines())
            except Exception as e:
                log.warning("Failed to read .gitignore: %s", e)

        return pathspec.PathSpec.from_lines(
            "gitignore", patterns
        )

    def _is_binary(self, filepath: Path) -> bool:
        """Heuristic check for binary files by scanning first 1024 bytes for nulls."""
        try:
            with open(filepath, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
        except Exception:
            return True  # If we can't read it, treat as binary/skip
        return False

    def _detect_language(self, filepath: Path) -> str | None:
        """Detect language based on extension."""
        ext = filepath.suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".java": "java",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".sh": "bash",
        }
        return mapping.get(ext)

    def scan(self) -> Iterator[FileMetadata]:
        """
        Recursively scan the workspace.
        Yields FileMetadata for each valid source file.
        """
        for root, dirs, files in os.walk(self.workspace_root):
            root_path = Path(root)
            
            # Filter directories based on gitignore
            rel_root = root_path.relative_to(self.workspace_root)
            
            # Remove ignored directories in-place to prevent os.walk from descending
            dirs[:] = [
                d for d in dirs
                if not self._gitignore_spec.match_file(
                    (rel_root / d).as_posix() + "/"
                )
            ]

            for file in files:
                file_path = root_path / file
                rel_path = file_path.relative_to(self.workspace_root).as_posix()

                # 1. Check ignore patterns
                if self._gitignore_spec.match_file(rel_path):
                    continue

                # 2. Check file size
                try:
                    size = file_path.stat().st_size
                except Exception:
                    continue

                if size > self.max_file_size:
                    log.debug("Skipping %s (too large: %d bytes)", rel_path, size)
                    continue
                if size == 0:
                    continue

                # 3. Check if binary
                if self._is_binary(file_path):
                    continue

                # 4. Compute SHA256
                try:
                    with open(file_path, "rb") as f:
                        sha256 = hashlib.sha256(f.read()).hexdigest()
                except Exception as e:
                    log.warning("Failed to read %s: %s", rel_path, e)
                    continue

                yield FileMetadata(
                    path=rel_path,
                    language=self._detect_language(file_path),
                    size_bytes=size,
                    sha256=sha256,
                )
