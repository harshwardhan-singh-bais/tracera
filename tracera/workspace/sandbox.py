"""
TRACERA Workspace Sandbox — Phase 2.

Provides a sandboxed view of the filesystem rooted at the workspace directory.
All file operations are validated to prevent path traversal attacks.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import AsyncIterator, Iterator

import aiofiles
import aiofiles.os

from tracera.errors import (
    FileSizeLimitError,
    FileNotFoundInWorkspaceError,
    PathTraversalError,
    WorkspaceError,
)
from tracera.logging import get_logger

log = get_logger("workspace")


class FileEntry:
    """Metadata about a single file or directory entry."""

    __slots__ = ("path", "relative", "is_dir", "size", "extension")

    def __init__(
        self,
        path: Path,
        root: Path,
        *,
        is_dir: bool,
        size: int = 0,
    ) -> None:
        self.path = path
        self.relative = path.relative_to(root)
        self.is_dir = is_dir
        self.size = size
        self.extension = path.suffix.lower() if not is_dir else ""

    def __repr__(self) -> str:
        kind = "dir" if self.is_dir else f"file({self.size:,}B)"
        return f"<FileEntry {self.relative} [{kind}]>"


class WorkspaceSandbox:
    """
    A sandboxed filesystem view rooted at a given directory.

    All read/write/delete operations are validated against the root to
    prevent path traversal.  Uses async I/O throughout.
    """

    MAX_FILE_SIZE_DEFAULT = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        root: Path,
        *,
        max_file_size: int = MAX_FILE_SIZE_DEFAULT,
    ) -> None:
        self.root = root.resolve()
        self.max_file_size = max_file_size
        if not self.root.exists():
            raise WorkspaceError(f"Workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {self.root}")
        log.info("Workspace initialised at %s", self.root)

    # ── Path resolution / validation ──────────────────────────────────────────

    def resolve(self, path: str | Path) -> Path:
        """
        Resolve *path* relative to workspace root.

        Raises PathTraversalError if the resolved path escapes the workspace.
        """
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root / p).resolve()

        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PathTraversalError(str(path))

        return resolved

    def relative(self, path: Path) -> Path:
        """Return *path* as a path relative to the workspace root."""
        return path.relative_to(self.root)

    # ── Sync helpers (for use in CLI / non-async contexts) ────────────────────

    def read_text_sync(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundInWorkspaceError(str(path))
        size = resolved.stat().st_size
        if size > self.max_file_size:
            raise FileSizeLimitError(str(path), size, self.max_file_size)
        return resolved.read_text(encoding=encoding, errors="replace")

    def write_text_sync(
        self, path: str | Path, content: str, *, encoding: str = "utf-8"
    ) -> Path:
        resolved = self.resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
        log.debug("Written %d chars to %s", len(content), resolved)
        return resolved

    def delete_sync(self, path: str | Path, *, recursive: bool = False) -> None:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundInWorkspaceError(str(path))
        if resolved.is_dir():
            if not recursive:
                raise WorkspaceError(
                    f"'{path}' is a directory — pass recursive=True to delete."
                )
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        log.debug("Deleted %s", resolved)

    def exists(self, path: str | Path) -> bool:
        try:
            return self.resolve(path).exists()
        except PathTraversalError:
            return False

    # ── Async file I/O ────────────────────────────────────────────────────────

    async def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundInWorkspaceError(str(path))
        stat = await aiofiles.os.stat(resolved)
        if stat.st_size > self.max_file_size:
            raise FileSizeLimitError(str(path), stat.st_size, self.max_file_size)
        async with aiofiles.open(resolved, encoding=encoding, errors="replace") as f:
            return await f.read()

    async def write_text(
        self, path: str | Path, content: str, *, encoding: str = "utf-8"
    ) -> Path:
        resolved = self.resolve(path)
        await aiofiles.os.makedirs(str(resolved.parent), exist_ok=True)
        async with aiofiles.open(resolved, "w", encoding=encoding) as f:
            await f.write(content)
        log.debug("Written %d chars to %s", len(content), resolved)
        return resolved

    async def read_bytes(self, path: str | Path) -> bytes:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundInWorkspaceError(str(path))
        stat = await aiofiles.os.stat(resolved)
        if stat.st_size > self.max_file_size:
            raise FileSizeLimitError(str(path), stat.st_size, self.max_file_size)
        async with aiofiles.open(resolved, "rb") as f:
            return await f.read()

    async def delete(self, path: str | Path, *, recursive: bool = False) -> None:
        resolved = self.resolve(path)
        if not await aiofiles.os.path.exists(str(resolved)):
            raise FileNotFoundInWorkspaceError(str(path))
        if resolved.is_dir():
            if not recursive:
                raise WorkspaceError(
                    f"'{path}' is a directory — pass recursive=True to delete."
                )
            await asyncio.to_thread(shutil.rmtree, resolved)
        else:
            await aiofiles.os.remove(str(resolved))
        log.debug("Deleted %s", resolved)

    async def list_directory(
        self,
        path: str | Path = ".",
        *,
        include_hidden: bool = False,
        max_depth: int = 1,
    ) -> list[FileEntry]:
        """
        List directory contents up to *max_depth* levels deep.
        """
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundInWorkspaceError(str(path))
        if not resolved.is_dir():
            raise WorkspaceError(f"Not a directory: {path}")

        entries: list[FileEntry] = []
        await self._walk_dir(resolved, entries, depth=0, max_depth=max_depth,
                             include_hidden=include_hidden)
        return entries

    async def _walk_dir(
        self,
        directory: Path,
        entries: list[FileEntry],
        *,
        depth: int,
        max_depth: int,
        include_hidden: bool,
    ) -> None:
        try:
            items = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return

        for item in items:
            if not include_hidden and item.name.startswith("."):
                continue
            is_dir = item.is_dir()
            size = 0 if is_dir else item.stat().st_size
            entries.append(FileEntry(item, self.root, is_dir=is_dir, size=size))
            if is_dir and depth < max_depth - 1:
                await self._walk_dir(
                    item, entries,
                    depth=depth + 1,
                    max_depth=max_depth,
                    include_hidden=include_hidden,
                )

    # ── Edit file (patch lines) ───────────────────────────────────────────────

    async def edit_text(
        self,
        path: str | Path,
        old_text: str,
        new_text: str,
        *,
        count: int = 1,
    ) -> int:
        """
        Replace the first *count* occurrences of *old_text* with *new_text*.
        Returns the number of replacements made.
        """
        content = await self.read_text(path)
        if old_text not in content:
            raise WorkspaceError(
                f"Text not found in '{path}': {old_text[:80]!r}"
            )
        if count == 0:
            new_content = content.replace(old_text, new_text)
            n = content.count(old_text)
        else:
            new_content = content.replace(old_text, new_text, count)
            n = min(count, content.count(old_text))
        await self.write_text(path, new_content)
        return n

    # ── Grep / search ─────────────────────────────────────────────────────────

    async def grep(
        self,
        pattern: str,
        path: str | Path = ".",
        *,
        case_insensitive: bool = False,
        include_extensions: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict]:
        """
        Search for *pattern* in files under *path*.
        Returns list of {file, line, content} dicts.
        """
        import re

        resolved = self.resolve(path)
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            raise WorkspaceError(f"Invalid regex pattern: {e}")

        results: list[dict] = []

        async def _search_file(file_path: Path) -> None:
            if len(results) >= max_results:
                return
            try:
                content = await self.read_text(file_path)
            except (FileSizeLimitError, UnicodeDecodeError):
                return
            for i, line in enumerate(content.splitlines(), start=1):
                if rx.search(line):
                    results.append({
                        "file": str(self.relative(file_path)),
                        "line": i,
                        "content": line.rstrip(),
                    })
                    if len(results) >= max_results:
                        break

        if resolved.is_file():
            await _search_file(resolved)
        else:
            for root, dirs, files in os.walk(resolved):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    file_path = Path(root) / fname
                    if include_extensions:
                        if file_path.suffix.lower() not in include_extensions:
                            continue
                    try:
                        file_path.relative_to(self.root)
                    except ValueError:
                        continue
                    await _search_file(file_path)
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

        return results

    def __repr__(self) -> str:
        return f"<WorkspaceSandbox root={self.root}>"
