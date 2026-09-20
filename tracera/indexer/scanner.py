"""
TRACERA Repository Scanner (Phase 11).

Recursively discovers source files, ignores .git and binaries,
respects .gitignore, and collects file metadata.

``.gitignore`` is honoured the way git honours it: every directory along the
path can contribute one, and the **deepest file that has an opinion wins**.
Only reading the repository-root ``.gitignore`` was not enough — TRACERA's own
``docs-site/.gitignore`` contains ``/.next/``, and because that pattern is
anchored it excludes nothing from the root, so Next.js build output ended up
in the index and dominated search results.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

from tracera.indexer.schema import FileMetadata
from tracera.logging import get_logger

log = get_logger("indexer.scanner")

#: Sentinel anchor for the repository-root spec: no directory prefix to strip.
_ROOT_ANCHOR = Path("")


class RepositoryScanner:
    """Scans a workspace for source code files."""

    #: Always skipped, regardless of the repo's own .gitignore. Split into
    #: build output (generated, never worth indexing) and binaries/media.
    _DEFAULT_PATTERNS: tuple[str, ...] = (
        # ── build / tool output ───────────────────────────────────────────
        ".git/",
        ".tracera/",
        ".gitignore",
        ".venv/",
        "venv/",
        "node_modules/",
        "__pycache__/",
        ".next/",
        ".turbo/",
        ".nuxt/",
        ".svelte-kit/",
        ".parcel-cache/",
        ".cache/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".gradle/",
        "target/",
        "dist/",
        "build/",
        "out/",
        "coverage/",
        ".coverage/",
        # ── binaries & media ──────────────────────────────────────────────
        "*.pyc",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",
        "*.bin",
        ".DS_Store",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.ico",
        "*.pdf",
        "*.mp3",
        "*.mp4",
        "*.zip",
        "*.tar.gz",
    )

    def __init__(
        self,
        workspace_root: Path,
        max_file_size: int = 2 * 1024 * 1024,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.max_file_size = max_file_size
        self._gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec:
        """Load the *root* .gitignore plus default ignores."""
        patterns = list(self._DEFAULT_PATTERNS)

        gitignore_path = self.workspace_root / ".gitignore"
        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding="utf-8")
                patterns.extend(content.splitlines())
            except Exception as e:
                log.warning("Failed to read .gitignore: %s", e)

        return pathspec.PathSpec.from_lines("gitignore", patterns)

    def _load_dir_gitignore(self, directory: Path) -> pathspec.PathSpec | None:
        """
        Load a nested ``.gitignore``, or None if the directory has none.

        Patterns are kept **relative to ``directory``**, which is how git
        interprets them — that is what makes an anchored pattern like
        ``/.next/`` in ``docs-site/.gitignore`` mean ``docs-site/.next/``
        rather than the repo root.
        """
        gitignore_path = directory / ".gitignore"
        if not gitignore_path.exists():
            return None
        try:
            lines = gitignore_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            log.warning("Failed to read %s: %s", gitignore_path, e)
            return None
        if not lines:
            return None
        try:
            return pathspec.PathSpec.from_lines("gitignore", lines)
        except Exception as e:
            log.warning("Failed to parse %s: %s", gitignore_path, e)
            return None

    @staticmethod
    def _spec_opinion(spec: pathspec.PathSpec, path: str) -> bool | None:
        """
        What ``spec`` says about ``path``: True = ignore, False = un-ignore,
        None = no opinion.

        ``PathSpec.match_file`` collapses "negated" and "no match" into False,
        which is fine within one file (pathspec already applies last-match-wins
        there) but wrong when layering nested files: a deeper ``!foo`` must be
        able to un-ignore something an ancestor ignored. So walk the patterns
        and take the last one that matches, honouring its ``include`` flag.
        """
        opinion: bool | None = None
        for pattern in spec.patterns:
            try:
                matched = pattern.match_file(path)
            except Exception:
                continue
            if matched:
                opinion = bool(getattr(pattern, "include", True))
        return opinion

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
            # TSX uses its own tree-sitter grammar (JSX). Treating it as
            # "typescript" made every component file parse with errors.
            ".tsx": "tsx",
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
        # (directory, spec) pairs, ordered outermost → innermost. A nested
        # .gitignore is scoped to its own directory, so each entry also records
        # where it came from.
        specs: list[tuple[Path, pathspec.PathSpec]] = [
            (Path(""), self._gitignore_spec)
        ]
        yield from self._scan_dir(self.workspace_root, Path(""), specs)

    def _scan_dir(
        self,
        directory: Path,
        rel_dir: Path,
        specs: list[tuple[Path, pathspec.PathSpec]],
    ) -> Iterator[FileMetadata]:
        """Yield indexable files under ``directory``, descending recursively."""
        nested = self._load_dir_gitignore(directory)
        if nested is not None:
            # Copy rather than mutate: siblings must not see each other's spec.
            specs = [*specs, (rel_dir, nested)]

        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as e:
            log.debug("Cannot list %s: %s", directory, e)
            return

        subdirs: list[str] = []
        files: list[str] = []
        for entry in entries:
            try:
                if entry.is_dir():
                    subdirs.append(entry.name)
                else:
                    files.append(entry.name)
            except OSError:
                continue

        for name in subdirs:
            child_rel = rel_dir / name
            if self._is_ignored(specs, child_rel, is_dir=True):
                continue
            yield from self._scan_dir(directory / name, child_rel, specs)

        for name in files:
            child_rel = rel_dir / name
            if self._is_ignored(specs, child_rel, is_dir=False):
                continue

            file_path = directory / name
            rel_path = child_rel.as_posix()

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

    def _is_ignored(
        self,
        specs: list[tuple[Path, pathspec.PathSpec]],
        rel_path: Path,
        *,
        is_dir: bool,
    ) -> bool:
        """
        Whether ``rel_path`` is excluded, layering every .gitignore above it.

        Git's rule is that the deepest .gitignore with an opinion wins, so we
        walk outermost → innermost and keep overwriting. Appending "/" for
        directories is what lets a dir-only pattern like ``build/`` match.
        """
        ignored = False
        for anchor, spec in specs:
            if anchor == _ROOT_ANCHOR:
                scoped = rel_path
            else:
                try:
                    scoped = rel_path.relative_to(anchor)
                except ValueError:
                    # This spec does not govern the path.
                    continue
            candidate = scoped.as_posix() + ("/" if is_dir else "")
            opinion = self._spec_opinion(spec, candidate)
            if opinion is not None:
                ignored = opinion
        return ignored
