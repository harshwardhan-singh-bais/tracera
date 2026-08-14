"""
TRACERA Git Integration — Phase 3.

Safe, read-mostly git operations using GitPython.
All mutating operations require explicit opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.errors import GitError, NotAGitRepositoryError
from tracera.logging import get_logger

log = get_logger("git")

try:
    import git as gitpython
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False
    Repo = None  # type: ignore[assignment,misc]


@dataclass
class GitStatus:
    """Represents the working-tree status of a repository."""

    branch: str
    is_dirty: bool
    staged: list[str] = field(default_factory=list)
    unstaged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"branch={self.branch}"]
        if self.staged:
            parts.append(f"staged={len(self.staged)}")
        if self.unstaged:
            parts.append(f"unstaged={len(self.unstaged)}")
        if self.untracked:
            parts.append(f"untracked={len(self.untracked)}")
        return " ".join(parts)


@dataclass
class GitCommit:
    """Lightweight commit representation."""

    sha: str
    short_sha: str
    author: str
    date: str
    message: str

    def __str__(self) -> str:
        return f"{self.short_sha} {self.author}: {self.message[:60]}"


@dataclass
class GitDiff:
    """Diff output for a file or the whole repo."""

    diff_text: str
    files_changed: int
    insertions: int
    deletions: int


class GitRepo:
    """
    Safe wrapper around a Git repository.

    Read operations (status, diff, log, branch) are always available.
    Write operations (commit, checkout, etc.) require ``allow_mutations=True``.
    """

    def __init__(self, path: Path, *, allow_mutations: bool = False) -> None:
        if not _GIT_AVAILABLE:
            raise GitError(
                "gitpython is not installed.",
                detail="Run: pip install gitpython",
            )
        self.path = path
        self.allow_mutations = allow_mutations
        self._repo: Any = None  # lazy-loaded

    def _get_repo(self) -> Any:
        if self._repo is None:
            try:
                self._repo = Repo(str(self.path), search_parent_directories=True)
            except InvalidGitRepositoryError:
                raise NotAGitRepositoryError(
                    f"Not a git repository: {self.path}"
                )
        return self._repo

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def repo_root(self) -> Path:
        return Path(self._get_repo().working_tree_dir)

    @property
    def current_branch(self) -> str:
        try:
            return self._get_repo().active_branch.name
        except TypeError:
            # Detached HEAD
            return f"HEAD@{self._get_repo().head.commit.hexsha[:7]}"

    def status(self) -> GitStatus:
        """Return the current working-tree status."""
        repo = self._get_repo()
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        unstaged = [item.a_path for item in repo.index.diff(None)]
        untracked = list(repo.untracked_files)
        return GitStatus(
            branch=self.current_branch,
            is_dirty=repo.is_dirty(untracked_files=True),
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
        )

    def diff(
        self,
        ref_a: str = "HEAD",
        ref_b: str | None = None,
        *,
        path: str | None = None,
    ) -> GitDiff:
        """
        Return the diff between two refs (or working tree vs HEAD).

        Args:
            ref_a: Base commit/branch (default: HEAD).
            ref_b: Target commit/branch (default: working tree).
            path: Limit diff to a specific file.
        """
        repo = self._get_repo()
        try:
            if ref_b:
                raw = repo.git.diff(ref_a, ref_b, "--", path or ".")
                stat = repo.git.diff(ref_a, ref_b, "--stat", path or ".")
            else:
                raw = repo.git.diff(ref_a, "--", path or ".")
                stat = repo.git.diff(ref_a, "--stat", path or ".")
        except GitCommandError as e:
            raise GitError(f"git diff failed: {e}")

        # Parse stat line: "X files changed, Y insertions(+), Z deletions(-)"
        files_changed = insertions = deletions = 0
        for line in stat.splitlines():
            line = line.strip()
            if "file" in line and "changed" in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "file" in part:
                        try:
                            files_changed = int(part.split()[0])
                        except ValueError:
                            pass
                    elif "insertion" in part:
                        try:
                            insertions = int(part.split()[0])
                        except ValueError:
                            pass
                    elif "deletion" in part:
                        try:
                            deletions = int(part.split()[0])
                        except ValueError:
                            pass

        return GitDiff(
            diff_text=raw,
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
        )

    def log(self, *, max_count: int = 20, branch: str | None = None) -> list[GitCommit]:
        """Return recent commit history."""
        repo = self._get_repo()
        rev = branch or "HEAD"
        try:
            commits = list(repo.iter_commits(rev, max_count=max_count))
        except GitCommandError as e:
            raise GitError(f"git log failed: {e}")

        return [
            GitCommit(
                sha=c.hexsha,
                short_sha=c.hexsha[:7],
                author=c.author.name or "unknown",
                date=c.authored_datetime.strftime("%Y-%m-%d %H:%M"),
                message=c.message.strip().splitlines()[0] if c.message else "",
            )
            for c in commits
        ]

    def branches(self) -> list[str]:
        """Return all local branch names."""
        repo = self._get_repo()
        return [b.name for b in repo.branches]

    def remote_branches(self) -> list[str]:
        """Return all remote tracking branch names."""
        repo = self._get_repo()
        result = []
        for remote in repo.remotes:
            result.extend(ref.name for ref in remote.refs)
        return result

    def stashes(self) -> list[str]:
        """Return stash list."""
        repo = self._get_repo()
        try:
            raw = repo.git.stash("list")
            return raw.splitlines() if raw else []
        except GitCommandError:
            return []

    def file_history(self, file_path: str, *, max_count: int = 10) -> list[GitCommit]:
        """Return commit history for a specific file."""
        repo = self._get_repo()
        try:
            commits = list(repo.iter_commits("HEAD", paths=file_path, max_count=max_count))
        except GitCommandError as e:
            raise GitError(f"git log --follow failed: {e}")

        return [
            GitCommit(
                sha=c.hexsha,
                short_sha=c.hexsha[:7],
                author=c.author.name or "unknown",
                date=c.authored_datetime.strftime("%Y-%m-%d %H:%M"),
                message=c.message.strip().splitlines()[0] if c.message else "",
            )
            for c in commits
        ]

    # ── Safe write operations (require allow_mutations=True) ──────────────────

    def _require_mutations(self) -> None:
        if not self.allow_mutations:
            raise GitError(
                "Mutation operations are disabled on this GitRepo instance.",
                detail="Instantiate with allow_mutations=True to enable.",
            )

    def stage(self, paths: list[str]) -> None:
        """Stage files for commit."""
        self._require_mutations()
        repo = self._get_repo()
        repo.index.add(paths)
        log.info("Staged %d file(s)", len(paths))

    def commit(self, message: str) -> GitCommit:
        """Create a commit with the staged changes."""
        self._require_mutations()
        repo = self._get_repo()
        try:
            c = repo.index.commit(message)
        except Exception as e:
            raise GitError(f"git commit failed: {e}")
        return GitCommit(
            sha=c.hexsha,
            short_sha=c.hexsha[:7],
            author=c.author.name or "unknown",
            date=c.authored_datetime.strftime("%Y-%m-%d %H:%M"),
            message=c.message.strip(),
        )

    def __repr__(self) -> str:
        try:
            branch = self.current_branch
        except Exception:
            branch = "unknown"
        return f"<GitRepo {self.path} [{branch}]>"


def detect_git_repo(path: Path) -> GitRepo | None:
    """
    Detect if *path* is inside a git repository.
    Returns a GitRepo instance or None.
    """
    if not _GIT_AVAILABLE:
        log.warning("gitpython not available — git integration disabled.")
        return None
    try:
        repo = GitRepo(path)
        repo._get_repo()  # trigger detection
        return repo
    except (NotAGitRepositoryError, GitError):
        return None
