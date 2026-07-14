from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _init_primary_repo(path: Path) -> Path:
    """Bare primary repo: init, configured identity, one commit on main,
    plus a fake origin/main ref (these tests have no real remote, and
    merged_branch_set/prune both default to comparing against origin/main)."""
    path.mkdir()
    _git("init", "-q", "-b", "main", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    (path / "f.txt").write_text("x", encoding="utf-8")
    _git("add", "f.txt", cwd=path)
    _git("commit", "-q", "-m", "init", cwd=path)
    _git("update-ref", "refs/remotes/origin/main", "main", cwd=path)
    return path


@pytest.fixture
def primary_repo(tmp_path: Path) -> Path:
    return _init_primary_repo(tmp_path / "primary")


@pytest.fixture
def repo_with_worktrees(primary_repo: Path) -> tuple[Path, Path, Path]:
    """primary repo, a worktree whose branch IS merged into main, and a
    worktree whose branch is NOT merged."""
    primary = primary_repo
    tmp_path = primary.parent

    merged_wt = tmp_path / "merged-wt"
    _git("worktree", "add", "-q", str(merged_wt), "-b", "feat/merged", cwd=primary)
    (merged_wt / "g.txt").write_text("y", encoding="utf-8")
    _git("add", "g.txt", cwd=merged_wt)
    _git("commit", "-q", "-m", "merged work", cwd=merged_wt)
    _git("merge", "-q", "feat/merged", cwd=primary)
    # Re-point the fake origin/main ref at the new tip -- it was set once at
    # initial-commit time in _init_primary_repo, before this merge existed.
    _git("update-ref", "refs/remotes/origin/main", "main", cwd=primary)

    unmerged_wt = tmp_path / "unmerged-wt"
    _git("worktree", "add", "-q", str(unmerged_wt), "-b", "feat/unmerged", cwd=primary)
    (unmerged_wt / "h.txt").write_text("z", encoding="utf-8")
    _git("add", "h.txt", cwd=unmerged_wt)
    _git("commit", "-q", "-m", "unmerged work", cwd=unmerged_wt)

    return primary, merged_wt, unmerged_wt
