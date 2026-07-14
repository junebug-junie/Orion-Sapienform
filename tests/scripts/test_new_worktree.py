from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "new_worktree.sh"


def test_creates_sibling_worktree_with_correct_branch(primary_repo: Path) -> None:
    proc = subprocess.run(
        ["sh", str(SCRIPT), "feat", "widget"], cwd=primary_repo, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    target = primary_repo.parent / f"{primary_repo.name}-widget"
    assert target.exists()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=target, capture_output=True, text=True
    ).stdout.strip()
    assert branch == "feat/widget"


def test_rejects_unknown_type(primary_repo: Path) -> None:
    proc = subprocess.run(
        ["sh", str(SCRIPT), "bogus", "widget"], cwd=primary_repo, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "type must be one of" in proc.stderr


def test_rejects_unsafe_name(primary_repo: Path) -> None:
    proc = subprocess.run(
        ["sh", str(SCRIPT), "feat", "../escape"], cwd=primary_repo, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "name must contain only" in proc.stderr


def test_rejects_when_target_already_exists(primary_repo: Path) -> None:
    (primary_repo.parent / f"{primary_repo.name}-widget").mkdir()
    proc = subprocess.run(
        ["sh", str(SCRIPT), "feat", "widget"], cwd=primary_repo, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_warns_on_name_collision_with_existing_worktree(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    """repo_with_worktrees already has a worktree at .../merged-wt on
    branch feat/merged -- creating a new worktree also named "merged"
    should warn about the existing one, not silently proceed as if no
    related work existed."""
    primary, merged_wt, _ = repo_with_worktrees
    proc = subprocess.run(
        ["sh", str(SCRIPT), "fix", "merged"], cwd=primary, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "warning" in proc.stderr.lower()
    assert "merged" in proc.stderr


def test_helpful_error_when_branch_already_exists(primary_repo: Path) -> None:
    subprocess.run(
        ["sh", str(SCRIPT), "feat", "widget"], cwd=primary_repo, capture_output=True, text=True
    )
    # remove the worktree but leave the branch, mirroring what
    # prune_merged_worktrees.py does (worktree gone, branch left behind)
    target = primary_repo.parent / f"{primary_repo.name}-widget"
    subprocess.run(["git", "worktree", "remove", str(target)], cwd=primary_repo, check=True)

    proc = subprocess.run(
        ["sh", str(SCRIPT), "feat", "widget"], cwd=primary_repo, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
    assert "git branch -d" in proc.stderr
