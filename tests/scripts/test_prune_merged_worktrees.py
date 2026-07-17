from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_prune(cwd: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prune_merged_worktrees.py"), *extra_args],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


def test_dry_run_lists_merged_not_unmerged(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    proc = _run_prune(primary)
    assert proc.returncode == 0
    assert str(merged_wt) in proc.stdout
    assert str(unmerged_wt) not in proc.stdout
    assert "[would remove]" in proc.stdout
    assert "Re-run with --yes" in proc.stdout


def test_dry_run_does_not_remove_anything(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    _run_prune(primary)
    assert merged_wt.exists()


def test_yes_removes_merged_but_not_unmerged(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    proc = _run_prune(primary, "--yes")
    assert proc.returncode == 0
    assert not merged_wt.exists()
    assert unmerged_wt.exists()  # untouched -- not merged
    assert "[removed]" in proc.stdout
    remaining = subprocess.run(
        ["git", "worktree", "list"], cwd=primary, capture_output=True, text=True
    ).stdout
    assert str(unmerged_wt) in remaining


def test_yes_skips_merged_worktree_with_uncommitted_changes(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    """Regression guard: a merged branch's worktree with real uncommitted
    changes must NOT be force-removed -- git's own `worktree remove`
    refusal (no --force passed) is what protects this, verified here so a
    future edit that adds --force doesn't silently defeat it."""
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    (merged_wt / "uncommitted.txt").write_text("oops", encoding="utf-8")
    proc = _run_prune(primary, "--yes")
    assert merged_wt.exists()
    assert "[skipped, not removed]" in proc.stdout


def test_no_mergeable_worktrees_reports_cleanly(primary_repo: Path) -> None:
    proc = _run_prune(primary_repo)
    assert proc.returncode == 0
    assert "No worktrees found" in proc.stdout


def test_yes_closes_agent_board_presence_for_removed_worktree(
    repo_with_worktrees: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary, merged_wt, _ = repo_with_worktrees
    board_path = tmp_path / "agent-board.jsonl"
    monkeypatch.setenv("ORION_AGENT_BOARD_PATH", str(board_path))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "agent_board.py"),
            "heartbeat",
            "--summary",
            "Merged worktree presence.",
            "--task",
            "Will be pruned.",
        ],
        cwd=merged_wt,
        check=True,
        capture_output=True,
        text=True,
    )

    proc = _run_prune(primary, "--yes")

    assert proc.returncode == 0
    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), "list", "--all"],
        cwd=primary,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)},
    )
    assert str(merged_wt.resolve()) in listed.stdout
    assert "closed" in listed.stdout
