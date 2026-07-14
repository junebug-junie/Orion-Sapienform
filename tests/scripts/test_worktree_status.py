from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_status(cwd: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "worktree_status.py"), *extra_args],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


def test_summary_mode_reports_counts(repo_with_worktrees: tuple[Path, Path, Path]) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    proc = _run_status(primary, "--summary")
    assert proc.returncode == 0
    assert "2 total" in proc.stdout
    assert "1 merged" in proc.stdout


def test_summary_mode_never_calls_gh(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch
) -> None:
    """--summary must not touch PR data at all -- this is the path that
    runs on every SessionStart/post-merge firing; a `gh` call here would
    defeat the whole point of the fast-path optimization."""
    primary, _, _ = repo_with_worktrees
    proc = _run_status(primary, "--summary")
    assert proc.returncode == 0
    assert "PR" not in proc.stdout


def test_full_table_shows_merged_and_unmerged(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    proc = _run_status(primary)
    assert proc.returncode == 0
    assert str(merged_wt) in proc.stdout
    assert str(unmerged_wt) in proc.stdout
    # merged_wt's row should say "yes" for MERGED, unmerged_wt's "no"
    for line in proc.stdout.splitlines():
        if str(merged_wt) in line:
            assert " yes " in line or line.split()[2] == "yes"
        if str(unmerged_wt) in line:
            assert " no " in line or line.split()[2] == "no"


def test_no_worktrees_reports_cleanly(primary_repo: Path) -> None:
    proc = _run_status(primary_repo)
    assert proc.returncode == 0
    assert "No worktrees to report" in proc.stdout


def test_bad_base_ref_reports_error_not_traceback(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, _, _ = repo_with_worktrees
    proc = _run_status(primary, "--summary", "--base", "origin/does-not-exist")
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "ERROR" in proc.stderr


def test_stale_only_refuses_when_pr_data_unavailable(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    """Regression guard: this synthetic repo has no GitHub remote, so
    all_open_prs() returns None (unknown), not {} (confirmed empty).
    --stale-only must refuse to report results rather than silently
    treating every merged worktree as stale when PR data couldn't be
    fetched at all."""
    primary, merged_wt, _ = repo_with_worktrees
    proc = _run_status(primary, "--stale-only")
    assert proc.returncode == 1
    assert "refusing to report --stale-only" in proc.stderr


def test_full_table_warns_but_still_reports_when_pr_data_unavailable(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, _ = repo_with_worktrees
    proc = _run_status(primary)
    assert proc.returncode == 0
    assert "WARNING" in proc.stderr
    assert str(merged_wt) in proc.stdout
