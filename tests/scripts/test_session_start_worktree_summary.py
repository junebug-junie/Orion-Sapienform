from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts" / "hooks" / "session_start_worktree_summary.py"


def _run_hook(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK)], cwd=cwd, capture_output=True, text=True, timeout=30
    )


def test_emits_valid_session_start_json(repo_with_worktrees: tuple[Path, Path, Path]) -> None:
    primary, _, _ = repo_with_worktrees
    proc = _run_hook(primary)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "worktrees:" in payload["hookSpecificOutput"]["additionalContext"]


def test_summary_content_matches_real_counts(
    repo_with_worktrees: tuple[Path, Path, Path]
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    proc = _run_hook(primary)
    payload = json.loads(proc.stdout)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "2 total" in context
    assert "1 merged" in context


def test_no_output_outside_any_git_repo(tmp_path: Path) -> None:
    """Fails silently (no stdout at all) rather than emitting malformed
    JSON or a traceback -- this hook must never block session start."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    proc = _run_hook(outside)
    assert proc.returncode == 0
    assert proc.stdout == ""
