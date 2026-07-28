"""Unit tests for scripts/hooks/shared_checkout_edit_guard.py.

Uses real, throwaway git repos under tmp_path (not mocks) so
_is_shared_checkout's real `git rev-parse --git-dir --git-common-dir`
subprocess call is exercised end-to-end, mirroring destructive_git_guard.py's
own reliance on real git plumbing rather than a stubbed check.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "hooks" / "shared_checkout_edit_guard.py"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_primary_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "primary"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@test", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "tracked.py", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    return repo


def _make_worktree(primary: Path, tmp_path: Path) -> Path:
    wt = tmp_path / "worktree"
    _git("worktree", "add", "-b", "feat/x", str(wt), cwd=primary)
    return wt


def _run(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(GUARD)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_blocks_tracked_file_edit_in_primary_checkout(tmp_path):
    primary = _make_primary_repo(tmp_path)
    proc = _run({"tool_input": {"file_path": str(primary / "tracked.py")}})

    assert proc.returncode == 0
    decision = json.loads(proc.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert str(primary / "tracked.py") in decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_allows_edit_in_linked_worktree(tmp_path):
    primary = _make_primary_repo(tmp_path)
    wt = _make_worktree(primary, tmp_path)
    proc = _run({"tool_input": {"file_path": str(wt / "tracked.py")}})

    assert proc.returncode == 0
    assert proc.stdout == ""


def test_allows_gitignored_file_in_primary_checkout(tmp_path):
    primary = _make_primary_repo(tmp_path)
    (primary / ".gitignore").write_text("local.json\n", encoding="utf-8")
    _git("add", ".gitignore", cwd=primary)
    _git("commit", "-q", "-m", "gitignore", cwd=primary)
    (primary / "local.json").write_text("{}", encoding="utf-8")

    proc = _run({"tool_input": {"file_path": str(primary / "local.json")}})

    assert proc.returncode == 0
    assert proc.stdout == ""


def test_escape_hatch_allows_primary_checkout_edit(tmp_path, monkeypatch):
    primary = _make_primary_repo(tmp_path)
    monkeypatch.setenv("ORION_ALLOW_SHARED_CHECKOUT_WRITE", "1")
    proc = subprocess.run(
        ["python3", str(GUARD)],
        input=json.dumps({"tool_input": {"file_path": str(primary / "tracked.py")}}),
        capture_output=True,
        text=True,
        timeout=10,
        env={**__import__("os").environ, "ORION_ALLOW_SHARED_CHECKOUT_WRITE": "1"},
    )

    assert proc.returncode == 0
    assert proc.stdout == ""


def test_fails_open_on_malformed_payload():
    proc = subprocess.run(
        ["python3", str(GUARD)], input="not json", capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_fails_open_on_missing_file_path():
    proc = _run({"tool_input": {}})
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_fails_open_outside_any_git_repo(tmp_path):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    proc = _run({"tool_input": {"file_path": str(outside / "scratch.txt")}})
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_write_tool_input_shape_also_handled(tmp_path):
    """Write's tool_input has a different second key (content vs old_string/
    new_string) than Edit's, but both share file_path -- confirm the guard
    only reads file_path and ignores the rest."""
    primary = _make_primary_repo(tmp_path)
    proc = _run(
        {"tool_input": {"file_path": str(primary / "tracked.py"), "content": "new content"}}
    )
    assert proc.returncode == 0
    decision = json.loads(proc.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
