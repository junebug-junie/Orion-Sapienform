from __future__ import annotations

import subprocess

import pytest

from orion.fcc.sandbox_sync import sync_fcc_sandbox


def _git(cwd, *args):
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result


def _init_repo_with_main(path):
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "file.txt").write_text("v1")
    _git(path, "add", "file.txt")
    _git(path, "commit", "-m", "initial")
    return path


@pytest.fixture
def origin_and_clone(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo_with_main(origin)

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    return origin, clone


def test_sync_no_workspace_is_skipped():
    assert sync_fcc_sandbox(None) == "skipped_no_workspace"
    assert sync_fcc_sandbox("") == "skipped_no_workspace"


def test_sync_non_git_dir_is_skipped(tmp_path):
    assert sync_fcc_sandbox(str(tmp_path)) == "skipped_not_a_git_repo"


def test_sync_pulls_new_origin_main_commits(origin_and_clone):
    origin, clone = origin_and_clone

    # New commit lands on origin/main after the clone was made.
    (origin / "file.txt").write_text("v2")
    _git(origin, "add", "file.txt")
    _git(origin, "commit", "-m", "second")

    result = sync_fcc_sandbox(str(clone))
    assert result == "synced"
    assert (clone / "file.txt").read_text() == "v2"


def test_sync_skips_dirty_worktree_on_a_branch(origin_and_clone):
    _, clone = origin_and_clone
    _git(clone, "checkout", "-b", "feature/in-flight")
    (clone / "file.txt").write_text("uncommitted-work")

    result = sync_fcc_sandbox(str(clone))
    assert result == "skipped_dirty_worktree"
    # Uncommitted work on the in-flight branch must survive the sync attempt.
    assert (clone / "file.txt").read_text() == "uncommitted-work"


def test_sync_skips_unpushed_branch(origin_and_clone):
    _, clone = origin_and_clone
    _git(clone, "checkout", "-b", "feature/unpushed")
    (clone / "file.txt").write_text("v2")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "unpushed work")

    result = sync_fcc_sandbox(str(clone))
    assert result == "skipped_unpushed_branch"
    assert _git(clone, "branch", "--show-current").stdout.strip() == "feature/unpushed"


def test_sync_skips_dirty_worktree_on_main(origin_and_clone):
    """Regression: the sandbox's steady state after a sync is `main`, so dirty/
    uncommitted work sitting directly on main (no branch checked out) must be
    protected too, not just work on a named branch."""
    _, clone = origin_and_clone
    (clone / "file.txt").write_text("uncommitted-on-main")

    result = sync_fcc_sandbox(str(clone))
    assert result == "skipped_dirty_worktree"
    assert (clone / "file.txt").read_text() == "uncommitted-on-main"


def test_sync_resets_a_pushed_branch_back_to_main(origin_and_clone):
    origin, clone = origin_and_clone
    _git(clone, "checkout", "-b", "feature/pushed")
    (clone / "file.txt").write_text("v2")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "pushed work")
    _git(clone, "push", "origin", "feature/pushed")

    result = sync_fcc_sandbox(str(clone))
    assert result == "synced"
    assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
