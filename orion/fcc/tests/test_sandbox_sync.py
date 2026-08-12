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


def test_sync_configures_push_auth_from_github_pat(origin_and_clone, tmp_path, monkeypatch):
    """Regression: GITHUB_PAT only ever reached the GitHub MCP server's own
    subprocess env, never the outer claude shell or git's credential store --
    `git push`/`gh` had no credentials at all. sync_fcc_sandbox must rewrite the
    SSH-style remote to a token-authenticated HTTPS one, repo-local only."""
    _, clone = origin_and_clone
    fcc_env_path = tmp_path / "fcc.env"
    fcc_env_path.write_text("GITHUB_PAT=ghp_test_token_123\nOTHER_KEY=ignored\n")
    monkeypatch.setenv("HARNESS_FCC_ENV_PATH", str(fcc_env_path))

    sync_fcc_sandbox(str(clone))

    rewrite = _git(
        clone,
        "config",
        "--local",
        "--get",
        "url.https://x-access-token:ghp_test_token_123@github.com/.insteadOf",
    )
    assert rewrite.stdout.strip() == "git@github.com:"

    # Must be repo-local, never global -- a global write would leak the token
    # into the image-baked gitconfig shared by every sandbox session.
    global_check = subprocess.run(
        ["git", "config", "--global", "--get-regexp", "url\\..*insteadOf"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "x-access-token:ghp_test_token_123" not in global_check.stdout


def test_sync_skips_push_auth_when_no_token(origin_and_clone, tmp_path, monkeypatch):
    _, clone = origin_and_clone
    monkeypatch.setenv("HARNESS_FCC_ENV_PATH", str(tmp_path / "does_not_exist.env"))

    result = sync_fcc_sandbox(str(clone))
    assert result == "synced"

    rewrite = subprocess.run(
        ["git", "config", "--local", "--get-regexp", "url\\..*insteadOf"],
        cwd=str(clone),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rewrite.returncode != 0 or not rewrite.stdout.strip()
