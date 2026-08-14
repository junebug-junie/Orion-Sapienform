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


def test_sync_rescues_dirty_worktree_then_syncs(origin_and_clone):
    """Regression for the live wedge: dirty work used to abort the sync forever.

    Confirmed on 2026-08-14 -- Orion left staged edits on a test branch on
    2026-08-13 and every browser refresh after that returned
    ``skipped_dirty_worktree``, leaving the sandbox 294 commits behind main with
    no way to recover. The sync must now stash the work and proceed.
    """
    origin, clone = origin_and_clone
    _git(clone, "checkout", "-b", "feature/in-flight")
    (clone / "file.txt").write_text("uncommitted-work")

    (origin / "file.txt").write_text("v2")
    _git(origin, "add", "file.txt")
    _git(origin, "commit", "-m", "second")

    result = sync_fcc_sandbox(str(clone))

    assert result == "synced_after_rescue"
    assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
    assert (clone / "file.txt").read_text() == "v2"

    # The rescued work is recoverable, labelled, and attributed to its branch.
    stashes = _git(clone, "stash", "list").stdout
    assert "orion-sandbox-autorescue/feature/in-flight/" in stashes
    # `stash show`, not `stash pop`: this edit touches the same file main moved,
    # so popping it conflicts. That is ordinary git, and the point of the rescue
    # is that the content survives to be resolved, not that it reapplies cleanly.
    assert "uncommitted-work" in _git(clone, "stash", "show", "-p", "stash@{0}").stdout


def test_sync_rescues_untracked_files_too(origin_and_clone):
    """`stash push -u`, not plain `stash`: `git clean -fd` runs after the reset,
    so an untracked-only dirty state left unstashed would be deleted outright."""
    _, clone = origin_and_clone
    (clone / "orion_scratch.md").write_text("notes Orion had not committed")

    assert sync_fcc_sandbox(str(clone)) == "synced_after_rescue"
    assert not (clone / "orion_scratch.md").exists()

    _git(clone, "stash", "pop")
    assert (clone / "orion_scratch.md").read_text() == "notes Orion had not committed"


def test_sync_leaves_unpushed_branch_intact_and_syncs(origin_and_clone):
    """Unpushed commits no longer block the sync -- `checkout main` does not move
    refs/heads/<branch>, so they stay reachable by branch name in the same clone."""
    _, clone = origin_and_clone
    _git(clone, "checkout", "-b", "feature/unpushed")
    (clone / "file.txt").write_text("v2")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "unpushed work")
    unpushed_head = _git(clone, "rev-parse", "HEAD").stdout.strip()

    result = sync_fcc_sandbox(str(clone))

    assert result == "synced"
    assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
    # The commit survives: same sha, still reachable by branch name.
    assert _git(clone, "rev-parse", "feature/unpushed").stdout.strip() == unpushed_head


def test_sync_rescues_dirty_worktree_on_main(origin_and_clone):
    """The sandbox's steady state after a sync is `main`, so dirty work sitting
    directly on main (no branch checked out) must be rescued too -- that is the
    most likely place for Orion's next uncommitted turn to leave something."""
    _, clone = origin_and_clone
    (clone / "file.txt").write_text("uncommitted-on-main")

    assert sync_fcc_sandbox(str(clone)) == "synced_after_rescue"
    assert "orion-sandbox-autorescue/main/" in _git(clone, "stash", "list").stdout
    _git(clone, "stash", "pop")
    assert (clone / "file.txt").read_text() == "uncommitted-on-main"


def test_sync_declines_reset_when_rescue_fails(origin_and_clone, monkeypatch):
    """If the stash itself fails, a reset would genuinely destroy work. Decline,
    under a status distinct from a healthy sync so the wedge is visible."""
    from orion.fcc import sandbox_sync

    _, clone = origin_and_clone
    (clone / "file.txt").write_text("precious-uncommitted-work")

    real_run_git = sandbox_sync._run_git

    def _fail_stash(workspace, *args):
        if args and args[0] == "stash":
            return subprocess.CompletedProcess(args, 1, "", "stash exploded")
        return real_run_git(workspace, *args)

    monkeypatch.setattr(sandbox_sync, "_run_git", _fail_stash)

    assert sandbox_sync.sync_fcc_sandbox(str(clone)) == "skipped_dirty_rescue_failed"
    assert (clone / "file.txt").read_text() == "precious-uncommitted-work"


def test_declined_sync_reports_real_staleness_not_a_stale_ref(origin_and_clone, monkeypatch):
    """A declined sync must still report how far behind the sandbox actually is.

    ``behind_main`` is read from refs/remotes/origin/main, so if the fetch only
    happened on the success path this would report 0 for a sandbox hundreds of
    commits back -- the exact silent-staleness failure the surface exists to
    expose. The fetch therefore runs before any decline.
    """
    from orion.fcc import sandbox_sync

    origin, clone = origin_and_clone

    # Two commits land on origin after the clone; the clone has never fetched them.
    for text in ("v2", "v3"):
        (origin / "file.txt").write_text(text)
        _git(origin, "add", "file.txt")
        _git(origin, "commit", "-m", text)

    (clone / "file.txt").write_text("dirty")

    real_run_git = sandbox_sync._run_git

    def _fail_stash(workspace, *args):
        if args and args[0] == "stash":
            return subprocess.CompletedProcess(args, 1, "", "stash exploded")
        return real_run_git(workspace, *args)

    monkeypatch.setattr(sandbox_sync, "_run_git", _fail_stash)

    assert sandbox_sync.sync_fcc_sandbox(str(clone)) == "skipped_dirty_rescue_failed"
    # Hand-counted: exactly the two commits made above, not 0.
    assert sandbox_sync.last_sync_state()["behind_main"] == 2


def test_last_sync_state_records_every_attempt(origin_and_clone):
    """The wedge was invisible for two days because the only evidence was a log
    line. Every attempt must leave an inspectable verdict behind."""
    from orion.fcc.sandbox_sync import last_sync_state, record_sync_skip

    _, clone = origin_and_clone

    assert sync_fcc_sandbox(str(clone)) == "synced"
    state = last_sync_state()
    assert state["result"] == "synced"
    assert state["workspace"] == str(clone)
    assert state["branch"] == "main"
    assert state["behind_main"] == 0
    assert state["at"]

    # A caller-side skip must overwrite it, not leave the stale "synced" verdict.
    record_sync_skip(str(clone), "skipped_turn_in_flight")
    assert last_sync_state()["result"] == "skipped_turn_in_flight"


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
