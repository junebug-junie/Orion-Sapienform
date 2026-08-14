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


@pytest.fixture(autouse=True)
def _no_real_fcc_env(tmp_path, monkeypatch):
    """Keep the operator's real GitHub PAT out of these tests.

    ``_configure_push_auth`` falls back to ``~/.fcc/.env`` when
    ``HARNESS_FCC_ENV_PATH`` is unset, and that file holds a live token on the host.
    Every test here builds a clone and syncs it, so the token was being written into
    each clone's ``.git/config`` under ``/tmp/pytest-of-<user>/`` -- verified live at
    26 files, and pytest retains the last three run directories. The tests that
    genuinely exercise push auth override this with their own fixture file.
    """
    monkeypatch.setenv("HARNESS_FCC_ENV_PATH", str(tmp_path / "no-such-fcc.env"))
    monkeypatch.delenv("HUB_FCC_ENV_PATH", raising=False)


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


def _break_rescue(monkeypatch, *, blocked=("stash", "commit")):
    """Make the named git subcommands fail, so a rescue cannot succeed at all."""
    from orion.fcc import sandbox_sync

    real_run_git = sandbox_sync._run_git

    def _failing(workspace, *args):
        if args and args[0] in blocked:
            return subprocess.CompletedProcess(args, 1, "", f"{args[0]} exploded")
        return real_run_git(workspace, *args)

    monkeypatch.setattr(sandbox_sync, "_run_git", _failing)
    return sandbox_sync


def test_sync_declines_reset_when_every_rescue_fails(origin_and_clone, monkeypatch):
    """If neither the stash nor the branch fallback can capture the work, a reset
    would genuinely destroy it. Decline, under a status distinct from a healthy sync
    so the wedge is visible."""
    _, clone = origin_and_clone
    (clone / "file.txt").write_text("precious-uncommitted-work")

    sandbox_sync = _break_rescue(monkeypatch)

    assert sandbox_sync.sync_fcc_sandbox(str(clone)) == "skipped_dirty_rescue_failed"
    assert (clone / "file.txt").read_text() == "precious-uncommitted-work"


def test_stash_refusal_falls_back_to_a_rescue_branch(origin_and_clone):
    """A conflicted merge makes `git stash` refuse outright, which used to mean the
    sync declined on every subsequent attempt forever -- the same permanent freeze
    this module exists to end, relabelled. The branch fallback must capture the work
    and let the sync proceed.
    """
    origin, clone = origin_and_clone

    # Build a genuine conflict: same line changed on main and on a side branch.
    _git(clone, "checkout", "-b", "side")
    (clone / "file.txt").write_text("side-edit")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "side")
    _git(clone, "checkout", "main")
    (clone / "file.txt").write_text("main-edit")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "main")
    merge = subprocess.run(
        ["git", "merge", "side"], cwd=str(clone), capture_output=True, text=True, check=False
    )
    assert merge.returncode != 0, "fixture must actually conflict"
    assert "UU" in _git(clone, "status", "--porcelain").stdout

    # Precondition: stash really does refuse here, so the fallback is what is tested.
    refused = subprocess.run(
        ["git", "stash", "push", "-u"], cwd=str(clone), capture_output=True, text=True, check=False
    )
    assert refused.returncode != 0

    result = sync_fcc_sandbox(str(clone))

    assert result == "synced_after_rescue"
    assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
    rescue_refs = [
        line.strip().lstrip("* ")
        for line in _git(clone, "branch", "--list", "orion-sandbox-autorescue/*").stdout.splitlines()
    ]
    assert rescue_refs, "conflicted work must survive on a rescue branch"
    assert "main-edit" in _git(clone, "show", f"{rescue_refs[0]}:file.txt").stdout

    # And it terminates: a second run is a clean no-op, not another decline.
    assert sync_fcc_sandbox(str(clone)) == "synced"


def test_detached_head_commits_get_a_real_ref(origin_and_clone):
    """In detached HEAD, `git branch --show-current` is empty, so the
    branch-preservation path never fires and a commit would survive only in HEAD's
    reflog until gc expires it."""
    origin, clone = origin_and_clone

    _git(clone, "checkout", "--detach")
    (clone / "file.txt").write_text("detached-work")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-m", "work made in detached HEAD")
    orphan = _git(clone, "rev-parse", "HEAD").stdout.strip()

    assert sync_fcc_sandbox(str(clone)) == "synced"

    containing = _git(clone, "for-each-ref", "--contains", orphan, "--format=%(refname)").stdout
    assert orphan not in _git(clone, "rev-parse", "HEAD").stdout
    assert containing.strip(), "detached commit must be reachable from some ref after the sync"


def test_stale_index_lock_does_not_freeze_the_sync_forever(origin_and_clone):
    """A git command killed by the 30s timeout leaves .git/index.lock behind, and
    nothing else removes it -- every later sync would fail on it indefinitely. Same
    silent-recurring-freeze family as the wedge this module exists to end."""
    import os
    import time as _time

    from orion.fcc import sandbox_sync

    origin, clone = origin_and_clone
    (origin / "file.txt").write_text("v2")
    _git(origin, "add", "file.txt")
    _git(origin, "commit", "-m", "second")

    lock = clone / ".git" / "index.lock"
    lock.write_text("")
    # Backdate past the git timeout: a fresh lock could still belong to a live git.
    old = _time.time() - (sandbox_sync._GIT_TIMEOUT_S + 60)
    os.utime(lock, (old, old))

    assert sync_fcc_sandbox(str(clone)) == "synced"
    assert not lock.exists()
    assert (clone / "file.txt").read_text() == "v2"


def test_fresh_index_lock_is_left_alone(origin_and_clone):
    """The age check is load-bearing: a lock younger than the timeout may belong to a
    git process still running, and removing it would corrupt that operation."""
    _, clone = origin_and_clone
    lock = clone / ".git" / "index.lock"
    lock.write_text("")

    sync_fcc_sandbox(str(clone))

    assert lock.exists(), "a fresh lock must never be removed"


def test_sync_yields_to_a_turn_holding_the_shared_lock(origin_and_clone):
    """The cross-container interlock: a turn holds the shared lock, so the sync must
    decline rather than reset the tree under a running claude subprocess."""
    from orion.fcc.turn_lock import turn_in_progress

    _, clone = origin_and_clone

    with turn_in_progress(str(clone)) as held:
        assert held, "fixture must actually acquire the turn lock"
        assert sync_fcc_sandbox(str(clone)) == "skipped_turn_in_flight_external"

    # Released -> the very next attempt proceeds.
    assert sync_fcc_sandbox(str(clone)) == "synced"


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

    sandbox_sync = _break_rescue(monkeypatch)

    assert sandbox_sync.sync_fcc_sandbox(str(clone)) == "skipped_dirty_rescue_failed"
    # Hand-counted: exactly the two commits made above, not 0.
    assert sandbox_sync.last_sync_state()["last_attempt"]["behind_main"] == 2


def test_last_sync_state_keeps_attempt_and_success_apart(origin_and_clone):
    """The wedge was invisible for two days because the only evidence was a log line.

    One slot was not enough: a skip would overwrite the last successful verdict, so
    the surface could report what the last attempt did but not whether the sandbox
    was actually current -- the question it exists to answer.
    """
    from orion.fcc.sandbox_sync import last_sync_state, record_sync_skip

    origin, clone = origin_and_clone
    (origin / "file.txt").write_text("v2")
    _git(origin, "add", "file.txt")
    _git(origin, "commit", "-m", "second")

    assert sync_fcc_sandbox(str(clone)) == "synced"
    state = last_sync_state()
    assert state["last_attempt"]["result"] == "synced"
    assert state["last_attempt"]["workspace"] == str(clone)
    assert state["last_attempt"]["branch"] == "main"
    assert state["last_attempt"]["at"]
    # Hand-counted: the clone starts one commit behind and the sync applies exactly
    # that one commit.
    assert state["last_attempt"]["commits_advanced"] == 1

    # A skip replaces the attempt but must NOT erase the last known-good state.
    record_sync_skip(str(clone), "skipped_turn_in_flight")
    state = last_sync_state()
    assert state["last_attempt"]["result"] == "skipped_turn_in_flight"
    assert state["last_success"]["result"] == "synced"
    assert state["last_success"]["commits_advanced"] == 1


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
