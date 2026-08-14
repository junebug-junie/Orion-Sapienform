"""Refresh Orion's FCC sandbox checkout to current ``origin/main`` on session start.

The FCC sandbox (default ``/mnt/orion-fcc/repo``, see ``HUB_AGENT_CLAUDE_WORKSPACE`` /
``HARNESS_FCC_WORKSPACE``) is a standalone git clone -- not a worktree, so it shares no
``.git`` object store with the primary checkout. Nothing Orion does there can touch the
real repo directly; the only path back is an explicit ``git push`` to a non-``main``
branch, gated by GitHub branch protection (PR + review required, force-push disabled).

Because the sandbox is disposable and disconnected from the primary checkout, it drifts
from ``main`` unless something refreshes it. Hub triggers that refresh once per browser
session (see ``websocket_handler.py``'s connect-time setup, before the per-turn loop) --
not per-turn, since Juniper's own description of the workflow is "each time hub is
refreshed on the browser, that would trigger a refresh on the sandbox".

The refresh is a hard reset to ``origin/main``. It does not destroy *tracked or
untracked* work to get there: anything dirty in the worktree is first rescued into a
labelled ``git stash`` entry (or, when stash refuses, onto a rescue branch), branch refs
survive a ``checkout main`` untouched, and a detached HEAD is given a ref before it is
left. Only if every rescue fails does the sync decline to reset.

One carve-out, stated rather than papered over: **gitignored files are not rescued**.
``git status --porcelain`` does not report them, so the rescue never sees them, and
``stash push -u`` would not capture them anyway (that needs ``-a``, far too broad here).
A gitignored path that ``origin/main`` later starts tracking is overwritten by the
checkout with no stash entry. Local scratch files in the sandbox are disposable by
design; anything that is not should not live in a gitignored path there.

That rescue exists because the original guard -- skip the sync outright whenever the
worktree was dirty or the branch unpushed -- was a permanent trap, confirmed live on
2026-08-14. Orion left staged edits behind on ``test/orion-live-push-proof-aacd41ef``
on 2026-08-13; every browser refresh after that logged ``skipped_dirty_worktree`` and
returned, so the sandbox sat 294 commits behind ``main`` with nothing able to clear it.
A guard whose only exit is manual intervention nobody is told to perform is not a guard,
it is a silent freeze. Rescue-then-reset keeps the work *and* keeps the sync honest.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from orion.fcc.turn_lock import sync_exclusive

logger = logging.getLogger("orion-hub.fcc_sandbox_sync")

_GIT_TIMEOUT_S = 30

# Verdicts that mean "the sandbox is now at origin/main".
_SUCCESS_RESULTS = frozenset({"synced", "synced_after_rescue"})


def _load_fcc_env_value(key: str) -> str:
    """Minimal dotenv read for ``key``, mirroring fcc_motor.load_fcc_env's parsing.

    Not importing that function directly: it lives in orion.harness (governor-only
    territory) and this module is shared with orion-hub, which has its own
    near-identical loader in a hub-scoped script. Duplicating a ~10-line KEY=VALUE
    parser here is cheaper than adding a cross-service import for one field.
    """
    raw_path = os.environ.get("HARNESS_FCC_ENV_PATH") or os.environ.get(
        "HUB_FCC_ENV_PATH", "~/.fcc/.env"
    )
    path = Path(os.path.expanduser(str(raw_path or "~/.fcc/.env").strip() or "~/.fcc/.env"))
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def _configure_push_auth(workspace: Path) -> None:
    """Rewrite the sandbox's SSH-style remote to token-authenticated HTTPS.

    Repo-LOCAL config only (``git -C workspace config --local ...``), never
    ``--global`` -- this keeps the token out of the image-baked global gitconfig
    (see the safe.directory/identity fix: a global .gitconfig bind mount already
    caused one outage here, nothing should write secrets into that file) and
    scopes it to this one disposable clone. ``git push``/``gh`` in the sandbox
    otherwise have no credentials at all: GITHUB_PAT is injected into the GitHub
    MCP server's own subprocess env (orion/fcc/mcp_config.py) but never reaches
    the outer claude subprocess's shell or git's credential store -- confirmed
    live (env | grep -i GITHUB found nothing, git push failed with no SSH agent
    and no HTTPS credential available).
    """
    token = _load_fcc_env_value("GITHUB_PAT")
    if not token:
        logger.info("fcc_sandbox_push_auth_skipped_no_token")
        return
    result = _run_git(
        workspace,
        "config",
        "--local",
        "url.https://x-access-token:" + token + "@github.com/.insteadOf",
        "git@github.com:",
    )
    if result.returncode != 0:
        logger.warning("fcc_sandbox_push_auth_config_failed err=%s", result.stderr.strip())


class _SyncAbort(Exception):
    """Raised internally to short-circuit the sync with a status string."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


def _run_git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )


def _git_or_abort(
    workspace: Path, *args: str, error_status: str
) -> subprocess.CompletedProcess:
    """Run a git step; abort the sync with ``error_status`` on non-zero exit."""
    result = _run_git(workspace, *args)
    if result.returncode != 0:
        logger.warning(
            "fcc_sandbox_sync_step_failed step=%s err=%s", error_status, result.stderr.strip()
        )
        raise _SyncAbort(error_status)
    return result


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rescue_dirty_worktree(workspace: Path, branch: str) -> str | None:
    """Stash everything dirty (including untracked) under a findable label.

    Returns how the work was rescued (``"stash"`` or ``"branch"``) when the worktree
    is clean afterwards and the caller may reset, or ``None`` if it could not be
    rescued at all. The caller needs the *method*, not just success: the branch
    fallback moves HEAD, which changes what the detached-HEAD check below should do.

    ``stash push -u`` is the first choice because it leaves the branch's own history
    untouched -- Orion's next turn on that branch sees the commits it expects, and the
    rescued edits are one ``git stash list`` away instead of buried in a commit nobody
    looks for.

    But ``git stash`` refuses outright on unmerged paths, so a conflicted merge,
    rebase, or cherry-pick left behind by an agent would make the rescue fail on every
    subsequent sync -- the same permanent freeze this module exists to end, just
    relabelled. A conflicted merge in the sandbox is an ordinary event, not an exotic
    one, so there is a second rescue: commit the whole tree onto a dedicated branch
    ref. Less convenient to recover than a stash, and it is not the tidy option, but
    it is lossless and it always terminates.
    """
    label = f"orion-sandbox-autorescue/{branch or 'main'}/{_stamp()}"
    result = _run_git(workspace, "stash", "push", "-u", "-m", label)
    if result.returncode == 0:
        logger.info(
            "fcc_sandbox_sync_rescued_dirty_work branch=%s stash=%s", branch or "main", label
        )
        return "stash"

    # git writes the stash refusal to stdout, not stderr. Reading only stderr here
    # produced a live log line of literally `err=` on the one verdict that means
    # "a human needs to look at this".
    reason = (result.stdout.strip() or result.stderr.strip() or "unknown").splitlines()[-1]
    logger.warning(
        "fcc_sandbox_sync_stash_refused branch=%s reason=%s", branch or "main", reason
    )

    if _rescue_to_branch(workspace, label):
        return "branch"

    logger.warning("fcc_sandbox_sync_rescue_failed branch=%s reason=%s", branch or "main", reason)
    return None


def _rescue_to_branch(workspace: Path, label: str) -> bool:
    """Fallback rescue: commit the entire dirty tree onto ``label`` as a branch.

    Used when ``git stash`` refuses -- overwhelmingly an unresolved merge/rebase.
    Aborts any in-progress merge/rebase state *after* the content is safely committed,
    never before, so nothing is discarded to make the commit possible.
    """
    checkout = _run_git(workspace, "checkout", "-b", label)
    if checkout.returncode != 0:
        return False
    if _run_git(workspace, "add", "-A").returncode != 0:
        return False
    commit = _run_git(
        workspace, "commit", "--no-verify", "-m", f"orion sandbox autorescue: {label}"
    )
    if commit.returncode != 0:
        return False

    # The commit resolves the conflicted state; clear any residual merge/rebase
    # bookkeeping so the following checkout of main is not itself refused.
    for state in (("merge",), ("rebase",), ("cherry-pick",)):
        _run_git(workspace, state[0], "--abort")

    logger.info("fcc_sandbox_sync_rescued_to_branch ref=%s", label)
    return True


def _rescue_detached_head(workspace: Path) -> str | None:
    """Give a detached HEAD a real ref before we check out ``main``.

    Without this, commits made in detached HEAD are reachable only through HEAD's
    reflog after the sync -- no branch, no tag, and eventually expired by ``git gc``.
    The module docstring's promise that work survives depends on a ref existing.
    """
    ref = f"orion-sandbox-autorescue-detached/{_stamp()}"
    if _run_git(workspace, "branch", ref, "HEAD").returncode != 0:
        return None
    logger.info("fcc_sandbox_sync_rescued_detached_head ref=%s", ref)
    return ref


_LAST_ATTEMPT: dict[str, object] = {"result": None, "at": None, "workspace": None}
_LAST_SUCCESS: dict[str, object] | None = None


def last_sync_state() -> dict[str, object]:
    """Snapshot of sync state, for the Hub status endpoint.

    Exists because the wedge this module now rescues from was invisible for two
    days: the only evidence was a log line inside the hub container, and Hub's UI
    reported nothing at all. A stale sandbox is a cognition-affecting condition --
    Orion reasons about a repo 294 commits out of date -- so it needs a surface,
    not just a log.

    Reports the last *attempt* and the last *success* separately. They answer
    different questions, and only the pair answers the one that matters. A single
    slot meant the common case -- Juniper refreshes while a turn is running --
    overwrote "synced, 0 behind" with a bare "skipped_turn_in_flight", leaving the
    surface unable to say whether the sandbox was actually current.
    """
    return {"last_attempt": dict(_LAST_ATTEMPT), "last_success": dict(_LAST_SUCCESS or {})}


def _record(workspace: str | None, result: str, **extra: object) -> str:
    """Publish a verdict as one atomic rebind.

    Rebinding beats mutating in place: ``api_fcc_sandbox_sync`` is a sync ``def``, so
    FastAPI runs it in the threadpool concurrently with the ``asyncio.to_thread``
    worker running the sync. A ``clear()``-then-``update()`` pair leaves a window
    where a reader sees an empty dict and the route returns a body with no verdict.
    """
    global _LAST_ATTEMPT, _LAST_SUCCESS
    record = {
        "result": result,
        "at": datetime.now(timezone.utc).isoformat(),
        "workspace": workspace,
        **extra,
    }
    _LAST_ATTEMPT = record
    if result in _SUCCESS_RESULTS:
        _LAST_SUCCESS = record
    return result


def record_sync_skip(workspace: str | None, result: str) -> str:
    """Record a skip decided by the *caller* (e.g. an FCC turn already in flight).

    Public so ``websocket_handler`` can keep the status surface truthful for the
    skips it makes before ever calling ``sync_fcc_sandbox``.
    """
    return _record(workspace, result)


def _behind_main(workspace: Path) -> int | None:
    """How many commits the checked-out HEAD trails ``origin/main`` by.

    Best-effort and read-only: reads the already-fetched remote ref, never
    fetches. ``None`` when it cannot be determined (no origin/main ref yet).
    """
    result = _run_git(workspace, "rev-list", "--count", "HEAD..origin/main")
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def sync_fcc_sandbox(workspace: str | None) -> str:
    """Best-effort sync of ``workspace`` to ``origin/main``.

    Returns a short status string for logging/telemetry; never raises -- a sync
    failure should degrade to "Orion works from a stale checkout this session",
    not break Hub's WebSocket connect.
    """
    if not workspace:
        return _record(workspace, "skipped_no_workspace")

    path = Path(workspace)
    if not (path / ".git").exists():
        return _record(workspace, "skipped_not_a_git_repo")

    # Idempotent, cheap -- runs every sync regardless of whether a full reset
    # happens below, so push auth is set up even on a skipped-dirty/unpushed
    # sync (the sandbox may still need to push what's already there).
    _configure_push_auth(path)

    # Exclusive for the whole destructive sequence. This is the only interlock that
    # actually works across containers: hub's in-process active_turns() cannot see a
    # governor turn, and the live chat path *is* the governor path, so that check was
    # empty during essentially every real turn. Non-blocking -- a refresh during a
    # turn reports and moves on rather than stalling the connect handler.
    with sync_exclusive(workspace) as have_lock:
        if not have_lock:
            logger.info("fcc_sandbox_sync_skipped_turn_lock_held workspace=%s", workspace)
            return _record(workspace, "skipped_turn_in_flight_external")
        return _sync_locked(workspace, path)


def _clear_stale_index_lock(path: Path) -> None:
    """Remove a ``.git/index.lock`` orphaned by a killed git process.

    ``subprocess.run(timeout=...)`` kills the child but cannot clean up after it, so a
    git command that hits ``_GIT_TIMEOUT_S`` mid-checkout leaves its lock behind and
    every subsequent sync fails on it forever -- another silent recurring freeze of
    the same family this module exists to end.

    Only safe to do while holding the exclusive turn lock (the caller does): that is
    what rules out a *live* git process in the other container owning this lock. The
    age check is a second belt -- anything younger than the timeout could still belong
    to a git we ourselves just launched.
    """
    lock = path / ".git" / "index.lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return
    if age < _GIT_TIMEOUT_S:
        return
    try:
        lock.unlink()
        logger.warning("fcc_sandbox_sync_cleared_stale_index_lock age_s=%.0f", age)
    except OSError as exc:
        logger.warning("fcc_sandbox_sync_stale_index_lock_unremovable err=%s", exc)


def _sync_locked(workspace: str, path: Path) -> str:
    """The sync proper, run while holding the exclusive sandbox lock."""
    _clear_stale_index_lock(path)
    try:
        current_branch = _git_or_abort(
            path, "branch", "--show-current", error_status="error_branch_check"
        ).stdout.strip()
        # Captured before anything moves, so the success record can report how far
        # the sandbox actually advanced.
        start_head = _run_git(path, "rev-parse", "HEAD").stdout.strip() or "HEAD"

        # Fetch before deciding anything. It is needed for the reset regardless, and
        # doing it up front is what makes ``behind_main`` truthful on the paths that
        # *decline* to reset -- those are exactly the ones a human reads. Computed
        # against a stale refs/remotes/origin/main it would report "0 behind" for a
        # sandbox hundreds of commits back, reproducing the silent-staleness failure
        # this surface exists to expose.
        _git_or_abort(path, "fetch", "origin", "main", error_status="error_fetch")

        # The dirty check applies unconditionally, including when already on main --
        # a prior sync leaves the sandbox on main, so "no branch checked out" must
        # not be read as "nothing to protect". Protection now means *rescue*, not
        # *refuse*: refusing left the sandbox frozen forever (see module docstring).
        status = _git_or_abort(
            path, "status", "--porcelain", error_status="error_status_check"
        )
        rescue_method = None
        if status.stdout.strip():
            rescue_method = _rescue_dirty_worktree(path, current_branch)
            if rescue_method is None:
                # Rescue failed, so a reset would genuinely destroy work. Decline,
                # under a status distinct from the ordinary skips so a wedge here
                # reads as "needs a human" rather than "working as designed".
                return _record(
                    workspace,
                    "skipped_dirty_rescue_failed",
                    branch=current_branch or "main",
                    behind_main=_behind_main(path),
                )

        detached_ref = None
        # Only when nothing has already given this HEAD a ref: the branch fallback
        # above moves HEAD onto its own rescue branch, so re-rescuing here would mint
        # a second ref for the same commit.
        if not current_branch and rescue_method != "branch":
            # Detached HEAD: `git branch --show-current` is empty, so the
            # branch-preservation path below never runs and any commit made here would
            # survive only in HEAD's reflog until gc expires it. Give it a real ref.
            head_sha = _run_git(path, "rev-parse", "HEAD").stdout.strip()
            origin_sha = _run_git(path, "rev-parse", "origin/main").stdout.strip()
            if head_sha and head_sha != origin_sha:
                detached_ref = _rescue_detached_head(path)

        if current_branch and current_branch != "main":
            # Unpushed commits are NOT a reason to skip: `checkout main` leaves
            # refs/heads/<branch> exactly where it was, so they stay reachable by
            # branch name in this same clone. Log what we stepped off so the branch
            # is findable later; the sync itself proceeds.
            # refs/heads/<branch> keeps a branch name that starts with "-" (e.g.
            # "-x") from being parsed as a flag instead of a ref.
            ref = f"refs/heads/{current_branch}"
            head = _run_git(path, "rev-parse", ref)
            _run_git(path, "fetch", "origin", ref)
            unpushed = _run_git(path, "rev-list", "--count", f"origin/{current_branch}..{ref}")
            logger.info(
                "fcc_sandbox_sync_left_branch branch=%s head=%s unpushed=%s",
                current_branch,
                head.stdout.strip()[:12] or "unknown",
                unpushed.stdout.strip() or "unknown",
            )

        _git_or_abort(path, "checkout", "main", error_status="error_checkout")
        _git_or_abort(
            path, "reset", "--hard", "origin/main", error_status="error_reset"
        )

        clean = _run_git(path, "clean", "-fd")
        if clean.returncode != 0:
            logger.warning("fcc_sandbox_sync_clean_failed err=%s", clean.stderr.strip())

        rescued = rescue_method is not None
        logger.info(
            "fcc_sandbox_sync_ok workspace=%s rescued=%s method=%s",
            workspace,
            rescued,
            rescue_method or "none",
        )
        head = _run_git(path, "rev-parse", "HEAD").stdout.strip()[:12]
        # `commits_advanced`, not `behind_main`: after a reset to origin/main the
        # latter is 0 by construction and proves nothing. How far the sandbox moved
        # is the number that says the sync actually did something.
        advanced = _run_git(path, "rev-list", "--count", f"{start_head}..HEAD")
        return _record(
            workspace,
            "synced_after_rescue" if rescued else "synced",
            branch="main",
            head=head or None,
            commits_advanced=int(advanced.stdout.strip() or 0)
            if advanced.returncode == 0 and advanced.stdout.strip().isdigit()
            else None,
            rescued_from=(current_branch or "main") if rescued else None,
            rescue_method=rescue_method,
            rescued_detached_ref=detached_ref,
        )
    except _SyncAbort as exc:
        return _record(workspace, exc.status)
    except subprocess.TimeoutExpired:
        logger.warning("fcc_sandbox_sync_timeout workspace=%s", workspace)
        return _record(workspace, "error_timeout")
    except OSError as exc:
        logger.warning("fcc_sandbox_sync_os_error workspace=%s err=%s", workspace, exc)
        return _record(workspace, "error_os")
