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

The refresh is a hard reset to ``origin/main`` guarded on there being no dirty or
unpushed work checked out -- on *any* branch, including ``main`` itself, since a prior
sync leaves the sandbox sitting on ``main`` and that's exactly where Orion's next turn
would add uncommitted or unpushed-on-main work if it doesn't branch first.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("orion-hub.fcc_sandbox_sync")

_GIT_TIMEOUT_S = 30


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


def sync_fcc_sandbox(workspace: str | None) -> str:
    """Best-effort sync of ``workspace`` to ``origin/main``.

    Returns a short status string for logging/telemetry; never raises -- a sync
    failure should degrade to "Orion works from a stale checkout this session",
    not break Hub's WebSocket connect.
    """
    if not workspace:
        return "skipped_no_workspace"

    path = Path(workspace)
    if not (path / ".git").exists():
        return "skipped_not_a_git_repo"

    try:
        current_branch = _git_or_abort(
            path, "branch", "--show-current", error_status="error_branch_check"
        ).stdout.strip()

        # Dirty/unpushed check applies unconditionally, including when already on
        # main -- a prior sync leaves the sandbox on main, so "no branch checked
        # out" must not be read as "nothing to protect".
        status = _git_or_abort(
            path, "status", "--porcelain", error_status="error_status_check"
        )
        if status.stdout.strip():
            logger.info("fcc_sandbox_sync_skipped_dirty branch=%s", current_branch or "main")
            return "skipped_dirty_worktree"

        if current_branch and current_branch != "main":
            # refs/heads/<branch> keeps a branch name that starts with "-" (e.g.
            # "-x") from being parsed as a flag instead of a ref.
            ref = f"refs/heads/{current_branch}"
            fetch = _run_git(path, "fetch", "origin", ref)
            unpushed = _run_git(path, "rev-list", f"origin/{current_branch}..{ref}")
            if fetch.returncode != 0 or unpushed.returncode != 0 or unpushed.stdout.strip():
                logger.info(
                    "fcc_sandbox_sync_skipped_unpushed_work branch=%s", current_branch
                )
                return "skipped_unpushed_branch"

        _git_or_abort(path, "fetch", "origin", "main", error_status="error_fetch")
        _git_or_abort(path, "checkout", "main", error_status="error_checkout")
        _git_or_abort(
            path, "reset", "--hard", "origin/main", error_status="error_reset"
        )

        clean = _run_git(path, "clean", "-fd")
        if clean.returncode != 0:
            logger.warning("fcc_sandbox_sync_clean_failed err=%s", clean.stderr.strip())

        logger.info("fcc_sandbox_sync_ok workspace=%s", workspace)
        return "synced"
    except _SyncAbort as exc:
        return exc.status
    except subprocess.TimeoutExpired:
        logger.warning("fcc_sandbox_sync_timeout workspace=%s", workspace)
        return "error_timeout"
    except OSError as exc:
        logger.warning("fcc_sandbox_sync_os_error workspace=%s err=%s", workspace, exc)
        return "error_os"
