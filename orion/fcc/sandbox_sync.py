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
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("orion-hub.fcc_sandbox_sync")

_GIT_TIMEOUT_S = 30


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

    # Idempotent, cheap -- runs every sync regardless of whether a full reset
    # happens below, so push auth is set up even on a skipped-dirty/unpushed
    # sync (the sandbox may still need to push what's already there).
    _configure_push_auth(path)

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
