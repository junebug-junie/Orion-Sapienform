#!/usr/bin/env python3
"""Stop hook: snapshots every dirty worktree into a local-only, hidden git
ref (`refs/orion-wip/<worktree-name>`) -- a safety net so uncommitted work
is never purely at the mercy of "did anyone remember to commit it", without
touching the real working tree, index, HEAD, or branch history at all.

WHY THIS SHAPE, NOT AN ACTUAL COMMIT ON THE BRANCH
-----------------------------------------------------
A concrete near-miss (2026-08-18): ~1,000 lines of real, uncommitted work
sat in a worktree whose branch was already merged, right before a main
pull/deploy. Juniper asked for a hook requiring worktree work to get
committed; the first design considered was periodic auto-commit, and Juniper
flagged the real risk directly: it can commit mid-edit (a half-finished
multi-file refactor caught between two edits) and pollutes branch history
with WIP noise that then needs squashing.

This sidesteps both problems structurally, not by being careful:

1. **Never touches the branch.** The snapshot is a real git commit object
   (built via `write-tree`/`commit-tree`, the same plumbing `git stash`
   itself uses under the hood), but the only thing that points at it is a
   ref under `refs/orion-wip/`, not `refs/heads/`. It never appears in
   `git log`, a PR diff, `git status`, or any normal branch-shaped view --
   purely a recovery mechanism, invisible until someone goes looking for it
   (`git show refs/orion-wip/<name>` for one file, or `git checkout
   refs/orion-wip/<name>/latest -- .` for the whole snapshot -- verified
   live: this is a plain commit, not stash-shaped, so `git stash apply`
   fails on it with "not a stash-like commit"). Nothing to squash later
   because nothing lands where squashing would matter.
2. **Never touches the real index or working tree.** Built entirely through
   a scratch `GIT_INDEX_FILE`, so `git add -A` there populates a THROWAWAY
   index reflecting the real working tree's current content (tracked +
   untracked, respecting .gitignore) without ever writing to `.git/index`.
   The agent's actual in-progress edit, staged changes, or mid-tool-call
   state are completely undisturbed.
3. **Can only ever catch a real pause point, not mid-edit.** This only runs
   from Stop (turn/session end), never on a timer during active work -- so
   it structurally cannot fire between two halves of one atomic multi-file
   change the way a periodic auto-commit could.

WHY IT SCANS EVERY WORKTREE, NOT JUST "THE CURRENT ONE"
-----------------------------------------------------------
Claude Code's own Stop-hook stdin payload/process cwd is documented (this
repo's own CLAUDE.md, "$CLAUDE_PROJECT_DIR hook cwd quirk") to stay fixed at
session-start regardless of where Bash-tool `cd`/worktree switches have
since moved -- so there is no reliable way to ask "which worktree is THIS
session actually in" from inside a hook. Scanning every worktree sidesteps
that bug class entirely, and also protects worktrees the CURRENT session
isn't the one actively touching -- which is exactly today's actual incident
shape: the at-risk worktree was discovered, not being actively edited, at
the moment it mattered.

THROTTLED
---------
Skips the full multi-worktree scan if the last one across ANY session
finished less than `_MIN_SCAN_INTERVAL_SEC` ago (tracked via a local marker
file's mtime) -- so N concurrent sessions each stopping do not each pay for
a full scan. A skipped scan is not a missed one: the next session to stop
after the interval elapses catches everything that changed since.

Fails silently on any error at every stage -- never blocks session stop,
and a broken snapshot mechanism must degrade to "no snapshot this time",
never to a hook error surfaced to the user.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worktree_lib import WorktreeLibError, list_worktrees, parallel_map  # noqa: E402

_GIT_TIMEOUT = 10

# How often a full scan may run at all, across every concurrent session.
_MIN_SCAN_INTERVAL_SEC = 300.0
_SCAN_MARKER_PATH = Path.home() / ".orion" / "worktree_wip_snapshot_last_scan"

# Snapshots kept per worktree before older ones are pruned. Small on purpose
# -- this is a short-lived safety net (catches "forgot to commit for a
# while"), not a substitute for real version history.
_MAX_SNAPSHOTS_PER_WORKTREE = 3

_REF_PREFIX = "refs/orion-wip"


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=_GIT_TIMEOUT, **kwargs)


def _should_scan_now() -> bool:
    try:
        mtime = _SCAN_MARKER_PATH.stat().st_mtime
    except OSError:
        return True
    return (time.time() - mtime) >= _MIN_SCAN_INTERVAL_SEC


def _touch_scan_marker() -> None:
    try:
        _SCAN_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SCAN_MARKER_PATH.touch()
    except OSError:
        pass


def _is_dirty(worktree_path: str) -> bool:
    try:
        result = _run(["git", "-C", worktree_path, "status", "--porcelain"])
    except Exception:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _ref_name(worktree_name: str) -> str:
    # Worktree directory names are already filesystem-safe; ref components
    # reject a few characters (space, ~, ^, :, ?, *, [, \) that are not
    # realistic in this repo's own worktree-naming convention, so no
    # escaping beyond a defensive replace.
    safe = worktree_name.replace(" ", "_")
    return f"{_REF_PREFIX}/{safe}"


def _latest_snapshot_tree(worktree_path: str, ref: str) -> str | None:
    result = _run(["git", "-C", worktree_path, "rev-parse", "--verify", "-q", f"{ref}/latest^{{tree}}"])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _snapshot_one(info) -> str | None:
    """Returns a short status string for logging, or None on no-op/failure.
    Never raises -- every step degrades to "skip this worktree" on error."""
    path = str(info.path)
    if not _is_dirty(path):
        return None

    try:
        head = _run(["git", "-C", path, "rev-parse", "HEAD"]).stdout.strip()
        if not head:
            return None

        # A path that does not exist yet, NOT a pre-created empty file: git
        # reads an existing-but-empty GIT_INDEX_FILE as a corrupt index
        # ("index file smaller than expected") rather than starting fresh --
        # confirmed live while building this hook. tempfile.NamedTemporaryFile
        # creates the file; this only reserves a name.
        index_path = os.path.join(tempfile.gettempdir(), f"orion-wip-index-{uuid.uuid4().hex}")
        try:
            env = dict(os.environ, GIT_INDEX_FILE=index_path)
            # Scratch index only -- never touches .git/index. `-C path` sets
            # which working tree `add -A` reads from; GIT_INDEX_FILE
            # redirects only where the resulting index metadata is written.
            add = _run(["git", "-C", path, "add", "-A"], env=env)
            if add.returncode != 0:
                return None
            tree = _run(["git", "-C", path, "write-tree"], env=env).stdout.strip()
            if not tree:
                return None
        finally:
            try:
                os.unlink(index_path)
            except OSError:
                pass

        ref_base = _ref_name(info.branch or Path(path).name)
        latest_ref = f"{ref_base}/latest"
        prior_tree = _latest_snapshot_tree(path, ref_base)
        if prior_tree == tree:
            return None  # identical to the last snapshot -- nothing changed

        message = f"orion-wip snapshot: {info.branch or '(detached)'}"
        commit = _run(
            ["git", "-C", path, "commit-tree", tree, "-p", head, "-m", message]
        ).stdout.strip()
        if not commit:
            return None

        timestamp_ref = f"{ref_base}/{int(time.time())}"
        for ref in (latest_ref, timestamp_ref):
            _run(["git", "-C", path, "update-ref", ref, commit])

        _prune_old_snapshots(path, ref_base)
        return f"{info.branch or Path(path).name}: snapshotted {commit[:12]}"
    except Exception:
        return None


def _prune_old_snapshots(worktree_path: str, ref_base: str) -> None:
    try:
        result = _run(["git", "-C", worktree_path, "for-each-ref", "--format=%(refname) %(creatordate:unix)", ref_base])
        if result.returncode != 0:
            return
        # Timestamped refs only -- "latest" is a moving pointer, not part of
        # the prune-by-age count.
        entries = [
            line.split(" ", 1)
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith(f"{ref_base}/latest ")
        ]
        entries.sort(key=lambda kv: int(kv[1]))  # oldest first
        excess = len(entries) - _MAX_SNAPSHOTS_PER_WORKTREE
        for refname, _ in entries[: max(0, excess)]:
            _run(["git", "-C", worktree_path, "update-ref", "-d", refname])
    except Exception:
        pass


def _run_worker_scan() -> None:
    """The actual scan -- confirmed live to take ~20s across this repo's
    real ~258-worktree fleet (16-way parallel `git status`/`add`/`write-
    tree`/`commit-tree` calls per dirty worktree), well past what a
    Claude-Code hook timeout should ever block on, and only growing as the
    worktree count does. Always invoked as a detached background process
    (see main()), never inline in the hook's own blocking execution."""
    try:
        worktrees = [w for w in list_worktrees() if not w.is_main]
    except WorktreeLibError:
        return
    if not worktrees:
        return
    try:
        parallel_map(_snapshot_one, worktrees, max_workers=16)
    except Exception:
        pass


def main() -> None:
    """Hook entry point -- must return in well under a second regardless of
    fleet size. Touches the throttle marker itself (not the worker) so a
    burst of near-simultaneous Stop events from several sessions only ever
    launches one worker, not one per session -- the marker's write is the
    lock, checked-then-touched before the background process starts."""
    if not _should_scan_now():
        return
    _touch_scan_marker()
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from this process's session/controlling terminal
    )


if __name__ == "__main__":
    try:
        if "--worker" in sys.argv[1:]:
            _run_worker_scan()
        else:
            main()
    except Exception:
        pass
