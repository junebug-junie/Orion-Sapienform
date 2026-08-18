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

WHY IT SCANS EVERY WORKTREE (INCLUDING THE MAIN/SHARED CHECKOUT), NOT JUST
"THE CURRENT ONE"
-----------------------------------------------------------
Claude Code's own Stop-hook stdin payload/process cwd is documented (this
repo's own CLAUDE.md, "$CLAUDE_PROJECT_DIR hook cwd quirk") to stay fixed at
session-start regardless of where Bash-tool `cd`/worktree switches have
since moved -- so there is no reliable way to ask "which worktree is THIS
session actually in" from inside a hook. Scanning every worktree sidesteps
that bug class entirely, and also protects worktrees the CURRENT session
isn't the one actively touching -- which is exactly today's actual incident
shape: the at-risk worktree was discovered, not being actively edited, at
the moment it mattered. The main/shared checkout is included, not filtered
out the way `worktree_lib.mergeable_worktrees()`'s own consumers (prune,
status) correctly exclude it -- code review, 2026-08-18: an earlier draft
copied that same `is_main` filter here without reconsidering it, silently
leaving the one location this repo's own CLAUDE.md names as the highest-risk
spot for concurrent-session dirty-state collisions completely unprotected,
while claiming "every worktree" anyway. Read-only git-status/plumbing calls
against the main checkout are safe -- this never writes to its real index,
working tree, or HEAD, same guarantee as every other worktree it snapshots.

THROTTLED, WITH A REAL LOCK NOT A CHECK-THEN-ACT RACE
----------------------------------------------------------
Skips the full multi-worktree scan if the last one across ANY session
finished less than `_MIN_SCAN_INTERVAL_SEC` ago (tracked via a local marker
file's mtime). The check-then-touch sequence is guarded by a non-blocking
`fcntl.flock` on a sibling lock file, not a bare stat-then-touch -- code
review, 2026-08-18, caught an earlier draft doing exactly the plain
existence-check race this repo's own `scripts/bus_core_health_watchdog.py`
already documents a real incident for ("a plain existence-check lock (not
flock) raced its own cron job and corrupted a shutdown sequence"). Same
`_StateLock` pattern reused here: `flock` is kernel-atomic, no gap for two
near-simultaneous Stop events to both pass the check and each launch their
own background worker. The marker is only touched AFTER the background
worker actually launches successfully -- if `Popen` itself raises (e.g.
resource exhaustion), the marker stays untouched so the next qualifying Stop
event gets a real retry instead of a silent 5-minute unprotected gap.

Fails silently on any error at every stage -- never blocks session stop,
and a broken snapshot mechanism must degrade to "no snapshot this time",
never to a hook error surfaced to the user.

WHAT THIS DOES NOT PROTECT AGAINST: SECRETS IN AN UNIGNORED FILE
----------------------------------------------------------------------
`git add -A` against the scratch index captures exactly what a real
`git add -A && git commit` would -- every tracked file's current content and
every untracked file NOT matched by `.gitignore`. That is the same rule
every other commit in this repo already follows, but here it happens
automatically, with no chance to review before the content is captured. A
stray file with something sensitive in it, sitting in a dirty worktree and
not yet covered by any `.gitignore` pattern, gets committed into a real git
object reachable from `refs/orion-wip/*` -- silently, with no user-visible
signal that it happened. Ref pruning (`_MAX_SNAPSHOTS_PER_WORKTREE` and
`_MAX_SNAPSHOT_AGE_SEC` below) bounds how long a REF points at that content,
but does not by itself `git gc` the underlying object out of
`.git/objects` -- this mechanism does not claim to erase anything, only to
stop pointing at it. Normal secret hygiene (`.gitignore` real secret files,
never leave credentials in an untracked scratch file) is not optional just
because this hook exists -- it is exactly as necessary as it already was for
every other `git add -A` in this repo.
"""
from __future__ import annotations

import fcntl
import hashlib
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
_SCAN_LOCK_PATH = _SCAN_MARKER_PATH.with_suffix(".lock")

# Snapshots kept per worktree before older ones are pruned by count. Small on
# purpose -- this is a short-lived safety net (catches "forgot to commit for
# a while"), not a substitute for real version history. See the module
# docstring's own "what this does not protect against" section for why ref
# pruning is not the same as erasing the underlying object.
_MAX_SNAPSHOTS_PER_WORKTREE = 3
# Belt-and-suspenders time bound, independent of count -- a worktree that
# goes dirty once and never changes again would otherwise keep exactly 1
# ref forever under count-based pruning alone.
_MAX_SNAPSHOT_AGE_SEC = 48 * 3600.0

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


def _ref_name(worktree_name: str, worktree_path: str) -> str:
    """`worktree_name` (branch, or the directory basename for a detached
    HEAD) is for human readability only -- the actual uniqueness guarantee
    is an 8-hex-char hash of the worktree's own resolved absolute path,
    always appended. Code review, 2026-08-18: an earlier draft escaped only
    literal spaces in the branch name, so two worktrees on branches that
    differ solely by space-vs-underscore (both legal git ref components)
    would collide on the identical ref and silently overwrite each other's
    snapshot history. A path can never collide with another worktree's path
    by definition, so hashing it removes the whole collision class rather
    than trying to escape branch names more carefully."""
    safe = worktree_name.replace(" ", "_")
    digest = hashlib.sha1(str(Path(worktree_path).resolve()).encode()).hexdigest()[:8]
    return f"{_REF_PREFIX}/{safe}-{digest}"


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

        ref_base = _ref_name(info.branch or Path(path).name, path)
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
    """Prunes by count (keep the newest `_MAX_SNAPSHOTS_PER_WORKTREE`) AND
    independently by age (drop anything older than `_MAX_SNAPSHOT_AGE_SEC`,
    regardless of count) -- count-only pruning would keep exactly 1 ref
    forever for a worktree that goes dirty once and is never touched again,
    which is a real path a stray-secret's ref could sit on indefinitely
    (see the module docstring's own disclosure on this)."""
    try:
        result = _run(["git", "-C", worktree_path, "for-each-ref", "--format=%(refname) %(creatordate:unix)", ref_base])
        if result.returncode != 0:
            return
        # Timestamped refs only -- "latest" is a moving pointer, not part of
        # either prune rule.
        entries = [
            line.split(" ", 1)
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith(f"{ref_base}/latest ")
        ]
        entries.sort(key=lambda kv: int(kv[1]))  # oldest first

        now = time.time()
        to_delete = {refname for refname, ts in entries if (now - int(ts)) > _MAX_SNAPSHOT_AGE_SEC}
        excess = len(entries) - _MAX_SNAPSHOTS_PER_WORKTREE
        to_delete.update(refname for refname, _ in entries[: max(0, excess)])

        for refname in to_delete:
            _run(["git", "-C", worktree_path, "update-ref", "-d", refname])
    except Exception:
        pass


def _run_worker_scan() -> None:
    """The actual scan -- confirmed live to take ~20s across this repo's
    real ~258-worktree fleet (16-way parallel `git status`/`add`/`write-
    tree`/`commit-tree` calls per dirty worktree), well past what a
    Claude-Code hook timeout should ever block on, and only growing as the
    worktree count does. Always invoked as a detached background process
    (see main()), never inline in the hook's own blocking execution.

    Scans ALL worktrees `list_worktrees()` returns, including the main/
    shared checkout -- NOT filtered by `is_main` the way prune/status
    tooling correctly filters it (main is never a prune candidate, but it
    is absolutely a snapshot candidate; see the module docstring's own
    account of this)."""
    try:
        worktrees = list_worktrees()
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
    fleet size.

    The check-then-touch throttle decision is guarded by a non-blocking
    `fcntl.flock` (see module docstring's "THROTTLED, WITH A REAL LOCK"
    section) -- kernel-atomic, so two near-simultaneous Stop events cannot
    both observe "due" and each launch their own background worker. The
    marker is touched only once `Popen` has actually launched the worker
    without raising -- a launch failure leaves the marker untouched so the
    next qualifying Stop event gets a real retry instead of the whole fleet
    going unprotected for a silent `_MIN_SCAN_INTERVAL_SEC`.
    """
    try:
        _SCAN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = open(_SCAN_LOCK_PATH, "w")
    except OSError:
        return
    try:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return  # another session's Stop hook is already deciding right now
        if not _should_scan_now():
            return
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from this process's session/controlling terminal
            )
        except Exception:
            return  # launch failed -- do NOT touch the marker; let the next Stop event retry
        _touch_scan_marker()
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


if __name__ == "__main__":
    try:
        if "--worker" in sys.argv[1:]:
            _run_worker_scan()
        else:
            main()
    except Exception:
        pass
