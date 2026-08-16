"""Cross-container advisory lock protecting Orion's FCC sandbox during a turn.

Two containers write to the same checkout (``/mnt/orion-fcc/repo``):

- ``orion-athena-hub`` spawns claude in-process via ``fcc_claude_bridge``
- ``orion-athena-harness-governor`` spawns claude via ``orion/harness/fcc_motor.py``,
  driven by ``orion:harness:run:request`` (autonomy, endogenous runs, and — confirmed
  live on 2026-08-14 — ordinary unified-chat turns, which dispatch through the
  governor's bus RPC path, not hub's in-process bridge)

Hub's sandbox sync used to gate on ``fcc_claude_bridge.active_turns()``, a dict local
to the *hub process*. It cannot see a governor turn, and since the live chat path is
the governor path, it was empty during essentially every real turn. That was survivable
only while the sync refused to touch a dirty tree; once the sync gained a
rescue-and-reset path, a browser refresh during any turn could stash a running agent's
edits and move the tree out from under it mid-write.

A dict in one process cannot coordinate two containers. A file lock on the shared
mount can: ``flock`` is kernel-level and applies to the inode, so both containers
contend correctly on the same bind-mounted file.

Turns take a **shared** lock — several may run concurrently, they are not exclusive
with each other. The sync takes an **exclusive**, non-blocking lock: it proceeds only
when no turn is in flight anywhere, and otherwise reports that rather than waiting.

The lock file lives beside the repo, never inside it, so ``git clean -fd`` during a
sync cannot delete the very lock coordinating that sync.
"""

from __future__ import annotations

import fcntl
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("orion-fcc.turn_lock")

_LOCK_FILENAME = ".fcc-turn.lock"


def turn_lock_path(workspace: str | os.PathLike[str] | None) -> Optional[Path]:
    """Locate the lock file for ``workspace``: a sibling of the checkout.

    Sibling, not child: the sync runs ``git clean -fd`` inside the workspace, which
    would delete an untracked lock file living there.
    """
    if not workspace:
        return None
    path = Path(workspace)
    parent = path.parent
    if not parent.is_dir():
        return None
    return parent / _LOCK_FILENAME


@contextmanager
def _locked(workspace: str | os.PathLike[str] | None, flags: int) -> Iterator[bool]:
    """Acquire ``flags`` on the workspace lock. Yields whether it was acquired.

    Never raises: a lock that cannot be created (missing mount, read-only parent,
    permissions) must not take down a turn or a WebSocket connect. It degrades to
    the pre-lock behaviour, which is what the caller already handled.
    """
    lock_path = turn_lock_path(workspace)
    if lock_path is None:
        yield False
        return

    handle = None
    acquired = False
    try:
        handle = open(lock_path, "a+")
        fcntl.flock(handle.fileno(), flags)
        acquired = True
        yield True
    except BlockingIOError:
        # Expected contention, not an error: someone else holds it.
        yield False
    except OSError as exc:
        logger.warning("fcc_turn_lock_unavailable path=%s err=%s", lock_path, exc)
        yield False
    finally:
        if handle is not None:
            try:
                if acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()


@contextmanager
def turn_in_progress(workspace: str | os.PathLike[str] | None) -> Iterator[bool]:
    """Hold a shared lock for the duration of an FCC turn.

    Shared so concurrent turns do not block each other; it only excludes the sync's
    exclusive acquire. Blocking: a turn should wait out an in-progress sync rather
    than start against a tree that is mid-reset. Syncs are seconds at most.
    """
    with _locked(workspace, fcntl.LOCK_SH) as acquired:
        yield acquired


@contextmanager
def sync_exclusive(workspace: str | os.PathLike[str] | None) -> Iterator[bool]:
    """Hold an exclusive lock across a sandbox sync, or yield False immediately.

    Non-blocking on purpose: a browser refresh during a long turn should report
    "a turn is running" and move on, not stall the connect handler.
    """
    with _locked(workspace, fcntl.LOCK_EX | fcntl.LOCK_NB) as acquired:
        yield acquired
