#!/usr/bin/env python3
"""SessionStart hook: surfaces a one-line worktree-hygiene summary (total /
merged counts) as additionalContext at the start of every session, so that
visibility doesn't depend on anyone remembering to run `graphify prs
--worktrees` or `worktree_status.py` by hand.

Imports worktree_lib directly rather than subprocessing to
worktree_status.py -- this used to shell out to a second python3 process
purely to get a one-line string, adding a full duplicate interpreter
startup (measured ~50ms) on every firing, including every subagent spawn.

Fails silently (no output) on any error -- never blocks session start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worktree_lib import WorktreeLibError, mergeable_worktrees  # noqa: E402


def main() -> None:
    try:
        worktrees, merged_set = mergeable_worktrees()
    except WorktreeLibError:
        return

    merged_count = sum(1 for w in worktrees if w.branch in merged_set)
    summary = (
        f"worktrees: {len(worktrees)} total, {merged_count} merged into origin/main "
        f"(candidates for: python3 scripts/prune_merged_worktrees.py)."
    )

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": summary,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
