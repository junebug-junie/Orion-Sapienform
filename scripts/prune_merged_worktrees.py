#!/usr/bin/env python3
"""Lists (and, with --yes, removes) worktrees whose branch is already merged
into origin/main.

Dry-run by default -- prints what *would* be removed. Pass --yes to actually
run `git worktree remove` on each one. Never passes --force to `git worktree
remove`: a worktree with uncommitted changes is left alone and reported as
skipped, not force-deleted, even if its branch is merged -- an agent (or a
human) may have left real work sitting there uncommitted.

Only removes the worktree directory, never the branch itself (`git branch
-d`/`-D`). Branch cleanup is a separate, more sensitive decision left to a
human or a future, explicitly-scoped tool.

Usage:
    python3 scripts/prune_merged_worktrees.py              # dry run
    python3 scripts/prune_merged_worktrees.py --yes         # actually remove
    python3 scripts/prune_merged_worktrees.py --base main   # different base
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worktree_lib import (  # noqa: E402
    WorktreeLibError,
    dir_size_bytes,
    human_size,
    mergeable_worktrees,
    parallel_map,
    repo_toplevel,
)
from agent_board_lib import board_config_from_env, close_presence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="actually remove, not just report")
    parser.add_argument("--base", default="origin/main", help="branch to compare merge status against")
    args = parser.parse_args()

    try:
        worktrees, merged_set = mergeable_worktrees(base=args.base)
        toplevel = repo_toplevel()
    except WorktreeLibError as exc:
        print(f"[prune-merged-worktrees] ERROR: {exc}", file=sys.stderr)
        return 1

    mergeable = [w for w in worktrees if w.branch in merged_set]

    if not mergeable:
        print(f"No worktrees found whose branch is merged into {args.base}.")
        return 0

    sizes = parallel_map(lambda w: dir_size_bytes(w.path), mergeable)

    total_bytes = 0
    removed, skipped = 0, 0
    for w, size in zip(mergeable, sizes):
        total_bytes += size or 0
        label = f"{w.path}  (branch: {w.branch}, {human_size(size)})"
        if not args.yes:
            print(f"[would remove] {label}")
            continue
        result = subprocess.run(
            ["git", "-C", toplevel, "worktree", "remove", str(w.path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[removed] {label}")
            removed += 1
            try:
                close_presence(board_config_from_env(), worktree_path=str(w.path.resolve()))
            except Exception as exc:
                print(f"[agent-board close skipped] {w.path} -- {exc}", file=sys.stderr)
        else:
            print(f"[skipped, not removed] {label} -- {result.stderr.strip()}")
            skipped += 1

    print()
    if args.yes:
        print(f"Removed {removed}, skipped {skipped} (uncommitted changes or other git refusal).")
    else:
        print(f"{len(mergeable)} worktree(s) merged into {args.base}, ~{human_size(total_bytes)} reclaimable.")
        print("Re-run with --yes to actually remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
