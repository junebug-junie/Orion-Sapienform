#!/usr/bin/env python3
"""Reconciled view of every worktree for this repo, regardless of which
location convention it uses (sibling directory, .worktrees/, .claude/
worktrees/agent-<id>), with merge status, open-PR status, and disk size.

Usage:
    python3 scripts/worktree_status.py              # full table
    python3 scripts/worktree_status.py --summary     # one-line counts only, no PR/disk lookups
    python3 scripts/worktree_status.py --stale-only   # only merged-with-no-open-PR rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worktree_lib import (  # noqa: E402
    WorktreeLibError,
    all_open_prs,
    dir_size_bytes,
    human_size,
    mergeable_worktrees,
    parallel_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="one-line summary only")
    parser.add_argument("--stale-only", action="store_true", help="only rows merged with no open PR")
    parser.add_argument("--base", default="origin/main", help="branch to compare merge status against")
    args = parser.parse_args()

    try:
        worktrees, merged_set = mergeable_worktrees(base=args.base)
    except WorktreeLibError as exc:
        print(f"[worktree-status] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        # No PR or disk-size lookups here on purpose -- this path runs on
        # every SessionStart, and even a single batched `gh pr list` call
        # is unnecessary work for a one-line count.
        merged_count = sum(1 for w in worktrees if w.branch in merged_set)
        print(
            f"worktrees: {len(worktrees)} total, {merged_count} merged into {args.base} "
            f"(candidates for: python3 scripts/prune_merged_worktrees.py)."
        )
        return 0

    prs = all_open_prs()
    if prs is None:
        if args.stale_only:
            print(
                "[worktree-status] ERROR: could not fetch open PR data (gh call failed) -- "
                "refusing to report --stale-only results, since a merged worktree with an "
                "open PR that failed to be detected would look identical to a genuinely "
                "stale one.",
                file=sys.stderr,
            )
            return 1
        print("[worktree-status] WARNING: could not fetch open PR data; PR column will show '?'.", file=sys.stderr)

    rows = []
    for w in worktrees:
        merged = w.branch in merged_set
        pr = prs.get(w.branch) if (prs is not None and w.branch) else None
        rows.append((w, merged, pr, prs is None))

    if args.stale_only:
        rows = [r for r in rows if r[1] and r[2] is None]

    if not rows:
        print("No worktrees to report (or --stale-only found none).")
        return 0

    sizes = parallel_map(lambda r: dir_size_bytes(r[0].path), rows)

    print(f"{'PATH':<60} {'BRANCH':<45} {'MERGED':<7} {'PR':<20} {'SIZE':>8}")
    for (w, merged, pr, pr_unknown), size in zip(rows, sizes):
        if pr_unknown:
            pr_label = "?"
        else:
            pr_label = f"#{pr['number']} {pr['state']}" if pr else "-"
        print(
            f"{str(w.path):<60} {str(w.branch or '(detached)'):<45} "
            f"{'yes' if merged else 'no':<7} {pr_label:<20} {human_size(size):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
