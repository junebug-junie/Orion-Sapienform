#!/usr/bin/env python3
"""Archive duplicate/stale ProposedGoal rows in the autonomy goals graph.

Run once after Phase 0 deploy with ``--apply`` (default is dry-run).
"""
from __future__ import annotations

import argparse
import sys

from orion.autonomy.goal_archive import archive_subject_goals, archive_subjects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--subject", default="")
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Archive all subjects in AUTONOMY_GOAL_ARCHIVE_SUBJECTS (default orion,relationship)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    if args.all_subjects:
        summaries = archive_subjects(dry_run=dry_run)
        for summary in summaries:
            print(summary)
        return 0

    subject = args.subject.strip() or "orion"
    summary = archive_subject_goals(subject, dry_run=dry_run)
    print(
        f"subject={summary.get('subject')} candidates={summary.get('candidates')} "
        f"applied={summary.get('applied')} dry_run={summary.get('dry_run')} "
        f"rows_scanned={summary.get('rows_scanned')}"
    )
    if summary.get("error"):
        print(f"error={summary['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
