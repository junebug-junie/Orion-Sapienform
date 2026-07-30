#!/usr/bin/env python3
"""Read-only replay of ``pr_lifecycle_delta()`` against this repo's real GitHub
PR history -- SSP §7's "measure before minting" discipline, and the design
spec's own named acceptance check
(docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md): verify the
real count exceeds ``MAX_DIGEST_INPUT_PRS`` (8) on a real window, proving this
domain isn't silently capped the way ``trim_github_compactor_input()``'s
LLM-facing item list is.

Requires a real, authenticated `gh` CLI (same one this repo's own PR workflow
already uses) -- this is intentionally not mocked, unlike the unit test suite.

Run:
    python scripts/analysis/measure_pr_lifecycle.py
    python scripts/analysis/measure_pr_lifecycle.py --owner junebug-junie --repo Orion-Sapienform --lookback-days 3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from orion.cognition.github_compactor.constants import MAX_DIGEST_INPUT_PRS  # noqa: E402
from orion.structural_mass.pr_lifecycle import (  # noqa: E402
    fetch_recent_prs,
    pr_lifecycle_delta,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="junebug-junie")
    parser.add_argument("--repo", default="Orion-Sapienform")
    parser.add_argument("--lookback-days", type=float, default=1.0)
    parser.add_argument("--fetch-limit", type=int, default=200)
    args = parser.parse_args()

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=args.lookback_days)

    prs = fetch_recent_prs(args.owner, args.repo, limit=args.fetch_limit)
    result = pr_lifecycle_delta(prs, since=since, until=until, fetch_limit=args.fetch_limit)

    print(f"Window: {since.isoformat()} .. {until.isoformat()} "
          f"({args.lookback_days} day(s)), fetched {len(prs)} PR record(s) "
          f"(fetch limit {args.fetch_limit})\n")
    print(f"submitted_count:            {result.submitted_count}  {result.submitted_numbers}")
    print(f"merged_count:                {result.merged_count}  {result.merged_numbers}")
    print(f"closed_without_merge_count:  {result.closed_without_merge_count}  {result.closed_without_merge_numbers}")
    print(f"possibly_truncated:          {result.possibly_truncated}")

    if result.possibly_truncated:
        print("\nFLAG: fetch may not reach far enough back for this window -- "
              "increase --fetch-limit and rerun before trusting these counts.")
        return 1

    max_count = max(result.submitted_count, result.merged_count, result.closed_without_merge_count)
    print(f"\n--- Acceptance check: exceeds MAX_DIGEST_INPUT_PRS ({MAX_DIGEST_INPUT_PRS}) ---")
    if max_count > MAX_DIGEST_INPUT_PRS:
        print(f"OK: max bucket count ({max_count}) exceeds the digest's {MAX_DIGEST_INPUT_PRS}-item "
              "cap -- this domain is not silently capped the way the LLM-facing digest is.")
        return 0

    print(f"Max bucket count ({max_count}) does not exceed {MAX_DIGEST_INPUT_PRS} in this window -- "
          "widen --lookback-days on a busier window to confirm this check, or accept this window "
          "genuinely had low PR volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
