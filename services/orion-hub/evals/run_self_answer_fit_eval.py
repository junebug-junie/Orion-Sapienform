#!/usr/bin/env python3
"""Live eval: are Orion's mirrored self answers whole?

Reads self_concept_history (read-only) and reports, for rows written since
--since:

  * at_cap_mid_word   rows exactly SELF_DEFINITION_TEXT_CAP long that end inside a
                      word -- the signature of the old blind cut. Must be 0.
  * full_over_budget  rows holding the full text because no fit was stored
                      (readers cut these at a sentence).
  * within_budget     the rest: answers that fit, and fit-hop rewrites.

Fit-hop outcomes are in Hub's log (`curiosity_self_fit_stored` /
`_refused` / `_skipped`), not in this table.

    python services/orion-hub/evals/run_self_answer_fit_eval.py --since 2026-11-04

Also reports at_cap_mid_word_all_time (any date), which must be 0 once the
backfill has run.

Env: POSTGRES_URI (or --pg-dsn). Exit 0 = no blind cuts; 1 = at least one;
2 = could not run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone

CAP = 1200
SQL = (
    "SELECT entry_id, concept_id, created_at, length(content) AS n, right(content, 1) AS last "
    "FROM self_concept_history WHERE produced_by = 'curiosity_self_inquiry' AND created_at >= $1 "
    "ORDER BY created_at"
)
# All time, not since --since: the backfill keeps created_at, so an old cut row
# that was never repaired is invisible to the windowed count.
ALL_TIME_SQL = (
    "SELECT count(*) FROM self_concept_history WHERE produced_by = 'curiosity_self_inquiry' "
    "AND length(content) = $1 AND right(content, 1) ~ '\\w'"
)


def classify(n: int, last: str) -> str:
    if n == CAP and re.match(r"\w", last or ""):
        return "at_cap_mid_word"
    if n > CAP:
        return "full_over_budget"
    return "within_budget"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", required=True, help="ISO date, e.g. 2026-11-04")
    parser.add_argument("--pg-dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("RECALL_PG_DSN"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.pg_dsn:
        print("missing --pg-dsn / POSTGRES_URI", file=sys.stderr)
        return 2
    since = datetime.fromisoformat(args.since)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    try:
        import asyncpg

        conn = await asyncpg.connect(args.pg_dsn)
        try:
            rows = await conn.fetch(SQL, since)
            all_time_bad = int(await conn.fetchval(ALL_TIME_SQL, CAP) or 0)
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"could not read self_concept_history: {exc}", file=sys.stderr)
        return 2
    counts: dict[str, int] = {"at_cap_mid_word": 0, "full_over_budget": 0, "within_budget": 0}
    bad: list[str] = []
    for r in rows:
        kind = classify(int(r["n"]), str(r["last"] or ""))
        counts[kind] += 1
        if kind == "at_cap_mid_word":
            bad.append(str(r["entry_id"]))
    report = {"since": since.isoformat(), "rows": len(rows), **counts,
              "blind_cut_entries_first_10": bad[:10], "at_cap_mid_word_all_time": all_time_bad}
    print(json.dumps(report, indent=None if args.json else 2))
    if not rows:
        print("UNVERIFIED: no rows written since --since", file=sys.stderr)
    return 1 if bad or all_time_bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
