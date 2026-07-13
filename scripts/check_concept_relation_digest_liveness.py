#!/usr/bin/env python3
"""Liveness gate for `scripts/concept_relation_digest.py`.

Context: the digest closes the concept-relation decision loop (see
docs/superpowers/specs/2026-07-13-recall-followups-loop-retirement-saturation-gate-
spec.md, section 1) -- but it is a standalone script run on demand or via cron, not a
live service loop. Nothing detects if the cron entry dies, is silently dropped on a
host migration, or the job starts failing. This script is that detector.

Unlike scripts/check_activation_saturation.py (which reports a distribution and lets a
human judge whether it got worse), this one measures actual, unambiguous harm: how old
is the oldest undigested `memory_concept_relation_decisions` row right now. If the
digest job is running on schedule, that age never exceeds one cron interval by much. If
it stops running (dead cron, dropped crontab entry after a host swap, crashing job),
the backlog age grows without bound -- this is a real symptom, not a heartbeat file that
can go stale independently of the thing it claims to represent.

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/check_concept_relation_digest_liveness.py
    python scripts/check_concept_relation_digest_liveness.py --postgres-uri postgresql://...
    python scripts/check_concept_relation_digest_liveness.py --max-age-hours 3 --json

Exit codes: 0 = healthy (no backlog, or backlog younger than --max-age-hours).
            1 = stale backlog (digest job likely not running) -- FAIL.
            2 = could not run the check at all (bad config / query error).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_concept_relation_digest_liveness.py` puts scripts/
# on sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_QUERY = """
SELECT count(*) AS backlog, min(decided_at) AS oldest_pending
FROM memory_concept_relation_decisions
WHERE digested = false
"""


async def _query_backlog(postgres_uri: str) -> tuple[int, datetime | None]:
    import asyncpg

    conn = await asyncpg.connect(postgres_uri)
    try:
        row = await conn.fetchrow(_QUERY)
    finally:
        await conn.close()
    return int(row["backlog"]), row["oldest_pending"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help=(
            "fail if the oldest undigested decision is older than this many hours "
            "(default: 3.0 -- generous headroom over the documented 30-minute cron "
            "cadence in services/orion-memory-consolidation/README.md)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "check_concept_relation_digest_liveness: no --postgres-uri given and "
            "$POSTGRES_URI is unset. Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    try:
        backlog, oldest_pending = asyncio.run(_query_backlog(args.postgres_uri))
    except Exception as exc:
        print(f"check_concept_relation_digest_liveness: query failed -- {exc}", file=sys.stderr)
        return 2

    if backlog == 0 or oldest_pending is None:
        age_hours = 0.0
        stale = False
    else:
        age_hours = (datetime.now(timezone.utc) - oldest_pending).total_seconds() / 3600.0
        stale = age_hours > args.max_age_hours

    if args.json:
        print(json.dumps({
            "backlog": backlog,
            "oldest_pending_age_hours": age_hours,
            "max_age_hours": args.max_age_hours,
            "stale": stale,
        }))
    else:
        if backlog == 0:
            print("check_concept_relation_digest_liveness: OK -- no undigested decisions pending.")
        else:
            print(
                f"check_concept_relation_digest_liveness: {backlog} undigested decision(s) "
                f"pending, oldest is {age_hours:.2f}h old (threshold {args.max_age_hours:.2f}h)"
            )
        if stale:
            print(
                "STALE: the concept-relation digest does not appear to be running on "
                "schedule. Check the cron entry (crontab -l), the job's log "
                "(logs/orion-concept-relation-digest.log), and whether this host still "
                "has the crontab installed after any migration -- see "
                "services/orion-memory-consolidation/README.md, 'Scheduled maintenance'.",
                file=sys.stderr,
            )

    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
