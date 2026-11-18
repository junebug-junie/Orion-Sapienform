"""Consult source replay: how many consults the last N days would have started, read-only.

    PYTHONPATH=services/orion-hub:. .venv/bin/python -P \
        services/orion-hub/evals/run_consult_source_replay.py [--days 14] [--dsn ...] [--json]

Source: ``endogenous_outreach_decisions`` rows with ``reason='not_for_juniper'`` -- curiosity runs
where Orion wanted to reach Juniper about their own plumbing and the door policy (#3177) held it
back. From 2026-11-11 each such run offers one consult (``source_ref = curiosity_run:<run_id>``).

The loop runs one consult at a time, ticks every ``HUB_CONSULT_TICK_SEC`` and lets at most
``HUB_CONSULT_WEEKLY_BRIEF_CAP`` briefs through per 7 days; a pending source expires after
``HUB_CONSULT_SOURCE_MAX_AGE_HOURS``. This replay counts the offers per day and the upper bound on
briefs (cap x weeks). It cannot replay endings: those depend on what Orion and Claude say.

Kill criterion (design doc): if fewer than 1 offer a week arrives, the curiosity source is too thin
to matter and the other two sources (self-monitor, stuck prior) carry the feature; if more than ~10 a
day arrive, most expire unstarted and the source needs a selection rule.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--weekly-cap", type=int, default=3)
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(a.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"UNVERIFIED: cannot reach Postgres ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    try:
        rows = await conn.fetch(
            "SELECT created_at::date AS day, result_json->>'run_id' AS run_id, result_json->>'line' AS line "
            "FROM endogenous_outreach_decisions WHERE reason = 'not_for_juniper' "
            "AND created_at >= now() - make_interval(days => $1) ORDER BY created_at",
            a.days,
        )
    finally:
        await conn.close()
    runs = {r["run_id"] for r in rows if r["run_id"]}
    per_day = Counter(str(r["day"]) for r in rows)
    per_line = Counter(r["line"] or "?" for r in rows)
    weeks = max(1.0, a.days / 7.0)
    report = {
        "days": a.days,
        "not_for_juniper_rows": len(rows),
        "distinct_runs_offered": len(runs),
        "offers_per_day": round(len(runs) / a.days, 2),
        "per_day": dict(sorted(per_day.items())),
        "per_line": dict(per_line),
        "brief_upper_bound": int(min(len(runs), a.weekly_cap * weeks)),
    }
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Last {a.days} days: {report['distinct_runs_offered']} curiosity runs kept from Juniper "
              f"({report['offers_per_day']}/day) would each offer one consult.")
        print(f"By line: {dict(per_line)}")
        print(f"Briefs reaching Juniper: at most {report['brief_upper_bound']} (cap {a.weekly_cap}/week); "
              "answered consults reach no one.")
        print(f"Per day: {report['per_day']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
