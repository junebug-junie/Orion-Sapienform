#!/usr/bin/env python3
"""Live replay, read-only: what the PASS fix and reading spacing would have done to the messages
Orion really delivered to Juniper.

    PYTHONPATH=.:services/orion-hub .venv/bin/python services/orion-hub/evals/run_outreach_pass_spacing_replay.py \
        [--days 30] [--spacing-sec 5400] [--dsn postgresql://...] [--json]

1. PASS: every delivered outreach message (decision row ``sent`` joined to its chat_message row) is
   checked with the production ``is_pass_response`` on Orion's own words (the recorded article
   appendix is cut off first). A hit is a decline that was delivered as a message.
2. Spacing: the reading-finding messages that remain are replayed in their real order with the
   production wait rule (a message waits until ``spacing`` after the previous one). Reports the
   most messages in any 60 minutes before and after, and the longest delay a message would get.

SELECT only. Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_HUB = Path(__file__).resolve().parents[1]
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

READING = "reading_finding_outreach"
APPENDIX_MARKERS = ("\n\nWhat the piece says -- ", "\n\nSource: ")

SENT_SQL = """
SELECT d.decided_at, coalesce(d.result_json->>'source', 'endogenous') AS source, m.content
FROM endogenous_outreach_decisions d
JOIN chat_message m ON m.correlation_id = d.result_json->>'correlation_id' AND m.role = 'assistant'
WHERE d.reason = 'sent' AND d.decided_at >= now() - make_interval(days => $1)
ORDER BY d.decided_at
"""


def orion_words(content: str) -> str:
    """The delivered text minus the recorded appendix (``offer_message`` joins them with a blank line)."""
    text = str(content or "")
    for marker in APPENDIX_MARKERS:
        text = text.split(marker, 1)[0]
    return text


def spaced(times: list[datetime], spacing_sec: float) -> list[datetime]:
    """Send times under the spacing rule, in the original order. Pure."""
    out: list[datetime] = []
    for t in times:
        if out and spacing_sec > 0:
            t = max(t, out[-1] + timedelta(seconds=spacing_sec))
        out.append(t)
    return out


def max_in_window(times: list[datetime], window: timedelta = timedelta(hours=1)) -> int:
    best, start = 0, 0
    for end, t in enumerate(times):
        while t - times[start] > window:
            start += 1
        best = max(best, end - start + 1)
    return best


def replay(rows: list[dict], spacing_sec: float) -> dict:
    from scripts.endogenous_outreach import is_pass_response

    by_source: dict[str, dict[str, int]] = {}
    real_reading: list[datetime] = []
    for row in rows:
        src = str(row["source"])
        passed = is_pass_response(orion_words(row["content"]))
        stats = by_source.setdefault(src, {"delivered": 0, "now_pass": 0})
        stats["delivered"] += 1
        stats["now_pass"] += int(passed)
        if src == READING and not passed:
            real_reading.append(row["decided_at"])
    after = spaced(real_reading, spacing_sec)
    delays = [(b - a).total_seconds() for a, b in zip(real_reading, after)]
    return {
        "by_source": by_source,
        "reading_real_messages": len(real_reading),
        "reading_max_per_hour_before": max_in_window(real_reading),
        "reading_max_per_hour_after": max_in_window(after),
        "reading_max_delay_min": round(max(delays, default=0.0) / 60.0, 1),
        "spacing_sec": spacing_sec,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--spacing-sec", type=float, default=5400.0)
    parser.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(args.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {exc}", file=sys.stderr)
        return 2
    try:
        rows = [dict(r) for r in await conn.fetch(SENT_SQL, args.days)]
    finally:
        await conn.close()
    result = replay(rows, args.spacing_sec)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for src, s in sorted(result["by_source"].items()):
            print(f"{src}: delivered={s['delivered']} would_now_be_pass={s['now_pass']}")
        print(f"reading real messages={result['reading_real_messages']} "
              f"max/hour before={result['reading_max_per_hour_before']} "
              f"after={result['reading_max_per_hour_after']} "
              f"longest delay={result['reading_max_delay_min']} min (spacing {args.spacing_sec:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
