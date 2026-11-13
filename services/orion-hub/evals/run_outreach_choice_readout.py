#!/usr/bin/env python3
"""Read-only readout of the outreach choice point (scripts/outreach_choice.py).

    PYTHONPATH=.:services/orion-hub .venv/bin/python -P \
        services/orion-hub/evals/run_outreach_choice_readout.py [--days 7] [--dsn ...] [--json]

Reads background-tick rows in ``endogenous_outreach_decisions`` that carry ``result_json.choice``
(the tick showed Orion topics) and reports what the personality spec rev 4 acceptance checks need:

- choices per day, and how many were published on ``orion:personality:choice``
- options shown per choice (a real choice needs at least one turned-down option)
- picks vs ``orion_declined`` vs unreadable (``no_pick`` / ``invalid_pick``)
- the share of picks that departed from rank 1 (M1 wants at least 25% across lanes)
- repeats: a subject sent twice inside 7 days (the 2026-11-11..12 failure; should be 0)

SELECT only. Exit 0 on a completed readout; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping

REPEAT_WINDOW = timedelta(days=7)

ROWS_SQL = """
SELECT decided_at, reason, result_json->'choice' AS choice
FROM endogenous_outreach_decisions
WHERE decided_at >= $1 AND result_json ? 'choice' AND coalesce(result_json->>'source', '') = ''
ORDER BY decided_at
"""


def _choice(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def summarize(rows: Iterable[Mapping[str, Any]], *, days: int) -> Dict[str, Any]:
    """Pure. ``rows``: ``{decided_at, reason, choice}`` in time order."""
    rows = [dict(r, choice=_choice(r.get("choice"))) for r in rows]
    picks = Counter(r["choice"].get("pick") or "?" for r in rows)
    shown = Counter(len(r["choice"].get("shown") or []) for r in rows)
    ranks = Counter(r["choice"].get("chosen_rank") for r in rows if r["choice"].get("pick") == "chosen")
    chosen = sum(ranks.values())
    departed = sum(n for rank, n in ranks.items() if rank not in (None, 1))
    published = sum(1 for r in rows if r["choice"].get("published"))
    repeats: List[Dict[str, Any]] = []
    last_sent: Dict[str, datetime] = {}
    for r in rows:
        subject = r["choice"].get("chosen_subject")
        if r.get("reason") != "sent" or not subject:
            continue
        prev = last_sent.get(subject)
        if prev is not None and r["decided_at"] - prev < REPEAT_WINDOW:
            repeats.append({"subject": subject, "first": prev.isoformat(), "again": r["decided_at"].isoformat()})
        last_sent[subject] = r["decided_at"]
    return {
        "days": days,
        "choices": len(rows),
        "choices_per_day": round(len(rows) / max(1, days), 2),
        "published": published,
        "picks": dict(picks),
        "options_shown": {str(k): v for k, v in sorted(shown.items())},
        "chosen_rank": {str(k): v for k, v in sorted(ranks.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "departed_from_rank_1": None if not chosen else round(departed / chosen, 3),
        "reasons": dict(Counter(r.get("reason") for r in rows)),
        "repeated_subjects_7d": repeats,
    }


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(a.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        await conn.execute("SET default_transaction_read_only = on")
        rows = await conn.fetch(ROWS_SQL, datetime.now(timezone.utc) - timedelta(days=a.days))
    finally:
        await conn.close()
    out = summarize([dict(r) for r in rows], days=a.days)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"outreach choice point, last {a.days} days: {out['choices']} choices ({out['choices_per_day']}/day), "
          f"{out['published']} published")
    print(f"  picks: {out['picks']}   reasons: {out['reasons']}")
    print(f"  options shown: {out['options_shown']}   chosen rank: {out['chosen_rank']}")
    print(f"  departed from rank 1: {out['departed_from_rank_1']}")
    print(f"  subjects sent twice inside 7 days: {len(out['repeated_subjects_7d'])}")
    for r in out["repeated_subjects_7d"]:
        print(f"    {r['subject']}  {r['first']} -> {r['again']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
