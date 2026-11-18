#!/usr/bin/env python3
"""Read-only readout of the outreach choice point (scripts/outreach_choice.py).

    PYTHONPATH=services/orion-hub:. .venv/bin/python -P \
        services/orion-hub/evals/run_outreach_choice_readout.py [--days 7] [--dsn ...] [--json] [--candidates]

Reads background-tick rows in ``endogenous_outreach_decisions`` that carry ``result_json.choice``
(the tick showed Orion topics) and reports what the personality spec rev 4 acceptance checks need:

- choices per day, and how many were published on ``orion:personality:choice``
- options shown per choice (a real choice needs at least one turned-down option)
- picks vs ``orion_declined`` vs unreadable (``no_pick`` / ``invalid_pick``)
- the share of picks that departed from rank 1 (M1 wants at least 25% across lanes)
- repeats: a subject sent twice inside 7 days (the 2026-11-11..12 failure; should be 0)
- per topic kind (daydream, reading, person, chat_thread): times shown and times picked. Picked /
  shown is the kind's revealed-preference rate (spec rev 4 move 2's family endorsement is a rate
  per offer, never a count)

``--candidates``: instead of the readout, count the person and chat-thread topics that would exist
right now (``scripts/outreach_world_topics.py``, the same SQL and rules the tick runs), and print them.
The 7-day subject ledger is not applied here.

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

# Hub's ``scripts`` package, not the repo root's (``--candidates`` imports scripts.outreach_world_topics).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    shown_kinds: Counter = Counter()
    picked_kinds: Counter = Counter()
    for r in rows:
        shown_items = [x for x in (r["choice"].get("shown") or []) if isinstance(x, dict)]
        shown_kinds.update(str(x.get("kind") or "?") for x in shown_items)
        chosen_subject = r["choice"].get("chosen_subject")
        if r["choice"].get("pick") == "chosen" and chosen_subject:
            picked_kinds.update(str(x.get("kind") or "?") for x in shown_items
                                if x.get("subject") == chosen_subject)
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
        "kinds": {k: {"shown": shown_kinds[k], "picked": picked_kinds.get(k, 0),
                      "picked_per_shown": round(picked_kinds.get(k, 0) / shown_kinds[k], 3)}
                  for k in sorted(shown_kinds)},
    }


async def candidates(conn: Any, now: datetime, tz_name: str) -> Dict[str, Any]:
    """The person and chat-thread topics the tick's sources would build now (read-only)."""
    from scripts.outreach_world_topics import (
        CHAT_SQL,
        HISTORY_DAYS,
        JUNIPER_CHAT_SOURCES,
        PEOPLE_SQL,
        THREAD_LOOKBACK_DAYS,
        chat_thread_topics,
        person_topics,
    )

    people_rows = await conn.fetch(PEOPLE_SQL, now - timedelta(days=HISTORY_DAYS))
    chat_cutoff = (now - timedelta(days=THREAD_LOOKBACK_DAYS)).replace(tzinfo=None)
    chat_rows = await conn.fetch(CHAT_SQL, chat_cutoff, list(JUNIPER_CHAT_SOURCES))
    out: Dict[str, Any] = {}
    for kind, topics in (("person", person_topics([dict(r) for r in people_rows], now, tz_name=tz_name)),
                         ("chat_thread", chat_thread_topics([dict(r) for r in chat_rows], now, tz_name=tz_name))):
        out[kind] = {"count": len(topics),
                     "topics": [{"subject": t.subject, "version": t.version, "evidence_ref": t.evidence_ref,
                                 "observed_at": t.observed_at.isoformat() if t.observed_at else None,
                                 "option_text": t.option_text} for t in topics]}
    return out


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--candidates", action="store_true",
                    help="count the person / chat-thread topics that would exist now")
    ap.add_argument("--tz", default=os.environ.get("HUB_ENDOGENOUS_OUTREACH_TZ", "America/Denver"))
    a = ap.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(a.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        await conn.execute("SET default_transaction_read_only = on")
        if a.candidates:
            found = await candidates(conn, datetime.now(timezone.utc), a.tz)
        else:
            rows = await conn.fetch(ROWS_SQL, datetime.now(timezone.utc) - timedelta(days=a.days))
    finally:
        await conn.close()
    if a.candidates:
        if a.json:
            print(json.dumps(found, indent=2, default=str))
            return 0
        for kind, block in found.items():
            print(f"{kind}: {block['count']} candidate(s)")
            for t in block["topics"]:
                print(f"  {t['subject']} [{t['evidence_ref']}]")
                print(f"    {t['option_text']}")
        return 0
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
    for kind, n in out["kinds"].items():
        print(f"  {kind}: shown {n['shown']}, picked {n['picked']} ({n['picked_per_shown']})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
