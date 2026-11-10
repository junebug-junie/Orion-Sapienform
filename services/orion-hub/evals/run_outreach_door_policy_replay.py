#!/usr/bin/env python3
"""Replay, read-only: the last N days of messages Juniper got, through PR A's door policy.

    PYTHONPATH=.:services/orion-hub .venv/bin/python -P \
        services/orion-hub/evals/run_outreach_door_policy_replay.py [--days 7] [--inventory-days 14] [--dsn ...]

For every ``sent`` row in ``endogenous_outreach_decisions``:

- ``self_finding_outreach``: retired, never sent.
- ``curiosity_outreach``: the run's ``line`` and its taken focus (``curiosity_focus_request``) through
  ``outreach_door_policy.curiosity_reaches_juniper``.
- background tick (no ``source``): its recorded daydream (``grounding.daydream_age_sec`` -> the
  ``reverie_visual_chain`` row it read -> ``daydream_content_id``). No daydream: nothing to speak
  from (tension alone is a trigger, not a topic). A daydream a kept tick already used in the last
  7 days: not used again.
- readings, office visits, strain report: kept (the readings rewrite changes words, not the count).

What this cannot replay: the ``Why Juniper`` gate (it depends on text Orion writes at send time), and
a tick composing at a different moment than it did. So the "after" numbers are an upper bound.

SELECT only. Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

_HUB = Path(__file__).resolve().parents[1]
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from scripts.outreach_door_policy import curiosity_reaches_juniper  # noqa: E402

TICK = "tick"
DAYDREAM_LEDGER_DAYS = 7

SENDS_SQL = """
SELECT d.decided_at, coalesce(d.result_json->>'source', 'tick') AS source,
       d.result_json->>'line' AS line, d.result_json->>'run_id' AS run_id,
       (d.result_json->'grounding'->>'daydream_age_sec')::float AS daydream_age_sec,
       coalesce((d.result_json->'grounding'->>'tension')::boolean, false) AS tension,
       left(regexp_replace(coalesce(c.response, ''), '\\s+', ' ', 'g'), 160) AS text
FROM endogenous_outreach_decisions d
LEFT JOIN chat_history_log c ON c.correlation_id = d.correlation_id
WHERE d.reason = 'sent' AND d.decided_at >= $1
ORDER BY d.decided_at
"""

FOCUS_SQL = """
SELECT offered_run_id, origin, chat_correlation_id
FROM curiosity_focus_request WHERE offered_run_id = ANY($1::text[])
"""

# The age was measured at gather time, which is up to one compose (agent lane + chat fallback,
# ~10 min) before ``decided_at``: the row it read was created in [decided_at - age - 900 s,
# decided_at - age]. The chain writes ~12 rows a day, so that window holds one row.
COMPOSE_SLACK_SEC = 900
DAYDREAM_SQL = """
SELECT created_at, chain_json->>'description' AS description
FROM reverie_visual_chain
WHERE created_at BETWEEN $1 AND $2 AND chain_json->>'description' IS NOT NULL
ORDER BY created_at DESC LIMIT 1
"""


def decide(
    sends: Iterable[Mapping[str, Any]],
    focus_by_run: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Each send with ``verdict`` (``kept`` or the reason it would not go out) and ``why``. Pure.

    ``sends`` oldest first; a tick carries ``daydream_id`` ("" = none recorded)."""
    out: list[dict[str, Any]] = []
    used: dict[str, datetime] = {}
    for s in sends:
        row = dict(s)
        src = row.get("source") or TICK
        verdict, why = "kept", ""
        if src == "self_finding_outreach":
            verdict, why = "retired", "self findings no longer reach Juniper"
        elif src == "curiosity_outreach":
            focus = focus_by_run.get(str(row.get("run_id") or "")) or {}
            ok, why = curiosity_reaches_juniper(
                row.get("line"), focus_origin=focus.get("origin"),
                focus_chat_correlation_id=focus.get("chat_correlation_id"))
            verdict = "kept" if ok else "not_for_juniper"
        elif src == TICK:
            did = str(row.get("daydream_id") or "")
            at = row["decided_at"]
            if not did:
                # Tension alone is a trigger, not a topic: has_talkable_content refuses it in the live
                # code (tension_without_content), and without tension the tick does not roll at all.
                verdict = "tension_without_content" if row.get("tension") else "no_material"
                why = "no daydream: nothing left to speak from"
            elif did in used and at - used[did] < timedelta(days=DAYDREAM_LEDGER_DAYS):
                verdict, why = "content_already_used", f"daydream {did} already used"
            else:
                why = f"daydream {did}"
                used[did] = at
        row["verdict"], row["why"] = verdict, why
        out.append(row)
    return out


def per_day(rows: Iterable[Mapping[str, Any]], *, kept_only: bool, days: int) -> dict[str, float]:
    counts: Counter = Counter()
    for r in rows:
        if kept_only and r["verdict"] != "kept":
            continue
        counts[r.get("source") or TICK] += 1
    return {k: round(v / days, 2) for k, v in sorted(counts.items())}


async def _daydream_id(conn: Any, at: datetime, age: Optional[float]) -> str:
    from scripts.endogenous_outreach import _clean_daydream
    from scripts.outreach_door_policy import daydream_content_id

    if age is None:
        return ""
    target = at - timedelta(seconds=float(age))
    row = await conn.fetchrow(DAYDREAM_SQL, target - timedelta(seconds=COMPOSE_SLACK_SEC), target + timedelta(seconds=5))
    if row is None:
        return ""
    return daydream_content_id(_clean_daydream(row["description"]) or "")


async def replay(conn: Any, *, days: int, now: datetime) -> list[dict[str, Any]]:
    rows = [dict(r) for r in await conn.fetch(SENDS_SQL, now - timedelta(days=days))]
    for r in rows:
        if r["source"] == TICK:
            r["daydream_id"] = await _daydream_id(conn, r["decided_at"], r["daydream_age_sec"])
    run_ids = sorted({r["run_id"] for r in rows if r.get("run_id")})
    focus = {str(f["offered_run_id"]): dict(f) for f in await conn.fetch(FOCUS_SQL, run_ids)}
    return decide(rows, focus)


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--inventory-days", type=int, default=14)
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
    now = datetime.now(timezone.utc)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        week = await replay(conn, days=a.days, now=now)
        inventory = await replay(conn, days=a.inventory_days, now=now)
    finally:
        await conn.close()
    out = {
        "days": a.days,
        "before_per_day": per_day(week, kept_only=False, days=a.days),
        "after_per_day": per_day(week, kept_only=True, days=a.days),
        "before_total_per_day": round(len(week) / a.days, 2),
        "after_total_per_day": round(sum(1 for r in week if r["verdict"] == "kept") / a.days, 2),
        "verdicts": dict(Counter(f"{r['source']}:{r['verdict']}" for r in week)),
        "curiosity_inventory": [
            {"at": r["decided_at"].strftime("%m-%d %H:%M"), "line": r["line"], "run_id": r["run_id"],
             "verdict": r["verdict"], "why": r["why"], "text": r["text"]}
            for r in inventory if r["source"] == "curiosity_outreach"],
        "tick_week": [
            {"at": r["decided_at"].strftime("%m-%d %H:%M"), "verdict": r["verdict"], "why": r["why"]}
            for r in week if r["source"] == TICK],
    }
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"last {a.days} days, messages/day by door")
    doors = sorted(set(out["before_per_day"]) | set(out["after_per_day"]))
    for d in doors:
        print(f"  {d:28s} before {out['before_per_day'].get(d, 0):5.2f}   after {out['after_per_day'].get(d, 0):5.2f}")
    print(f"  {'TOTAL':28s} before {out['before_total_per_day']:5.2f}   after {out['after_total_per_day']:5.2f}")
    print("\nverdicts:", json.dumps(out["verdicts"]))
    inv = out["curiosity_inventory"]
    kept = sum(1 for r in inv if r["verdict"] == "kept")
    print(f"\ncuriosity sends, last {a.inventory_days} days: {len(inv)} sent, {kept} would still be sent")
    by: dict[str, Counter] = defaultdict(Counter)
    for r in inv:
        by[r["line"] or "?"][r["verdict"]] += 1
    for line, c in sorted(by.items()):
        print(f"  line={line:13s} {dict(c)}")
    for r in inv:
        print(f"  {r['at']}  {r['line']:12s} {r['run_id']}  {r['verdict']:16s} {r['why']:22s} {r['text'][:90]}")
    print("\ntick sends, last week:")
    for r in out["tick_week"]:
        print(f"  {r['at']}  {r['verdict']:22s} {r['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
