"""Read-only dry run of the office reach-out against the live database. Sends nothing, writes nothing.

For each office visit in the window: when it would first have had a story (the earliest moment its
timeline reached MIN_ENTRIES entries over MIN_MINUTES), its novelty magnitude and salience as the want
producer would compute them now, and the exact prompt and camera notes the reach-out would offer.
Orion's own sentence is written by the unified turn at send time and is NOT produced here.

Also a rest-point check for the novelty magnitude: a visit's second half measured against its own first
half (does a visit that repeats itself read near 0?).

    PYTHONPATH=.:services/orion-hub python services/orion-hub/evals/run_office_reach_out_dry_run.py \
        --dsn postgresql://postgres:<pw>@127.0.0.1:55432/conjourney --since 2026-11-02T00:00:00Z
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from scripts.office_reach_out import (
    SESSIONS_SQL_NO_REACH, appendix, build_prompt, distinct_activities, has_story, novelty, plan_wants, session_from_row,
)


def first_story_at(session: dict, *, min_entries: int, min_minutes: float) -> datetime | None:
    """The earliest moment the visit had a story, replaying its timeline in time order."""
    start = session["started_at"]
    tl = session["timeline"]
    for k in range(min_entries, len(tl) + 1):
        at = datetime.fromisoformat(str(tl[k - 1]["at"]))
        t = max(at, start + timedelta(minutes=min_minutes))
        partial = {**session, "timeline": tl[:k], "status": "open", "ended_at": None}
        end = session["ended_at"]
        if end is not None and t > end:
            t = end
            partial = {**session, "timeline": tl[:k]}
        if has_story(partial, t, min_entries=min_entries, min_minutes=min_minutes):
            return t
    return None


async def main() -> None:
    import asyncpg

    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", default="2026-11-01T00:00:00Z")
    ap.add_argument("--session", default="", help="print the prompt for this session only")
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--min-entries", type=int, default=4)
    ap.add_argument("--min-minutes", type=float, default=3.0)
    a = ap.parse_args()
    tz = ZoneInfo(a.tz)
    now = datetime.now(timezone.utc)
    conn = await asyncpg.connect(a.dsn)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        since = datetime.fromisoformat(a.since.replace("Z", "+00:00"))
        sessions = [session_from_row(r) for r in await conn.fetch(SESSIONS_SQL_NO_REACH, since, now)]
    finally:
        await conn.close()
    plan = {p["session_id"]: p for p in plan_wants(sessions, now=now, lookback=now - since, min_entries=a.min_entries,
                                                     min_minutes=a.min_minutes)}
    print(f"{len(sessions)} office visits since {since.isoformat()}; with a story: {len(plan)}")
    for s in sessions:
        story_at = first_story_at(s, min_entries=a.min_entries, min_minutes=a.min_minutes)
        p = plan.get(s["session_id"])
        print(f"\n{s['session_id']}  {s['started_at']:%Y-%m-%d %H:%M:%S}Z -> "
              f"{s['ended_at']:%H:%M:%S}Z" if s["ended_at"] else f"\n{s['session_id']}  open")
        print(f"  who: {s['who_text']}  entries={len(s['timeline'])}  unknown={s['unknown_people']}  "
              f"first story at: {story_at.isoformat() if story_at else 'never'}")
        if p:
            print(f"  novelty={p['novelty']}  salience={p['salience']}  history_n={p['history_n']} "
                  f"({'no want: under 20 past visits' if p['salience'] is None else 'want posted'})")
        acts = distinct_activities(s["timeline"])
        if len(acts) >= 4:
            half = len(s["timeline"]) // 2
            first, second = distinct_activities(s["timeline"][:half]), distinct_activities(s["timeline"][half:])
            print(f"  rest-point check: second half vs first half novelty={novelty(second, first)} "
                  f"(whole visit vs itself={novelty(acts, acts)})")
        if (a.session and s["session_id"] == a.session) or (not a.session and story_at):
            at = story_at or now
            partial = {**s, "timeline": [e for e in s["timeline"] if datetime.fromisoformat(str(e["at"])) <= at]}
            if s["ended_at"] is None or at < s["ended_at"]:
                partial = {**partial, "status": "open", "ended_at": None}
            print(f"\n----- PROMPT (as it would be composed at {at.isoformat()}) -----")
            print(build_prompt(partial, tz, now=at))
            print("\n----- CAMERA NOTES delivered under Orion's message (record_appendix) -----")
            print(appendix(partial, tz, at))
            print("\n----- PROMPT if composed after the visit closed -----")
            print(build_prompt(s, tz, now=now))
            print("\n----- CAMERA NOTES after close -----")
            print(appendix(s, tz, now))


if __name__ == "__main__":
    asyncio.run(main())
