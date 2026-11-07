#!/usr/bin/env python3
"""Eval: the daily reminiscence tick over three simulated weeks of days. Deterministic, no database.

Runs the SHIPPED tick (``app.reminiscence_poll.ReminiscenceTicker``) every 10 min over 21 Denver days
(the DST change on 2026-11-01 included) against a REAL ledger file in a temp dir and a fake run registry.
Each submitted run "happens" by the script: on most days all five hops get written (the run completes);
on some days it fails after two hops (its -a2 retry writes the rest, or it stays unfinished); on one day
Juniper asks in chat first and all five are written before the window opens.

Gates (each one a property the design states):

1. no run is submitted outside 03:00-09:00 Denver, in MST and in MDT;
2. at most one run is unfinished at a time, and at most 2 runs a day;
3. weeks are visited oldest first with none skipped: the complete weeks are exactly weeks 1..N;
4. an unfinished week is picked again the next day (design: "an unfinished episode leaves its week open");
5. a day whose five hops were written in chat gets no run;
6. every run's brief names the week the dig would pick for that day (the ledger's week_for);
7. each run's admission deadline is 09:00 Denver of its own day.

Mutations (each must break at least one gate, or the eval itself is broken):

* "window ignored": the tick submits at any hour;
* "partial counts as visited": a week with any hop counts as complete (week 1 is skipped after a failed day);
* "no retry gate": a failed run is retried at once, any number of times.

    python -P services/orion-durable-runs/evals/reminiscence_daily_replay.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

import app.reminiscence_poll as poll  # noqa: E402
from app.reminiscence_ledger import ReminiscenceLedger, WeekPick  # noqa: E402
from orion.schemas.reminiscence_run import REMINISCENCE_HOPS  # noqa: E402

START = date(2026, 10, 25)          # 21 days, across the 2026-11-01 DST change
DAYS = 21
TICK = timedelta(minutes=10)
FAIL_DAYS = {2, 3, 9}               # the run writes two hops and fails; day 3 also fails its -a2
CHAT_DAY = 12                       # Juniper asked in chat at 01:00: all five written before the window


class Registry:
    """durable_admission_runs as a list; answers the tick's two queries."""

    def __init__(self):
        self.rows: list[dict] = []

    @asynccontextmanager
    async def connection(self):
        reg = self

        class Cur:
            def __init__(self, rows):
                self.rows = rows

            async def fetchall(self):
                return self.rows

            async def fetchone(self):
                return self.rows[0] if self.rows else None

        class Conn:
            async def execute(self, sql, params):
                prefix = params[0].rstrip("%")
                rows = [r for r in reg.rows if r["run_id"].startswith(prefix)]
                if "count(*)" in sql:
                    return Cur([{"n": sum(1 for r in rows if r["terminal"] is None)}])
                return Cur(rows)

        yield Conn()


def hop_line(hop, episode, period, n):
    return json.dumps({"schema_version": "reminiscence_hop.v1", "hop_id": f"rm:{episode}:{hop}:{n:010x}",
                       "episode": episode, "hop": hop, "period_start": period[0], "period_end": period[1],
                       "note": "[reflection] A hop written by the replay, long enough.", "claims": [],
                       "written_at": f"{episode}T12:00:00+00:00", "supersedes": None})


def simulate(mutation: str | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="reminiscence-replay-"))
    ledger = ReminiscenceLedger(tmp)
    if mutation == "partial counts as visited":
        real = ledger.week_for

        def week_for(episode):
            got = real(episode)
            if got is None or got.fixed_by_hop:
                return got
            lines = [json.loads(ln) for ln in ledger.path.read_text().splitlines()] if ledger.path.exists() else []
            seen = {ln["period_start"] for ln in lines}
            start = got.period_start
            while start.isoformat() in seen:
                start += timedelta(days=7)
            return WeekPick(start, start + timedelta(days=6), False, got.weeks_complete, got.weeks_total)
        ledger.week_for = week_for
    reg = Registry()
    clock = {"now": None}
    log: list[dict] = []
    counter = [0]

    def write(hops, episode, period):
        with ledger.path.open("a") as fh:
            for h in hops:
                counter[0] += 1
                fh.write(hop_line(h, episode, period, counter[0]) + "\n")

    async def submit(req):
        if any(r["run_id"] == req.run_id for r in reg.rows):
            raise ValueError("duplicate")
        b = req.brief
        expected = ledger.week_for(b.episode)
        log.append({"run_id": req.run_id, "at": clock["now"], "episode": b.episode, "deadline": req.admission.deadline_at,
                    "period": (b.period_start, b.period_end), "dig_week": (expected.period_start, expected.period_end)})
        day_i = (b.episode - START).days
        period = (b.period_start.isoformat(), b.period_end.isoformat())
        have = set(ledger.hop_ids(b.episode))
        todo = [h for h in REMINISCENCE_HOPS if h not in have]
        retry = req.run_id.endswith("-a2")
        if day_i in FAIL_DAYS and not retry or (day_i == 3 and retry):
            write(todo[:2] if not retry else [], b.episode.isoformat(), period)
            terminal = "failed"
        else:
            write(todo, b.episode.isoformat(), period)
            terminal = "completed"
        reg.rows.append({"run_id": req.run_id, "terminal": None, "created_at": clock["now"], "updated_at": clock["now"],
                         "_ends": clock["now"] + timedelta(minutes=40), "_terminal": terminal})

    settings = SimpleNamespace(reminiscence_hour_local=3, reminiscence_end_hour_local=9, reminiscence_poll_sec=600,
                               reminiscence_turn_timeout_sec=900.0)
    ticker = poll.ReminiscenceTicker(settings=settings, pool=reg, submit=submit, ledger=ledger,
                                     now=lambda: clock["now"])
    orig_window, orig_kw = poll.window, dict(poll.next_run_id.__kwdefaults__)
    if mutation == "window ignored":
        poll.window = lambda day, s, e: (datetime.combine(day, datetime.min.time(), tzinfo=poll.ZONE).astimezone(timezone.utc),
                                         datetime.combine(day + timedelta(days=1), datetime.min.time(),
                                                          tzinfo=poll.ZONE).astimezone(timezone.utc))
    if mutation == "no retry gate":
        poll.next_run_id.__kwdefaults__ = {"max_runs": 99, "retry_after_sec": 0.0}
    try:
        t = datetime.combine(START, datetime.min.time(), tzinfo=poll.ZONE).astimezone(timezone.utc)
        end = t + timedelta(days=DAYS)
        while t < end:
            clock["now"] = t
            local = t.astimezone(poll.ZONE)
            if (local.date() - START).days == CHAT_DAY and local.hour == 1 and local.minute == 0:
                pick = ledger.week_for(local.date())
                write(REMINISCENCE_HOPS, local.date().isoformat(), (pick.period_start.isoformat(), pick.period_end.isoformat()))
            for r in reg.rows:
                if r["terminal"] is None and t >= r["_ends"]:
                    r["terminal"], r["updated_at"] = r["_terminal"], t
            asyncio.run(ticker.tick_once())
            t += TICK
    finally:
        poll.window = orig_window
        poll.next_run_id.__kwdefaults__ = orig_kw
    lines = [json.loads(ln) for ln in ledger.path.read_text().splitlines()]
    return {"log": log, "lines": lines, "ledger": ledger}


def gates(sim: dict) -> list[str]:
    fails = []
    log, lines = sim["log"], sim["lines"]
    for e in log:
        local = e["at"].astimezone(poll.ZONE)
        if not 3 <= local.hour < 9:
            fails.append(f"1 window: {e['run_id']} submitted at {local.isoformat()}")
        if e["deadline"].astimezone(poll.ZONE).replace(tzinfo=None) != datetime.combine(e["episode"], datetime.min.time()).replace(hour=9):
            fails.append(f"7 deadline: {e['run_id']} deadline {e['deadline'].isoformat()}")
        if e["period"] != e["dig_week"]:
            fails.append(f"6 week: {e['run_id']} brief {e['period']} vs dig {e['dig_week']}")
    per_day: dict = {}
    for e in log:
        per_day.setdefault(e["episode"], []).append(e)
    if any(len(v) > 2 for v in per_day.values()):
        fails.append("2 runs per day: " + ", ".join(f"{k}:{len(v)}" for k, v in per_day.items() if len(v) > 2))
    by_ep: dict = {}
    for ln in lines:
        by_ep.setdefault(ln["episode"], {})[ln["hop"]] = ln["period_start"]
    complete = sorted({next(iter(h.values())) for h in by_ep.values() if set(h) == set(REMINISCENCE_HOPS)})
    weeks = [(date(2025, 6, 24) + timedelta(days=7 * i)).isoformat() for i in range(len(complete))]
    if complete != weeks:
        fails.append(f"3 order: complete weeks {complete} != {weeks}")
    unfinished = [ep for ep, h in by_ep.items() if set(h) != set(REMINISCENCE_HOPS)]
    for ep in unfinished:
        nxt = (date.fromisoformat(ep) + timedelta(days=1)).isoformat()
        if nxt in by_ep and next(iter(by_ep[nxt].values())) != next(iter(by_ep[ep].values())):
            fails.append(f"4 retry week: {ep} left {next(iter(by_ep[ep].values()))} open, {nxt} visited another")
    chat = (START + timedelta(days=CHAT_DAY))
    if chat in per_day:
        fails.append(f"5 chat day: {chat} got a run")
    return fails


def main() -> int:
    sim = simulate()
    fails = gates(sim)
    days = len({e["episode"] for e in sim["log"]})
    complete = len({ln["period_start"] for ln in sim["lines"]})
    print(f"oracle: {len(sim['log'])} runs over {days} days, {len(sim['lines'])} hop lines, weeks touched {complete}")
    for f in fails:
        print("  FAIL", f)
    ok = not fails
    print("oracle:", "PASS" if ok else "FAIL")
    for m in ("window ignored", "partial counts as visited", "no retry gate"):
        got = gates(simulate(m))
        caught = bool(got)
        print(f"mutation {m!r}: {'CAUGHT' if caught else 'MISSED'}" + (f" ({got[0][:90]})" if got else ""))
        ok = ok and caught
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
