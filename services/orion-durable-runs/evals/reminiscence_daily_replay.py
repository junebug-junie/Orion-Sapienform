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
7. each run's admission deadline is 09:00 Denver of its own day;
8. a week with no commit on main is never visited alone: every period ends on a week with a commit, and
   holds no earlier week with one (the replay's history is main's real shape: weeks 2-4, 6-10, 12 quiet).
9. across the day Orion's records start (2026-07-23, week 57), the shipped what_it_was_like hop prompt for
   each week 54-60 says "no lived record" before it, and from it on names `dig days <first> <end>` for
   exactly that period's lived days (week 57 starts at 07-23, partway through) and refuses a record from
   outside them.

Gate 3 counts weeks covered by complete periods: a joined period covers every week inside it.

Mutations (each must break at least one gate, or the eval itself is broken):

* "window ignored": the tick submits at any hour;
* "partial counts as visited": a week with any hop counts as complete (week 1 is skipped after a failed day);
* "no retry gate": a failed run is retried at once, any number of times;
* "quiet week visited alone": the tick picks the oldest open week whatever its commits (the old rule);
* "join runs past the first busy week": a quiet run is joined onto the second week with a commit;
* "lived hop reads the old arc pointer": the lived what_it_was_like prompt names `dig arc <id>`, not `dig days`;
* "dig days window not clamped": week 57's prompt reads `dig days` from 07-21, before the records start.

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

import app.reminiscence_graph as rgraph  # noqa: E402
import app.reminiscence_poll as poll  # noqa: E402
from app.reminiscence_ledger import PeriodRefused, ReminiscenceLedger, WeekPick  # noqa: E402
from orion.schemas.reminiscence_run import REMINISCENCE_HOPS, ReminiscenceRunBriefV1  # noqa: E402

START = date(2026, 10, 25)          # 21 days, across the 2026-11-01 DST change
DAYS = 21
TICK = timedelta(minutes=10)
FAIL_DAYS = {2, 3, 9}               # the run writes two hops and fails; day 3 also fails its -a2
CHAT_DAY = 12                       # Juniper asked in chat at 01:00: all five written before the window
FIRST = date(2025, 6, 24)
# main's real shape (counted on /mnt/orion-fcc/repo 2026-11-11): weeks 2-4, 6-10 and 12 had no commit.
QUIET = {FIRST + timedelta(days=7 * (w - 1)) for w in (2, 3, 4, 6, 7, 8, 9, 10, 12)}


def commits_in(start: date, end: date) -> int:
    return 0 if start in QUIET else 4


def weeks_in(start: date, end: date) -> list[date]:
    return [start + timedelta(days=7 * i) for i in range(((end - start).days + 1) // 7)]


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
    ledger = ReminiscenceLedger(tmp, commits_in=commits_in)
    if mutation in ("quiet week visited alone", "join runs past the first busy week"):
        real_pick = ledger.week_for

        def week_for(episode):
            got = real_pick(episode)
            if got is None or got.fixed_by_hop:
                return got
            if mutation == "quiet week visited alone":
                end = got.period_start + timedelta(days=6)
            else:
                busy = [s for s in weeks_in(got.period_start, date(2026, 6, 1)) if s not in QUIET]
                end = busy[1] + timedelta(days=6) if got.quiet_weeks else got.period_end
            return WeekPick(got.period_start, end, False, got.weeks_complete, got.weeks_total)
        ledger.week_for = week_for
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
                try:
                    pick = ledger.week_for(local.date())
                except PeriodRefused:   # a mutation broke the ledger's order: the gates report it
                    pick = None
                if pick is not None:
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
    periods: dict = {}
    for ln in lines:
        by_ep.setdefault(ln["episode"], {})[ln["hop"]] = ln["period_start"]
        periods[ln["episode"]] = (date.fromisoformat(ln["period_start"]), date.fromisoformat(ln["period_end"]))
    complete_eps = [ep for ep, h in by_ep.items() if set(h) == set(REMINISCENCE_HOPS)]
    covered = sorted(w for ep in complete_eps for w in weeks_in(*periods[ep]))
    weeks = [FIRST + timedelta(days=7 * i) for i in range(len(covered))]
    if covered != weeks:
        fails.append(f"3 order: covered weeks {[w.isoformat() for w in covered]} != {[w.isoformat() for w in weeks]}")
    for ep, (s, e) in sorted(periods.items()):
        inside = weeks_in(s, e)
        if inside[-1] in QUIET:
            fails.append(f"8 quiet: {ep} visited {s}..{e}, which ends on a week with no commit (a quiet week alone)")
        elif any(w not in QUIET for w in inside[:-1]):
            fails.append(f"8 quiet: {ep} visited {s}..{e}, which joins past a week with a commit")
    if len(covered) < 13:
        fails.append(f"8 quiet: the replay covered only {len(covered)} weeks, not past the quiet weeks 2-12")
    unfinished = [ep for ep, h in by_ep.items() if set(h) != set(REMINISCENCE_HOPS)]
    for ep in unfinished:
        nxt = (date.fromisoformat(ep) + timedelta(days=1)).isoformat()
        if nxt in by_ep and next(iter(by_ep[nxt].values())) != next(iter(by_ep[ep].values())):
            fails.append(f"4 retry week: {ep} left {next(iter(by_ep[ep].values()))} open, {nxt} visited another")
    chat = (START + timedelta(days=CHAT_DAY))
    if chat in per_day:
        fails.append(f"5 chat day: {chat} got a run")
    return fails


RECORDS_FROM = date(2026, 7, 23)  # hardcoded: a mutation of the module's constant must not move the oracle


def lived_gate(mutation: str | None = None) -> list[str]:
    """Gate 9: the shipped hop prompt for weeks 54-60, the stretch where Orion's records begin."""
    real = rgraph._what_to_read
    if mutation == "lived hop reads the old arc pointer":
        def what(hop, brief):
            if hop == "what_it_was_like" and brief.period_end >= RECORDS_FROM:
                return "what that week was like (`python -P -m orion.dig arc <id>`). Needs [lived_record]."
            return real(hop, brief)
        rgraph._what_to_read = what
    elif mutation == "dig days window not clamped":
        def what(hop, brief):
            got = real(hop, brief)
            return got.replace(f"days {RECORDS_FROM}", f"days {brief.period_start}") if hop == "what_it_was_like" else got
        rgraph._what_to_read = what
    fails = []
    try:
        for week in range(54, 61):
            start = FIRST + timedelta(days=7 * (week - 1))
            end = start + timedelta(days=6)
            b = ReminiscenceRunBriefV1.model_validate({
                "reminiscence": "reminiscence.episode", "episode": "2026-12-20", "period_start": start.isoformat(),
                "period_end": end.isoformat(), "weeks_complete": week - 1, "weeks_total": 77, "timeout_sec": 900})
            p = rgraph.build_hop_prompt(b, "what_it_was_like")
            if end < RECORDS_FROM:
                if "no lived record" not in p or "orion.dig days" in p:
                    fails.append(f"9 lived: week {week} ({start}..{end}) is before the records but its prompt "
                                 "does not say so")
                continue
            first = max(start, RECORDS_FROM)
            if f"python -P -m orion.dig days {first} {end}`" not in p:
                fails.append(f"9 lived: week {week} ({start}..{end}) prompt does not read `dig days {first} {end}`")
            if f"outside {first}..{end} is refused" not in p:
                fails.append(f"9 lived: week {week} prompt does not say a record outside {first}..{end} is refused")
    finally:
        rgraph._what_to_read = real
    return fails


def main() -> int:
    sim = simulate()
    fails = gates(sim) + lived_gate()
    days = len({e["episode"] for e in sim["log"]})
    complete = sorted({(ln["period_start"], ln["period_end"]) for ln in sim["lines"]})
    joined = [f"{s}..{e}" for s, e in complete if (date.fromisoformat(e) - date.fromisoformat(s)).days > 6]
    print(f"oracle: {len(sim['log'])} runs over {days} days, {len(sim['lines'])} hop lines, periods touched "
          f"{len(complete)}, joined {joined}")
    for f in fails:
        print("  FAIL", f)
    ok = not fails
    print("oracle:", "PASS" if ok else "FAIL")
    for m in ("window ignored", "partial counts as visited", "no retry gate", "quiet week visited alone",
              "join runs past the first busy week"):
        got = gates(simulate(m))
        caught = bool(got)
        print(f"mutation {m!r}: {'CAUGHT' if caught else 'MISSED'}" + (f" ({got[0][:90]})" if got else ""))
        ok = ok and caught
    for m in ("lived hop reads the old arc pointer", "dig days window not clamped"):
        got = lived_gate(m)
        print(f"mutation {m!r}: {'CAUGHT' if got else 'MISSED'}" + (f" ({got[0][:90]})" if got else ""))
        ok = ok and bool(got)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
