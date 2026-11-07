"""Daily reminiscence Patch 2: the overnight tick (app/reminiscence_poll.py) and the one durable-runs module
that reads the reminiscence ledger (app/reminiscence_ledger.py).

What must hold: one run a Denver day, only inside the overnight window, for the oldest week with no complete
episode (or the week today's first hop fixed); nothing when today's five hops are live; one run at a time; a
failed day gets one more run only while a whole step still fits before the window ends; the run's deadline is
the window's end; a dry run submits nothing; an unmounted ledger never reads as "nothing written".
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.reminiscence_ledger import ReminiscenceLedger
from app.reminiscence_poll import ReminiscenceTicker, next_run_id, window
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reminiscence_run import REMINISCENCE_HOPS, base_run_id, episode_of_run

# 2026-11-08 04:00 Denver (MST = UTC-7).
NOW = datetime(2026, 11, 8, 11, 0, tzinfo=timezone.utc)
EPISODE = date(2026, 11, 8)


def settings(**kw):
    base = dict(reminiscence_hour_local=3, reminiscence_end_hour_local=9, reminiscence_poll_sec=600.0,
                reminiscence_turn_timeout_sec=900.0)
    return SimpleNamespace(**{**base, **kw})


def hop_line(hop, *, episode="2026-11-08", start="2025-06-24", end="2025-06-30", n=0, supersedes=None):
    """ReminiscenceHopV1 as JSON (tests outside the atlas and dig do not import the ledger module)."""
    return json.dumps({"schema_version": "reminiscence_hop.v1", "hop_id": f"rm:{episode}:{hop}:{n:010x}",
                       "episode": episode, "hop": hop, "period_start": start, "period_end": end,
                       "note": "[reflection] A hop long enough to be saved by the dig.", "claims": [],
                       "written_at": f"{episode}T11:00:00+00:00", "supersedes": supersedes})


class FakePool:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

            async def fetchone(self):
                return self._rows[0] if self._rows else None

        class Conn:
            async def execute(self, sql, params):
                prefix = params[0].rstrip("%")
                if "count(*)" in sql:
                    return Cur([{"n": sum(1 for r in pool.rows if r["terminal"] is None and r["run_id"].startswith(prefix))}])
                return Cur([r for r in pool.rows if r["run_id"].startswith(prefix)])

        yield Conn()


class Submitted:
    def __init__(self, pool):
        self.pool, self.requests = pool, []

    async def __call__(self, request: DurableRunRequestV1):
        if any(r["run_id"] == request.run_id for r in self.pool.rows):
            raise ValueError("run_id already exists")
        self.requests.append(request)
        self.pool.rows.append({"run_id": request.run_id, "terminal": None, "created_at": NOW, "updated_at": NOW})


def ticker(tmp_path, *, rows=None, now=NOW, lines=(), **kw):
    for ln in lines:
        with (tmp_path / "reminiscence.jsonl").open("a") as fh:   # the ledger file name, as the dig writes it
            fh.write(ln + "\n")
    pool = FakePool(rows)
    submit = Submitted(pool)
    t = ReminiscenceTicker(settings=settings(**kw), pool=pool, submit=submit, ledger=ReminiscenceLedger(tmp_path),
                           now=lambda: now)
    return t, submit, pool


def test_the_window_is_denver_time_across_dst():
    lo, hi = window(date(2026, 11, 8), 3, 9)        # MST
    assert (lo.hour, hi.hour) == (10, 16)
    lo, hi = window(date(2026, 7, 8), 3, 9)         # MDT
    assert (lo.hour, hi.hour) == (9, 15)


def test_first_tick_submits_week_one_with_the_window_end_as_deadline(tmp_path):
    t, submit, _ = ticker(tmp_path)
    r = asyncio.run(t.tick_once())
    assert r.decision == "submitted" and r.run_id == "reminiscence-2026-11-08"
    [req] = submit.requests
    assert req.workflow == "reminiscence.episode" and req.admission.preferred_lane == "agent"
    assert req.admission.priority == "background"
    assert req.admission.deadline_at == datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)   # 09:00 Denver
    b = req.brief
    assert (b.episode, b.period_start, b.period_end) == (EPISODE, date(2025, 6, 24), date(2025, 6, 30))
    assert b.weeks_complete == 0 and b.weeks_total == 71 and b.timeout_sec == 900.0
    # A second tick submits nothing new: the run is in flight.
    r2 = asyncio.run(t.tick_once())
    assert r2.decision == "nothing_due" and len(submit.requests) == 1


def test_outside_the_window_nothing_happens(tmp_path):
    for now in (NOW.replace(hour=9, minute=59), NOW.replace(hour=16)):     # 02:59 and 09:00 Denver
        t, submit, _ = ticker(tmp_path, now=now)
        assert asyncio.run(t.tick_once()).decision == "outside_window" and submit.requests == []


def test_a_complete_week_is_skipped_and_an_unfinished_one_is_picked_again(tmp_path):
    week1 = [hop_line(h, episode="2026-11-06") for h in REMINISCENCE_HOPS]
    week2_part = [hop_line("what_it_was", episode="2026-11-07", start="2025-07-01", end="2025-07-07")]
    t, submit, _ = ticker(tmp_path, lines=week1 + week2_part)
    r = asyncio.run(t.tick_once())
    assert r.decision == "submitted" and (r.period_start, r.period_end) == ("2025-07-01", "2025-07-07")
    assert r.week_fixed_by_hop is False and r.weeks_complete == 1
    assert submit.requests[0].brief.weeks_complete == 1


def test_a_hop_written_today_fixes_the_week(tmp_path):
    t, _, _ = ticker(tmp_path, lines=[hop_line("where_i_am_now", start="2025-07-15", end="2025-07-21")])
    r = asyncio.run(t.tick_once())
    assert r.decision == "submitted" and r.period_start == "2025-07-15" and r.week_fixed_by_hop is True
    assert r.hops_written == ["where_i_am_now"]


def test_five_hops_written_today_means_no_run(tmp_path):
    t, submit, _ = ticker(tmp_path, lines=[hop_line(h) for h in REMINISCENCE_HOPS])
    assert asyncio.run(t.tick_once()).decision == "complete" and submit.requests == []


def test_a_failed_day_gets_one_more_run_only_while_a_step_still_fits(tmp_path):
    failed = {"run_id": "reminiscence-2026-11-08", "terminal": "failed", "created_at": NOW - timedelta(hours=1),
              "updated_at": NOW - timedelta(hours=1)}
    t, submit, _ = ticker(tmp_path, rows=[failed])
    assert asyncio.run(t.tick_once()).run_id == "reminiscence-2026-11-08-a2"
    late = NOW.replace(hour=15, minute=40)          # 08:40 Denver: 20 min left < 2 x 900 s
    t, submit, _ = ticker(tmp_path, rows=[dict(failed, updated_at=late - timedelta(hours=1))], now=late)
    assert asyncio.run(t.tick_once()).decision == "nothing_due" and submit.requests == []


def test_next_run_id_rules():
    lo, hi = window(EPISODE, 3, 9)
    kw = dict(now=NOW, deadline=hi, min_left_sec=1800)
    assert next_run_id(EPISODE, [], **kw) == "reminiscence-2026-11-08"
    row = lambda rid, term, ago=2.0: {"run_id": rid, "terminal": term, "updated_at": NOW - timedelta(hours=ago)}  # noqa: E731
    assert next_run_id(EPISODE, [row("reminiscence-2026-11-08", None)], **kw) is None
    assert next_run_id(EPISODE, [row("reminiscence-2026-11-08", "completed")], **kw) is None
    assert next_run_id(EPISODE, [row("reminiscence-2026-11-08", "failed", 0.1)], **kw) is None   # too soon
    two = [row("reminiscence-2026-11-08", "failed"), row("reminiscence-2026-11-08-a2", "failed")]
    assert next_run_id(EPISODE, two, **kw) is None                                               # 2 per day


def test_one_unfinished_run_of_any_day_blocks_a_new_one(tmp_path):
    stuck = {"run_id": "reminiscence-2026-11-07", "terminal": None, "created_at": NOW - timedelta(days=1),
             "updated_at": NOW - timedelta(days=1)}
    t, submit, _ = ticker(tmp_path, rows=[stuck])
    assert asyncio.run(t.tick_once()).decision == "waiting_in_flight" and submit.requests == []


def test_a_dry_run_builds_the_request_and_submits_nothing(tmp_path):
    t, submit, _ = ticker(tmp_path, now=NOW.replace(hour=20))   # 13:00 Denver, outside the window
    r = asyncio.run(t.tick_once(dry_run=True, ignore_window=True))
    assert r.decision == "would_submit" and r.dry_run and submit.requests == []
    assert t.last_request.brief.period_start == date(2025, 6, 24) and t.last is None   # health keeps real ticks only


def test_an_unmounted_ledger_never_reads_as_nothing_written(tmp_path):
    t, submit, _ = ticker(tmp_path / "absent")
    r = asyncio.run(t.tick_once())
    assert r.decision == "ledger_unreadable" and submit.requests == []


def test_the_ledger_hands_back_ids_and_rewrites(tmp_path):
    ledger = ReminiscenceLedger(tmp_path)
    (tmp_path / "reminiscence.jsonl").write_text("\n".join([
        hop_line("otherwise"), hop_line("otherwise", n=1, supersedes="rm:2026-11-08:otherwise:0000000000"),
        hop_line("otherwise", episode="2026-11-07", n=2), "not json"]) + "\n")
    assert ledger.hop_ids(EPISODE) == {"otherwise": "rm:2026-11-08:otherwise:0000000001"}
    assert ledger.receipt_ok(EPISODE, "otherwise", "rm:2026-11-08:otherwise:0000000000")   # superseded, still saved
    assert not ledger.receipt_ok(EPISODE, "what_it_was", "rm:2026-11-08:otherwise:0000000000")
    assert not ledger.receipt_ok(EPISODE, "otherwise", "rm:2026-11-07:otherwise:0000000002")
    assert ledger.rewrite_ok(EPISODE, "rm:2026-11-08:otherwise:0000000001")
    assert not ledger.rewrite_ok(EPISODE, "rm:2026-11-08:otherwise:0000000000")
    assert not ledger.episode_complete(EPISODE)


def test_run_ids_round_trip():
    assert base_run_id(EPISODE) == "reminiscence-2026-11-08"
    assert episode_of_run("reminiscence-2026-11-08-a2") == "2026-11-08" and episode_of_run("pr-field-note-1") is None


def test_the_settings_ship_the_tick_on_with_an_overnight_window(monkeypatch):
    from app.settings import Settings

    s = Settings(_env_file=None, POSTGRES_URI="postgresql://unused")
    assert s.reminiscence_enabled is True and (s.reminiscence_hour_local, s.reminiscence_end_hour_local) == (3, 9)
    assert s.reminiscence_turn_timeout_sec == 900.0 and s.reminiscence_ledger_dir == s.field_note_ledger_dir
    with pytest.raises(ValueError):
        Settings(_env_file=None, POSTGRES_URI="postgresql://unused", REMINISCENCE_HOUR_LOCAL=9,
                 REMINISCENCE_END_HOUR_LOCAL=3)
