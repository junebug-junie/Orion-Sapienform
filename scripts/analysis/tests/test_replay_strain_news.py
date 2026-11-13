"""Strain news budget replay: reconstruction, projection and the failing exit."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import replay_strain_news as rp  # noqa: E402

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 10, 18, 12, 7, tzinfo=timezone.utc)  # 06:07 local


def _t(at, level, reasons, temp=30.0):
    return {"arousal_level": level, "since": at.isoformat(), "observed_at": at.isoformat(),
            "reasons": reasons, "cabinet_temp_c": temp}


def test_cabinet_hot_is_inferred_from_critical_temps_once_per_episode():
    transitions = [_t(T0, "strained", ["gpu_queue_sustained"])]
    temps = [(T0 + timedelta(minutes=m), 34.4 if 60 <= m < 90 or 150 <= m < 170 else 32.0)
             for m in range(0, 240, 1)]
    events = rp.replay_events(transitions, temps, until=T0 + timedelta(hours=4) - timedelta(minutes=1), tz=TZ)
    kinds = [(e.kind, e.at - T0) for e in events]
    assert ("onset", timedelta(0)) in kinds
    esc = [e for e in events if e.kind == "escalation"]
    assert len(esc) == 1 and esc[0].new_causes == ("cabinet_hot",) and esc[0].critical


def _ticks(start, n, winner_every=5):
    return [(start + timedelta(seconds=30 * i), "node:substrate.execution" if i % winner_every == 0 else None)
            for i in range(n)]


def test_projection_holds_at_most_ten_ticks_per_event_and_counts_displaced():
    transitions = [_t(T0, "strained", ["gpu_queue_sustained"])]
    events = rp.replay_events(transitions, [], until=T0 + timedelta(minutes=1), tz=TZ)
    assert len(events) == 1
    ticks = _ticks(T0 - timedelta(hours=1), 600)  # 5 h, one local day
    (day,) = rp.project(events, ticks, tz=TZ)
    assert day.held == 10
    assert day.displaced == 2      # ticks 120..129 after start; winners every 5th
    assert not day.failures()


def test_budget_breach_fails():
    r = rp.DayReport(day=T0.date(), events=9, ticks=2000, winner_ticks=100, held=90, displaced=20)
    fails = r.failures()
    assert any("events" in f for f in fails)
    assert any("share" in f for f in fails)
    assert any("displaced" in f for f in fails)


def test_partial_days_are_judged_against_a_full_day():
    days = []
    for d in range(4):
        days += _ticks(T0 + timedelta(days=d), 2000 if d < 3 else 500)
    transitions = [_t(T0 + timedelta(days=3), "strained", ["gpu_queue_sustained"])]
    events = rp.replay_events(transitions, [], until=T0 + timedelta(days=3, minutes=1), tz=TZ)
    reports = rp.project(events, days, tz=TZ)
    last = [r for r in reports if r.events][0]
    assert last.partial and last.share == last.held / last.full_day_ticks


def test_main_exits_2_on_read_failure():
    assert rp.main(["--psql", "false"]) == 2
