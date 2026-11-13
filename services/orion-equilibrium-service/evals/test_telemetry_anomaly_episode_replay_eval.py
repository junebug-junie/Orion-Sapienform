"""Replay eval: 15 days of real telemetry_anomaly fires (metacog_trigger) and
harness turns (harness_turn_trace), exported 2026-10-13 with
scripts/analysis/replay_telemetry_anomaly_episodes.py --write-fixture, folded
through the live episode gate (orion/metacog/anomaly_episode.py).

Before = one fire per flagged 60s window (the shipped behaviour).
After  = one fire per episode, task_execution-led windows during a turn left to
chat_turn, sustained override after 15 flagged windows in 30 min."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "analysis")]

from replay_telemetry_anomaly_episodes import fires_between, replay  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "telemetry_anomaly_replay_2026-09-28_2026-10-13.jsonl"
ITEMS = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
STEADY = [f"2026-10-{d:02d}" for d in range(1, 14)]  # full days with a 48h lookback behind them


def _total(result, key):
    return sum(d[key] for d in result["per_day"].values())


def test_fire_volume_drops_by_an_order_of_magnitude():
    r = replay(ITEMS)
    before, after = _total(r, "before_fires"), _total(r, "after_fires")
    assert before == 2485
    assert after <= before * 0.08, (before, after)  # 2485 -> 140
    for day, d in r["per_day"].items():
        assert d["after_fires"] <= d["episodes"] + 3, (day, d)  # escalations only


def test_10_08_gpu_outage_still_fires_promptly():
    """10-08 14:40Z onward: failure_pressure-led, turns hung 20-70 min so turn
    brackets cover almost every window. Must fire within ~15 min of onset."""
    r = replay(ITEMS)
    fires = fires_between(r, "2026-10-08T14:30:00+00:00", "2026-10-08T19:00:00+00:00")
    assert fires, "the 10-08 outage produced no trigger"
    assert fires[0]["window_end"] < 1791471300  # 2026-10-08T14:55:00Z
    assert any(f["severity"] == "critical" for f in fires)


def test_without_sustained_override_the_outage_never_fires():
    """Pins why the override exists: hung turns cover every outage window, so
    pure turn suppression mutes the whole 10-08 outage."""
    r = replay(ITEMS, sustained_windows=0)
    assert not fires_between(r, "2026-10-08T14:30:00+00:00", "2026-10-08T19:00:00+00:00")


def test_non_chat_availability_outage_still_fires():
    r = replay(ITEMS)
    fires = fires_between(r, "2026-10-03T21:00:00+00:00", "2026-10-03T23:00:00+00:00")
    assert any(f["lead"] == "availability" for f in fires)


def test_turn_suppression_is_what_removes_most_episodes():
    episodes_only = _total(replay(ITEMS, turn_suppress=False), "after_fires")
    with_suppress = _total(replay(ITEMS), "after_fires")
    assert episodes_only > 5 * with_suppress  # 885 vs 140


def test_dream_novelty_pressure_from_telemetry_is_roughly_unchanged():
    """orion-dream counts only NEW keys (direction + lead channel); the keys
    suppression removes are chronic ones already in the 48h lookback."""
    r = replay(ITEMS)
    before = sum(r["per_day"][d]["before_new_key_pressure"] for d in STEADY)
    after = sum(r["per_day"][d]["after_new_key_pressure"] for d in STEADY)
    assert abs(after - before) <= 0.25 * before, (before, after)


def _quick_escalations(result):
    f = result["after_fires"]
    return sum(1 for a, b in zip(f, f[1:])
               if b["reason"] == "severity_escalation" and b["window_end"] - a["window_end"] <= 75)


def test_nominal_open_deferral_removes_double_fires_without_losing_novelty():
    """2026-10-20: a nominal open waits one window and fires with that window's band.
    15 days: 140 -> 106 fires, fewer escalations one window after a fire, and dream
    novelty unchanged (nominal keys carry no dream weight)."""
    from orion.metacog import anomaly_episode as ae
    import replay_telemetry_anomaly_episodes as rte

    after = replay(ITEMS)
    orig = rte.step
    try:
        rte.step = lambda s, **kw: ae.step(s, defer_nominal_open=False, **kw)
        old = replay(ITEMS)
    finally:
        rte.step = orig
    assert _total(old, "after_fires") == 140
    assert _total(after, "after_fires") == 106
    assert _quick_escalations(after) < _quick_escalations(old)
    assert sum(after["per_day"][d]["after_new_key_pressure"] for d in STEADY) == \
        sum(old["per_day"][d]["after_new_key_pressure"] for d in STEADY)
