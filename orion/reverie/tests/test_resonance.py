"""Phase H — resonance detector unit tests.

The tripwire must fire on a runaway loop, stay silent on a damped one, ignore
noise themes, and cap its sample. All deterministic.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.reverie.resonance import ThemeEvent, detect_resonance
from orion.schemas.reverie import MAX_RESONANCE_SAMPLES

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)
REFRACTORY = 900.0  # 15 min


def _events(theme: str, gaps_sec: list[float]) -> list[ThemeEvent]:
    out = [ThemeEvent(theme, NOW)]
    t = NOW
    for g in gaps_sec:
        t = t + timedelta(seconds=g)
        out.append(ThemeEvent(theme, t))
    return out


def test_runaway_loop_trips_the_wire():
    # Three recurrences all inside the refractory window → runaway.
    events = _events("loop:ol-1", [60, 90, 120])
    alert = detect_resonance(events, refractory_sec=REFRACTORY)
    assert alert is not None
    assert alert.theme_key == "loop:ol-1"
    assert alert.violation_count == 3
    assert alert.min_gap_sec == 60
    assert alert.occurrences == 4


def test_damped_loop_stays_silent():
    # Every recurrence waited out the full refractory bound → healthy.
    events = _events("loop:ol-1", [1000, 1200, 1500])
    assert detect_resonance(events, refractory_sec=REFRACTORY) is None


def test_single_breach_below_min_violations_is_silent():
    # One breach only; min_violations defaults to 2.
    events = _events("loop:ol-1", [60, 1000, 1000])
    assert detect_resonance(events, refractory_sec=REFRACTORY) is None


def test_unknown_and_empty_themes_are_ignored():
    events = _events("unknown", [10, 20, 30]) + _events("", [10, 20, 30])
    assert detect_resonance(events, refractory_sec=REFRACTORY) is None


def test_picks_the_tightest_runaway_among_many():
    a = _events("loop:a", [100, 100])  # 2 breaches, min gap 100
    b = _events("loop:b", [100, 100, 100])  # 3 breaches — more severe
    alert = detect_resonance(a + b, refractory_sec=REFRACTORY)
    assert alert is not None
    assert alert.theme_key == "loop:b"
    assert alert.violation_count == 3


def test_zero_refractory_never_trips():
    events = _events("loop:ol-1", [1, 1, 1])
    assert detect_resonance(events, refractory_sec=0.0) is None


def test_full_tie_winner_is_deterministic():
    # Two themes tied on violation_count AND min_gap → winner must not depend on
    # set-iteration order (PYTHONHASHSEED).
    a = _events("loop:a", [100, 100])
    b = _events("loop:b", [100, 100])
    winners = {detect_resonance(a + b, refractory_sec=REFRACTORY).theme_key for _ in range(20)}
    assert len(winners) == 1  # stable across repeated calls


def test_alert_id_is_deterministic_for_dedup():
    # Same events → same alert_id, so ON CONFLICT collapses a persisting runaway.
    events = _events("loop:ol-1", [60, 90, 120])
    a1 = detect_resonance(events, refractory_sec=REFRACTORY)
    a2 = detect_resonance(events, refractory_sec=REFRACTORY)
    assert a1.alert_id == a2.alert_id
    assert "resonance:loop:ol-1:" in a1.alert_id


def test_sample_ats_capped():
    events = _events("loop:ol-1", [1.0] * (MAX_RESONANCE_SAMPLES + 20))
    alert = detect_resonance(events, refractory_sec=REFRACTORY)
    assert alert is not None
    assert len(alert.sample_ats) <= MAX_RESONANCE_SAMPLES
