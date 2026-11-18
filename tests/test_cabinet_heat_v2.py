"""Thermal controller v2, D1: one owner for "how hot is the cabinet" (orion/autonomy/cabinet_heat.py).

Spec: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md (D1, Decisions: critical
34 C re-arm 33 C; the 32 C ``hot`` line and the ThermalState names are unchanged)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.cabinet_heat import (
    REFLEX_CABINET_HOT, REFLEX_CABINET_UNKNOWN, minutes_to_hot, read_cabinet_heat,
)
from orion.autonomy.thermal_gate import DEFAULT_CRITICAL_C, DEFAULT_CRITICAL_REARM_C, DEFAULT_HOT_C
from orion.hardware_watch.rules import TempPoint

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def pts(*values: float, step: float = 30.0, end: datetime = NOW) -> list[TempPoint]:
    n = len(values)
    return [TempPoint(end - timedelta(seconds=step * (n - 1 - i)), v) for i, v in enumerate(values)]


def test_lines_are_where_the_decision_put_them():
    assert (DEFAULT_CRITICAL_C, DEFAULT_CRITICAL_REARM_C, DEFAULT_HOT_C) == (34.0, 33.0, 32.0)


@pytest.mark.parametrize("temp,state,critical,reflex", [
    (29.4, "normal", False, None),
    (29.5, "elevated", False, None),
    (31.99, "elevated", False, None),
    (32.0, "hot", False, None),          # the 32 C hot line is unchanged and does NOT shed
    (33.99, "hot", False, None),
    (34.0, "hot", True, REFLEX_CABINET_HOT),
])
def test_boundaries(temp, state, critical, reflex):
    r = read_cabinet_heat(pts(temp), NOW)
    assert (r.thermal_state, r.critical, r.reflex) == (state, critical, reflex)


def test_critical_hysteresis_holds_until_below_33():
    r = read_cabinet_heat(pts(34.1, 33.5, 33.1), NOW)
    assert r.critical and r.reflex == REFLEX_CABINET_HOT
    r = read_cabinet_heat(pts(34.1, 33.5, 33.0), NOW)
    assert not r.critical and r.reflex is None


def test_previous_reading_seeds_the_latch_beyond_the_window():
    prev = read_cabinet_heat(pts(34.2), NOW - timedelta(minutes=40))
    r = read_cabinet_heat(pts(33.6, 33.4), NOW, previous=prev)
    assert r.critical
    assert not read_cabinet_heat(pts(33.6, 33.4), NOW).critical   # without it the window decides


def test_grace_holds_the_last_state_then_unknown():
    old = pts(30.5, end=NOW - timedelta(seconds=290))
    assert read_cabinet_heat(old, NOW).thermal_state == "elevated"       # within grace: one missed read is nothing
    r = read_cabinet_heat(pts(30.5, end=NOW - timedelta(seconds=301)), NOW)
    assert r.thermal_state == "unknown" and r.reading_age_sec == pytest.approx(301)


def test_unknown_counts_as_elevated_and_sheds_background_only():
    r = read_cabinet_heat([], NOW)
    assert r.thermal_state == "unknown" and r.effective_state == "elevated" and r.reflex == REFLEX_CABINET_UNKNOWN


def test_unknown_plus_ac_low_is_hot():
    r = read_cabinet_heat([], NOW, ac_low=True)
    assert r.effective_state == "hot" and r.reflex == REFLEX_CABINET_HOT


def test_unknown_with_the_ac_silent_too_is_not_hot():
    """Monitoring outage (10-03): AC silent is not 'AC low'."""
    r = read_cabinet_heat([], NOW, ac_low=None)
    assert r.reflex == REFLEX_CABINET_UNKNOWN


def test_ac_low_is_ignored_while_the_cabinet_is_readable():
    r = read_cabinet_heat(pts(26.0), NOW, ac_low=True)
    assert r.effective_state == "normal" and r.reflex is None


def test_minutes_to_hot_projection():
    assert minutes_to_hot(31.0, 0.75) == pytest.approx(20.0)   # 0.05 C/min -> 1 C in 20 min
    assert minutes_to_hot(32.5, 0.0) == 0.0
    assert minutes_to_hot(30.0, 0.0) is None and minutes_to_hot(None, 1.0) is None
    r = read_cabinet_heat([TempPoint(NOW - timedelta(minutes=15), 30.0), TempPoint(NOW, 31.0)], NOW)
    assert r.minutes_to_hot == pytest.approx(15.0) and r.as_dict()["minutes_to_hot"] == 15.0
