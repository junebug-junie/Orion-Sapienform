"""The attention:juniper pacing threshold (one attention budget, 2026-11-10)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from orion.physiology import attention_pacing as ap

TZ = "America/Denver"


def test_threshold_end_points_and_monotone():
    assert ap.message_price(0.0) == pytest.approx(ap.VALUE_LOW / math.e)
    assert ap.message_price(1.0) == pytest.approx(ap.VALUE_HIGH)
    assert ap.message_price(5.0) == pytest.approx(ap.VALUE_HIGH)       # ahead of pace: clamped at U
    zs = [i / 10 for i in range(11)]
    prices = [ap.message_price(z) for z in zs]
    assert prices == sorted(prices)


def test_inverse_matches_the_threshold():
    for v in (0.2, 0.31, 0.38, 0.5, 0.6):
        z = ap.pace_for_value(v)
        assert ap.message_price(z) == pytest.approx(v)
    assert ap.pace_for_value(ap.VALUE_LOW / math.e) is None              # never beats even psi(0)
    assert ap.pace_for_value(ap.VALUE_HIGH + 0.1) == 1.0


def test_waking_day_fraction_and_allowance():
    assert ap.waking_fraction(datetime(2026, 11, 10, 14, 0, tzinfo=timezone.utc), tz=TZ, start_hour=8, end_hour=23) == 0.0
    mid = datetime(2026, 11, 11, 0, 30, tzinfo=timezone.utc)            # 17:30 MST: 9.5 of 15 h
    assert ap.waking_fraction(mid, tz=TZ, start_hour=8, end_hour=23) == pytest.approx(9.5 / 15)
    assert ap.waking_fraction(datetime(2026, 11, 11, 6, 30, tzinfo=timezone.utc), tz=TZ, start_hour=8, end_hour=23) == 1.0
    assert ap.allowance(0.0, 6) == 1.0 and ap.allowance(1.0, 6) == 6.0
    # Quiet hours off: the whole local day.
    assert ap.waking_span_hours(mid, tz=TZ, start_hour=-1, end_hour=-1) == pytest.approx(24.0)
    # DST end (2026-11-01): the local day is 25 h; a waking day 08-23 is still 15 h.
    dst = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)
    assert ap.waking_span_hours(dst, tz=TZ, start_hour=-1, end_hour=-1) == pytest.approx(25.0)
    assert ap.waking_span_hours(dst, tz=TZ, start_hour=8, end_hour=23) == pytest.approx(15.0)


def test_relief_is_when_the_allowance_lets_the_value_through():
    # 3 sent, value 0.38 -> needs allowance 3 / z*(0.38); with 2.0 now growing 1/3 per hour.
    z = ap.pace_for_value(0.38)
    expected = (3 / z - 2.0) / (5 / 15) * 3600.0
    assert ap.relief_sec(3, 0.38, allowance_now=2.0, rate_per_hour=5 / 15) == pytest.approx(expected)
    assert ap.relief_sec(1, 0.38, allowance_now=2.0, rate_per_hour=5 / 15) == 0.0
    assert ap.relief_sec(3, 0.38, allowance_now=2.0, rate_per_hour=0.0) is None    # not accruing (night)
    assert ap.relief_sec(3, 0.05, allowance_now=2.0, rate_per_hour=5 / 15) is None  # never worth it


def test_a_strong_want_goes_late_in_the_day_and_a_weak_one_waits():
    late = datetime(2026, 11, 11, 4, 30, tzinfo=timezone.utc)           # 21:30 MST
    f = ap.waking_fraction(late, tz=TZ, start_hour=8, end_hour=23)
    price = ap.message_price(ap.pace(4, f, ap.pace_budget()))           # the 5th message of the day
    assert 0.33 < price < 0.60
    assert 0.60 > price          # an unusual event (value ~0.60) still goes
    assert 0.33 < price          # a weak want (value 0.33) waits


def test_the_cap_is_six_paced_to_five_and_only_the_strain_report_is_exempt():
    assert ap.JUNIPER_DAILY_CAP == 6
    assert ap.pace_budget() == 5 and ap.pace_budget(1) == 1
    assert ap.CAP_EXEMPT_SOURCES == ("strain_report",)
