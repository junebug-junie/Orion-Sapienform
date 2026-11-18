"""Forecast surprise -> node:prediction.room / node:prediction.body fold (pure)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate.prediction_surprise import (
    BODY_NODE_ID,
    ROOM_NODE_ID,
    WINDOW,
    PredictionSurpriseState,
    calm_excess_variance,
    independent_origin,
)

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
CALM = 0.95


def _ev(series="office.temp_c", hm=5, origin_min_ago=10, excess=0.0, fid="1"):
    origin = NOW - timedelta(minutes=origin_min_ago)
    return {"series_id": series, "horizon_min": hm, "origin_ts": origin,
            "target_ts": origin + timedelta(minutes=hm), "excess": excess, "calm_surprisal": CALM,
            "forecast_id": fid}


def test_calm_variance_is_the_floored_exponential_variance():
    # -ln max(U, 0.05): mean 0.95, variance 2 - 2P(1 + ln 1/P) - (1 - P)^2.
    assert calm_excess_variance(CALM) == pytest.approx(0.69793, abs=1e-5)
    with pytest.raises(ValueError):
        calm_excess_variance(1.0)


def test_series_map_to_one_node_each_and_unknown_series_is_ignored():
    st = PredictionSurpriseState()
    assert st.add(_ev("office.temp_c"), now=NOW) == (ROOM_NODE_ID, "accepted")
    assert st.add(_ev("athena.cabinet_temp_c"), now=NOW) == (BODY_NODE_ID, "accepted")
    assert st.add(_ev("somewhere.else"), now=NOW) == (None, "unknown_series")


def test_only_clock_anchored_origins_count_per_horizon():
    assert independent_origin(datetime(2026, 11, 11, 8, 35, tzinfo=timezone.utc), 5)
    assert independent_origin(datetime(2026, 11, 11, 8, 45, tzinfo=timezone.utc), 15)
    assert not independent_origin(datetime(2026, 11, 11, 8, 40, tzinfo=timezone.utc), 15)
    st = PredictionSurpriseState()
    # NOW - 20 min = 08:40: not a 15-min block start -> overlapping, never folded.
    assert st.add(_ev(hm=15, origin_min_ago=20), now=NOW) == (ROOM_NODE_ID, "overlapping_origin")
    assert st.reading(ROOM_NODE_ID, now=NOW) is None


def test_reading_is_summed_excess_over_its_calm_spread():
    st = PredictionSurpriseState()
    for i, x in enumerate((2.05, -0.2, -0.2, -0.2), start=1):
        st.add(_ev(origin_min_ago=5 * i + 5, excess=x, fid=str(i)), now=NOW)
    r = st.reading(ROOM_NODE_ID, now=NOW)
    z = (2.05 - 0.6) / math.sqrt(4 * calm_excess_variance(CALM))
    assert r.n == 4 and r.z == pytest.approx(z) and r.value == pytest.approx(z / 3)
    assert r.mean_excess == pytest.approx((2.05 - 0.6) / 4)
    assert r.forecast_ids == ("4", "3", "2", "1")       # oldest target first


def test_calm_outcomes_rest_at_zero_and_a_run_of_misses_saturates():
    st = PredictionSurpriseState()
    for i in range(1, 6):
        st.add(_ev(origin_min_ago=5 * i + 5, excess=-0.3, fid=str(i)), now=NOW)
    assert st.reading(ROOM_NODE_ID, now=NOW).value == 0.0       # below calm: clipped to rest, signed z kept
    assert st.reading(ROOM_NODE_ID, now=NOW).z < 0
    for i in range(1, 6):
        st.add(_ev(origin_min_ago=5 * i + 5, excess=2.05, fid=str(i)), now=NOW)   # same origins: replaced
    r = st.reading(ROOM_NODE_ID, now=NOW)
    assert r.n == 5 and r.value == 1.0


def test_old_outcomes_leave_the_window_and_a_backlog_never_writes_the_past_as_now():
    st = PredictionSurpriseState()
    st.add(_ev(origin_min_ago=20, excess=2.05), now=NOW)
    assert st.reading(ROOM_NODE_ID, now=NOW) is not None
    assert st.reading(ROOM_NODE_ID, now=NOW + WINDOW) is None          # aged out: no reading, not calm
    late = _ev(origin_min_ago=90, excess=2.05)                          # scored after an outage
    assert st.add(late, now=NOW) == (ROOM_NODE_ID, "older_than_window")


def test_non_finite_excess_is_refused():
    st = PredictionSurpriseState()
    assert st.add(_ev(excess=float("nan")), now=NOW) == (ROOM_NODE_ID, "non_finite_excess")


def test_calm_crossing_rate_does_not_depend_on_how_many_outcomes_the_window_holds():
    """Calibrated (uniform p) outcomes: the node crosses curiosity's 0.55 bar (z > 1.65) ~5 % of
    the time with 2 samples or 6 -- the reason for the z-score over a plain mean."""
    rng = random.Random(7)
    for n in (2, 6):
        crossings, trials = 0, 4000
        for _ in range(trials):
            st = PredictionSurpriseState()
            for i in range(1, n + 1):
                x = -math.log(max(rng.random(), 0.05)) - CALM
                st.add(_ev(origin_min_ago=5 * i, excess=x, fid=str(i)), now=NOW)
            crossings += st.reading(ROOM_NODE_ID, now=NOW).value >= 0.55
        assert 0.03 <= crossings / trials <= 0.08, (n, crossings / trials)


def test_one_outcome_per_target_bucket_the_shortest_horizon_wins():
    """A 5- and a 15-min forecast of the same bucket share its actual: count it once."""
    st = PredictionSurpriseState()
    st.add(_ev(hm=15, origin_min_ago=15, excess=2.0, fid="long"), now=NOW)   # 08:45 -> 09:00
    st.add(_ev(hm=5, origin_min_ago=5, excess=-0.5, fid="short"), now=NOW)   # 08:55 -> 09:00
    r = st.reading(ROOM_NODE_ID, now=NOW)
    assert r.n == 1 and r.forecast_ids == ("short",)


def test_an_origin_with_seconds_still_counts():
    assert independent_origin(datetime(2026, 11, 11, 8, 45, 3, tzinfo=timezone.utc), 15)
    assert not independent_origin(datetime(2026, 11, 11, 8, 50, 3, tzinfo=timezone.utc), 15)
