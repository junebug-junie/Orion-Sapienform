"""Family A joint surprise window (orion/substrate/prediction_joint_surprise.py): rest point, scale, staleness."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate.prediction_joint_surprise import (
    BODY_JOINT_NODE_ID,
    ROOM_JOINT_NODE_ID,
    WINDOW,
    JointSurpriseState,
    calm_excess_variance,
)

NOW = datetime(2026, 11, 18, 9, 30, tzinfo=timezone.utc)


def _ev(i, excess, group="body", horizon=5, n_cal=1000, minutes_ago=None):
    origin = NOW - timedelta(minutes=5 * (i + 1) if minutes_ago is None else minutes_ago)
    return {"joint_id": str(i), "group": group, "horizon_min": horizon, "origin_ts": origin,
            "target_ts": origin + timedelta(minutes=horizon), "excess": excess, "n_cal": n_cal,
            "top_shares": {"a": 0.6, "b": 0.4}}


def _calm_point(n):
    return math.log(n + 1) - math.lgamma(n + 2) / (n + 1)


def test_calm_variance_is_one_in_the_limit_and_exact_on_the_grid():
    assert calm_excess_variance(10 ** 5) == pytest.approx(1.0, abs=2e-3)
    m = 4                                      # p in {1/4, .., 1}: direct
    xs = [-math.log(k / m) for k in range(1, m + 1)]
    mu = sum(xs) / m
    assert calm_excess_variance(m - 1) == pytest.approx(sum((x - mu) ** 2 for x in xs) / m)
    with pytest.raises(ValueError):
        calm_excess_variance(0)


def test_a_calibrated_stream_rests_at_exactly_zero_about_half_the_time_and_rarely_crosses():
    """Rest point by theory: p uniform on the conformal grid -> z ~ N(0, 1) -> value 0 half the time."""
    rng = random.Random(0)
    n_cal = 1000
    calm = _calm_point(n_cal)
    zeros, crossed, total = 0, 0, 0
    for trial in range(400):
        st = JointSurpriseState()
        for i in range(6):
            p = rng.randint(1, n_cal + 1) / (n_cal + 1)
            st.add(_ev(i, -math.log(p) - calm, n_cal=n_cal), now=NOW)
        r = st.reading(BODY_JOINT_NODE_ID, now=NOW)
        total += 1
        zeros += r.value == 0.0
        crossed += r.value > 0.55
    assert 0.40 < zeros / total < 0.62
    assert crossed / total < 0.10


def test_one_coupled_spike_reads_once_and_leaves_with_the_window():
    st = JointSurpriseState()
    calm = _calm_point(1000)
    st.add(_ev(0, -math.log(0.01) - calm), now=NOW)        # one joint miss (the whole coupled event)
    for i in range(1, 6):
        st.add(_ev(i, 1.0 - calm), now=NOW)                 # calm outcomes (excess ~0)
    r = st.reading(BODY_JOINT_NODE_ID, now=NOW)
    assert r.n == 6 and 0.3 < r.value < 0.7 and r.top_reading == "a"
    later = NOW + WINDOW + timedelta(minutes=6)
    assert st.reading(BODY_JOINT_NODE_ID, now=later) is None   # gone: stale, never calm


def test_one_outcome_per_target_bucket_shortest_horizon_wins():
    st = JointSurpriseState()
    origin = NOW - timedelta(minutes=15)
    st.add({**_ev(0, 2.0, horizon=15, minutes_ago=15), "origin_ts": origin,
            "target_ts": origin + timedelta(minutes=15)}, now=NOW)
    o5 = NOW - timedelta(minutes=5)
    st.add({**_ev(1, 0.1), "origin_ts": o5, "target_ts": NOW}, now=NOW)   # same target bucket, 5 min
    r = st.reading(BODY_JOINT_NODE_ID, now=NOW)
    assert r.n == 1 and r.joint_ids == ("1",)


def test_groups_never_mix():
    st = JointSurpriseState()
    st.add(_ev(0, 5.0, group="room"), now=NOW)
    assert st.reading(BODY_JOINT_NODE_ID, now=NOW) is None
    assert st.reading(ROOM_JOINT_NODE_ID, now=NOW).value > 0.9


@pytest.mark.parametrize("ev,verdict", [
    (lambda: {**_ev(0, 1.0), "group": "blend"}, "unknown_group"),
    (lambda: _ev(0, 1.0, minutes_ago=120), "older_than_window"),
    (lambda: {**_ev(0, 1.0), "excess": float("nan")}, "non_finite_excess"),
    (lambda: {**_ev(0, 1.0), "n_cal": None}, "no_calibration_size"),
])
def test_refusals(ev, verdict):
    _node, why = JointSurpriseState().add(ev(), now=NOW)
    assert why == verdict
