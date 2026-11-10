"""Surprise math: conformal p from stored quantiles, the calm point, and the publish gate."""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest
from scipy import stats

from app.surprise import (
    CALM_SURPRISAL,
    P_FLOOR,
    conformal_p_value,
    excess,
    horizon_gate,
    summarize,
    surprisal,
)
from tests.helpers import T0

LEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


def _normal_q(mu=0.0, sd=1.0):
    return {str(l): float(stats.norm.ppf(l, mu, sd)) for l in LEVELS}


def test_median_is_unsurprising_and_the_95_edge_hits_the_floor():
    q = _normal_q()
    assert conformal_p_value(0.0, q) == (1.0, False)
    p, censored = conformal_p_value(q["0.9"], q)
    assert p == pytest.approx(0.2) and not censored
    assert conformal_p_value(5.0, q) == (P_FLOOR, True)
    assert conformal_p_value(-5.0, q) == (P_FLOOR, True)


def test_calm_point_is_one_minus_the_floor():
    assert CALM_SURPRISAL == pytest.approx(1 - P_FLOOR)
    # E[-ln max(U, floor)] for U ~ Uniform(0, 1), integrated by hand.
    expected = P_FLOOR * -math.log(P_FLOOR) + (1 - (P_FLOOR - P_FLOOR * math.log(P_FLOOR)))
    assert CALM_SURPRISAL == pytest.approx(expected)


def test_a_calibrated_forecaster_rests_at_zero_excess_and_a_shifted_one_does_not():
    rng = np.random.default_rng(7)
    q = _normal_q()
    calm = np.mean([excess(conformal_p_value(x, q)[0]) for x in rng.normal(0, 1, 20000)])
    # Linear interpolation between the 9 stored levels is not the exact CDF: for a normal
    # forecaster it reads ~0.033 below the theoretical calm point (measured, seed 7, n=20000).
    # Documented in app/surprise.py; small next to live offsets (up to +-0.27).
    assert -0.045 < calm < -0.02
    shifted = np.mean([excess(conformal_p_value(x, q)[0]) for x in rng.normal(1.5, 1, 20000)])
    too_wide = np.mean([excess(conformal_p_value(x, q)[0]) for x in rng.normal(0, 0.5, 20000)])
    assert shifted > 0.5 and too_wide < -0.3


def test_collapsed_quantiles_do_not_divide_by_zero():
    q = _normal_q()
    q["0.025"] = q["0.05"]          # the pool clipped both to its minimum (live case, 2026-11-03)
    p, censored = conformal_p_value(q["0.05"], q)
    assert 0 < p <= 0.1 and not censored
    assert surprisal(1.0) == 0.0 and surprisal(0.0) == pytest.approx(-math.log(P_FLOOR))


@pytest.mark.parametrize("live,served,verdict,expected", [
    ({"n": 500, "coverage80": 0.80}, "direct_arx", None, (True, "passed")),
    ({"n": 500, "coverage80": 0.80}, "naive", None, (False, "served_baseline")),
    ({"n": 500, "coverage80": 0.80}, None, None, (False, "no_selection")),
    ({"n": 500, "coverage80": 0.89}, "direct_arx", None, (False, "coverage_out_of_band")),
    ({"n": 50, "coverage80": 0.80}, "direct_arx", None, (False, "too_few_live_scores")),
    ({"n": 500, "coverage80": 0.80}, "direct_arx", {"verdict": "worse"}, (False, "live_worse_than_baseline")),
])
def test_horizon_gate(live, served, verdict, expected):
    assert horizon_gate(15, live, served, verdict) == expected


def test_summary_thins_overlapping_origins_and_flags_calm():
    rng = np.random.default_rng(1)
    q = _normal_q()
    rows = [{"horizon_min": 15, "origin_ts": T0 + timedelta(minutes=5 * i), "actual": float(x), "quantiles": q}
            for i, x in enumerate(rng.normal(0, 1, 900))]
    h = summarize(rows, grain_min=5)["15"]
    assert h["n"] == 900 and h["n_independent"] == 300
    assert h["calm"] is True and abs(h["mean_excess"]) < 0.15
