from __future__ import annotations

from datetime import timedelta

import numpy as np

from app.backtest import BacktestResult, select_champion
from app.live import MIN_INDEPENDENT, verdicts
from tests.helpers import T0, spec


def _pairs(model, hm, n, shadow_mean, base_mean, every_min=5, seed=0):
    rng = np.random.default_rng(seed)
    return [{"model": model, "horizon_min": hm, "origin_ts": T0 + timedelta(minutes=every_min * i),
             "shadow_err": abs(shadow_mean + rng.normal(0, 0.05)), "base_err": abs(base_mean + rng.normal(0, 0.05))}
            for i in range(n)]


def test_verdicts_better_worse_unclear_pending():
    v = verdicts(_pairs("a", 5, 300, 0.10, 0.20) + _pairs("b", 5, 300, 0.30, 0.20) + _pairs("c", 5, 300, 0.20, 0.20)
                 + _pairs("d", 5, MIN_INDEPENDENT - 1, 0.10, 0.20))
    assert v[("a", 5)]["verdict"] == "better" and v[("b", 5)]["verdict"] == "worse"
    assert v[("c", 5)]["verdict"] == "unclear" and v[("d", 5)]["verdict"] == "pending"


def test_overlapping_long_horizon_forecasts_are_thinned_to_independent_ones():
    v = verdicts(_pairs("a", 360, 400, 0.5, 1.0))            # 400 forecasts 5 min apart, 6 h horizon
    assert v[("a", 360)]["n_scored"] == 400 and v[("a", 360)]["n_independent"] == 6
    assert v[("a", 360)]["verdict"] == "pending"


def _gate(cand_scale, live=None, incumbent=None, n=40, seed=3):
    sp = spec(horizons=["5min"], thresholds=[])
    h = sp.max_horizon_steps
    rng = np.random.default_rng(seed)
    naive = rng.normal(0, 1.0, (n, h))
    cand = naive * cand_scale + rng.normal(0, 0.6, (n, h))
    res = {"naive": BacktestResult("naive", naive, [], 0.0),
           "seasonal_naive": BacktestResult("seasonal_naive", naive * 3, [], 0.0),
           "cand": BacktestResult("cand", cand, [], 0.0)}
    return select_champion(res, sp, live=live, incumbent=incumbent)


def _find_scale(lo, hi):
    """A candidate scale whose backtest p lands in (lo, hi)."""
    for sc in np.linspace(0.0, 1.2, 241):
        p = _gate(sc).metrics["cand"]["5"]["p_win"]
        if lo < p < hi:
            return sc
    raise AssertionError("no scale in range")


def test_live_worse_blocks_a_backtest_pass_and_live_better_serves_without_one():
    strong_sc, weak_sc = _find_scale(0.0, 0.03), _find_scale(0.5, 1.0)
    strong = _gate(strong_sc)
    assert strong.assignment[5] == "cand" and strong.metrics["cand"]["5"]["passed_by"] == "backtest"
    worse = {("cand", 5): {"verdict": "worse", "rel_mae": 1.2}}
    assert _gate(strong_sc, live=worse).assignment[5] == "naive"
    assert _gate(weak_sc).assignment[5] == "naive"
    better = {("cand", 5): {"verdict": "better", "rel_mae": 0.9}}
    sel = _gate(weak_sc, live=better)
    assert sel.assignment[5] == "cand" and sel.metrics["cand"]["5"]["passed_by"] == "live"


def test_an_incumbent_keeps_its_horizon_in_the_hysteresis_band_a_newcomer_does_not_get_in():
    sc = _find_scale(0.05, 0.10)
    assert _gate(sc).assignment[5] == "naive"                          # newcomer at p in (0.05, 0.10): no
    assert _gate(sc, incumbent={5: "cand"}).assignment[5] == "cand"    # incumbent: keeps it
