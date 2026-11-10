"""Adaptive champion ranges (app/adaptive.py): calm hours get narrow ranges, busy hours wide ones,
and both stay near nominal coverage -- the property one backtest width could not have."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import numpy as np
import pandas as pd

from app.adaptive import MIN_POOL, adaptive_offsets, difficulty_at, difficulty_series
from app.backtest import QUANTILES
from tests.helpers import T0
from tests.test_engine import Clock, _engine, _setup, _sync, fast_zoo  # noqa: F401 -- fixture

Q10, Q90 = QUANTILES.index(0.1), QUANTILES.index(0.9)


def _rows(n_origins: int, sigma, horizons=(5, 15, 60), seed: int = 0, start=T0):
    """Scored champion rows every 5 min; the miss at origin o has scale sigma(o) for every horizon."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_origins):
        o = pd.Timestamp(start) + pd.Timedelta(minutes=5 * i)
        for hm in horizons:
            if o + pd.Timedelta(minutes=hm) > pd.Timestamp(start) + pd.Timedelta(minutes=5 * (n_origins - 1)):
                continue   # target not known yet at the last origin
            rows.append({"role": "champion", "origin_ts": o, "horizon_min": hm, "point": 20.0,
                         "actual": 20.0 + rng.normal(0, sigma(o))})
    return rows


def _regime(o):   # 3 h calm, 3 h busy, repeating
    return 0.05 if (o.hour // 3) % 2 == 0 else 0.5


def test_difficulty_is_only_what_was_known_and_not_stale():
    rows = _rows(48, lambda o: 0.1, horizons=(5,))
    diff = difficulty_series(rows, 5)
    last_target = diff.index[-1]
    t = pd.DatetimeIndex([last_target - pd.Timedelta(minutes=1), last_target + pd.Timedelta(minutes=9),
                          last_target + pd.Timedelta(minutes=11), pd.Timestamp(T0)])
    got = difficulty_at(diff, t, timedelta(minutes=10))
    assert np.isfinite(got[0]) and got[0] == diff.iloc[-2]      # the newest target is not known yet
    assert np.isfinite(got[1])
    assert np.isnan(got[2])                                      # older than max_age: no carry-forward
    assert np.isnan(got[3])                                      # nothing known before the first target


def test_calm_hour_gets_a_narrow_range_and_busy_hour_a_wide_one():
    rows = _rows(4 * 288, _regime)
    calm_origin = pd.Timestamp(T0) + pd.Timedelta(days=3, hours=1)     # hour 1: calm block
    busy_origin = pd.Timestamp(T0) + pd.Timedelta(days=3, hours=4)     # hour 4: busy block
    known = lambda o: [r for r in rows if r["origin_ts"] + pd.Timedelta(minutes=r["horizon_min"]) <= o]
    calm, _ = adaptive_offsets(known(calm_origin), calm_origin, (5, 15, 60), 5)
    busy, _ = adaptive_offsets(known(busy_origin), busy_origin, (5, 15, 60), 5)
    for hm in (5, 15):
        w_calm = calm[hm][Q90] - calm[hm][Q10]
        w_busy = busy[hm][Q90] - busy[hm][Q10]
        assert w_busy > 4 * w_calm, (hm, w_calm, w_busy)
        assert abs(w_calm - 2 * 1.2816 * 0.05) < 0.05             # near the true calm 80 % width


def test_coverage_holds_in_calm_and_busy_hours_separately():
    """The calibration the calm check needs: ~80 % inside the 80 % range in BOTH regimes, once the
    last hour reflects the regime. The first hour after a switch lags (the difficulty window still
    holds the old regime): measured here, reported, not hidden."""
    rows = _rows(4 * 288, _regime, horizons=(5, 15), seed=1)
    rng = np.random.default_rng(7)
    hits: dict[str, list] = {"calm": [], "busy": [], "calm_first_hour": [], "busy_first_hour": []}
    for i in range(3 * 288, 4 * 288 - 3, 2):
        o = pd.Timestamp(T0) + pd.Timedelta(minutes=5 * i)
        pool = [r for r in rows if o - pd.Timedelta(days=3) <= r["origin_ts"]
                and r["origin_ts"] + pd.Timedelta(minutes=r["horizon_min"]) <= o]
        offs, _ = adaptive_offsets(pool, o, (5, 15), 5)
        y = rng.normal(0, _regime(o))
        key = ("calm" if _regime(o) < 0.1 else "busy") + ("_first_hour" if o.hour % 3 == 0 else "")
        hits[key].append(offs[15][Q10] <= y <= offs[15][Q90])
    cov = {k: float(np.mean(v)) for k, v in hits.items()}
    assert 0.70 <= cov["calm"] <= 0.90, cov
    assert 0.70 <= cov["busy"] <= 0.90, cov
    overall = np.mean([x for v in hits.values() for x in v])
    assert 0.74 <= overall <= 0.86, (overall, cov)
    assert cov["calm_first_hour"] > cov["calm"] and cov["busy_first_hour"] < cov["busy"], cov   # the lag


def test_thin_pool_or_no_recent_difficulty_falls_back_with_a_reason():
    rows = _rows(MIN_POOL // 2, lambda o: 0.1)
    last = rows[-1]["origin_ts"]
    offs, why = adaptive_offsets(rows, last, (5, 15), 5)
    assert offs == {} and why[5].startswith("live_pool_") and why[15].startswith("live_pool_")
    offs, why = adaptive_offsets(rows, last + pd.Timedelta(hours=2), (5, 15), 5)
    assert offs == {} and why == {5: "no_recent_difficulty", 15: "no_recent_difficulty"}
    assert adaptive_offsets([], last, (5,), 5) == ({}, {5: "no_recent_difficulty"})


def test_offsets_are_monotone_in_level():
    rows = _rows(2 * 288, _regime)
    o = rows[-1]["origin_ts"]
    offs, _ = adaptive_offsets(rows, o, (5, 15, 60), 5)
    assert set(offs) == {5, 15, 60}
    for v in offs.values():
        assert np.all(np.diff(v) >= 0)


def test_engine_serves_adaptive_ranges_once_the_live_pool_is_deep(fast_zoo):

    reg, spec, s, store, end = _setup()
    eng = _engine(reg, store, Clock(end + timedelta(seconds=90)), [])
    asyncio.run(eng.tick(_sync))
    st = eng.state["test.temp"]
    assert st.interval_source and all(v.startswith("backtest:") for v in st.interval_source.values())
    backtest_q = {f["horizon_min"]: f["quantiles"] for f in store.forecasts if f["role"] == "champion"}

    # Three days of scored champion history with a tiny, steady miss -> narrow adaptive ranges.
    origin = pd.Timestamp(max(f["origin_ts"] for f in store.forecasts))
    hist = _rows(3 * 288, lambda o: 0.001, horizons=(5, 60, 360), start=(origin - pd.Timedelta(days=3)).to_pydatetime())
    for r in hist:
        store.forecasts.append({**r, "series_id": "test.temp", "target_ts": (r["origin_ts"] + pd.Timedelta(
            minutes=r["horizon_min"])).to_pydatetime(), "origin_ts": r["origin_ts"].to_pydatetime(),
            "score_status": "scored", "abs_error": abs(r["actual"] - r["point"]), "in_80": True, "in_95": True,
            "forecast_id": -1, "scored_at": T0, "quantiles": {}})
    built = eng.build_forecast(spec, s, origin, end + timedelta(seconds=90))
    assert built["interval_source"][5] == built["interval_source"][60] == "adaptive"
    assert built["interval_source"][360] == "backtest:horizon_beyond_difficulty_window"
    champ = {r["horizon_min"]: r for r in built["rows"] if r["role"] == "champion"}
    w_new = champ[5]["quantiles"]["0.9"] - champ[5]["quantiles"]["0.1"]
    w_old = backtest_q[5]["0.9"] - backtest_q[5]["0.1"]
    assert w_new < w_old / 5
    base = {r["horizon_min"]: r for r in built["rows"] if r["role"] == "baseline"}
    assert base[5]["quantiles"]["0.9"] - base[5]["quantiles"]["0.1"] > w_new   # baseline untouched

    eng.adaptive_intervals = False
    built = eng.build_forecast(spec, s, origin, end + timedelta(seconds=90))
    assert set(built["interval_source"].values()) == {"backtest:disabled"}
