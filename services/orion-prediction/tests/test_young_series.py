"""Regression on REAL data: a young series (office.temp_c, pan-1 sensor installed 2026-10-28) whose
reselect failed every 6 h from 2026-11-02 with "too few blend residual paths at [5] min ...
{'champion': [8, 16, 16, 16, 16], 'baseline': [16, ...]}".

Root cause: direct_ar needs ~3 days of training (MIN_ROWS complete rows at its 24 h lead, after the
one-day-lag feature), the backtest only 2, so the oldest windows of a young series always fail for it.
That alone is honest -- the backtest gate refuses a model with residuals on < 90% of the reference's
windows. But a live ``better`` verdict promoted it WITHOUT that check, the blend's 5-min column then had
8 residuals, and check_paths refused the whole selection.

Fixture: evals/fixtures/cabinet_5min.csv.gz (export_cabinet_fixture.py), office_temp_c column.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from app.backtest import MIN_PATHS, run_backtest, select_champion, step_owner, stitch
from app.frame import mask_windows, to_grain
from app.models import build_models, split_seasonalities
from app.series import load_registry
from tests.conftest import REGISTRY_PATH, SERVICE

FIXTURE = SERVICE / "evals" / "fixtures" / "cabinet_5min.csv.gz"
# The live selection that failed with 16 baseline windows ran at ~21:12Z (selections are 6 h apart).
CUTOFF = pd.Timestamp("2026-11-02T21:10:00Z")
FAST = ("naive", "seasonal_naive", "direct_ar")


def _office():
    spec = load_registry(REGISTRY_PATH).get("office.temp_c")
    with gzip.open(FIXTURE, "rt") as fh:
        df = pd.read_csv(fh)
    ts = pd.to_datetime(df["bucket_start"], utc=True)
    keep = (ts < CUTOFF) & df["office_temp_c"].notna()
    pts = list(zip(ts[keep], df["office_temp_c"][keep].astype(float)))
    start = (CUTOFF - pd.Timedelta(days=6)).to_pydatetime()
    series = to_grain(pts, spec.grain_min, spec.agg, spec.max_fill_min, start=start, end=CUTOFF.to_pydatetime())
    actuals = mask_windows(to_grain(pts, spec.grain_min, spec.agg, 0, start=start, end=CUTOFF.to_pydatetime()),
                           spec.exclude)
    return spec, series, actuals


def _backtest(spec, series, actuals):
    usable, _ = split_seasonalities(spec, len(series))
    return {n: run_backtest(series, spec, n, lambda u, n=n: build_models(spec, u)[n], 40, 36, actuals=actuals)
            for n in FAST}


def test_young_series_live_promotion_cannot_serve_a_model_without_window_coverage():
    spec, series, actuals = _office()
    results = _backtest(spec, series, actuals)
    base_n = results["naive"].n_windows
    # The live failure, reproduced: direct_ar loses its oldest windows (too little training for it).
    assert base_n >= MIN_PATHS
    assert results["direct_ar"].n_windows < 0.9 * base_n
    assert any("complete training rows" in e for e in results["direct_ar"].errors)
    # A live 'better' verdict at 5 min (what the shadow record said on 2026-11-02/03).
    live = {("direct_ar", 5): {"verdict": "better", "rel_mae": 0.8}}
    sel = select_champion(results, spec, 180, live=live)
    assert sel.assignment[5] != "direct_ar"
    assert sel.metrics["direct_ar"]["5"]["passed"] is False
    # And so the blend every served horizon draws on keeps MIN_PATHS residuals (check_paths passes).
    res = {n: r.residuals for n, r in results.items()}
    blend = stitch(res, step_owner(sel.assignment, spec.grain_min, spec.max_horizon_steps))
    counts = [int(np.isfinite(blend[:, spec.steps(hm) - 1]).sum()) for hm in spec.horizons_min]
    assert min(counts) >= MIN_PATHS, counts


def test_live_promotion_still_works_when_the_model_covers_the_windows():
    spec, series, actuals = _office()
    results = _backtest(spec, series, actuals)
    # Same verdict for a model with full coverage: give direct_ar the baseline's windows by
    # treating naive's residuals (scaled down: a real lower error) as its own.
    results["direct_ar"].residuals = results["naive"].residuals * 0.5
    live = {("direct_ar", 5): {"verdict": "better", "rel_mae": 0.8}}
    sel = select_champion(results, spec, 180, live=live)
    assert sel.assignment[5] == "direct_ar" and sel.metrics["direct_ar"]["5"]["passed_by"] in ("live", "backtest")
