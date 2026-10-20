from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from app.events import MIN_POST_BREAK_DAYS, with_events
from app.models import build_models
from tests.helpers import T0, spec


def ev(effect, start, end=None, scope="*"):
    return {"scope": scope, "effect": effect, "started_at": start, "ended_at": end}


def test_exclude_windows_are_added_and_scoped():
    sp = spec(train_from="2026-09-01T00:00:00Z")
    w = (T0 + timedelta(days=2), T0 + timedelta(days=2, hours=3))
    out, summary = with_events(sp, [ev("exclude", *w), ev("exclude", *w, scope="other.series")], T0 + timedelta(days=5))
    assert out.exclude == (w,) and len(summary["excluded_windows"]) == 1


def test_a_break_is_pending_until_enough_data_then_becomes_the_training_start():
    sp = spec(train_from="2026-09-01T00:00:00Z")
    b = T0 + timedelta(days=3)
    early, s1 = with_events(sp, [ev("break", b)], b + timedelta(days=MIN_POST_BREAK_DAYS - 1))
    assert early.train_from == sp.train_from and early.breaks == (b,) and s1["breaks_pending"]
    late, s2 = with_events(sp, [ev("break", b)], b + timedelta(days=MIN_POST_BREAK_DAYS))
    assert late.train_from == b and late.breaks == () and s2["breaks_adopted"]


def test_future_events_and_breaks_before_train_from_do_nothing():
    sp = spec(train_from="2026-09-10T00:00:00Z")
    now = T0 + timedelta(days=30)
    out, _ = with_events(sp, [ev("break", T0 + timedelta(days=2)), ev("break", now + timedelta(days=1))], now)
    assert out.train_from == sp.train_from and out.breaks == ()


def test_prophet_learns_a_level_shift_at_a_pending_break_that_it_misses_without_one():
    # One day after a 2 C step (e.g. a moved sensor): measured 2026-10-20, error 0.009 C with the break
    # regressor vs 0.93 C without -- Prophet's own changepoints lag a fresh step.
    n = 5 * 288
    idx = pd.date_range(T0, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    b = idx[4 * 288]
    y = 28 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 288) + rng.normal(0, 0.05, n)
    y[idx >= b] += 2.0
    sp = spec(seasonalities=[{"name": "daily", "period": "1d", "fourier_order": 3}])
    with_break, _ = with_events(sp, [ev("break", b.to_pydatetime())], idx[-1].to_pydatetime())
    truth = 30 + 0.5 * np.sin(2 * np.pi * np.arange(n, n + 72) / 288)
    err = lambda s: np.mean(np.abs(build_models(s, list(s.seasonalities))["prophet"].forecast(y, 72, idx) - truth))
    e_with, e_without = err(with_break), err(sp)
    assert e_with < 0.1 and e_without > 3 * e_with
