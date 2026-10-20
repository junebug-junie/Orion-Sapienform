from __future__ import annotations

import asyncio
import copy
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import app.engine as engine_mod
from app.engine import Engine
from app.models import build_models as real_build_models
from app.series import parse_registry
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.prediction import PredictionForecastV1
from tests.helpers import daily_series, points_of, spec_doc

FAST = ("naive", "seasonal_naive", "window_average_1h")


@pytest.fixture(autouse=True)
def fast_zoo(monkeypatch):
    monkeypatch.setattr(engine_mod, "build_models",
                        lambda spec, usable: {k: v for k, v in real_build_models(spec, usable).items() if k in FAST})


class FakeStore:
    def __init__(self, points):
        self.raw = sorted(points)
        self.runs: list[dict] = []
        self.forecasts: list[dict] = []
        self._ids = itertools.count(1)

    cov_raw: dict = {}

    def points(self, spec, since, until=None):
        raw = self.cov_raw.get(spec.series_id, self.raw)
        return [(t, v) for t, v in raw if t >= since and (until is None or t < until)]

    def insert_runs(self, rows):
        self.runs.extend(copy.deepcopy(rows))

    def latest_selection(self, series_id):
        rows = [r for r in self.runs if r["series_id"] == series_id]
        if not rows:
            return []
        last = max(rows, key=lambda r: r["fitted_at"])["selection_id"]
        return [r for r in rows if r["selection_id"] == last]

    def insert_forecasts(self, rows):
        keys = {(f["series_id"], f["role"], f["origin_ts"], f["horizon_min"]) for f in self.forecasts}
        for r in rows:
            k = (r["series_id"], r["role"], r["origin_ts"], r["horizon_min"])
            if k not in keys:
                self.forecasts.append({**r, "forecast_id": next(self._ids), "scored_at": None})
        return len(rows)

    def last_origin(self, sid):
        o = [f["origin_ts"] for f in self.forecasts if f["series_id"] == sid and f["role"] == "champion"]
        return max(o) if o else None

    def last_issued(self, sid):
        o = [f["issued_at"] for f in self.forecasts if f["series_id"] == sid]
        return max(o) if o else None

    def latest_forecast(self, sid):
        o = self.last_origin(sid)
        return [f for f in self.forecasts if f["role"] == "champion" and f["origin_ts"] == o]

    def due_for_scoring(self, sid, before, limit=2000):
        return [f for f in self.forecasts if f["scored_at"] is None and f["target_ts"] < before][:limit]

    def write_scores(self, scores):
        by_id = {f["forecast_id"]: f for f in self.forecasts}
        for s in scores:
            by_id[s["forecast_id"]].update(s)

    def live_scores(self, sid, since):
        return [f for f in self.forecasts if f.get("score_status") == "scored" and f["target_ts"] >= since]


def _setup(days=12.0, noise=0.05):
    reg = parse_registry(spec_doc(train_window="10d"))
    spec = reg.get("test.temp")
    s = daily_series(days, amp=1.0, noise=noise, seed=3)
    store = FakeStore(points_of(s))
    end = s.index[-1].to_pydatetime() + timedelta(minutes=5)
    return reg, spec, s, store, end


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


async def _sync(fn, *a):
    return fn(*a)


def _engine(reg, store, clock, published):
    async def publish(channel, env):
        published.append((channel, env))
    return Engine(reg, store, backtest_windows=14, backtest_step_min=180, select_interval_sec=6 * 3600,
                  settle_sec=60, publish_enabled=True, publish=publish,
                  source=ServiceRef(name="orion-prediction", version="test", node="test"), clock=clock)


def test_tick_selects_issues_both_roles_and_publishes_a_valid_forecast(monkeypatch):
    reg, spec, s, store, end = _setup()
    published = []
    eng = _engine(reg, store, Clock(end + timedelta(seconds=90)), published)
    asyncio.run(eng.tick(_sync))
    kinds = sorted(r["kind"] for r in store.runs)
    assert kinds == ["baseline_blend", "champion_blend", "model", "model", "model"]
    roles = {(f["role"], f["horizon_min"]) for f in store.forecasts}
    assert roles == {(r, h) for r in ("champion", "baseline") for h in (5, 60, 360)}
    fc = [PredictionForecastV1.model_validate(e.payload) for c, e in published if c == "orion:prediction:forecast"]
    assert len(fc) == 1
    p = fc[0]
    assert [st.horizon_min for st in p.steps] == [5, 60, 360]
    for st in p.steps:
        vals = [st.quantiles[k] for k in sorted(st.quantiles, key=float)]
        assert vals == sorted(vals) and vals[0] < st.point < vals[-1]
    assert p.exceedance[0].name == "hot" and p.exceedance[0].n_paths >= 10
    # The origin is the last complete bucket, and targets are origin + horizon.
    assert p.origin_ts == (end - timedelta(minutes=5))
    assert p.steps[0].target_ts == p.origin_ts + timedelta(minutes=5)


def test_same_bucket_is_not_reissued_and_next_bucket_is():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    n = len(store.forecasts)
    clock.t += timedelta(seconds=30)
    asyncio.run(eng.tick(_sync))
    assert len(store.forecasts) == n
    # A new bucket's readings arrive.
    t = end
    store.raw += [(t + timedelta(seconds=k), 28.0) for k in range(0, 300, 30)]
    clock.t = end + timedelta(minutes=5, seconds=90)
    asyncio.run(eng.tick(_sync))
    assert len(store.forecasts) == 2 * n


def test_scoring_reducer_scores_real_buckets_and_marks_gaps_missing():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    # Next hour of truth arrives, except the bucket the 5-min forecast targets.
    future = daily_series(1, amp=1.0, noise=0.05, seed=4, start=end)
    future.iloc[0] = np.nan
    store.raw += points_of(future)
    clock.t = end + timedelta(hours=1, minutes=10)
    eng.score(spec, clock.t)
    by_h = {(f["role"], f["horizon_min"]): f for f in store.forecasts if f["origin_ts"] == end - timedelta(minutes=5)}
    assert by_h[("champion", 5)]["score_status"] == "missing"
    assert by_h[("champion", 60)]["score_status"] == "scored"
    assert by_h[("champion", 60)]["actual"] == pytest.approx(future.iloc[11])
    assert by_h[("champion", 360)]["scored_at"] is None   # not due yet


def test_stale_input_issues_nothing_and_logs_the_alert_once(caplog):
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    assert eng.state["test.temp"].alerts == {}
    n = len(store.forecasts)
    clock.t = end + timedelta(hours=2)
    caplog.set_level("WARNING", logger="orion-prediction.engine")
    asyncio.run(eng.tick(_sync))
    asyncio.run(eng.tick(_sync))
    assert len(store.forecasts) == n
    assert set(eng.state["test.temp"].alerts) == {"input_stale"}
    assert sum("prediction_alerts_changed" in r.message for r in caplog.records) == 1


def test_restart_restores_the_stored_selection_instead_of_reselecting():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    asyncio.run(_engine(reg, store, clock, []).tick(_sync))
    n_runs = len(store.runs)
    fresh = _engine(reg, store, clock, [])
    asyncio.run(fresh.tick(_sync))
    assert len(store.runs) == n_runs
    st = fresh.state["test.temp"]
    assert st.served is not None and st.served.residuals["champion"].shape[1] == spec.max_horizon_steps


def test_degenerate_output_is_refused_and_alerted():
    reg, spec, s, store, end = _setup(noise=0.0)
    flat = [(t, 28.0) for t, _ in store.raw]   # perfectly flat history -> zero-width intervals
    store.raw = flat
    published = []
    eng = _engine(reg, store, Clock(end + timedelta(seconds=90)), published)
    asyncio.run(eng.tick(_sync))
    assert store.forecasts == []
    assert "degenerate_forecast" in eng.state["test.temp"].alerts
    assert not [1 for c, _ in published if c == "orion:prediction:forecast"]


def test_live_metrics_compare_champion_to_baseline_on_the_same_origins():
    reg, spec, s, store, end = _setup()
    eng = _engine(reg, store, Clock(end), [])
    o = end - timedelta(hours=1)
    for role, err in (("champion", 0.5), ("baseline", 1.0)):
        store.forecasts.append({"series_id": "test.temp", "role": role, "origin_ts": o, "horizon_min": 60,
                                "target_ts": o + timedelta(hours=1), "abs_error": err, "in_80": True,
                                "in_95": True, "score_status": "scored", "scored_at": end})
    live = eng.live_metrics(spec, end + timedelta(minutes=1))
    assert live["horizons"]["60"]["rel_mae_vs_baseline"] == pytest.approx(0.5)
    assert live["rel_mae_vs_baseline_24h"] == pytest.approx(0.5)


def test_restore_refuses_a_selection_that_no_longer_fits_the_registry():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    asyncio.run(_engine(reg, store, clock, []).tick(_sync))
    changed = parse_registry(spec_doc(train_window="10d", horizons=["5min", "1h", "12h"]))
    fresh = _engine(changed, store, clock, [])
    assert fresh.restore(changed.get("test.temp")) is False


def test_build_forecast_refuses_stale_input_even_outside_issue():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    later = end + timedelta(hours=3)
    series = eng.load_series(spec, later, timedelta(days=10))
    with pytest.raises(engine_mod.InputStale):
        eng.build_forecast(spec, series, eng.latest_complete_bucket(spec, later), later)


def test_one_series_failing_does_not_stop_the_others():
    doc = spec_doc(train_window="10d")
    other = dict(doc["series"][0], series_id="test.other")
    reg = parse_registry({"version": 1, "series": [doc["series"][0], other]})
    _, spec, s, store, end = _setup()
    real_points = store.points

    def points(sp, since, until=None):
        if sp.series_id == "test.temp":
            raise ConnectionError("pg down")
        return real_points(sp, since, until)
    store.points = points
    eng = _engine(reg, store, Clock(end + timedelta(seconds=90)), [])
    asyncio.run(eng.tick(_sync))
    assert "test.temp" not in {f["series_id"] for f in store.forecasts}
    assert {f["series_id"] for f in store.forecasts} == {"test.other"}
    assert "selection_failed" in eng.state["test.temp"].alerts or "tick_failed" in eng.state["test.temp"].alerts


def _cov_setup():
    doc = spec_doc(train_window="10d", covariates=["test.watts"])
    doc["series"].append({"series_id": "test.watts", "role": "covariate", "unit": "W",
                          "source": {"kind": "biometrics", "node": "n", "key": "w"}, "grain": "5min",
                          "consumers": ["tests"]})
    reg = parse_registry(doc)
    spec = reg.get("test.temp")
    s = daily_series(12, amp=1.0, noise=0.05, seed=3)
    store = FakeStore(points_of(s))
    watts = pd.Series(400.0 + 50 * np.sin(np.arange(len(s)) / 20.0), index=s.index)
    store.cov_raw = {"test.watts": points_of(watts)}
    return reg, spec, s, store, s.index[-1].to_pydatetime() + timedelta(minutes=5)


def test_a_dead_covariate_falls_back_to_the_baseline_for_its_steps_only(monkeypatch):
    import dataclasses
    monkeypatch.setattr(engine_mod, "build_models", lambda spec, usable: {
        k: v for k, v in real_build_models(spec, usable).items() if k in FAST + ("direct_arx",)})
    reg, spec, s, store, end = _cov_setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    st = eng.state["test.temp"]
    assert st.served is not None
    # Serve direct_arx at the 5-min horizon, then kill its covariate for the last hour.
    sel = dataclasses.replace(st.served.selection, assignment={**st.served.selection.assignment, 5: "direct_arx"})
    st.served = dataclasses.replace(st.served, selection=sel)
    origin = eng.latest_complete_bucket(spec, clock.t)
    series = eng.load_series(spec, clock.t, timedelta(days=10))
    ok = eng.build_forecast(spec, series, origin, clock.t)
    assert ok["failed_models"] == {} and ok["payload"].model.startswith("direct_arx@5")
    cut = end - timedelta(hours=1)
    store.cov_raw["test.watts"] = [(t, v) for t, v in store.cov_raw["test.watts"] if t < cut]
    out = eng.build_forecast(spec, series, origin, clock.t)
    assert "direct_arx" in out["failed_models"] and "no reading" in out["failed_models"]["direct_arx"]
    assert out["fallback_steps"] >= 1 and not out["payload"].model.startswith("direct_arx")
    roles = {r["role"] for r in out["rows"]}
    assert roles == {"champion", "baseline"}          # the baseline blend is still issued


def test_restore_refuses_a_selection_made_with_different_covariates():
    reg, spec, s, store, end = _cov_setup()
    clock = Clock(end + timedelta(seconds=90))
    asyncio.run(_engine(reg, store, clock, []).tick(_sync))
    plain = parse_registry(spec_doc(train_window="10d"))
    assert _engine(plain, store, clock, []).restore(plain.get("test.temp")) is False
    assert _engine(reg, store, clock, []).restore(spec) is True


def test_restore_refuses_a_selection_from_another_gate_version():
    reg, spec, s, store, end = _setup()
    clock = Clock(end + timedelta(seconds=90))
    asyncio.run(_engine(reg, store, clock, []).tick(_sync))
    for r in store.runs:
        if r["kind"] in ("champion_blend", "baseline_blend"):
            r["metrics"] = {k: v for k, v in r["metrics"].items() if k != "gate_version"}
    assert _engine(reg, store, clock, []).restore(spec) is False


def test_change_events_mask_training_and_an_adopted_break_forces_a_reselect():
    from app.events import MIN_POST_BREAK_DAYS
    reg, spec, s, store, end = _setup(days=12.0)
    clock = Clock(end + timedelta(seconds=90))
    store.events = []
    store.change_events = lambda: store.events
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    first = eng.state["test.temp"].served
    # A window two days ago is masked out of the training view.
    w0, w1 = end - timedelta(days=2), end - timedelta(days=2) + timedelta(hours=2)
    store.events.append({"scope": "*", "effect": "exclude", "started_at": w0, "ended_at": w1})
    eff = eng.effective(spec, clock.t)
    from app.frame import training_slice
    view = training_slice(eng.load_series(eff, clock.t, timedelta(days=10)), eff)
    assert view[(view.index >= w0) & (view.index < w1)].isna().all()
    # A break old enough to adopt moves the training start and makes the selection stale at once.
    brk = end - timedelta(days=MIN_POST_BREAK_DAYS, minutes=1)
    store.events.append({"scope": "*", "effect": "break", "started_at": brk, "ended_at": None})
    eff = eng.effective(spec, clock.t)
    assert eff.train_from == brk and eng.selection_due(eff, clock.t)
    asyncio.run(eng.tick(_sync))
    second = eng.state["test.temp"].served
    assert second.selection_id != first.selection_id and second.train_from == brk
    assert eng.state["test.temp"].events["breaks_adopted"]
