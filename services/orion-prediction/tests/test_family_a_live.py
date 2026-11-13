"""Family A in the live loop (app/family_a_live.py): select -> issue -> score on the in-memory store."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app import family_a as fa
from app import family_a_live as fl
from app.backtest import q_key

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
G = timedelta(minutes=5)
DAY7 = T0 + timedelta(days=7)


def _frame(days=9, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 288
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
    idx = pd.date_range(T0, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"s.x": 20 + x, "s.y": 40 + 3 * y}, index=idx)


def _cfg() -> fa.FamilyAConfig:
    from tests.test_family_a import CONFIG_PATH
    import yaml

    doc = yaml.safe_load(CONFIG_PATH.read_text())
    doc["inputs"] = [{"id": "s.x", "unit": "C", "source": {"table": "biometrics", "node": "n", "key": "x"}},
                     {"id": "s.y", "unit": "W", "source": {"table": "biometrics", "node": "n", "key": "y"}}]
    return fa.parse_config(doc)


class Recording:
    """frame_loader that records every [start, end) asked for."""

    def __init__(self, frame: pd.DataFrame):
        self.inner = fl.frame_loader(frame, 5)
        self.calls: list[tuple[datetime, datetime]] = []

    def __call__(self, start, end):
        self.calls.append((start, end))
        return self.inner(start, end)


def _live(frame=None, store=None, **kw):
    loader = Recording(frame if frame is not None else _frame())
    return fl.FamilyALive(store or fl.MemoryStore(), _cfg(), loader, settle_sec=60, select_interval_sec=6 * 3600,
                          seeds=(0, 1), train_days=7, **kw), loader


@pytest.fixture(scope="module")
def ran():
    """Select at day 7, then 2 hours of 5-min ticks (issue + score every bucket)."""
    live, loader = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)                              # no selection yet: select only
    assert live.served is not None
    for _ in range(26):
        now += G
        live.tick(now)
    return live, loader, now


def test_selection_stores_the_rungs_and_everything_scoring_needs(ran):
    live, _l, _n = ran
    runs = live.store.runs
    assert sorted(r["kind"] for r in runs) == ["champion_blend", "model", "model"]
    champ = next(r for r in runs if r["kind"] == "champion_blend")
    assert champ["series_id"] == fl.FAMILY and champ["run_id"] == live.served.run_id
    for h in ("5", "15"):
        m = champ["metrics"][h]
        assert m["rung"] in fa.RUNGS and set(m["targets"]) == {"s.x", "s.y"}
        assert m["joint"]["cal_tail"] and m["joint"]["cal_tail_adaptive"]
        assert len(m["backtest"]) == 2                      # two seeds
        assert champ["assignment"][h] == m["rung"]
    # a coupled pair: the other reading leads, so ridge_var must win and serve
    assert champ["assignment"] == {"5": "ridge_var", "15": "ridge_var"}
    for r in runs:    # the Postgres store writes these with allow_nan=False
        json.dumps({k: r[k] for k in ("metrics", "assignment", "errors", "residual_paths")}, allow_nan=False)


def test_every_bucket_is_issued_before_its_target_completes_from_data_before_it(ran):
    live, loader, _n = ran
    fc = live.store.forecasts
    assert fc and all(r["series_id"].startswith(fl.PREFIX) and r["role"] == "champion" for r in fc)
    for r in fc:
        assert r["issued_at"] < r["target_ts"] + G
        assert r["issued_at"] >= r["origin_ts"] + G           # the origin bucket is complete
        assert r["target_ts"] == r["origin_ts"] + timedelta(minutes=r["horizon_min"])
    origins = sorted({r["origin_ts"] for r in fc})
    assert len(origins) == 26 and all(b - a == G for a, b in zip(origins, origins[1:]))
    # the issue loads end at the origin bucket's end, never inside a target bucket
    issue_loads = [c for c in loader.calls if c[1] - c[0] == timedelta(hours=fl.GATE_HOURS)]
    assert len(issue_loads) == 26
    for (_s, end), origin in zip(issue_loads, origins):
        assert end == origin + G
    assert {(r["series_id"], r["model"]) for r in live.store.shadows} == {
        (fl.PREFIX + "s.x", "ar"), (fl.PREFIX + "s.y", "ar")}


def test_reading_rows_score_against_the_real_bucket_mean(ran):
    live, _l, _n = ran
    frame = _frame()
    scored = [r for r in live.store.forecasts if r.get("score_status") == "scored"]
    assert len(scored) > 60
    for r in scored[:20]:
        v = frame.at[pd.Timestamp(r["target_ts"]), r["series_id"][len(fl.PREFIX):]]
        assert r["actual"] == pytest.approx(v)
        assert r["abs_error"] == pytest.approx(abs(v - r["point"]))
        q = r["quantiles"]
        assert r["in_80"] == (q[q_key(0.1)] <= v <= q[q_key(0.9)])
    cov80 = np.mean([r["in_80"] for r in scored])
    assert 0.6 < cov80 <= 1.0


def test_joint_rows_score_and_turn_adaptive_once_an_hour_is_known(ran):
    live, _l, _n = ran
    js = live.store.joints
    scored = [j for j in js if j["status"] == "scored"]
    assert len(scored) >= 40
    for j in scored:
        assert 0 < j["p"] <= 1 and np.isfinite(j["excess"]) and j["n_observed"] == 2
        assert j["excess"] == pytest.approx(-np.log(j["p"]) - j["calm_point"])
        assert j["top_shares"]
    early = [j for j in scored if j["origin_ts"] < DAY7 + timedelta(minutes=30)]
    assert early and all(j["difficulty"] is None and j["calibration"] == "static:no_difficulty" for j in early)
    late = [j for j in scored if j["origin_ts"] >= DAY7 + timedelta(minutes=80)]
    assert late and all(j["difficulty"] is not None and j["calibration"] == fa.ADAPTIVE for j in late)
    assert all(j["p_static"] is not None for j in scored)


def test_gate_refuses_a_stale_input_and_stores_why_never_a_degraded_forecast():
    f = _frame()
    f.loc[DAY7 + timedelta(minutes=40):, "s.y"] = np.nan     # s.y goes silent
    live, _l = _live(f)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    for _ in range(14):
        now += G
        live.tick(now)
    refused = [j for j in live.store.joints if j["status"] == "refused"]
    assert refused
    assert all("stale" in j["refusal"]["s.y"] for j in refused)
    origins = {j["origin_ts"] for j in refused}
    assert not [r for r in live.store.forecasts if r["origin_ts"] in origins]   # nothing forecast then
    assert live.state.counts["refused"] == len(origins)
    # last reading ends at day7+40; stale after max_stale_min (30) -> the first refused issue is at
    # now > day7+70, i.e. origin >= day7+65 (issued ~6 min after its origin)
    assert min(origins) >= DAY7 + timedelta(minutes=65)


def test_a_horizon_whose_target_already_completed_is_skipped():
    live, _l = _live()
    live.tick(DAY7 + timedelta(minutes=6))                    # select + first issue (origin day7 + 0)
    live.last_origin = DAY7 - G
    late = DAY7 + timedelta(minutes=10, seconds=30)           # origin day7+0; 5 min target done at +10
    live.store.joints.clear()
    live.store.forecasts.clear()
    live.issue(late)
    assert {j["horizon_min"] for j in live.store.joints} == {15}
    assert live.state.counts["late_horizons"] == 1


def test_rows_issued_before_a_restart_are_scored_with_the_run_that_issued_them(ran):
    live, _l, now = ran
    store = live.store
    old_run = live.served.run_id
    pending = [j for j in store.joints if j["status"] == "issued"]
    assert pending
    fresh, _l2 = _live(store=store)
    fresh.score(now + timedelta(minutes=30))
    for j in pending:
        assert j["status"] == "scored" and j["run_id"] == old_run


def test_prune_drops_old_reading_rows_and_keeps_joint_rows():
    store = fl.MemoryStore()
    old = {"series_id": fl.PREFIX + "s.x", "role": "champion", "origin_ts": T0, "horizon_min": 5,
           "target_ts": T0 + G}
    store.insert_family_issue([old], [{"family": fl.FAMILY, "origin_ts": T0, "horizon_min": 5, "target_ts": T0 + G,
                                       "status": "issued"}])
    assert store.prune_family(fl.PREFIX, T0 + timedelta(days=1)) == 1
    assert not store.forecasts and store.joints


def test_a_scoring_backlog_past_the_joint_limit_never_strands_a_joint_row():
    # Review finding: readings past the joint limit were scored without their joint row, which later
    # found no readings and was written missing ("too_few_readings") for good.
    live, _l = _live()
    live.due_limit = 5
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)                                  # select
    for _ in range(15):                             # 15 buckets issued, scoring down
        now += G
        live.issue(now)
    for _ in range(12):                             # recovery: 5 joint rows per pass
        live.score(now + timedelta(minutes=20))
    js = live.store.joints
    assert js and all(j["status"] == "scored" for j in js), {j["status"] for j in js}
    assert all(j["n_observed"] == 2 for j in js)


def test_a_thin_calibration_pool_refuses_the_selection(monkeypatch):
    monkeypatch.setattr(fa, "MIN_CAL", 10 ** 6)
    live, _l = _live()
    live.tick(DAY7 + timedelta(minutes=6))
    assert live.served is None
    assert "calibration pool" in live.state.errors["select"]
    assert not live.store.runs


def test_family_prefix_queries_are_collation_independent():
    """Regression (2026-11-13): a ``series_id >= 'family_a:' AND series_id < 'family_a;'`` range matched 0
    of 1,326 live rows under the database's en_US.utf8 collation (punctuation is ignored at the first sort
    level), so no Family A reading, joint or shadow row was ever scored or pruned. Python's byte-order
    comparison could not catch it. Every prefix query must use ``starts_with`` and pass the bare prefix."""
    import inspect
    import re

    from app import store as st

    calls = []

    class Cur:
        rowcount = 0

        def execute(self, sql, params=()):
            calls.append((sql, params))
            return self

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            calls.append((sql, params))
            return Cur()

    class Pool:
        def connection(self):
            return Conn()

    s = st.PostgresStore.__new__(st.PostgresStore)
    s.pool = Pool()
    s._all = lambda sql, params=(): calls.append((sql, params)) or []
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)
    s.family_due(fl.FAMILY, fl.PREFIX, now)
    s.family_due_shadow(fl.PREFIX, now)
    s.prune_family(fl.PREFIX, now)
    prefixed = [(sql, p) for sql, p in calls if "series_id" in sql and "FROM prediction_joint_surprise" not in sql]
    assert len(prefixed) == 4  # due readings, due shadows, prune x2 tables
    for sql, params in prefixed:
        assert "starts_with(series_id, %s)" in sql, sql
        assert not re.search(r"series_id\s*[<>]=?\s*%s", sql), sql
        assert params[0] == fl.PREFIX
    assert "series_id >=" not in inspect.getsource(st)


def _calm_eval():
    import importlib.util
    from tests.conftest import SERVICE

    spec = importlib.util.spec_from_file_location("calm_eval", SERVICE / "evals" / "run_family_a_calm_live_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _joint_rows(days: float, p_of, seed=0):
    rng = np.random.default_rng(seed)
    calm = fa.calm_point(1000)
    rows = []
    for h in (5, 15):
        for i in range(int(days * 288)):
            p = float(p_of(rng))
            rows.append({"horizon_min": h, "status": "scored", "target_ts": T0 + i * G, "p": p, "p_static": p,
                         "p_adaptive": p, "calm_point": calm, "excess": -np.log(p) - calm,
                         "calibration": fa.ADAPTIVE, "refusal": None})
    rows.append({"horizon_min": 5, "status": "refused", "target_ts": T0, "refusal": {"s.y": ["stale"]}})
    return rows


def test_calm_eval_passes_a_calibrated_week_and_waits_for_seven_days():
    ev = _calm_eval()

    order = iter(np.tile(np.random.default_rng(5).permutation(1000), 10))

    def uniform(_rng):     # exactly uniform over whole cycles: a calibrated score, not a lucky draw
        return (1 + next(order)) / 1001

    v, why = ev.verdict(ev.summarize(_joint_rows(8, uniform), 5), 7.0)
    assert v == "PASS", why
    s = ev.summarize(_joint_rows(2, uniform), 5)
    assert s["5"]["refusal_reasons"] == {"s.y:stale": 1}
    assert ev.verdict(s, 7.0)[0].startswith("NOT YET")
    assert ev.verdict(ev.summarize(_joint_rows(8, lambda rng: 0.01), 5), 7.0)[0] == "FAIL"
