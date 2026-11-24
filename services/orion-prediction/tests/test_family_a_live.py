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
    doc["inputs"] = [{"id": "s.x", "unit": "C", "group": "body", "source": {"table": "biometrics", "node": "n", "key": "x"}},
                     {"id": "s.y", "unit": "W", "group": "room", "source": {"table": "biometrics", "node": "n", "key": "y"}}]
    return fa.parse_config(doc)


def _blend(joints: list[dict]) -> list[dict]:
    """The blended joint rows (family_a); group rows (family_a_body / _room) are tested on their own."""
    return [j for j in joints if j["family"] == fl.FAMILY]


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
    js = _blend(live.store.joints)
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
    refused = [j for j in _blend(live.store.joints) if j["status"] == "refused"]
    assert refused
    assert all("stale" in j["refusal"]["s.y"] for j in refused)
    origins = {j["origin_ts"] for j in refused}
    assert not [r for r in live.store.forecasts if r["origin_ts"] in origins]   # nothing forecast then
    assert live.state.counts["refused"] == len(origins)
    # last reading ends at day7+40; stale after max_stale_min (30) -> the first refused issue is at
    # now > day7+70, i.e. origin >= day7+65 (issued ~6 min after its origin)
    assert min(origins) >= DAY7 + timedelta(minutes=65)


def test_a_target_resting_flat_inside_its_band_is_issued_and_scored_not_refused():
    # Regression, live 2026-11-14: office PM2.5 resting at 0 for 24 h refused every bucket ("one_value").
    # Flatten s.x AT ITS FLOOR for the last 23 h before selection and on: its 24 h gate window is ~96 % one
    # value. (Flat at a non-floor value is a frozen sensor, refused since #3370 -- see the test below.)
    f = _frame()
    rest = float(f.loc[:DAY7 - timedelta(hours=23), "s.x"].quantile(0.05))
    f["s.x"] = f["s.x"].clip(lower=rest)          # a floor it rests on every day, like PM2.5's 0
    f.loc[DAY7 - timedelta(hours=23):, "s.x"] = rest
    live, _l = _live(f)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    assert live.served is not None and live.served.named["s.x"] == fa.TARGET
    for _ in range(14):
        now += G
        live.tick(now)
    js = _blend(live.store.joints)
    assert js and not [j for j in js if j["status"] == "refused"]
    assert live.state.counts["refused"] == 0
    scored = [j for j in js if j["status"] == "scored"]
    assert scored
    # scoring standardises by the TRAINING std, so a flat reading never divides by zero
    for j in scored:
        assert j["n_observed"] == 2 and np.isfinite(j["d2"]) and np.isfinite(j["excess"]) and 0 < j["p"] <= 1
    flat = [r for r in live.store.forecasts if r["series_id"] == fl.PREFIX + "s.x" and r.get("score_status") == "scored"]
    assert flat and all(r["actual"] == rest and np.isfinite(r["abs_error"]) for r in flat)


def test_a_sensor_frozen_after_select_refuses_buckets_until_the_next_select_drops_it():
    """#3370: frozen at a normal-looking value it never read as calm. Issue refuses (flat_beyond_training)
    until the next select (6 h), which drops the input and serves the rest without it."""
    f = _frame()
    live, _l = _live(f)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    assert live.served.named["s.x"] == fa.TARGET and live.served.flat_ceilings["s.x"] < 0.2
    f.loc[DAY7 - timedelta(hours=24):, "s.x"] = 21.37                 # stuck, inside its band, not its floor
    first_select = live.served.run_id
    for _ in range(80):                                                # ~6 h 40 min: one more select
        now += G
        live.tick(now)
    refused = [j for j in _blend(live.store.joints) if j["status"] == "refused"]
    assert refused and all(j["refusal"] == {"s.x": ["flat_beyond_training"]} or
                           j["refusal"] == {"s.x": ("flat_beyond_training",)} for j in refused)
    assert not [r for r in live.store.forecasts if r["origin_ts"] in {j["origin_ts"] for j in refused}]
    assert live.served.run_id != first_select and "s.x" not in live.served.named
    assert live.state.last_select["refused_inputs"]["s.x"] == ["flat_beyond_training"]
    after = [j for j in _blend(live.store.joints) if j["run_id"] == live.served.run_id]
    assert after and all(j["status"] != "refused" for j in after)   # the rest is served without it


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
    js = _blend(live.store.joints)
    assert js and all(j["status"] == "scored" for j in js), {j["status"] for j in js}
    assert all(j["n_observed"] == 2 for j in js)
    groups = [j for j in live.store.joints if j["family"] != fl.FAMILY]
    assert groups and all(j["status"] == "scored" and j["n_observed"] == 1 for j in groups)


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
    from tests.conftest import SERVICE

    for path in [*(SERVICE / "app").glob("*.py"), *(SERVICE / "evals").glob("*.py")]:
        assert not re.search(r"series_id\s*>=", path.read_text()), f"byte-order range on series_id in {path.name}"


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


def test_a_probe_that_died_to_a_new_low_before_select_is_not_served_as_rest():
    """Review round 2: s.x dies to -127 six hours before select. The window minimum is -127, but the
    served floor is the rest floor of normal days, so the dead probe is never exempt as 'rest'."""
    f = _frame()
    f.loc[DAY7 - timedelta(hours=6):, "s.x"] = -127.0
    live, _l = _live(f)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    assert live.served is not None and live.served.floors["s.x"] > -100
    for _ in range(140):                                               # ~11.7 h
        now += G
        live.tick(now)
    issued_with_x = [r for r in live.store.forecasts if r["series_id"] == fl.PREFIX + "s.x"]
    refused = [j for j in _blend(live.store.joints) if j["status"] == "refused" and "s.x" in (j["refusal"] or {})]
    assert refused                                                     # caught within the day, never calm
    # dead since day7 - 6 h; the half-dead day it died on is its flattest day (0.25), so the limit is
    # 0.625 and the refusals start ~15 h in, like real readings frozen at their median (12-15 h)
    late = [r for r in issued_with_x if r["origin_ts"] > DAY7 + timedelta(hours=9, minutes=30)]
    assert not late                                                    # refused from then on


def _seed_pool(live, origin, miss_by_hour):
    """One day of scored champion rows for every target and horizon before ``origin``, served by the rung
    serving now. ``miss_by_hour(hours_before_origin) -> |actual - point|`` sets how hard each hour was."""
    rows = []
    for h, sh in live.served.horizons.items():
        for t in sh.fit.targets:
            for i in range(1, 300):
                o = origin - i * G
                miss = miss_by_hour((origin - o).total_seconds() / 3600.0) * (1 if i % 2 else -1)
                rows.append({"series_id": fl.PREFIX + t, "role": "champion", "model": sh.rung, "horizon_min": h,
                             "origin_ts": o - timedelta(minutes=h), "target_ts": o, "point": 0.0, "actual": miss,
                             "score_status": "scored", "scored_at": o, "interval_source": fl.INTERVAL_SOURCE,
                             "forecast_id": -i, "quantiles": {}})
    live.store.forecasts.extend(rows)


def test_family_ranges_come_from_each_readings_own_live_misses_scaled_by_its_last_hour():
    """#3310: a busy last hour widens the reading's range, a calm one narrows it; validation is the fallback."""
    widths = {}
    for name, last_hour in (("calm", 0.1), ("busy", 2.0)):
        live, _l = _live()
        now = DAY7 + timedelta(minutes=6)
        live.tick(now)                                           # select only (no selection before this tick)
        origin = live.latest_complete(now + G)
        _seed_pool(live, origin, lambda hrs, lh=last_hour: lh if hrs <= 1.0 else 0.5)
        live.issue(now + G)
        mine = [r for r in live.store.forecasts if r["origin_ts"] == origin and r["role"] == "champion"]
        assert mine and {r["interval_source"] for r in mine} == {fl.ADAPTIVE_SOURCE}
        assert live.state.last_issue["n_adaptive"] == len(mine)
        r = next(r for r in mine if r["series_id"] == fl.PREFIX + "s.x" and r["horizon_min"] == 5)
        assert r["quantiles"]["0.1"] < r["point"] < r["quantiles"]["0.9"]
        widths[name] = r["quantiles"]["0.9"] - r["quantiles"]["0.1"]
    # busy 2.0 vs calm 0.1 last hour; a calm hour narrows only down to the pool's difficulty floor
    # (adaptive.FLOOR_Q), so the ratio is ~4, not 20
    assert widths["busy"] > 2 * widths["calm"]


def test_family_ranges_stay_validation_when_disabled_or_when_the_pool_read_fails():
    for kw in ({"adaptive_intervals": False}, {}):
        live, _l = _live(**kw)
        now = DAY7 + timedelta(minutes=6)
        live.tick(now)
        origin = live.latest_complete(now + G)
        _seed_pool(live, origin, lambda hrs: 1.0)
        if not kw:
            def boom(*a, **k):
                raise RuntimeError("db down")
            live.store.family_champion_pool = boom
        live.issue(now + G)
        mine = [r for r in live.store.forecasts if r["origin_ts"] == origin and r["role"] == "champion"]
        assert mine and {r["interval_source"] for r in mine} == {fl.INTERVAL_SOURCE}
        if not kw:
            assert "db down" in live.state.errors["adaptive"]



def test_a_pool_served_by_the_other_rung_does_not_size_this_rungs_range():
    """Owner filter: rungs miss by different amounts, so another rung's rows never size the range."""
    live, _l = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    origin = live.latest_complete(now + G)
    _seed_pool(live, origin, lambda hrs: 1.0)
    for r in live.store.forecasts:
        if r.get("forecast_id", 0) < 0:
            r["model"] = "ridge_var" if r["model"] == "ar" else "ar"
    live.issue(now + G)
    mine = [r for r in live.store.forecasts if r["origin_ts"] == origin and r["role"] == "champion"]
    assert mine and {r["interval_source"] for r in mine} == {fl.INTERVAL_SOURCE}


def test_a_zero_width_adaptive_range_keeps_the_validation_range():
    """>= 80 % identical misses give equal 0.1 / 0.9 offsets; family C refuses a zero-width range."""
    live, _l = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    origin = live.latest_complete(now + G)
    _seed_pool(live, origin, lambda hrs: 1.0)
    for r in live.store.forecasts:
        if r.get("forecast_id", 0) < 0:
            r["actual"] = 1.0                                  # every miss identical: +1.0
    live.issue(now + G)
    mine = [r for r in live.store.forecasts if r["origin_ts"] == origin and r["role"] == "champion"]
    assert mine and all(r["interval_source"] == fl.INTERVAL_SOURCE for r in mine)
    assert all(r["quantiles"]["0.9"] > r["quantiles"]["0.1"] for r in mine)


# --- joint groups (#3575 / #3580): body and room scored apart, published as prediction.joint_surprise.v1 ---

def test_selection_calibrates_each_group_on_its_own_readings(ran):
    live, _l, _n = ran
    champ = next(r for r in live.store.runs if r["kind"] == "champion_blend")
    for h in ("5", "15"):
        g = champ["metrics"][h]["groups"]
        assert {k: v["targets"] for k, v in g.items()} == {"body": ["s.x"], "room": ["s.y"]}
        for v in g.values():
            assert len(v["joint"]["cal_tail"]) >= fa.MIN_CAL and len(v["joint"]["cal_tail_adaptive"]) >= fa.MIN_CAL
            assert len(v["joint"]["cov"]) == 1                       # the group's own block, nothing borrowed
        assert champ["metrics"][h]["groups_skipped"] == {}


def test_each_bucket_gets_one_row_per_group_scored_on_that_groups_readings_only(ran):
    live, _l, _n = ran
    rows = [j for j in live.store.joints if j["family"] in fl.GROUP_OF_FAMILY]
    blend = {(j["origin_ts"], j["horizon_min"]) for j in _blend(live.store.joints)}
    for fam, reading in (("family_a_body", "s.x"), ("family_a_room", "s.y")):
        mine = [j for j in rows if j["family"] == fam]
        assert {(j["origin_ts"], j["horizon_min"]) for j in mine} == blend   # one per blended bucket
        scored = [j for j in mine if j["status"] == "scored"]
        assert len(scored) >= 40
        for j in scored:
            assert j["n_observed"] == 1 and set(j["top_shares"]) == {reading}
            assert j["excess"] == pytest.approx(-np.log(j["p"]) - j["calm_point"])
        # the group's own hour-before difficulty turns its p adaptive, as for the blend
        late = [j for j in scored if j["origin_ts"] >= DAY7 + timedelta(minutes=80)]
        assert late and all(j["calibration"] == fa.ADAPTIVE and j["difficulty"] is not None for j in late)


def test_only_fresh_adaptive_group_rows_become_events_and_they_validate():
    from orion.schemas.prediction import PredictionJointSurpriseV1

    live, _l = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    events = []
    for _ in range(26):
        now += G
        live.tick(now)
        events += live.take_pending_events()
    assert live.take_pending_events() == []                         # taking clears the queue
    assert events and {e["group"] for e in events} == {"body", "room"}
    for e in events:
        ev = PredictionJointSurpriseV1.model_validate(e)
        assert ev.calibration == "adaptive" and ev.family == "family_a"
        row = next(j for j in live.store.joints if str(j["joint_id"]) == ev.joint_id)
        assert row["family"] == fl.GROUP_FAMILIES[ev.group] and row["excess"] == pytest.approx(ev.excess)
    # every scored adaptive group row of the run was published once (none were older than the window)
    adaptive = [j for j in live.store.joints if j["family"] in fl.GROUP_OF_FAMILY and j.get("calibration") == fa.ADAPTIVE]
    assert sorted(e["joint_id"] for e in events) == sorted(str(j["joint_id"]) for j in adaptive)
    assert not [j for j in live.store.joints if j["family"] == fl.FAMILY and str(j["joint_id"]) in
                {e["joint_id"] for e in events}]                    # the blend is never published


def test_a_backlog_scored_late_is_stored_but_never_published_as_now():
    live, _l = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    for _ in range(20):                       # issue only: scoring is down
        now += G
        live.issue(now)
    live.score(now + timedelta(hours=2))      # the backlog scores two hours late
    assert [j for j in live.store.joints if j["family"] in fl.GROUP_OF_FAMILY and j["status"] == "scored"]
    assert live.take_pending_events() == []


def test_events_off_queues_nothing_and_rows_are_still_scored():
    live, _l = _live(events_enabled=False)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    for _ in range(26):
        now += G
        live.tick(now)
    assert live.take_pending_events() == [] and live.state.counts["events_queued"] == 0
    assert live.state.counts["group_scored"] > 0


def test_a_group_with_no_target_is_skipped_and_writes_no_rows():
    cfg = _cfg()
    cfg = fa.FamilyAConfig(grain_min=cfg.grain_min, horizons_min=cfg.horizons_min, hygiene=cfg.hygiene,
                           inputs=tuple(fa.InputSpec(id=i.id, unit=i.unit, source=i.source, group="body")
                                        for i in cfg.inputs))
    live = fl.FamilyALive(fl.MemoryStore(), cfg, Recording(_frame()), settle_sec=60, select_interval_sec=6 * 3600,
                          seeds=(0, 1), train_days=7)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    assert live.state.last_select["groups_skipped"] == {"5": {"room": "no_targets"}, "15": {"room": "no_targets"}}
    for _ in range(3):
        now += G
        live.tick(now)
    fams = {j["family"] for j in live.store.joints}
    assert fams == {fl.FAMILY, "family_a_body"}


def test_a_refused_bucket_writes_a_refused_row_for_every_group_too():
    f = _frame()
    f.loc[DAY7 + timedelta(minutes=40):, "s.y"] = np.nan
    live, _l = _live(f)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    for _ in range(14):
        now += G
        live.tick(now)
    refused = [j for j in live.store.joints if j["status"] == "refused"]
    by = {}
    for j in refused:
        by.setdefault((j["origin_ts"], j["horizon_min"]), set()).add(j["family"])
    assert by and all(v == {fl.FAMILY, "family_a_body", "family_a_room"} for v in by.values())


def test_a_run_without_groups_scores_its_old_group_rows_missing_not_with_another_yardstick():
    """Rows issued under a run whose stored metrics have no groups (before this patch) get no group rows;
    a group row whose run lost the group is scored missing ('no_group'), never with the blend's calibration."""
    live, _l = _live()
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    now += G
    live.issue(now)
    champ = next(r for r in live.store.runs if r["kind"] == "champion_blend")
    for h in ("5", "15"):
        champ["metrics"][h]["groups"].pop("room")
    fresh, _l2 = _live(store=live.store)           # a restart: reads the run back from its stored metrics
    fresh.score(now + timedelta(minutes=30))
    room = [j for j in live.store.joints if j["family"] == "family_a_room"]
    assert room and all(j["status"] == "missing" and j["calibration"] == "no_group" for j in room)


def test_an_input_in_no_group_is_scored_in_the_blend_only():
    """office sound (#3581): forecast, scored in the blend, never in a group's joint."""
    cfg = _cfg()
    cfg = fa.FamilyAConfig(grain_min=cfg.grain_min, horizons_min=cfg.horizons_min, hygiene=cfg.hygiene,
                           inputs=(fa.InputSpec(id="s.x", unit="C", source=cfg.inputs[0].source, group="body"),
                                   fa.InputSpec(id="s.y", unit="W", source=cfg.inputs[1].source, group=fa.NO_GROUP)))
    live = fl.FamilyALive(fl.MemoryStore(), cfg, Recording(_frame()), settle_sec=60, select_interval_sec=6 * 3600,
                          seeds=(0, 1), train_days=7)
    now = DAY7 + timedelta(minutes=6)
    live.tick(now)
    for _ in range(3):
        now += G
        live.tick(now)
    champ = next(r for r in live.store.runs if r["kind"] == "champion_blend")
    assert {g: v["targets"] for g, v in champ["metrics"]["5"]["groups"].items()} == {"body": ["s.x"]}
    assert champ["metrics"]["5"]["groups_skipped"] == {"room": "no_targets"}
    assert "s.y" in champ["metrics"]["5"]["targets"]                    # still forecast, still in the blend
    assert {j["family"] for j in live.store.joints} == {fl.FAMILY, "family_a_body"}


def test_the_config_rejects_an_unclassified_input():
    from tests.test_family_a import CONFIG_PATH
    import yaml

    doc = yaml.safe_load(CONFIG_PATH.read_text())
    doc["inputs"] = [{"id": "s.x", "unit": "C", "source": {"table": "biometrics", "node": "n", "key": "x"}}]
    with pytest.raises(fa.FamilyAConfigError, match="group"):
        fa.parse_config(doc)
    live_cfg = fa.parse_config(yaml.safe_load(CONFIG_PATH.read_text()))
    assert live_cfg.group_of("office.sound_dba") == fa.NO_GROUP            # until #3581
    assert live_cfg.group_of("office.temp_c") == "body"                    # until family C measures Orion's share
    assert {i.id for i in live_cfg.inputs if i.group == "room"} == {
        "office.humidity_pct", "office.eco2_ppm", "office.tvoc_ppb", "office.pm25_ug_m3", "office.lux"}
