"""Family C (app/family_c.py): effect table from randomised switchbacks, the choice guard, the
efference copy and its miss score, the model gate."""
from __future__ import annotations

import math
from statistics import NormalDist
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app import family_c as fc
from app.causal import CausalError, design_switchback

GRAIN = 5
EXP_START = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _experiment(effect: float, seed: int = 0, n_blocks: int = 12, block_min: int = 20, burn: int = 10,
                noise: float = 0.05) -> tuple[dict, list[dict], pd.DataFrame]:
    """A frame of 3 readings over PROFILE_DAYS of history + the experiment: 'hot' moves by `effect`
    during ON blocks after the burn-in, 'flat' never moves, 'gappy' has no reading in the experiment."""
    design = design_switchback(EXP_START, block_min, n_blocks, seed=seed)
    exp = {"experiment_id": f"exp-{seed}", "action": "vision.focus:walkway", "burn_in_min": burn,
           "design": design, "status": "analyzed"}
    start = EXP_START - timedelta(days=fc.PROFILE_DAYS)
    end = EXP_START + timedelta(minutes=block_min * n_blocks)
    idx = pd.date_range(start, end - timedelta(minutes=GRAIN), freq=f"{GRAIN}min", tz="UTC")
    rng = np.random.default_rng(seed + 100)
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    daily = np.sin(2 * np.pi * minutes / 1440)
    hot = 26 + daily + rng.normal(0, noise, len(idx))
    for b in design["blocks"]:
        if b["arm"] == "on":
            m = (idx >= pd.Timestamp(b["start"]) + pd.Timedelta(minutes=burn)) & (idx < pd.Timestamp(b["end"]))
            hot[m] += effect
    flat = 100 + 5 * daily + rng.normal(0, noise, len(idx))
    gappy = 40 + rng.normal(0, noise, len(idx))
    gappy[idx >= pd.Timestamp(EXP_START)] = np.nan
    frame = pd.DataFrame({"hot": hot, "flat": flat, "gappy": gappy}, index=idx)
    return exp, design["blocks"], frame


UNITS = {"hot": "C", "flat": "W", "gappy": "%"}


def _rows(effect=0.5, seed=0, **kw):
    exp, blocks, frame = _experiment(effect, seed, **kw)
    return {r.target: r for r in fc.rows_from_experiment(exp, blocks, frame, UNITS, GRAIN, n_perm=500)}


def test_table_recovers_a_known_effect_and_a_null():
    rows = _rows(0.5)
    hot, flat = rows["hot"], rows["flat"]
    assert hot.method == fc.METHOD_RANDOMISED
    assert hot.ci_lo < 0.5 < hot.ci_hi and hot.ci_lo > 0
    assert (hot.window_start_min, hot.window_end_min) == (10, 20)
    assert hot.n_on == 6 and hot.n_off == 6 and hot.experiment_ids == ("exp-0",)
    assert hot.profile_adjusted and hot.unit == "C"
    assert flat.ci_lo < 0 < flat.ci_hi


def test_reading_absent_during_experiment_gets_a_reason_not_a_number():
    g = _rows()["gappy"]
    assert g.estimate is None and g.se is None and "measured blocks" in g.reason


def test_profile_dropped_when_history_misses_a_bucket():
    exp, blocks, frame = _experiment(0.5)
    frame.loc[frame.index < pd.Timestamp(EXP_START), "hot"] = np.nan   # no history at all
    row = {r.target: r for r in fc.rows_from_experiment(exp, blocks, frame, UNITS, GRAIN, n_perm=200)}["hot"]
    assert not row.profile_adjusted and row.estimate is not None


def test_ci_coverage_over_many_synthetic_switchbacks():
    rows = [_rows(0.3, seed=s)["hot"] for s in range(20)]
    hits = sum(r.ci_lo <= 0.3 <= r.ci_hi for r in rows)
    assert hits >= 17   # nominal 95 %


def _row(target="hot", method=fc.METHOD_RANDOMISED, est=0.5, se=0.1, ids=("e1",), window=(10, 20)):
    return fc.EffectRow("act", target, "C", window[0], window[1], method, est,
                        None if est is None else est - 2 * se, None if est is None else est + 2 * se,
                        None if est is None else se, 6, 6, 0.01, ids)


def test_choose_refuses_observational_rows():
    with pytest.raises(fc.NotChoosable, match="never used to choose"):
        fc.choose([_row(method="observational")], "act", "hot", 12)
    assert fc.choose([_row(method="observational"), _row()], "act", "hot", 12).method == "randomised"


def test_choose_refuses_outside_window_and_rows_without_estimate():
    with pytest.raises(fc.NotChoosable, match="no effect row"):
        fc.choose([_row()], "act", "hot", 25)
    with pytest.raises(fc.NotChoosable, match="no effect row"):
        fc.choose([_row()], "act", "hot", 5)     # inside the burn-in: not measured
    with pytest.raises(fc.NotChoosable, match="no estimate"):
        fc.choose([_row(est=None)], "act", "hot", 12)


def test_pool_is_inverse_variance_and_single_rows_pass_through():
    a, b = _row(est=1.0, se=0.1, ids=("e1",)), _row(est=0.0, se=0.2, ids=("e2",))
    (p,) = fc.pool_rows([a, b])
    assert p.estimate == pytest.approx((1.0 / 0.01) / (1 / 0.01 + 1 / 0.04))
    # pooled on the rows' interval-implied SE (here 2 se / 1.96), not the raw Welch SE
    assert p.se == pytest.approx(math.sqrt(1 / (1 / 0.01 + 1 / 0.04)) * 2 / 1.96)
    assert p.experiment_ids == ("e1", "e2") and p.n_on == 12
    assert fc.pool_rows([a]) == [a]
    (q,) = fc.pool_rows([a, _row(est=None, ids=("e3",))])
    assert q.estimate == a.estimate and "1 experiment" in q.reason


LEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)   # what Family A stores


def _forecast(origin, target_ts, point=26.0, half80=0.4, target="hot"):
    """A Gaussian forecast written as Family A writes it: 9 quantiles keyed '0.025' .. '0.975'."""
    sigma = half80 / fc.Z80
    return {"target": target, "origin_ts": origin, "target_ts": target_ts, "point": point,
            "quantiles": {f"{q:g}": point + sigma * NormalDist().inv_cdf(q) for q in LEVELS}}


T_ORIGIN = EXP_START                              # forecast bucket start: data up to +5 min
T_ACT = EXP_START + timedelta(minutes=5)          # act starts once that bucket is complete
T_END = T_ACT + timedelta(minutes=20)


def _eff(forecast, table, act_start=T_ACT, act_end=T_END, act_id="a1"):
    return fc.efference_copy(act_id, "act", act_start, act_end, forecast, table)


def test_efference_copy_adds_the_effect_and_refuses_a_forecast_that_saw_the_act():
    eff = _eff(_forecast(T_ORIGIN, T_ACT + timedelta(minutes=15)), [_row(se=0.1)])
    assert eff.expected == pytest.approx(26.5) and eff.offset_min == 15
    sb = 0.4 / fc.Z80
    k = math.sqrt(1 + ((0.2 / 1.96) / sb) ** 2)   # widened by the interval-implied SE
    assert eff.effect_se == pytest.approx(0.2 / 1.96)
    assert eff.expected_quantiles["0.5"] == pytest.approx(26.5)
    assert eff.expected_quantiles["0.9"] - eff.expected_quantiles["0.5"] == pytest.approx(0.4 * k)
    assert eff.baseline_quantiles["0.5"] == pytest.approx(26.0)
    # origin == act start: Family A's origin is a bucket START, so that forecast saw 5 min of the act
    with pytest.raises(fc.EfferenceRefused, match="saw the act"):
        _eff(_forecast(T_ACT, T_ACT + timedelta(minutes=15)), [_row()])
    with pytest.raises(fc.EfferenceRefused, match="80 %"):
        _eff({**_forecast(T_ORIGIN, T_ACT + timedelta(minutes=15)), "quantiles": {}}, [_row()])


def test_efference_copy_refuses_an_act_that_ended_before_the_target():
    with pytest.raises(fc.EfferenceRefused, match="held through"):
        _eff(_forecast(T_ORIGIN, T_ACT + timedelta(minutes=15)), [_row()], act_end=T_ACT + timedelta(minutes=2))


def test_efference_copy_accepts_naive_utc_times():
    naive = T_ACT.replace(tzinfo=None)
    eff = _eff(_forecast(T_ORIGIN, T_ACT + timedelta(minutes=15)), [_row()], act_start=naive,
               act_end=T_END.replace(tzinfo=None))
    assert eff.offset_min == 15


def test_choose_skips_rows_without_estimate_and_prefers_the_most_precise():
    noisy, sharp = _row(se=0.3, ids=("e1",)), _row(est=0.4, se=0.05, ids=("e2",), window=(5, 20))
    assert fc.choose([_row(est=None, ids=("e0",)), noisy, sharp], "act", "hot", 12) is sharp
    assert fc.choose([_row(est=None, ids=("e0",)), noisy], "act", "hot", 12) is noisy


def test_pit_is_linear_inside_and_gaussian_beyond_the_stored_range():
    q = _forecast(EXP_START, EXP_START, point=0.0, half80=fc.Z80)["quantiles"]   # sigma 1
    assert fc.pit(q, 0.0) == pytest.approx(0.5)
    assert fc.pit(q, q["0.9"]) == pytest.approx(0.9)
    assert fc.pit(q, 3.0) == pytest.approx(NormalDist().cdf(3.0), rel=1e-6)    # tail continues the Gaussian
    assert fc.pit(q, -3.0) == pytest.approx(NormalDist().cdf(-3.0), rel=1e-6)
    assert fc.pit(q, q["0.975"] + 1e-9) == pytest.approx(0.975, abs=1e-6)      # continuous at the edge
    with pytest.raises(fc.EfferenceRefused, match="increasing"):
        fc.pit({"0.1": 1.0, "0.5": 0.0, "0.9": 2.0}, 0.0)


def test_score_missing_is_scored_not_dropped():
    eff = _eff(_forecast(T_ORIGIN, T_ACT + timedelta(minutes=15)), [_row()])
    assert fc.score(eff, None)["status"] == "missing"
    assert fc.score(eff, float("nan"))["status"] == "missing"


def _simulate(true_effect, table_effect, n_acts=200, seed=0):
    """Acts whose reading truly moves by `true_effect`; the table says `table_effect`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_acts):
        point = 26 + rng.normal()
        sb = 0.3
        f = _forecast(T_ORIGIN, T_ACT + timedelta(minutes=15), point, half80=sb * fc.Z80)
        eff = _eff(f, [_row(est=table_effect, se=0.05)], act_id=f"a{i}")
        actual = point + true_effect + rng.normal(0, sb)
        out.append(fc.score(eff, actual))
    return out


def test_calm_rests_at_zero_when_the_table_is_right():
    g = fc.gate(_simulate(0.5, 0.5))
    assert abs(g["mean_excess"]) < 0.15
    assert 0.02 <= g["share_p_le_0_05"] <= 0.09
    assert g["gap"] > 0 and g["prob_better"] > 0.99   # knowing the effect beats the forecast alone


def test_wrong_table_spikes_and_loses_to_the_forecast_alone():
    g = fc.gate(_simulate(0.0, 0.8))
    assert g["mean_excess"] > 1.0
    assert g["gap"] < 0 and g["prob_better"] < 0.01


def test_a_table_that_only_adds_uncertainty_cannot_win_the_gate():
    """Regression: gain was once the two-sided-p surprisal difference, which any widening lowers, so
    an effect of 0 with a large SE 'beat' a calibrated forecast. The log score must charge for it."""
    rng = np.random.default_rng(5)
    scores = []
    for i in range(300):
        f = _forecast(T_ORIGIN, T_ACT + timedelta(minutes=15), 26.0, half80=0.3 * fc.Z80)
        eff = _eff(f, [_row(est=0.0, se=0.3)], act_id=f"w{i}")
        scores.append(fc.score(eff, 26.0 + rng.normal(0, 0.3)))
    g = fc.gate(scores)
    assert g["gap"] < 0 and g["prob_better"] < 0.01
    assert np.mean([s["surprisal"] < s["baseline_surprisal"] for s in scores]) == 1.0   # the old, wrong reading


def test_gate_clusters_readings_by_act_and_needs_two_acts():
    one = [{"status": "scored", "act_id": "a", "gain": 1.0, "excess": 0.0, "p": 0.5}] * 3
    g = fc.gate(one)
    assert g["n_acts"] == 1 and g["n_scored"] == 3 and g["se"] is None and g["prob_better"] is None


class _Store:
    def __init__(self, exps, logged):
        self.exps, self.logged = exps, logged

    def experiment(self, eid):
        return next((e for e in self.exps if e["experiment_id"] == eid), None)

    def experiments(self, status=None, limit=500):
        return [e for e in self.exps if status is None or e["status"] == status]

    def interventions(self, action=None, experiment_id=None, limit=500):
        return [r for r in self.logged if r["experiment_id"] == experiment_id]


def _logged(exp, blocks):
    return [{"intervention_id": f"iv{b['index']}", "action": exp["action"], "arm": b["arm"],
             "experiment_id": exp["experiment_id"], "started_at": pd.Timestamp(b["start"]).to_pydatetime(),
             "ended_at": pd.Timestamp(b["end"]).to_pydatetime()} for b in blocks]


def test_effect_table_builds_from_analyzed_experiments_only_and_caches():
    exp, blocks, frame = _experiment(0.5)
    cancelled = {**exp, "experiment_id": "exp-x", "status": "cancelled"}
    calls = []

    def loader(start, end):
        calls.append((start, end))
        return frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))], {}

    table = fc.EffectTable(_Store([exp, cancelled], _logged(exp, blocks)), loader, UNITS, GRAIN, n_perm=200)
    out = table.build()
    assert out["experiments"] == ["exp-0"] and out["errors"] == {}
    assert out["n_rows"] == 3 and out["n_with_estimate"] == 2
    hot = next(r for r in out["rows"] if r["target"] == "hot")
    assert hot["method"] == "randomised" and hot["experiment_ids"] == ["exp-0"]
    table.build()
    assert len(calls) == 1
    with pytest.raises(CausalError, match="cancelled"):
        table.rows_for("exp-x")


def test_effect_table_refuses_blocks_that_do_not_follow_the_design():
    exp, blocks, frame = _experiment(0.5)
    logged = _logged(exp, blocks)
    logged[0] = {**logged[0], "arm": "off" if logged[0]["arm"] == "on" else "on"}
    table = fc.EffectTable(_Store([exp], logged), lambda s, e: (frame, {}), UNITS, GRAIN)
    out = table.build()
    assert out["n_rows"] == 0 and "randomised design" in out["errors"]["exp-0"]


def test_http_effect_table_route_is_not_captured_by_effects_action(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    class T:
        def build(self):
            return {"method": "randomised", "rows": [], "n_rows": 0}

    monkeypatch.setattr(main, "engine", object())
    monkeypatch.setattr(main, "effect_table", None)
    c = TestClient(main.app)
    assert c.get("/effects/table").status_code == 503
    monkeypatch.setattr(main, "effect_table", T())
    r = c.get("/effects/table")
    assert r.status_code == 200 and r.json()["method"] == "randomised"


def test_effect_table_aligns_an_unaligned_logged_start_to_the_loader_grid():
    """Logged blocks may start up to 2 min off the design; the loader's SQL buckets sit on grain
    boundaries (date_bin) while its grid starts at the requested start. Unaligned start -> all NaN."""
    exp, blocks, frame = _experiment(0.5)
    logged = _logged(exp, blocks)
    for r in logged:   # each block logged 1 min 37 s late (still within the 2-min design tolerance)
        r["started_at"] += timedelta(seconds=97)
        r["ended_at"] += timedelta(seconds=97)

    def date_bin_loader(start, end):
        index = pd.date_range(start, end - timedelta(minutes=GRAIN), freq=f"{GRAIN}min", tz="UTC")
        return frame.reindex(index), {}   # like load_frame: SQL buckets reindexed onto the grid

    out = fc.EffectTable(_Store([exp], logged), date_bin_loader, UNITS, GRAIN, n_perm=200).build()
    assert out["errors"] == {} and out["n_with_estimate"] == 2
