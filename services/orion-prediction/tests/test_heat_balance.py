"""Heat per watt (app/heat_balance.py): planted slopes are recovered with honest intervals, a null
or thin dataset is refused, flat machines are left out with a reason, and the conversion of an
experiment's measured watts into cabinet C is the slope arithmetic plus its uncertainty."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.heat_balance import (
    HEAT_BALANCE,
    MIN_HOURS,
    HeatBalanceRefused,
    fit_from_frame,
    hourly,
    identifiable,
    predict_heat,
)

CFG = HEAT_BALANCE["athena.cabinet_temp_c"]
T = datetime(2026, 10, 29, tzinfo=timezone.utc)


def planted(n: int = 168, slope_a: float = 0.0043, slope_c: float = 0.0, noise: float = 0.08, phi: float = 0.6,
            seed: int = 0, room_a: float = 0.009) -> pd.DataFrame:
    """Hourly frame: athena watts swing with load, circe independently, hecate off but for 2 hours,
    the AC flat; gap = slopes * watts + AR(1) noise (neighbouring hours share drift)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T, periods=n, freq="60min")
    a = 450 + 150 * (np.sin(np.arange(n) / 5.0) > 0.3) + rng.normal(0, 30, n)
    c = 1000 + rng.normal(0, 200, n)
    h = np.zeros(n)
    h[10:12] = 500.0
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, noise)
    gap = -1.3 + slope_a * a + slope_c * c + e
    # The room: outdoor air plus the machines' heat that mixed into the office (athena 0.009, circe
    # 0.003 C per W), plus its own slow wander.
    outdoor = 50 + 6 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.7 * r[i - 1] + rng.normal(0, 0.05)
    room = 10 + 0.2 * outdoor + room_a * a + 0.003 * c + r
    return pd.DataFrame({"gap": gap, "room": room, "cabinet": room + gap, "outdoor.temp_f": outdoor,
                         "athena.chassis_watts": a, "circe.chassis_watts": c,
                         "hecate.chassis_watts": h, "athena.cabinet_cooling_watts": 118.0 + rng.normal(0, 0.3, n)},
                        index=idx)


def _fit(frame):
    return fit_from_frame(frame, CFG, "athena.cabinet_temp_c", T, T)


def test_planted_slope_is_recovered_and_the_unseen_machine_reads_zero():
    fit = _fit(planted())
    a, c = fit.slopes["athena.chassis_watts"], fit.slopes["circe.chassis_watts"]
    assert a["ci_lo"] < 0.0043 < a["ci_hi"] and a["per_100w"] == pytest.approx(0.43, abs=0.08)
    assert c["ci_lo"] < 0 < c["ci_hi"]
    assert fit.r2 > 0.5 and fit.claim_kind == "observational_effect"
    # hecate (2 hours on) and the flat AC are left out with a reason, never fitted.
    assert "hecate.chassis_watts" in fit.refused and "flat" in fit.refused["hecate.chassis_watts"]
    assert "refused" in fit.controls["athena.cabinet_cooling_watts"]


def test_total_fit_lets_the_room_warm_and_controls_outdoor_air():
    fit = _fit(planted())
    t = fit.total
    assert t["outdoor_control"] and t["n_hours"] > 100
    a, c = t["slopes"]["athena.chassis_watts"], t["slopes"]["circe.chassis_watts"]
    assert a["ci_lo"] < 0.0043 + 0.009 < a["ci_hi"]          # local + room path
    assert c["ci_lo"] < 0.003 < c["ci_hi"] and c["ci_lo"] > 0  # circe: only through the room
    assert "With the room free to warm too" in fit.as_dict()["summary"]


def test_summing_watts_the_sensor_cannot_see_would_have_hidden_the_slope():
    """The reason each machine has its own slope: one shared slope on athena+circe fits badly."""
    f = planted()
    tot = f["athena.chassis_watts"] + f["circe.chassis_watts"]
    r_sum = np.corrcoef(tot, f["gap"])[0, 1] ** 2
    assert r_sum < 0.5 * _fit(f).r2


def test_null_and_thin_data_are_refused():
    # Pure noise (no slope, white): the watts explain almost nothing -> refused, not a tiny slope.
    with pytest.raises(HeatBalanceRefused, match="explain almost none"):
        _fit(planted(slope_a=0.0, noise=0.3, phi=0.0, seed=0))
    with pytest.raises(HeatBalanceRefused, match="complete hours"):
        _fit(planted(n=MIN_HOURS))      # the lag columns cost LAG_H hours
    flat = planted()
    flat["gap"] = 0.8
    with pytest.raises(HeatBalanceRefused, match="flat"):
        _fit(flat)
    off = planted()
    off["athena.chassis_watts"] = 450.0
    off["circe.chassis_watts"] = 1000.0
    with pytest.raises(HeatBalanceRefused, match="no machine"):
        _fit(off)


def test_interval_coverage_on_autocorrelated_noise():
    """AR(1) GLS + HAC, not plain OLS: neighbouring hours share drift. Over 60 draws the "95%"
    interval covers the planted slope in >= 80% (measured ~89% over 150; plain OLS+HAC ~82%)."""
    hits = 0
    for seed in range(60):
        s = _fit(planted(seed=seed, phi=0.7, n=120)).slopes["athena.chassis_watts"]
        hits += s["ci_lo"] <= 0.0043 <= s["ci_hi"]
    assert hits >= 48


def test_identifiable_names_why():
    assert identifiable(pd.Series([np.nan, np.nan])) == "no readings in the window"
    assert identifiable(pd.Series([0.0] * 95 + [500.0] * 5)).startswith("flat")
    assert identifiable(pd.Series([400.0] * 50 + [600.0] * 50)) is None


def test_hourly_needs_half_the_buckets():
    idx = pd.date_range(T, periods=24, freq="5min")
    s = pd.Series(1.0, index=idx)
    s.iloc[12:19] = np.nan          # second hour: 5 of 12 buckets
    h = hourly(s)
    assert h.iloc[0] == 1.0 and np.isnan(h.iloc[1])


def test_predict_heat_is_slope_times_watts_with_both_uncertainties():
    fit = _fit(planted()).as_dict()
    a = fit["slopes"]["athena.chassis_watts"]["per_w"]
    c = fit["slopes"]["circe.chassis_watts"]["per_w"]
    watts = {"athena.chassis_watts": {"effect": 7.8, "ci_lo": 1.0, "ci_hi": 14.6},
             "circe.chassis_watts": {"effect": 50.0, "ci_lo": 20.0, "ci_hi": 80.0},
             "athena.gpu_watts_total": {"effect": 3.6, "ci_lo": -3.1, "ci_hi": 10.3}}
    out = predict_heat(fit, watts)
    assert out["local"]["predicted_c"] == pytest.approx(7.8 * a + 50.0 * c)
    assert out["local"]["ignored"] == ["athena.gpu_watts_total"]      # no slope: contained in chassis
    ta = fit["total"]["slopes"]["athena.chassis_watts"]["per_w"]
    tc = fit["total"]["slopes"]["circe.chassis_watts"]["per_w"]
    assert out["basis"] == "total" and out["predicted_c"] == pytest.approx(7.8 * ta + 50.0 * tc)
    exact = predict_heat(fit, {k: {"effect": v["effect"], "ci_lo": v["effect"], "ci_hi": v["effect"]}
                               for k, v in watts.items()})
    assert out["local"]["se"] > exact["local"]["se"] > 0      # watt uncertainty widens it
    assert predict_heat(fit, {"x": {"effect": 1.0}})["predicted_c"] is None
    # No total fit (refused): the local answer is the headline, and says so.
    no_total = {**fit, "total": {"refused": "only 3 complete hours (need 48)"}}
    out2 = predict_heat(no_total, watts)
    assert out2["basis"] == "local" and out2["predicted_c"] == out["local"]["predicted_c"]
    assert "only 3" in out2["total"]["reason"]


# --- wiring: best_effect cites the stored fit beside the experiment; the GET route is not shadowed ---
class _Store:
    def __init__(self, rows, exp, logged):
        self.rows, self.exp, self.logged = rows, exp, logged

    def effects(self, action, series_id):
        return [r for r in self.rows if r["action"] == action]

    def experiment(self, experiment_id):
        return self.exp

    def interventions(self, action=None, experiment_id=None, limit=500):
        return self.logged


def test_best_effect_cites_the_heat_balance_prediction_beside_the_experiment(monkeypatch):
    import app.effects as effects_mod
    import app.experiment_runner as runner_mod
    from app.effects import Effects

    fit = _fit(planted()).as_dict()
    exp_row = {"estimate_id": "e1", "claim_kind": "experimental_effect", "experiment_id": "x1", "effect": 0.21,
               "ci_lo": -0.16, "ci_hi": 0.58, "p_value": 0.2, "n": 12, "created_at": T,
               "action": "vision.focus:walkway"}
    fit_row = {"estimate_id": "h1", "action": "heat_per_watt:athena.chassis_watts", "created_at": T, "details": fit,
               "claim_kind": "observational_effect", "effect": 0.0043, "ci_lo": 0.0036, "ci_hi": 0.005}

    class Eng:
        class registry:
            @staticmethod
            def get(sid):
                return type("S", (), {"role": "target", "unit": "C", "series_id": sid})()

    fx = Effects.__new__(Effects)
    fx.engine, fx._manip_watts = Eng(), {}
    fx.store = _Store([exp_row, fit_row], {"action": "vision.focus:walkway", "design": {"blocks": []}}, [{"arm": "on"}])
    monkeypatch.setattr(Effects, "_match_design", staticmethod(lambda exp, logged: [{"index": 0}]))
    monkeypatch.setattr(runner_mod, "manipulation_watts", lambda eff, blocks: {
        "athena.chassis_watts": {"effect": 7.8, "ci_lo": 1.0, "ci_hi": 14.6}})
    out = fx.best_effect("vision.focus:walkway", "athena.cabinet_temp_c")
    assert out["claim_kind"] == "experimental_effect" and out["effect"] == 0.21
    h = out["heat_balance_prediction"]
    assert h["claim_kind"] == "observational_effect"
    assert h["predicted_c"] == pytest.approx(7.8 * fit["total"]["slopes"]["athena.chassis_watts"]["per_w"])
    assert h["basis"] == "total" and h["local"]["predicted_c"] == pytest.approx(
        7.8 * fit["slopes"]["athena.chassis_watts"]["per_w"])
    assert "heat-per-watt predicts" in out["summary"]
    # No stored fit: the experiment answer stands, the prediction says why it is missing.
    fx.store.rows = [exp_row]
    assert fx.best_effect("vision.focus:walkway", "athena.cabinet_temp_c")["heat_balance_prediction"][
        "predicted_c"] is None
    assert effects_mod.HEAT_BALANCE is HEAT_BALANCE


def test_http_heat_per_watt_route_is_not_captured_by_effects_action(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    calls = []

    class Fx:
        def heat_per_watt(self, series_id, days, experiment_id, store):
            calls.append((series_id, days, experiment_id, store))
            if days > 10:
                raise HeatBalanceRefused("only 3 complete hours (need 48)", {"n_hours": 3})
            return {"slopes": {}, "claim_kind": "observational_effect"}

    monkeypatch.setattr(main, "engine", object())
    monkeypatch.setattr(main, "effects", Fx())
    c = TestClient(main.app)
    r = c.get("/effects/heat-per-watt", params={"days": 5})
    assert r.status_code == 200 and r.json()["claim_kind"] == "observational_effect"
    assert calls[-1] == ("athena.cabinet_temp_c", 5.0, None, False)
    r = c.get("/effects/heat-per-watt", params={"days": 12})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "refused" and r.json()["detail"]["n_hours"] == 3
    assert c.post("/effects/heat-per-watt", json={"days": 7}).status_code == 200 and calls[-1][3] is True


def test_a_confounded_step_is_flagged_not_believed():
    """The live 2026-10-30 pattern: the gap steps down 1.1 C while athena stays flat, and athena steps
    down 300 W hours later with the gap unmoved. A level fit reads that as a large slope; the GLS
    (change-based) slope stays small and the summary says the level fit disagrees."""
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range(T, periods=n, freq="60min")
    a = np.where(np.arange(n) < 28, 740.0, 430.0) + rng.normal(0, 15, n)
    gap = np.where(np.arange(n) < 26, 1.9, 0.7) + 0.0005 * (a - 430) + rng.normal(0, 0.05, n)
    f = pd.DataFrame({"gap": gap, "cabinet": 25 + gap, "athena.chassis_watts": a, "circe.chassis_watts": 1000 + rng.normal(0, 200, n),
                      "hecate.chassis_watts": 0.0, "athena.cabinet_cooling_watts": 118.0}, index=idx)
    fit = _fit(f)
    s = fit.slopes["athena.chassis_watts"]
    assert s["levels_per_100w"] > 0.3 and s["per_100w"] < 0.2
    assert s["levels_disagree"] and "rests on something else" in fit.as_dict()["summary"]
