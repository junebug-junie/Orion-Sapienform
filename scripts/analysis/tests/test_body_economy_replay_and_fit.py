"""Offline checks for fit_heat_sensitivity.py and replay_body_economy.py (no DB)."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS))
sys.path.insert(0, str(_ANALYSIS.parents[1]))

import fit_heat_sensitivity as fit  # noqa: E402
import replay_body_economy as rep  # noqa: E402
from orion.physiology.prices import EXOGENOUS_BIND, NO_READING, PriceParams  # noqa: E402

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _z(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _synthetic_extract(days: int = 12, s_circe: float = 0.4, s_athena: float = 0.0, seed: int = 1) -> dict:
    """Cabinet = 26 + day offset + outdoor effect + s_circe per 100 W of circe compute (60-min lag) + noise."""
    rng = np.random.default_rng(seed)
    bins, energy, weather = [], [], []
    n = days * 96
    circe = 400 + 150 * np.sign(np.sin(np.arange(n) / 7.0)) + rng.normal(0, 20, n)
    athena = 240 + rng.normal(0, 15, n)
    day_off = {d: rng.normal(0, 1.5) for d in range(days)}
    wx_noise = rng.normal(0, 2.0, days * 24 + 1)  # outdoor not a pure function of the hour
    for i in range(n):
        t = T0 + timedelta(minutes=15 * i)
        lag = circe[i - 4] if i >= 4 else circe[0]
        out_c = 15 + 5 * math.sin(2 * math.pi * (t.hour / 24)) + wx_noise[i // 4]
        cab = 26 + day_off[i // 96] + 0.05 * out_c + s_circe * lag / 100 + s_athena * athena[i] / 100 + rng.normal(0, 0.05)
        bins.append({"t": _z(t), "cabinet_c": cab, "w_athena": athena[i] + 150, "w_circe": circe[i] + 120,
                     "w_hecate": 96.0, "cw_athena": athena[i], "cw_circe": circe[i], "fan_athena": 40.0, "fan_circe": 40.0, "n": 60})
    for h in range(days * 24):
        t = T0 + timedelta(hours=h)
        weather.append({"t": _z(t), "temp_f": (15 + 5 * math.sin(2 * math.pi * t.hour / 24) + wx_noise[h]) * 9 / 5 + 32,
                        "source": "history"})
        for src, kwh in (("athena", 0.4), ("circe", 0.9), ("hecate", 0.1), ("cooling", 0.85 + rng.normal(0, 0.01))):
            energy.append({"t": _z(t), "source": src, "kwh": kwh, "covered_sec": 3600.0, "sample_count": 240})
    return {"window": {"start": _z(T0), "end": _z(T0 + timedelta(days=days))},
            "bins15": bins, "energy_hourly": energy, "weather_hourly": weather}


def test_fit_recovers_a_known_sensitivity_and_null_host():
    ex = _synthetic_extract(s_circe=0.4, s_athena=0.0)
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    m = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=60, seed=3, day_fe=True)
    assert m["status"] == "ok"
    assert m["hosts"]["circe"]["c_per_100w"] == pytest.approx(0.4, abs=0.05)
    lo, hi = m["hosts"]["circe"]["ci95_block_bootstrap"]
    assert lo < 0.4 < hi
    assert abs(m["hosts"]["athena"]["c_per_100w"]) < 0.1


def test_build_frame_keeps_missing_as_nan_not_zero():
    ex = _synthetic_extract(days=2)
    ex["bins15"][10]["cw_circe"] = None
    ex["bins15"][11]["cabinet_c"] = None
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    assert math.isnan(frame["compute"]["circe"][10])
    assert math.isnan(frame["cab"][11])


def _hours(temps: list, watts: float = 600.0) -> dict:
    out = {}
    for i, c in enumerate(temps):
        out[T0 + timedelta(hours=i)] = {"cabinet_c": c, "circe": None if c is None else watts, "kwh": {}}
    return out


def test_heat_replay_flags_unmeetable_hours_and_returns_to_rest():
    temps = [28.0] * 10 + [34.0] * 10 + [28.0] * 14
    hours = _hours(temps)
    params = PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.25, slack_decay_floor=1 / 6)
    res = rep.run_heat_replay(hours, sensitivity={"circe": 0.1}, baselines={"circe": 500.0}, params=params,
                              soft_line_c=31.0, start=T0, end=T0 + timedelta(hours=len(temps)), forecast_mode="persistence")
    statuses = [r.status for r in res]
    assert statuses[11:21] == [EXOGENOUS_BIND] * 10
    r1 = rep.eval_r1(res, soft_line_c=31.0, ceiling=1.0)
    # realized-unmeetable hours are 10..19; persistence flags 11..20, so it misses the first hour
    assert r1["hours_exogenous_realized"] == 10
    assert r1["exogenous_flagged"] == 9 and r1["exogenous_recall"] == pytest.approx(0.9)
    binding = [r.status == EXOGENOUS_BIND or (r.excess is not None and r.excess > 0) for r in res]
    r3 = rep.eval_r3([r.price for r in res], statuses, binding, ceiling=1.0, amplitude=0.5)
    assert r3["max_hours_to_zero"] is not None and r3["max_hours_to_zero"] <= rep.HORIZON_H
    assert r3["longest_unflagged_ceiling_run_h"] == 0


def test_heat_replay_no_reading_hours_hold_price():
    temps = [34.0] * 3 + [None] * 3 + [34.0] * 2
    hours = _hours(temps)
    params = PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.25)
    res = rep.run_heat_replay(hours, sensitivity={"circe": 0.1}, baselines={"circe": 500.0}, params=params,
                              soft_line_c=31.0, start=T0, end=T0 + timedelta(hours=8), forecast_mode="oracle")
    nr = [r for r in res if r.status == NO_READING]
    assert nr, "hours with no cabinet or watts reading must be no_reading"
    for r in nr:
        assert r.price == 1.0  # held at the bound value, not dropped to 0


def test_r3_flags_oscillation_and_ignores_small_wiggles():
    saw = [0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6]
    r = rep.eval_r3(saw, ["live"] * 8, [False] * 8, ceiling=1.0, amplitude=0.5)
    assert r["max_peak_to_peak_oscillating_window"] == pytest.approx(0.6) and not r["pass"]
    falling = [1.0, 0.8, 0.6, 0.61, 0.4, 0.2, 0.0]
    r = rep.eval_r3(falling, ["live"] * 7, [False] * 7, ceiling=1.0, amplitude=0.5)
    assert r["max_peak_to_peak_oscillating_window"] == 0.0


def test_r3_unflagged_ceiling_run():
    prices = [1.0] * 8
    r = rep.eval_r3(prices, ["live"] * 8, [True] * 8, ceiling=1.0, amplitude=0.5)
    assert r["longest_unflagged_ceiling_run_h"] == 8 and not r["pass"]
    r = rep.eval_r3(prices, [EXOGENOUS_BIND] * 8, [True] * 8, ceiling=1.0, amplitude=0.5)
    assert r["longest_unflagged_ceiling_run_h"] == 0 and r["pass"]


def test_r4_constant_load_forecasts_exactly_and_fill_is_never_zero():
    hours = {T0 + timedelta(hours=i): {"kwh": {"athena": 0.4, "circe": 0.9, "hecate": 0.1, "cooling": 0.85}}
             for i in range(30 * 24)}
    # hecate missing for the first two days: filled with its median, not 0
    for i in range(48):
        del hours[T0 + timedelta(hours=i)]["kwh"]["hecate"]
    kwh, fill = rep.fill_energy(hours, T0, T0 + timedelta(days=30))
    assert fill["filled_source_hours"]["hecate"] == 48
    assert kwh[T0] == pytest.approx(2.25)
    r4 = rep.eval_r4(kwh, cycle_start=T0, cycle_end=T0 + timedelta(days=30),
                     replay_start=T0 + timedelta(days=14), replay_end=T0 + timedelta(days=29))
    assert r4["pass"] and abs(r4["worst_error"]) < 1e-9
    assert r4["realized_usd"] == pytest.approx(2.25 * 720 * rep.TARIFF_USD_PER_KWH)


def test_r1_with_no_data_is_not_a_pass():
    assert rep.eval_r1([], soft_line_c=31.0, ceiling=1.0)["pass"] is False
    nr = [rep.HourResult(hour=str(i), cabinet_c=None, forecast_c=None, own_c=None, budget_c=None,
                         status=NO_READING, price=0.0, excess=None) for i in range(24)]
    r = rep.eval_r1(nr, soft_line_c=31.0, ceiling=1.0)
    assert r["pass"] is False and r["pass_excluding_decay_tail"] is False


def test_r3_price_stuck_positive_after_clearing_fails():
    prices = [1.0] + [0.9] * 20 + [1.0]
    binding = [True] + [False] * 20 + [True]
    r = rep.eval_r3(prices, ["live"] * 22, binding, ceiling=1.0, amplitude=0.5)
    assert r["stuck_positive_past_horizon"] == 1 and r["pass"] is False


def test_fit_with_empty_reference_day_keeps_controls_finite():
    ex = _synthetic_extract(days=6)
    for r in ex["bins15"][:96]:  # first day has no circe reading at all
        r["cw_circe"] = None
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    m = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=20, seed=1, day_fe=True)
    assert m["status"] == "ok" and m["condition_number"] < 1e8
