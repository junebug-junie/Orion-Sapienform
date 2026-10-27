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


SENSOR_BREAK = T0 + timedelta(days=8, hours=0, minutes=35)
EXCLUDE = (T0 + timedelta(days=7, hours=21, minutes=30), SENSOR_BREAK)
PROXY_EFFECT = -0.15  # degC per F above 75, 3 h earlier: the central AC cools the house


def _synthetic_extract(days: int = 12, s_circe: float = 0.4, s_athena: float = 0.0, seed: int = 1,
                       sensor_offset: float = 2.0) -> dict:
    """Cabinet = 26 + day offset + central-AC proxy effect + s_circe per 100 W of circe compute
    (60-min lag) + a measurement offset after SENSOR_BREAK + noise; an exclude window holds junk."""
    rng = np.random.default_rng(seed)
    bins, energy, weather, cooling = [], [], [], []
    n = days * 96
    circe = 400 + 150 * np.sign(np.sin(np.arange(n) / 7.0)) + rng.normal(0, 20, n)
    athena = 240 + rng.normal(0, 15, n)
    day_off = {d: rng.normal(0, 0.5) for d in range(-1, days)}
    wx_noise = rng.normal(0, 2.0, days * 24 + 7)
    temp_f = {}
    for h in range(-6, days * 24):
        t = T0 + timedelta(hours=h)
        f = (21 + 7 * math.sin(2 * math.pi * t.hour / 24) + wx_noise[h + 6]) * 9 / 5 + 32
        temp_f[t] = f
        weather.append({"t": _z(t), "temp_f": f, "source": "history"})
    for i in range(n):
        t = T0 + timedelta(minutes=15 * i)
        lag = circe[i - 4] if i >= 4 else circe[0]
        src = (t - timedelta(hours=3)).replace(minute=0)
        proxy = max(temp_f[src] - 75.0, 0.0)
        cab = (26 + day_off[i // 96] + PROXY_EFFECT * proxy + s_circe * lag / 100 + s_athena * athena[i] / 100
               + rng.normal(0, 0.05))
        if t >= SENSOR_BREAK:
            cab += sensor_offset
        if EXCLUDE[0] <= t < EXCLUDE[1]:
            cab = 15.0  # door open: junk that must be masked
        bins.append({"t": _z(t), "cabinet_c": cab, "w_athena": athena[i] + 150, "w_circe": circe[i] + 120,
                     "w_hecate": 96.0, "cw_athena": athena[i], "cw_circe": circe[i], "fan_athena": 40.0, "fan_circe": 40.0, "n": 60})
        cooling.append({"t": _z(t), "w": 850 + rng.normal(0, 10)})
    for h in range(days * 24):
        t = T0 + timedelta(hours=h)
        for src, kwh in (("athena", 0.4), ("circe", 0.9), ("hecate", 0.1), ("cooling", 0.85 + rng.normal(0, 0.01))):
            energy.append({"t": _z(t), "source": src, "kwh": kwh, "covered_sec": 3600.0, "sample_count": 240})
    events = [
        {"scope": "*", "effect": "exclude", "title": "cabinet open", "notes": "",
         "started_at": _z(EXCLUDE[0]), "ended_at": _z(EXCLUDE[1])},
        {"scope": "*", "effect": "break", "title": "cabinet sensor geometry changed", "notes": "",
         "started_at": _z(SENSOR_BREAK), "ended_at": None},
    ]
    return {"window": {"start": _z(T0), "end": _z(T0 + timedelta(days=days))},
            "bins15": bins, "energy_hourly": energy, "weather_hourly": weather, "cooling15": cooling,
            "change_events": events, "house_hourly": []}


def test_fit_recovers_a_known_sensitivity_and_null_host():
    ex = _synthetic_extract(s_circe=0.4, s_athena=0.0)
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    m = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=60, seed=3)
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
    params = PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.25, slack_decay_floor=1 / 6,
                         bind_ramp=False)   # the pre-Patch B jump rule, explicitly (this test predates the ramp)
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
    params = PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.25, bind_ramp=True)
    res = rep.run_heat_replay(hours, sensitivity={"circe": 0.1}, baselines={"circe": 500.0}, params=params,
                              soft_line_c=31.0, start=T0, end=T0 + timedelta(hours=8), forecast_mode="oracle")
    nr = [i for i, r in enumerate(res) if r.status == NO_READING]
    assert nr, "hours with no cabinet or watts reading must be no_reading"
    held = res[nr[0] - 1].price
    assert held > 0   # bound and ramping (Patch B rule A: bounded ramp, not a jump)
    for i in nr:
        assert res[i].price == held  # held at the last value, not dropped to 0


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


def test_proxy_matches_orion_prediction_definition():
    spec = fit.central_ac_proxy_spec()  # read through orion-prediction's own registry parser
    assert spec == {"lag_min": 180, "above_f": 75.0, "hold_min": 60}
    times = [T0 + timedelta(minutes=15 * i) for i in range(8)]
    wx = [{"t": _z(T0 - timedelta(hours=3)), "temp_f": 80.0}, {"t": _z(T0 - timedelta(hours=2)), "temp_f": 70.0}]
    p = fit.central_ac_proxy(times, wx, **spec)
    assert list(p[:4]) == [5.0] * 4 and list(p[4:]) == [0.0] * 4  # value from 3 h earlier, floored at 0


def test_fit_recovers_proxy_effect_and_sensor_offset_and_masks_exclusion():
    ex = _synthetic_extract(sensor_offset=2.0)
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    assert frame["excluded"].sum() > 0
    free = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=30, seed=2)
    assert free["controls"]["central_ac_proxy_f"] == pytest.approx(PROXY_EFFECT, abs=0.03)
    assert free["controls"]["break:cabinet sensor geometry changed"] == pytest.approx(2.0, abs=0.5)
    fixed = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=30, seed=2,
                          fixed_offsets={"cabinet sensor geometry changed": 2.0})
    assert "break:cabinet sensor geometry changed" not in fixed["controls"]
    assert fixed["hosts"]["circe"]["c_per_100w"] == pytest.approx(0.4, abs=0.08)


def test_decomposition_adds_up_and_reports_the_fixed_offset():
    ex = _synthetic_extract(sensor_offset=2.0)
    frame = fit.build_frame(ex, ("athena", "circe", "hecate"))
    m = fit.fit_model(frame, {"athena": "compute", "circe": "compute"}, label="t", reps=20, seed=2,
                      fixed_offsets={"cabinet sensor geometry changed": 2.0})
    d = fit.decompose(m, frame, base=(T0, T0 + timedelta(days=6)), target=(T0 + timedelta(days=9), T0 + timedelta(days=12)),
                      reps=20, seed=1)
    assert d["status"] == "ok"
    assert d["contributions"]["fixed_offset:cabinet sensor geometry changed"]["degC"] == pytest.approx(2.0)
    total = sum(v["degC"] for v in d["contributions"].values()) + d["residual_c"]
    assert total == pytest.approx(d["rise_c"])


def test_smoke_rebuilds_the_leases_open_at_a_past_tick():
    """The live smoke re-reads lanes at past ticks: a lease whose close came after the tick was
    open at the tick (from its span start), not absent; a close inside the window is as-is."""
    import smoke_body_economy_prices as smoke

    tick = T0 + timedelta(hours=5)
    closes = [
        {"lease_id": "a", "role": "agent", "event": "released", "at": tick - timedelta(minutes=10), "held_ms": 600_000, "hold_lease_id": None},
        {"lease_id": "b", "role": "agent", "event": "released", "at": tick + timedelta(minutes=30), "held_ms": 3_600_000, "hold_lease_id": None},
        {"lease_id": "c", "role": "agent", "event": "released", "at": tick + timedelta(minutes=30), "held_ms": 60_000, "hold_lease_id": None},
    ]
    ev = smoke.lease_events_at(closes, [], tick, timedelta(hours=1))
    assert sorted((e.lease_id, e.event) for e in ev) == [("a", "released"), ("b", "granted")]
    b = next(e for e in ev if e.lease_id == "b")
    assert b.at == tick - timedelta(minutes=30)


# --- Patch C1 wants replay (R2, R6, R7) on synthetic act history ----------------------------------

def _acts_extract(t0: datetime, days: int = 8):
    acts, uses = [], []
    for d in range(days):
        for h in (2, 9, 16):
            o = t0 + timedelta(days=d, hours=h)
            acts.append({"act_id": f"r{d}{h}", "actor": "hub.world_pulse_read", "opened_at": o.isoformat(),
                         "closed_at": (o + timedelta(minutes=10)).isoformat(), "outcome_value": 1.0,
                         "outcome_source": "producer", "signal_id": None, "cost_json": {"seconds": 600.0}})
            acts.append({"act_id": f"s{d}{h}", "actor": "durable.self_inquiry", "opened_at": o.isoformat(),
                         "closed_at": (o + timedelta(hours=1)).isoformat(), "outcome_value": 1.0,
                         "outcome_source": "producer", "signal_id": None,
                         "cost_json": {"seconds": 3600.0, "gpu_slot_s_by_lane": {"agent": 3000.0}}})
            uses.append({"act_id": f"s{d}{h}", "actor": "durable.self_inquiry", "started_at": o.isoformat(),
                         "link_status": "linked", "used": False, "window_closed": True})
    return {"acts": acts, "uses": uses, "lane_prices": [], "background_refusals": []}


def test_wants_replay_r6_bounds_a_zero_price_burst_and_r7_caps_drift():
    from orion.physiology.families import load_families

    t0 = datetime(2026, 10, 16, tzinfo=timezone.utc)
    fams, err = load_families(_ANALYSIS.parents[1] / "config" / "physiology" / "families.yaml")
    assert err is None
    acts, uses = rep.acts_from_extract(_acts_extract(t0))
    r6 = rep.eval_r6(acts, fams, now=t0 + timedelta(days=8))
    assert r6["pass"] and r6["worst_reserved_over_budget"] <= 1.2
    assert r6["by_family"]["durable.self_inquiry"]["deferred_capacity"] > 0
    hours = {t0 + timedelta(hours=i): {"cabinet_c": 35.0 if 48 <= i < 72 else 29.0} for i in range(8 * 24)}
    armed = rep.reflex_armed(hours, t0, t0 + timedelta(days=8))
    rows, series = rep._replay_decisions(acts, uses, fams, start=t0 + timedelta(days=1), end=t0 + timedelta(days=8),
                                         heat_by_hour={}, cabinet={t: v["cabinet_c"] for t, v in hours.items()},
                                         armed=armed)
    held = [r for r in rows if r["reason"] == "spinal:heat_reflex"]
    assert held and all(r["gpu"] for r in held)                 # the reflex holds GPU wants only
    r2 = rep.eval_r2(rows, heat_start=t0 + timedelta(days=2), heat_end=t0 + timedelta(days=3))
    assert r2["non_gpu_hot_wants"] > 0 and r2["nonzero"]
    stress, _ = rep._replay_decisions(acts, uses, fams, start=t0 + timedelta(days=1), end=t0 + timedelta(days=8),
                                      heat_by_hour={}, cabinet={}, armed={}, lane_price=1.0)
    r7 = rep.eval_r7(stress, series, acts, uses, fams)
    assert r7["max_daily_move"] <= 0.1 + 1e-9 and r7["floor_violations"] == 0
    assert r7["final_weights"]["durable.self_inquiry"]["use"] == 0.0


def test_reflex_arms_at_34_and_rearms_below_33():
    t0 = datetime(2026, 10, 18, tzinfo=timezone.utc)
    temps = [33.5, 34.1, 33.5, 32.9, 33.5]
    hours = {t0 + timedelta(hours=i): {"cabinet_c": c} for i, c in enumerate(temps)}
    armed = rep.reflex_armed(hours, t0, t0 + timedelta(hours=5))
    assert [armed[t0 + timedelta(hours=i)] for i in range(5)] == [False, True, True, False, False]


def test_weekly_floor_report_counts_only_hungry_price_deferrals_under_the_quota():
    """Patch C3a R7: graded literally from the replay's own funded wants, trailing 7 days."""
    from orion.physiology.families import load_families

    fams, _ = load_families(_ANALYSIS.parents[1] / "config" / "physiology" / "families.yaml")
    t0 = datetime(2026, 10, 20, tzinfo=timezone.utc)

    def row(i, fam, outcome, reason, intensity=0.5):
        return {"t": (t0 + timedelta(minutes=i)).isoformat(), "family": fam, "tier": 2, "outcome": outcome,
                "reason": reason, "intensity": intensity}

    rows = [row(i, "durable.curiosity_investigate", "funded", "net_positive") for i in range(40)]
    rows += [row(41, "durable.self_inquiry", "deferred", "price", intensity=0.0),   # sated: not a miss
             row(42, "durable.self_inquiry", "deferred", "capacity_reserved"),    # capacity: not a miss
             row(43, "durable.self_inquiry", "deferred", "price")]                # hungry, under 5%: a miss
    rep_ = rep.weekly_floor_report(rows, fams, tier=2)
    si = rep_["durable.self_inquiry"]
    assert (si["quota_misses"], si["deferred_price_sated"], si["deferred_capacity"]) == (1, 1, 1)
    assert si["final_week_share"] == 0.0 and rep_["durable.curiosity_investigate"]["quota_misses"] == 0
    # Once the family is floor-funded up to 5% (3 of 43 = 7%), a later price deferral is not a miss.
    rows2 = rows[:-1] + [row(50 + i, "durable.self_inquiry", "funded", "exploration_floor") for i in range(3)] + [
        row(60, "durable.self_inquiry", "deferred", "price")]
    assert rep.weekly_floor_report(rows2, fams, tier=2)["durable.self_inquiry"]["quota_misses"] == 0
