"""Air-path Patch 1 (docs/superpowers/specs/2026-11-12-orion-air-path-design.md, #3234): the closet top
is a second heat-per-watt target. A planted circe slope is recovered inside its interval, hecate at a
constant 0 W is refused with the spread reason (never fitted as 0), the closet is read through the real
registry + store path, and the cabinet answer and stored rows are unchanged by the new entry."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import app.heat_balance as hb
from app.heat_balance import HEAT_BALANCE, HeatBalanceRefused, fit_from_frame, load_frame, store_fit
from app.series import load_registry
from tests.conftest import REGISTRY_PATH

CLOSET = "closet.top_temp_c"
CABINET = "athena.cabinet_temp_c"
T = datetime(2026, 11, 5, 1, tzinfo=timezone.utc)
CIRCE_SLOPE = 0.0025      # +0.25 C per 100 W (live 2026-11-05..12, 95% +0.16..+0.32)
ATHENA_SLOPE = -0.0015    # -0.15 C per 100 W (athena sits in the cabinet below, not in the plume)


def closet_frame(n: int = 168, seed: int = 0, hecate: float | np.ndarray = 0.0, hecate_slope: float = 0.0,
                 phi: float = 0.6, noise: float = 0.08) -> pd.DataFrame:
    """Hourly closet frame: circe swings with load, athena independently, hecate as given, the AC off;
    gap (closet top - office) = slopes * watts + AR(1) noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T, periods=n, freq="60min")
    c = 800 + 250 * (np.sin(np.arange(n) / 4.0) > 0.0) + rng.normal(0, 60, n)
    a = 450 + rng.normal(0, 60, n)
    h = np.broadcast_to(np.asarray(hecate, dtype=float), (n,)).copy()
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, noise)
    gap = 0.7 + CIRCE_SLOPE * (c - 800) + ATHENA_SLOPE * (a - 450) + hecate_slope * h + e
    outdoor = 50 + 6 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    room = 20 + 0.1 * outdoor + 0.002 * c + rng.normal(0, 0.05, n)
    return pd.DataFrame({"gap": gap, "room": room, "target": room + gap, "outdoor.temp_f": outdoor,
                         "circe.chassis_watts": c, "hecate.chassis_watts": h, "athena.chassis_watts": a,
                         "athena.cabinet_cooling_watts": np.zeros(n)}, index=idx)


def _fit(frame):
    return fit_from_frame(frame, HEAT_BALANCE[CLOSET], CLOSET, T, T)


def test_closet_entry_is_exactly_the_design():
    assert HEAT_BALANCE[CLOSET] == {
        "room": "office.temp_c",
        "watts": ("circe.chassis_watts", "hecate.chassis_watts", "athena.chassis_watts"),
        "controls": ("athena.cabinet_cooling_watts",)}
    reg = load_registry(REGISTRY_PATH)
    for sid, cfg in HEAT_BALANCE.items():       # every id the fit reads is registered
        assert reg.get(sid) is not None and reg.get(cfg["room"]) is not None
        assert all(reg.get(w) is not None for w in (*cfg["watts"], *cfg["controls"]))


def test_synthetic_closet_fit_recovers_the_circe_slope_inside_its_interval():
    fit = _fit(closet_frame())
    c = fit.slopes["circe.chassis_watts"]
    assert c["ci_lo"] < CIRCE_SLOPE < c["ci_hi"]
    assert c["per_100w"] == pytest.approx(0.25, abs=0.08)
    a = fit.slopes["athena.chassis_watts"]
    assert a["ci_lo"] < ATHENA_SLOPE < a["ci_hi"]
    assert "closet top minus room" in fit.as_dict()["summary"]


def test_hecate_at_constant_zero_is_refused_with_the_spread_reason_not_fitted_as_zero():
    fit = _fit(closet_frame(hecate=0.0))
    assert "hecate.chassis_watts" not in fit.slopes
    reason = fit.refused["hecate.chassis_watts"]
    assert reason.startswith("flat") and "20 W" in reason and "10%" in reason
    assert "hecate.chassis_watts" not in fit.total["slopes"]
    # The AC off all window is a refused control, not a fitted one.
    assert "refused" in fit.controls["athena.cabinet_cooling_watts"]


def test_hecate_becomes_fittable_once_it_runs_enough_hours():
    """The arrival proof: hecate on ~30% of hours at ~400 W -> a hecate slope, no code change."""
    n = 168
    rng = np.random.default_rng(7)
    h = np.where(rng.random(n) < 0.3, 400.0 + rng.normal(0, 30, n), 0.0)
    fit = _fit(closet_frame(n=n, hecate=h, hecate_slope=0.003))
    s = fit.slopes["hecate.chassis_watts"]
    # Fitted, positive, near the planted +0.30 per 100 W (one draw: a ~90% interval may just miss).
    assert "hecate.chassis_watts" not in fit.refused and s["ci_lo"] > 0
    assert s["per_100w"] == pytest.approx(0.30, abs=0.06)


# --- the real registry + store path: load_frame reads the closet top as the target -----------------
class _Store:
    """Points per registry spec, keyed by series_id; hourly-constant values at 5-min buckets."""

    def __init__(self, frame: pd.DataFrame):
        self.frame, self.calls = frame, []

    def points(self, spec, since, until=None):
        self.calls.append(spec.series_id)
        col = {CLOSET: "target", CABINET: "target", "office.temp_c": "room"}.get(spec.series_id, spec.series_id)
        if spec.series_id == "outdoor.temp_f":
            col = "outdoor.temp_f"
        if col not in self.frame:
            return []
        s = self.frame[col].reindex(pd.date_range(since, until, freq="5min", inclusive="left"), method="ffill")
        return [(t.to_pydatetime(), float(v)) for t, v in s.items() if np.isfinite(v)]

    def first_reading(self, spec):
        return T - timedelta(days=30)


class _Engine:
    def __init__(self, end):
        self.registry = load_registry(REGISTRY_PATH)
        self.settle = timedelta(seconds=60)
        self._end = end

    def effective(self, spec, now):
        return spec

    def clock(self):
        return self._end

    def latest_complete_bucket(self, spec, now):
        return pd.Timestamp(now) - pd.Timedelta(minutes=spec.grain_min)


def test_closet_is_read_through_the_registry_as_the_target_and_a_covariate_role_is_accepted():
    from app.effects import CausalError, Effects

    frame = closet_frame()
    end = (T + timedelta(hours=len(frame)))
    fx = Effects.__new__(Effects)
    fx.engine, fx.store, fx._manip_watts = _Engine(end), _Store(frame), {}
    assert fx.engine.registry.get(CLOSET).role == "covariate"
    out = fx.heat_per_watt(CLOSET, days=7.0)
    assert fx.store.calls[0] == CLOSET and "office.temp_c" in fx.store.calls
    assert out["series_id"] == CLOSET and out["room_id"] == "office.temp_c"
    c = out["slopes"]["circe.chassis_watts"]
    assert c["ci_lo"] < CIRCE_SLOPE < c["ci_hi"]
    assert "hecate.chassis_watts" in out["refused"]
    # A registered series with no declared heat balance is still refused; an unknown id too.
    with pytest.raises(HeatBalanceRefused, match="no heat balance declared"):
        fx.heat_per_watt("office.temp_c", days=7.0)
    with pytest.raises(CausalError):
        fx.heat_per_watt("closet.nope", days=7.0)


def test_load_frame_target_column_is_the_declared_series():
    frame = closet_frame()
    end = T + timedelta(hours=len(frame))

    class Fx:
        engine, store = _Engine(end), _Store(frame)

    out, cfg = load_frame(Fx(), CLOSET, T, end)
    assert cfg is HEAT_BALANCE[CLOSET]
    assert {"target", "room", "gap", *cfg["watts"], *cfg["controls"]} <= set(out.columns)
    np.testing.assert_allclose(out["gap"].dropna(), (out["target"] - out["room"]).dropna())


# --- the cabinet is unchanged by the new entry ------------------------------------------------------
def test_cabinet_entry_and_fit_are_unchanged_by_the_closet_entry(monkeypatch):
    assert HEAT_BALANCE[CABINET] == {
        "room": "office.temp_c",
        "watts": ("athena.chassis_watts", "circe.chassis_watts", "hecate.chassis_watts"),
        "controls": ("athena.cabinet_cooling_watts",)}
    from tests.test_heat_balance import planted

    f = planted()
    with_closet = fit_from_frame(f, HEAT_BALANCE[CABINET], CABINET, T, T).as_dict()
    monkeypatch.setattr(hb, "HEAT_BALANCE", {CABINET: HEAT_BALANCE[CABINET]})
    without = fit_from_frame(f, hb.HEAT_BALANCE[CABINET], CABINET, T, T).as_dict()
    with_closet.pop("fit_id"), without.pop("fit_id")
    assert with_closet == without
    assert "cabinet minus room" in with_closet["summary"]


class _Rows:
    def __init__(self):
        self.rows = []

    def insert_effect(self, row):
        self.rows.append(row)
        return {"estimate_id": len(self.rows), **row}


def test_stored_rows_keep_the_cabinet_words_and_the_closet_lands_under_its_own_series_id():
    from tests.test_heat_balance import planted

    st = _Rows()
    store_fit(st, fit_from_frame(planted(), HEAT_BALANCE[CABINET], CABINET, T, T))
    cab = st.rows
    assert {r["series_id"] for r in cab} == {CABINET}
    assert {r["details"]["what"] for r in cab} == {"cabinet minus room", "cabinet, room free"}
    assert {r["action"] for r in cab} == {"heat_per_watt:athena.chassis_watts", "heat_per_watt:circe.chassis_watts",
                                          "heat_per_watt_total:athena.chassis_watts",
                                          "heat_per_watt_total:circe.chassis_watts"}
    st2 = _Rows()
    store_fit(st2, _fit(closet_frame()))
    assert {r["series_id"] for r in st2.rows} == {CLOSET}
    assert {r["details"]["what"] for r in st2.rows} == {"closet top minus room", "closet top, room free"}


def test_latest_cabinet_fit_never_reads_a_closet_row():
    from app.effects import Effects

    class Store:
        def __init__(self, rows):
            self.rows = rows

        def effects(self, action, series_id):
            return [r for r in self.rows if r["action"] == action and r["series_id"] == series_id]

    from tests.test_heat_balance import planted

    cab, clo = _Rows(), _Rows()
    store_fit(cab, fit_from_frame(planted(), HEAT_BALANCE[CABINET], CABINET, T, T))
    store_fit(clo, _fit(closet_frame()))
    for i, r in enumerate(cab.rows):
        r["created_at"] = T + timedelta(minutes=i)
    for i, r in enumerate(clo.rows):          # newer than every cabinet row
        r["created_at"] = T + timedelta(days=1, minutes=i)
    fx = Effects.__new__(Effects)
    fx.store = Store(cab.rows + clo.rows)
    assert fx.latest_heat_fit(CABINET)["series_id"] == CABINET
    assert fx.latest_heat_fit(CLOSET)["series_id"] == CLOSET
