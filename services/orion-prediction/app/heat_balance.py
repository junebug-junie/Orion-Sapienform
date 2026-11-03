"""Heat per watt: how much a sustained watt of machine power warms the cabinet, split in two.

Theory anchor -- steady-state air heat balance across a heat source: the air leaving a machine is
warmer than the air entering it by ``P / (m_dot * c_p)`` (all electrical power ends as heat; m_dot is
the air mass flow past the sensor). The room sensor (office.temp_c) reads the air the cabinet draws
in; the cabinet sensor reads air the machines have heated. Two fits, same method:

- **local** -- ``gap = cabinet - room`` on each machine's watts: how much warmer than its own intake
  the cabinet runs, holding the room fixed. Each machine gets its OWN slope (a machine whose exhaust
  does not reach the sensor reads ~0: circe, 2026-11-03). A fleet SUM of watts fits badly (R^2 0.89 ->
  0.18): one shared slope on watts the sensor does not see.
- **total** -- ``cabinet`` on the same watts, outdoor temperature as a control: the room is free to
  warm too. The machines' heat goes into the office and comes back in through the intake, so this is
  what the cabinet actually does when load is held. Live 2026-11-03 most of it is the room path
  (athena local +0.07 vs total +0.89 C per 100 W; circe ~0 vs +0.28), in line with the earlier
  whole-cabinet fit in config/physiology/heat_sensitivity.yaml (athena 0.64, circe 0.31, 09-26..10-27).

Fit: hourly means over a recent window; for each machine, lags 0..LAG_H hours (distributed lag; the
effect is the SUM of its lag coefficients = the response to a step held LAG_H hours). Feasible GLS
with AR(1) errors plus Newey-West (HAC) standard errors -- neighbouring hours share slow thermal
drift, so plain OLS errors would be too small. Measured coverage of the "95%" interval is ~89% at
120 h (evals/run_heat_per_watt_eval.py); it is labelled 95% with that caveat. A plain level fit is
reported beside each slope: when it falls outside the interval, the level relation rests on something
else that moved at the same time (see ``levels_disagree``).

The local fit is refused, with reasons, when: too few complete hours, the gap is flat, no machine's
watts vary enough to identify a slope, or the fit explains almost nothing (R^2 < MIN_R2). The total
fit is best effort: refused on its own (reason reported) without refusing the local answer.

claim_kind is always ``observational_effect``: a common cause (a person in the office running a job)
would look the same. The switchback experiment is the causal answer; this is the physical prediction
to compare it with.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.frame import covariate_series, mask_windows, to_grain
from app.series import SeriesSpec, Source

# Outdoor air, raw (the registry only carries the central-AC proxy, F above 75 -- flat in November).
# Read ad hoc from the same observed-hours table, like the runner's GPU_SPEC.
OUTDOOR_SPEC = SeriesSpec(series_id="outdoor.temp_f", role="covariate", source=Source(kind="weather", key="temp_f"),
                          unit="F", grain_min=60, agg="mean", max_fill_min=0, consumers=("heat-per-watt",))

# Per target: the room it draws air from, the machines that heat it, controls (other heat flows).
# Metric gate for these choices: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md
# ("Heat per watt ...") and docs/superpowers/pr-reports/2026-11-03-prediction-heat-balance-pr.md.
HEAT_BALANCE = {
    "athena.cabinet_temp_c": {
        "room": "office.temp_c",
        "watts": ("athena.chassis_watts", "circe.chassis_watts", "hecate.chassis_watts"),
        "controls": ("athena.cabinet_cooling_watts",),
    },
}
HOUR_MIN = 60
MIN_BUCKETS_PER_HOUR = 6        # of 12 five-minute buckets: an hour mostly missing is not an hourly mean
LAG_H = 2                       # lags 0..2 h
MIN_HOURS = 48
MIN_TARGET_SD_C = 0.05
MIN_R2 = 0.10
# A regressor identifies a slope only if enough hours sit away from its usual level (hecate is off
# almost always: a 57 W sd from two hours at 500 W is not a slope).
MIN_SPREAD_W = 20.0
MIN_ACTIVE_SHARE = 0.10
MAX_DAYS = 21


class HeatBalanceRefused(Exception):
    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class HeatFit:
    series_id: str
    room_id: str
    start: datetime
    end: datetime
    n_hours: int
    r2: float
    hac_lags: int
    ar1_rho: float
    slopes: dict            # LOCAL (cabinet - room): watts id -> {per_w, ci_lo, ci_hi, p_value, per_100w, ...}
    controls: dict          # control id -> same shape, or {"refused": reason}
    refused: dict           # watts id -> reason it was left out
    cov: dict               # "a|b" -> covariance of the local per-W slopes (for conversions)
    gap_mean_c: float
    total: dict             # TOTAL (cabinet, room free): {slopes, cov, r2, n_hours, ...} or {"refused": reason}
    claim_kind: str = "observational_effect"
    fit_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_dict(self) -> dict:
        return {"fit_id": self.fit_id, "series_id": self.series_id, "room_id": self.room_id,
                "claim_kind": self.claim_kind, "start": self.start.isoformat(), "end": self.end.isoformat(),
                "n_hours": self.n_hours, "r2": self.r2, "hac_lags": self.hac_lags, "ar1_rho": self.ar1_rho,
                "lags_h": list(range(LAG_H + 1)), "slopes": self.slopes, "controls": self.controls,
                "refused": self.refused, "cov": self.cov, "gap_mean_c": self.gap_mean_c, "total": self.total,
                "summary": summarize(self)}


def hourly(s5: pd.Series) -> pd.Series:
    """5-min real buckets -> hourly means, NaN for an hour with fewer than MIN_BUCKETS_PER_HOUR."""
    g = s5.resample(f"{HOUR_MIN}min", origin="epoch")
    return g.mean().where(g.count() >= MIN_BUCKETS_PER_HOUR)


def identifiable(x: pd.Series) -> str | None:
    """None when x varies enough to identify a slope, else the reason it does not."""
    v = x.dropna()
    if v.empty:
        return "no readings in the window"
    active = float(np.mean(np.abs(v - v.median()) > MIN_SPREAD_W))
    if active < MIN_ACTIVE_SHARE:
        return (f"flat: only {active:.0%} of hours more than {MIN_SPREAD_W:g} W from its median "
                f"{v.median():.0f} W (need {MIN_ACTIVE_SHARE:.0%})")
    return None


def newey_west_lags(n: int) -> int:
    return max(1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def fit_frame(frame: pd.DataFrame, watts: list[str], controls: list[str], target: str = "gap") -> dict:
    """Pure fit on an hourly frame (regular hourly index): ``target`` on *watts, *controls, each at
    lags 0..LAG_H. Effects are per unit of the regressor (W for watts)."""
    import statsmodels.api as sm

    cols, groups = {}, {}
    for name in watts + controls:
        groups[name] = []
        for lag in range(LAG_H + 1):
            key = f"{name}@{lag}h"
            cols[key] = frame[name].shift(lag)
            groups[name].append(key)
    X = pd.DataFrame(cols, index=frame.index)
    data = pd.concat([frame[target].rename("y"), X], axis=1).dropna()
    n = len(data)
    if n < MIN_HOURS:
        raise HeatBalanceRefused(f"only {n} complete hours (need {MIN_HOURS})", {"n_hours": n})
    sd = float(data["y"].std())
    if sd < MIN_TARGET_SD_C:
        raise HeatBalanceRefused(f"{target} is flat (sd {sd:.3f} C)", {"n_hours": n, "target_sd_c": sd})
    keys = list(X.columns)
    lags = newey_west_lags(n)
    y, Xc = data["y"].to_numpy(), sm.add_constant(data[keys].to_numpy(), has_constant="add")
    levels = sm.OLS(y, Xc).fit()
    r2 = float(levels.rsquared)   # share of the target's LEVEL the regressors explain (a refusal floor)
    if not np.isfinite(r2) or r2 < MIN_R2:
        raise HeatBalanceRefused(f"the watts explain almost none of the {target} (R^2 {r2:.2f} < {MIN_R2})",
                                 {"n_hours": n, "r2": r2})
    # Slopes: feasible GLS with AR(1) errors (Cochrane-Orcutt, iterated), then Newey-West on what is
    # left. Plain OLS+HAC covered a planted slope in only ~82% of 95% intervals at 120 h of AR(1)
    # noise (phi 0.6-0.85, persistent watts); this reaches ~89% (evals/run_heat_per_watt_eval.py).
    res = sm.GLSAR(y, Xc, rho=1).iterative_fit(maxiter=10).get_robustcov_results(
        cov_type="HAC", maxlags=lags, use_t=True)
    contrasts = {}
    for name, ks in groups.items():
        c = np.zeros(len(keys) + 1)
        for k in ks:
            c[1 + keys.index(k)] = 1.0
        contrasts[name] = c
    out = {}
    for name, c in contrasts.items():
        t = res.t_test(c)
        eff = float(np.squeeze(t.effect))
        lo, hi = (float(v) for v in np.squeeze(t.conf_int(alpha=0.05)))
        lvl = float(c @ levels.params)
        out[name] = {"per_w": eff, "ci_lo": lo, "ci_hi": hi, "se": float(np.squeeze(t.sd)),
                     "p_value": float(np.squeeze(t.pvalue)), "per_100w": 100 * eff,
                     "ci_lo_per_100w": 100 * lo, "ci_hi_per_100w": 100 * hi,
                     "lag_coefs_per_100w": [100 * float(res.params[1 + keys.index(k)]) for k in groups[name]],
                     "mean": float(data[f"{name}@0h"].mean()), "sd": float(data[f"{name}@0h"].std()),
                     # Diagnostic only: the plain level fit. When it falls outside the GLS interval, the
                     # level relation rests on a slow common cause (live 2026-10-30 02:20Z: an unlogged
                     # airflow step dropped the gap 1.1 C with athena flat at ~740 W; athena's real
                     # 300 W drop at 04:00 moved it ~0.1 C -- the level fit read that step as +0.43/100 W).
                     "levels_per_100w": 100 * lvl, "levels_disagree": not (lo <= lvl <= hi)}
    V = np.asarray(res.cov_params())
    cov = {f"{a}|{b}": float(contrasts[a] @ V @ contrasts[b]) for a in watts for b in watts}
    return {"n": n, "r2": r2, "hac_lags": lags, "effects": out, "cov": cov, "target_mean": float(data["y"].mean()),
            "ar1_rho": float(np.squeeze(res.model.rho))}


def load_frame(effects, series_id: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, dict]:
    """Hourly cabinet, room, gap, watts, controls and outdoor air from the store. Exclusions: the
    target's (registry + change events) everywhere, the room's own on the room; each watts series via
    covariate_series (derived rules, own exclusions, zero before a machine's first reading)."""
    engine, store = effects.engine, effects.store
    cfg = HEAT_BALANCE[series_id]
    spec = engine.effective(engine.registry.get(series_id), engine.clock())
    room = engine.registry.get(cfg["room"])
    idx5 = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{spec.grain_min}min", inclusive="left")
    cab = hourly(mask_windows(to_grain(store.points(spec, start, end), spec.grain_min, spec.agg, 0,
                                       start=start, end=end), spec.exclude).reindex(idx5))
    rm = hourly(mask_windows(to_grain(store.points(room, start, end), room.grain_min, room.agg, 0,
                                      start=start, end=end), (*room.exclude, *spec.exclude)).reindex(idx5))
    frame = {"cabinet": cab, "room": rm, "gap": cab - rm}
    for wid in (*cfg["watts"], *cfg["controls"]):
        w = engine.registry.get(wid)
        frame[wid] = hourly(mask_windows(covariate_series(store, w, start, end, idx5), spec.exclude))
    out = pd.DataFrame(frame)
    outdoor = to_grain(store.points(OUTDOOR_SPEC, start, end), HOUR_MIN, "mean", 0, start=start, end=end)
    out[OUTDOOR_SPEC.series_id] = outdoor.reindex(out.index)
    return out, cfg


def fit_heat_per_watt(effects, series_id: str, days: float = 7.0) -> HeatFit:
    if series_id not in HEAT_BALANCE:
        raise HeatBalanceRefused(f"no heat balance declared for {series_id} (known: {sorted(HEAT_BALANCE)})")
    if not (2 <= days <= MAX_DAYS):
        raise HeatBalanceRefused(f"days must be in [2, {MAX_DAYS}]")
    engine = effects.engine
    spec = engine.registry.get(series_id)
    end = (engine.latest_complete_bucket(spec, engine.clock()) + pd.Timedelta(minutes=spec.grain_min)).floor("60min")
    end = end.to_pydatetime()
    start = end - timedelta(days=days)
    frame, cfg = load_frame(effects, series_id, start, end)
    return fit_from_frame(frame, cfg, series_id, start, end)


def _total(frame: pd.DataFrame, watts: list[str], controls: list[str]) -> dict:
    """Cabinet on the same machines with the room free to move; outdoor air as a control when present."""
    if "cabinet" not in frame:
        return {"refused": "no cabinet column"}
    ctl = list(controls)
    od = OUTDOOR_SPEC.series_id
    if od in frame and frame[od].notna().sum() >= MIN_HOURS:
        ctl.append(od)
    try:
        f = fit_frame(frame, watts, ctl, target="cabinet")
    except HeatBalanceRefused as exc:
        return {"refused": exc.reason}
    return {"slopes": {w: f["effects"][w] for w in watts}, "controls": {c: f["effects"][c] for c in ctl},
            "cov": f["cov"], "r2": f["r2"], "n_hours": f["n"], "hac_lags": f["hac_lags"], "ar1_rho": f["ar1_rho"],
            "outdoor_control": od in ctl}


def fit_from_frame(frame: pd.DataFrame, cfg: dict, series_id: str, start: datetime, end: datetime) -> HeatFit:
    refused = {w: r for w in cfg["watts"] if (r := identifiable(frame[w])) is not None}
    watts = [w for w in cfg["watts"] if w not in refused]
    if not watts:
        raise HeatBalanceRefused("no machine's watts vary enough to identify a slope", {"refused": refused})
    ctrl_refused = {c: r for c in cfg["controls"] if (r := identifiable(frame[c])) is not None}
    controls = [c for c in cfg["controls"] if c not in ctrl_refused]
    fit = fit_frame(frame, watts, controls)
    return HeatFit(series_id=series_id, room_id=cfg["room"], start=start, end=end, n_hours=fit["n"], r2=fit["r2"],
                   hac_lags=fit["hac_lags"], ar1_rho=fit["ar1_rho"], slopes={w: fit["effects"][w] for w in watts},
                   controls={**{c: fit["effects"][c] for c in controls},
                             **{c: {"refused": r} for c, r in ctrl_refused.items()}},
                   refused=refused, cov=fit["cov"], gap_mean_c=fit["target_mean"],
                   total=_total(frame, watts, controls))


def _slope_txt(w: str, s: dict) -> str:
    out = f"{w} {s['per_100w']:+.2f} (95% {s['ci_lo_per_100w']:+.2f}..{s['ci_hi_per_100w']:+.2f})"
    if s.get("levels_disagree"):
        out += (f" [a plain level fit says {s['levels_per_100w']:+.2f}: outside this interval, so the level "
                "relation rests on something else that moved at the same time]")
    return out


def summarize(fit: HeatFit) -> str:
    local = "; ".join(_slope_txt(w, s) for w, s in fit.slopes.items())
    if "slopes" in fit.total:
        total = "; ".join(_slope_txt(w, s) for w, s in fit.total["slopes"].items())
        total_txt = f" With the room free to warm too (outdoor air controlled), C per 100 W: {total}."
    else:
        total_txt = f" Total (room free) fit refused: {fit.total.get('refused')}."
    return (f"Over {fit.n_hours} h, C per 100 W held 2 h, cabinet minus room: {local}.{total_txt} "
            f"AR(1) GLS + Newey-West {fit.hac_lags} lags; the 95% interval covers ~89% in simulation. "
            "Observational: a common cause would look the same.")


def _convert(slopes: dict, cov: dict, watts: dict) -> dict:
    used = {w: v for w, v in watts.items() if w in slopes and v.get("effect") is not None}
    ignored = sorted(set(watts) - set(used))
    if not used:
        return {"predicted_c": None, "reason": "no measured watt change matches a fitted slope", "ignored": ignored}
    ids = list(used)
    dw = np.array([used[w]["effect"] for w in ids], dtype=float)
    s = np.array([slopes[w]["per_w"] for w in ids], dtype=float)
    S = np.array([[cov[f"{a}|{b}"] for b in ids] for a in ids], dtype=float)
    var_w = np.array([((used[w]["ci_hi"] - used[w]["ci_lo"]) / (2 * 1.96)) ** 2
                      if used[w].get("ci_lo") is not None and used[w].get("ci_hi") is not None else 0.0
                      for w in ids])
    pred = float(s @ dw)
    se = float(math.sqrt(max(0.0, dw @ S @ dw + np.sum(s ** 2 * var_w))))
    return {"predicted_c": pred, "ci_lo": pred - 1.96 * se, "ci_hi": pred + 1.96 * se, "se": se,
            "terms": {w: {"delta_w": float(d), "per_100w": 100 * float(k), "c": float(d * k)}
                      for w, d, k in zip(ids, dw, s)},
            "ignored": ignored}


def predict_heat(fit: dict, watts: dict) -> dict:
    """Convert measured watt changes into a predicted cabinet change (C), held ~LAG_H hours.

    ``fit``: HeatFit.as_dict() (or the stored details). ``watts``: id -> {effect, ci_lo, ci_hi} (W),
    e.g. an experiment's manipulation check. Only ids the fit has a slope for are used; the variance
    combines the slopes' HAC covariance and each watt change's own uncertainty (delta method).
    ``local`` holds the room fixed; ``total`` lets it warm. ``predicted_c`` is the total when that fit
    exists (it is what the cabinet sensor would read), else the local one; ``basis`` says which."""
    local = _convert(fit["slopes"], fit["cov"], watts)
    tot = fit.get("total") or {}
    total = (_convert(tot["slopes"], tot["cov"], watts) if "slopes" in tot
             else {"predicted_c": None, "reason": tot.get("refused", "no total fit")})
    head = total if total.get("predicted_c") is not None else local
    return {"predicted_c": head.get("predicted_c"), "ci_lo": head.get("ci_lo"), "ci_hi": head.get("ci_hi"),
            "basis": ("total" if head is total else "local") if head.get("predicted_c") is not None else None,
            "reason": head.get("reason"), "local": local, "total": total, "fit_id": fit.get("fit_id"),
            "note": (f"a load held ~{LAG_H} h; a block shorter than the cabinet's response sees less. "
                     "local = cabinet over its intake (room fixed); total = the room warms too.")}


def store_fit(store, fit: HeatFit) -> list[dict]:
    """prediction_effect_estimate rows per fitted machine: ``heat_per_watt:<watts id>`` (local, cabinet
    minus room) and, when that fit exists, ``heat_per_watt_total:<watts id>`` (cabinet, room free).
    Every row carries the whole fit in details so a conversion can be redone from the newest one."""
    d = fit.as_dict()
    rows = []
    kinds = [("heat_per_watt", fit.slopes, "cabinet minus room")]
    if "slopes" in fit.total:
        kinds.append(("heat_per_watt_total", fit.total["slopes"], "cabinet, room free"))
    for prefix, slopes, what in kinds:
        for w, s in slopes.items():
            rows.append(store.insert_effect({
                "action": f"{prefix}:{w}", "series_id": fit.series_id, "claim_kind": fit.claim_kind,
                "effect": s["per_w"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "p_value": s["p_value"],
                "n": fit.n_hours, "details": {"estimator": "heat_per_watt", "unit": "C per W", "what": what, **d}}))
    return rows
