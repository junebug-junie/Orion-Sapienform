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
  warm too (the machines' heat goes into the office and comes back in through the intake). A 0-2 h
  response, not a steady state -- the room keeps warming after that. Both fits run on the SAME hours.
  Live 2026-11-03 (7 days, 129 h) most of the 0-2 h response is the room path (athena local +0.09 vs
  total +0.92 C per 100 W; circe -0.02 vs +0.32), in line with the earlier whole-cabinet fit in
  config/physiology/heat_sensitivity.yaml (athena 0.64, circe 0.31, 09-26..10-27).

Fit: hourly means over a recent window; for each machine, lags 0..LAG_H hours (distributed lag; the
effect is the SUM of its lag coefficients = the response to a step held LAG_H hours). Feasible GLS
with AR(1) errors plus Newey-West (HAC) standard errors -- neighbouring hours share slow thermal
drift, so plain OLS errors would be too small. Gap-aware: hours on either side of a missing hour are
never treated as neighbours (``_gap_aware_gls``). Measured coverage of the "95%" interval is ~90% at
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
        return _finite({"fit_id": self.fit_id, "series_id": self.series_id, "room_id": self.room_id,
                        "claim_kind": self.claim_kind, "start": self.start.isoformat(), "end": self.end.isoformat(),
                        "n_hours": self.n_hours, "r2": self.r2, "hac_lags": self.hac_lags, "ar1_rho": self.ar1_rho,
                        "lags_h": list(range(LAG_H + 1)), "slopes": self.slopes, "controls": self.controls,
                        "refused": self.refused, "cov": self.cov, "gap_mean_c": self.gap_mean_c, "total": self.total,
                        "summary": summarize(self)})


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


def _finite(v):
    """Non-finite floats -> None, numpy scalars -> Python (a NaN must not turn a response into a 500)."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _lagged(frame: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cols, groups = {}, {}
    for name in names:
        groups[name] = []
        for lag in range(LAG_H + 1):
            key = f"{name}@{lag}h"
            cols[key] = frame[name].shift(lag)   # on the regular hourly index: a lag is a real hour
            groups[name].append(key)
    return pd.DataFrame(cols, index=frame.index), groups


def _gap_aware_gls(y: np.ndarray, X: np.ndarray, seg: np.ndarray, lags: int):
    """Feasible GLS with AR(1) errors (iterated Cochrane-Orcutt) that never pairs hours across a gap.

    ``seg``: a run id per row; rows are consecutive hours only within a run. rho comes from residual
    pairs inside runs, the quasi-difference y_t - rho*y_{t-1} is formed only inside runs (each run's
    first hour is dropped), and the Newey-West kernel runs per run (``hac-panel``), so an exclusion
    window or a sensor outage is a break, not two neighbouring hours."""
    import statsmodels.api as sm

    same = np.concatenate([[False], seg[1:] == seg[:-1]])   # row t has its predecessor t-1 in the same run
    if same.sum() < MIN_HOURS // 2:
        raise HeatBalanceRefused(f"only {int(same.sum())} consecutive hour pairs (need {MIN_HOURS // 2})")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = 0.0
    for _ in range(20):
        e = y - X @ beta
        prev, cur = e[:-1][same[1:]], e[1:][same[1:]]
        new_rho = float(np.clip((prev @ cur) / max(prev @ prev, 1e-12), -0.99, 0.99))
        ys = y[1:][same[1:]] - new_rho * y[:-1][same[1:]]
        Xs = X[1:][same[1:]] - new_rho * X[:-1][same[1:]]
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        if abs(new_rho - rho) < 1e-6:
            rho = new_rho
            break
        rho = new_rho
    groups = seg[1:][same[1:]]
    # use_t is NOT used here: statsmodels' panel df is (groups - 1), which is 0 for one unbroken run.
    # The t tests below use df = rows - parameters instead (the runs are pieces of one series).
    res = sm.OLS(ys, Xs).fit(cov_type="hac-panel", cov_kwds={"groups": groups, "maxlags": lags})
    return res, rho, len(np.unique(seg))


def fit_frame(frame: pd.DataFrame, watts: list[str], controls: list[str], target: str = "gap",
              rows: pd.Index | None = None) -> dict:
    """Pure fit on an hourly frame (regular hourly index): ``target`` on *watts, *controls, each at
    lags 0..LAG_H. ``rows``: restrict to these hours (both fits share one set). Watts effects are per W;
    control effects per unit of the control (``per_unit``)."""
    import statsmodels.api as sm

    X, groups = _lagged(frame, watts + controls)
    data = pd.concat([frame[target].rename("y"), X], axis=1).dropna()
    if rows is not None:
        data = data.loc[data.index.intersection(rows)]
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
    step = data.index.to_series().diff() != pd.Timedelta(minutes=HOUR_MIN)
    seg = step.cumsum().to_numpy()
    # Plain OLS+HAC covered a planted slope in only ~82% of 95% intervals at 120 h of AR(1) noise
    # (phi 0.6-0.85, persistent watts); AR(1) GLS + HAC reaches ~90% (evals/run_heat_per_watt_eval.py).
    res, rho, n_runs = _gap_aware_gls(y, Xc, seg, lags)
    from scipy import stats

    V = np.asarray(res.cov_params())
    df = max(1, int(res.nobs) - Xc.shape[1])
    tq = float(stats.t.ppf(0.975, df))
    out = {}
    for name, ks in groups.items():
        c = np.zeros(len(keys) + 1)
        for k in ks:
            c[1 + keys.index(k)] = 1.0
        eff = float(c @ res.params)
        se = float(np.sqrt(max(0.0, c @ V @ c)))
        lo, hi = eff - tq * se, eff + tq * se
        p_value = float(2 * stats.t.sf(abs(eff / se), df)) if se > 0 else float("nan")
        lvl = float(c @ levels.params)
        base = {"ci_lo": lo, "ci_hi": hi, "se": se, "p_value": p_value,
                "mean": float(data[f"{name}@0h"].mean()), "sd": float(data[f"{name}@0h"].std()),
                # Diagnostic only: the plain level fit. When it falls outside the GLS interval, the
                # level relation rests on a slow common cause (live 2026-10-30 02:20Z: an unlogged
                # airflow step dropped the gap 1.1 C with athena flat at ~740 W; athena's real 300 W
                # drop at 04:00 moved it ~0.1 C -- the level fit read that step as +0.43/100 W).
                "levels_disagree": not (lo <= lvl <= hi)}
        if name in watts:
            out[name] = {"per_w": eff, **base, "per_100w": 100 * eff, "ci_lo_per_100w": 100 * lo,
                         "ci_hi_per_100w": 100 * hi, "levels_per_100w": 100 * lvl,
                         "lag_coefs_per_100w": [100 * float(res.params[1 + keys.index(k)]) for k in ks]}
        else:
            out[name] = {"per_unit": eff, **base, "levels_per_unit": lvl}
    contrast = {}
    for name in watts:
        c = np.zeros(len(keys) + 1)
        for k in groups[name]:
            c[1 + keys.index(k)] = 1.0
        contrast[name] = c
    cov = {f"{a}|{b}": float(contrast[a] @ V @ contrast[b]) for a in watts for b in watts}
    return {"n": n, "r2": r2, "hac_lags": lags, "effects": out, "cov": cov, "target_mean": float(data["y"].mean()),
            "ar1_rho": rho, "n_runs": n_runs, "rows": data.index}


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


def _common_rows(frame: pd.DataFrame, watts: list[str], controls: list[str], outdoor: bool) -> pd.Index:
    """Hours where BOTH fits have everything (gap, cabinet, every lagged regressor, outdoor air when
    used): the local and total slopes are then measured on the same hours, so they can be compared."""
    names = watts + controls + ([OUTDOOR_SPEC.series_id] if outdoor else [])
    X, _ = _lagged(frame, names)
    need = pd.concat([frame[["gap", "cabinet"]], X], axis=1)
    return need.dropna().index


def _total(frame: pd.DataFrame, watts: list[str], controls: list[str], rows: pd.Index, outdoor: bool) -> dict:
    """Cabinet on the same machines with the room free to move; outdoor air as a control when present.
    A 0-2 h response (lags summed), not a steady state: the room keeps warming after that."""
    ctl = list(controls) + ([OUTDOOR_SPEC.series_id] if outdoor else [])
    try:
        f = fit_frame(frame, watts, ctl, target="cabinet", rows=rows)
    except HeatBalanceRefused as exc:
        return {"refused": exc.reason}
    return {"slopes": {w: f["effects"][w] for w in watts}, "controls": {c: f["effects"][c] for c in ctl},
            "cov": f["cov"], "r2": f["r2"], "n_hours": f["n"], "n_runs": f["n_runs"], "hac_lags": f["hac_lags"],
            "ar1_rho": f["ar1_rho"], "outdoor_control": outdoor}


def fit_from_frame(frame: pd.DataFrame, cfg: dict, series_id: str, start: datetime, end: datetime) -> HeatFit:
    refused = {w: r for w in cfg["watts"] if (r := identifiable(frame[w])) is not None}
    watts = [w for w in cfg["watts"] if w not in refused]
    if not watts:
        raise HeatBalanceRefused("no machine's watts vary enough to identify a slope", {"refused": refused})
    ctrl_refused = {c: r for c in cfg["controls"] if (r := identifiable(frame[c])) is not None}
    controls = [c for c in cfg["controls"] if c not in ctrl_refused]
    if "cabinet" not in frame:
        raise HeatBalanceRefused("no cabinet column")
    od = OUTDOOR_SPEC.series_id
    outdoor = od in frame and len(_common_rows(frame, watts, controls, True)) >= MIN_HOURS
    rows = _common_rows(frame, watts, controls, outdoor)
    fit = fit_frame(frame, watts, controls, rows=rows)
    total = _total(frame, watts, controls, fit["rows"], outdoor)
    return HeatFit(series_id=series_id, room_id=cfg["room"], start=start, end=end, n_hours=fit["n"], r2=fit["r2"],
                   hac_lags=fit["hac_lags"], ar1_rho=fit["ar1_rho"], slopes={w: fit["effects"][w] for w in watts},
                   controls={**{c: fit["effects"][c] for c in controls},
                             **{c: {"refused": r} for c, r in ctrl_refused.items()}},
                   refused=refused, cov=fit["cov"], gap_mean_c=fit["target_mean"], total=total)


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
        total_txt = (f" With the room free to warm too (outdoor air controlled; a 0-{LAG_H} h response, not a "
                     f"steady state), same {fit.total['n_hours']} h: {total}.")
    else:
        total_txt = f" Total (room free) fit refused: {fit.total.get('refused')}."
    return (f"Over {fit.n_hours} h, C per 100 W held 2 h, cabinet minus room: {local}.{total_txt} "
            f"AR(1) GLS + Newey-West {fit.hac_lags} lags; the 95% interval covers ~90% in simulation. "
            "Observational: a common cause would look the same.")


def _convert(slopes: dict, cov: dict, watts: dict) -> dict:
    used = {w: v for w, v in watts.items() if w in slopes and v.get("effect") is not None}
    ignored = sorted(set(watts) - set(used))
    unmeasured = sorted(set(slopes) - set(used))
    if not used:
        return {"predicted_c": None, "reason": "no measured watt change matches a fitted slope", "ignored": ignored,
                "unmeasured": unmeasured}
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
            "ignored": ignored, "unmeasured": unmeasured}


CONVERT_NOTE = ("Watt-change variance is read off each estimate's 95% interval with z=1.96 (the switchback "
                "interval uses a t quantile, so this slightly overstates it: conservative). The athena and "
                "circe watt changes come from the same blocks, but their correlation is ignored. A fitted "
                "machine the experiment did not measure (listed in 'unmeasured', e.g. hecate) adds 0.")


def predict_heat(fit: dict, watts: dict, block_min: float | None = None) -> dict:
    """Convert measured watt changes into a predicted cabinet change (C). Returns BOTH bases:
    ``local`` (cabinet over its intake, room fixed) and ``total`` (room free; a 0-2 h response, not a
    steady state). ``fit``: HeatFit.as_dict() (or stored details). ``watts``: id -> {effect, ci_lo,
    ci_hi} (W), e.g. an experiment's manipulation check. Only ids with a slope are used; the variance
    combines the slopes' HAC covariance and each watt change's uncertainty (delta method).

    ``basis`` picks the headline ``predicted_c``: a switchback block shorter than the fit's lag window
    (``block_min`` < LAG_H h) alternates load faster than the room can follow -- the room integrates
    the average load and cancels between arms -- so the local basis is the comparable one; a longer or
    unknown hold uses the total (when that fit exists)."""
    local = _convert(fit["slopes"], fit["cov"], watts)
    tot = fit.get("total") or {}
    total = (_convert(tot["slopes"], tot["cov"], watts) if "slopes" in tot
             else {"predicted_c": None, "reason": tot.get("refused", "no total fit")})
    short = block_min is not None and block_min < LAG_H * HOUR_MIN
    head = local if short or total.get("predicted_c") is None else total
    basis = None if head.get("predicted_c") is None else ("local" if head is local else "total")
    why = (f"{block_min:g}-min blocks are shorter than the {LAG_H} h fit window: the room cannot follow, so "
           "local is the comparable basis" if short else
           f"load held about {LAG_H} h (total = 0-{LAG_H} h response with the room free, not a steady state)")
    return _finite({"predicted_c": head.get("predicted_c"), "ci_lo": head.get("ci_lo"), "ci_hi": head.get("ci_hi"),
                    "basis": basis, "basis_reason": why, "block_min": block_min, "reason": head.get("reason"),
                    "local": local, "total": total, "fit_id": fit.get("fit_id"), "note": CONVERT_NOTE})


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
