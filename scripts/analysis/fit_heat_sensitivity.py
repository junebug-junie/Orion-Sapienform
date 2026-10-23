#!/usr/bin/env python3
"""Does Orion's own load move the cabinet temperature? Passive fit, per host (read-only).

Body-economy design (docs/superpowers/specs/2026-10-23-organism-body-economy-design.md), Organ 6
item 3, part 1: regress athena's ``cabinet_temp_c`` on per-host CPU+GPU watts at 0/15/30/60-minute
lags. Controls reuse orion-prediction's own definitions rather than parallel ones:
- weather enters through ``athena.central_ac_proxy`` (#2829; config/prediction/series.v1.yaml, parsed
  by services/orion-prediction/app/series.py): outdoor F above 75, 3 h earlier -- the house's central
  AC -- not a linear outdoor term;
- ``prediction_change_event`` (#2780): 'exclude' windows are masked; the 10-17 sensor-geometry break
  is a MEASUREMENT offset (fixed at the recorded ~2 C in the primary model), other breaks are steps;
- the portable AC is ``home_cooling_sample`` (athena.cabinet_cooling_watts).
A house-meter variant uses run_house_heat_balance_eval.py's household+central kWh instead of the
proxy. ``decompose`` splits the 09-26..10-10 -> 10-18..10-22 rise into per-term contributions.

The per-host sensitivity is the SUM of its lag coefficients: degC per 100 W held steady. Intervals
are a bootstrap over whole days (resampled i.i.d. with replacement; residuals are strongly
autocorrelated, so a plain OLS SE would be far too narrow) with a Newey-West HAC SE beside it. The
HAC SE is a LOWER bound: it is computed on the rows left after filtering, where row lag is not
exactly time lag.

It also re-runs the simple hourly correlations the design quoted (Fact 2) with the exact SQL, to
settle the circe 0.10 vs 0.22 discrepancy.

Usage:
  .venv/bin/python scripts/analysis/fit_heat_sensitivity.py \
      --out-json /tmp/heat_fit.json --out-md /tmp/heat_fit.md [--cache extract.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _body_economy_data as data_mod  # noqa: E402

BIN_MIN = 15
LAGS_BINS = (0, 1, 2, 4)  # 0, 15, 30, 60 minutes
OUTDOOR_LAGS_H = (0, 1, 3)
HARMONICS = 3
BINS_PER_DAY = 24 * 60 // BIN_MIN


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def build_frame(extract: dict, hosts: tuple[str, ...]) -> dict:
    """15-minute frame on a regular grid. Missing values stay NaN (never 0)."""
    bins = extract["bins15"]
    start = _ts(bins[0]["t"])
    end = _ts(bins[-1]["t"])
    n = int((end - start).total_seconds() // (BIN_MIN * 60)) + 1
    times = [start + timedelta(minutes=BIN_MIN * i) for i in range(n)]
    idx = {t: i for i, t in enumerate(times)}
    cab = np.full(n, np.nan)
    watts = {h: np.full(n, np.nan) for h in hosts}
    compute = {h: np.full(n, np.nan) for h in data_mod.COMPUTE_HOSTS}
    fan = {h: np.full(n, np.nan) for h in data_mod.COMPUTE_HOSTS}
    for r in bins:
        i = idx.get(_ts(r["t"]))
        if i is None:
            continue
        if r.get("cabinet_c") is not None:
            cab[i] = r["cabinet_c"]
        for h in hosts:
            v = r.get(f"w_{h}")
            if v is not None:
                watts[h][i] = v
        for h in data_mod.COMPUTE_HOSTS:
            if r.get(f"cw_{h}") is not None:
                compute[h][i] = r[f"cw_{h}"]
            if r.get(f"fan_{h}") is not None:
                fan[h][i] = r[f"fan_{h}"]
    # outdoor: hourly, linear interpolation onto the bin grid (only between two known hours)
    wx = {_ts(r["t"]): f_to_c(r["temp_f"]) for r in extract["weather_hourly"] if r.get("temp_f") is not None}
    outdoor = np.full(n, np.nan)
    for i, t in enumerate(times):
        h0 = t.replace(minute=0, second=0, microsecond=0)
        a, b = wx.get(h0), wx.get(h0 + timedelta(hours=1))
        if a is not None and b is not None:
            frac = t.minute / 60.0
            outdoor[i] = a * (1 - frac) + b * frac
        elif a is not None and t.minute == 0:
            outdoor[i] = a
    # Portable cabinet AC: orion-prediction's athena.cabinet_cooling_watts (home_cooling_sample).
    ac = np.full(n, np.nan)
    for r in extract.get("cooling15") or ():
        i = idx.get(_ts(r["t"]))
        if i is not None and r.get("w") is not None:
            ac[i] = r["w"]
    hod = np.array([t.hour + t.minute / 60.0 for t in times])
    proxy_spec = central_ac_proxy_spec()
    proxy = central_ac_proxy(times, extract["weather_hourly"], **proxy_spec)
    events = parse_events(extract.get("change_events") or ())
    house_rest = house_rest_kwh(times, extract.get("house_hourly") or (), watts, ac)
    excluded = np.array([any(a <= t < b for a, b in events["exclude"]) for t in times])
    return {"times": times, "cab": cab, "watts": watts, "compute": compute, "fan": fan,
            "outdoor": outdoor, "ac": ac, "hod": hod, "proxy": proxy, "proxy_spec": proxy_spec, "house_rest": house_rest,
            "house_hourly": extract.get("house_hourly") or (),
            "breaks": events["breaks"], "excludes": events["exclude"], "excluded": excluded}


SERIES_YAML = Path(__file__).resolve().parents[2] / "config" / "prediction" / "series.v1.yaml"
PREDICTION_SERIES_PY = Path(__file__).resolve().parents[2] / "services" / "orion-prediction" / "app" / "series.py"


def central_ac_proxy_spec() -> dict:
    """orion-prediction's own definition of athena.central_ac_proxy (#2829), read through its own
    registry parser so the lag / threshold / hold here cannot drift from the service's."""
    import importlib.util

    mod_spec = importlib.util.spec_from_file_location("orion_prediction_series", PREDICTION_SERIES_PY)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["orion_prediction_series"] = mod  # dataclasses need the module registered
    mod_spec.loader.exec_module(mod)
    spec = mod.load_registry(SERIES_YAML).get("athena.central_ac_proxy")
    if spec is None or spec.source.kind != "weather" or spec.above is None:
        raise SystemExit("athena.central_ac_proxy missing or changed shape in config/prediction/series.v1.yaml")
    return {"lag_min": spec.lag_min, "above_f": spec.above, "hold_min": spec.max_fill_min}


def central_ac_proxy(times: list[datetime], weather_hourly: list[dict], *, lag_min: int, above_f: float,
                     hold_min: int) -> np.ndarray:
    """Same rule as orion-prediction's app/frame.py load_points + covariate_series: the value at t is
    max(outdoor F - above, 0) from the hour that started ``lag`` earlier, held for up to ``hold_min``.
    Unknown stays NaN."""
    wx = {_ts(r["t"]): r["temp_f"] for r in weather_hourly if r.get("temp_f") is not None}
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        src = t - timedelta(minutes=lag_min)
        h0 = src.replace(minute=0, second=0, microsecond=0)
        if (src - h0) < timedelta(minutes=hold_min) and h0 in wx:
            out[i] = max(wx[h0] - above_f, 0.0)
    return out


HOUSE_REST_HOURS = 4  # trailing mean over the current meter hour and the 3 before it


def house_rest_kwh(times: list[datetime], house_hourly, watts: dict, ac: np.ndarray) -> np.ndarray:
    """Household + central-AC kWh per hour: the house meter minus the cabinet machines' chassis watts
    and the portable AC -- the same subtraction as orion-prediction's run_house_heat_balance_eval.py:
    hourly means of each source, hecate averaged over the bins where it has a reading and 0 only when
    the whole hour has none (the eval's ``coalesce(avg(hecate), 0)``). Trailing HOUSE_REST_HOURS-hour
    mean of the current meter hour and the ones before it (the current hour includes energy after
    the bin: fine for a control). NaN where the meter has no reading."""
    rest = house_rest_hourly(times, house_hourly, watts, ac)
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        h0 = t.replace(minute=0, second=0, microsecond=0)
        vals = [rest.get(h0 - timedelta(hours=k)) for k in range(HOUSE_REST_HOURS)]
        if all(v is not None for v in vals):
            out[i] = float(np.mean(vals))
    return out


def house_rest_hourly(times: list[datetime], house_hourly, watts: dict, ac: np.ndarray) -> dict:
    house = {_ts(r["t"]): r["kwh"] for r in house_hourly if r.get("kwh") is not None}
    acc: dict[datetime, dict[str, list[float]]] = {}
    for i, t in enumerate(times):
        h = acc.setdefault(t.replace(minute=0, second=0, microsecond=0), {"athena": [], "circe": [], "ac": [], "hecate": []})
        for k, arr in (("athena", watts["athena"]), ("circe", watts["circe"]), ("ac", ac), ("hecate", watts.get("hecate"))):
            if arr is not None and np.isfinite(arr[i]):
                h[k].append(float(arr[i]))
    rest = {}
    for h, v in acc.items():
        if h in house and v["athena"] and v["circe"] and v["ac"]:
            hec = np.mean(v["hecate"]) if v["hecate"] else 0.0
            rest[h] = house[h] - (np.mean(v["athena"]) + np.mean(v["circe"]) + np.mean(v["ac"]) + hec) / 1000.0
    return rest


def parse_events(rows) -> dict:
    """orion-prediction change events (#2780): 'exclude' windows are masked; each 'break' becomes a
    level-shift step at started_at (app/events.py's 'pending break' treatment)."""
    excl, breaks = [], []
    for e in rows:
        start = _ts(e["started_at"])
        if e["effect"] == "exclude" and e.get("ended_at"):
            excl.append((start, _ts(e["ended_at"])))
        elif e["effect"] == "break":
            breaks.append({"at": start, "title": e["title"], "notes": e.get("notes") or ""})
    return {"exclude": excl, "breaks": sorted(breaks, key=lambda b: b["at"])}


def restrict(frame: dict, end: datetime) -> dict:
    """Same frame with every bin at or after ``end`` blanked (NaN cabinet), so it drops out."""
    out = dict(frame)
    cab = frame["cab"].copy()
    cab[np.array([t >= end for t in frame["times"]])] = np.nan
    out["cab"] = cab
    return out


def regressors(frame: dict, spec: dict[str, str]) -> dict[str, np.ndarray]:
    """spec maps host -> 'chassis' | 'compute'."""
    return {h: frame["watts"][h] if kind == "chassis" else frame["compute"][h] for h, kind in spec.items()}


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def break_name(b: dict) -> str:
    return "break:" + b["title"]


def design(frame: dict, regs: dict[str, np.ndarray], *, outdoor: bool = False, proxy: bool = True,
           house_ac: bool = False, ac: bool = True, breaks: bool = True, day_fe: bool = False,
           fixed_offsets: dict[str, float] | None = None):
    """Returns X, y, names, day, rows (indices into the frame grid).

    ``fixed_offsets`` maps a break title to a known MEASUREMENT offset in degC: the cabinet reading
    after that break is lowered by it and the break gets no free step (its size is not estimated)."""
    fixed_offsets = fixed_offsets or {}
    cols, names = [], []
    for h, series in regs.items():
        for k in LAGS_BINS:
            cols.append(_lag(series, k) / 100.0)
            names.append(f"w_{h}_lag{k * BIN_MIN}m")
    if outdoor:
        for lh in OUTDOOR_LAGS_H:
            cols.append(_lag(frame["outdoor"], lh * 60 // BIN_MIN))
            names.append(f"outdoor_c_lag{lh}h")
    if proxy:
        cols.append(frame["proxy"])
        names.append("central_ac_proxy_f")
    if house_ac:
        cols.append(frame["house_rest"])
        names.append("house_rest_kwh")
    if ac:
        cols.append(frame["ac"] / 100.0)
        names.append("ac_per100w")
    if breaks:
        for b in frame["breaks"]:
            if b["title"] in fixed_offsets:
                continue
            step = np.array([t >= b["at"] for t in frame["times"]], dtype=float)
            if 0 < step.sum() < len(step):
                cols.append(step)
                names.append(break_name(b))
    for k in range(1, HARMONICS + 1):
        ang = 2 * math.pi * k * frame["hod"] / 24.0
        cols += [np.sin(ang), np.cos(ang)]
        names += [f"hod_sin{k}", f"hod_cos{k}"]
    day = np.array([t.date().toordinal() for t in frame["times"]])
    if day_fe:
        for d in sorted(set(day)):  # one per day; the reference is chosen after filtering
            cols.append((day == d).astype(float))
            names.append(f"day_{d}")
    cols.append(np.ones(len(frame["times"])))
    names.append("const")
    X = np.column_stack(cols)
    y = frame["cab"].copy()
    for b in frame["breaks"]:
        if b["title"] in fixed_offsets:
            y[np.array([t >= b["at"] for t in frame["times"]])] -= fixed_offsets[b["title"]]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & ~frame["excluded"]
    rows = np.nonzero(ok)[0]
    X, y, day = X[ok], y[ok], day[ok]
    # A column with no variation left after filtering (e.g. a break step outside a restricted
    # window) is not identified: drop it rather than carry a meaningless coefficient.
    keep = [j for j, nm in enumerate(names) if nm == "const" or nm.startswith("day_") or np.ptp(X[:, j]) > 0]
    X, names = X[:, keep], [names[j] for j in keep]
    if day_fe:
        # Drop dummies that ended up empty after filtering, then drop the first remaining one so the
        # reference day is a day that actually has rows (otherwise dummies + const are collinear).
        keep = [j for j, nm in enumerate(names) if not nm.startswith("day_") or X[:, j].any()]
        first_day = next(j for j in keep if names[j].startswith("day_"))
        keep.remove(first_day)
        # A break step that is constant within every day is a sum of day dummies: drop it.
        for j in list(keep):
            if names[j].startswith("break:") and all(np.ptp(X[day == d, j]) == 0 for d in np.unique(day)):
                keep.remove(j)
        X, names = X[:, keep], [names[j] for j in keep]
    return X, y, names, day, rows


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n = X.shape[0]
    xtx_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    return xtx_inv @ S @ xtx_inv * n / max(n - X.shape[1], 1)


def block_bootstrap(X: np.ndarray, y: np.ndarray, day: np.ndarray, contrasts: dict[str, np.ndarray],
                    reps: int, seed: int) -> dict[str, np.ndarray]:
    """Resample whole days with replacement (keeps within-day autocorrelation intact)."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    rows_by_day = {d: np.nonzero(day == d)[0] for d in days}
    out = {k: np.empty(reps) for k in contrasts}
    for r in range(reps):
        pick = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([rows_by_day[d] for d in pick])
        b, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        for k, c in contrasts.items():
            out[k][r] = c @ b
    return out


def fit_model(frame: dict, spec: dict[str, str], *, label: str, reps: int, seed: int,
              hac_lags: int = 96, **design_kw) -> dict:
    hosts = tuple(spec)
    X, y, names, day, rows = design(frame, regressors(frame, spec), **design_kw)
    if X.shape[0] < 4 * X.shape[1]:
        return {"label": label, "status": "insufficient_rows", "n": int(X.shape[0])}
    beta, resid = ols(X, y)
    cov = newey_west_cov(X, resid, hac_lags)
    contrasts = {}
    for h in hosts:
        c = np.array([1.0 if nm.startswith(f"w_{h}_lag") else 0.0 for nm in names])
        contrasts[h] = c
    boots = block_bootstrap(X, y, day, contrasts, reps, seed)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    dw = float((np.diff(resid) ** 2).sum() / (resid ** 2).sum())
    ac1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    hosts_out = {}
    for h, c in contrasts.items():
        est = float(c @ beta)
        se_hac = float(math.sqrt(max(c @ cov @ c, 0.0)))
        lo, hi = (float(v) for v in np.percentile(boots[h], [2.5, 97.5]))
        lag_coefs = {nm: float(beta[j]) for j, nm in enumerate(names) if nm.startswith(f"w_{h}_lag")}
        hosts_out[h] = {
            "c_per_100w": est,
            "ci95_block_bootstrap": [lo, hi],
            "se_hac": se_hac,
            "ci95_hac": [est - 1.96 * se_hac, est + 1.96 * se_hac],
            "interval_includes_zero": bool(lo <= 0.0 <= hi),
            "lag_coefficients_c_per_100w": lag_coefs,
        }
    ctl = {nm: float(beta[j]) for j, nm in enumerate(names) if not nm.startswith(("w_", "day_", "hod_"))}
    ctl_ci = {}
    ctl_contrasts = {nm: np.eye(len(names))[j] for j, nm in enumerate(names)
                     if nm == "central_ac_proxy_f" or nm.startswith("break:")}
    if ctl_contrasts:
        cb = block_bootstrap(X, y, day, ctl_contrasts, max(reps // 2, 20), seed + 1)
        ctl_ci = {k: [float(v) for v in np.percentile(cb[k], [2.5, 97.5])] for k in cb}
    return {
        "_fit": {"X": X, "y": y, "names": names, "day": day, "rows": rows, "beta": beta},
        "controls_ci95": ctl_ci,
        "label": label, "status": "ok", "regressor": spec, "hosts": hosts_out, "controls": ctl,
        "n": int(X.shape[0]), "n_days": int(len(np.unique(day))), "p": int(X.shape[1]),
        "r2": r2, "durbin_watson": dw, "resid_lag1_autocorr": ac1,
        "condition_number": float(np.linalg.cond(X)), "bootstrap_reps": reps,
        "design": dict(design_kw),
    }


def first_difference_hourly(frame: dict, spec: dict[str, str], *, reps: int, seed: int) -> dict:
    """Short-run check: hourly change in cabinet temp on the hourly change in each host's watts
    (this hour and last hour), change in outdoor temp and AC. Differencing removes slow weather
    and level confounds; what remains is how fast the cabinet answers a load step."""
    times = frame["times"]
    hours = sorted({t.replace(minute=0) for t in times})
    hidx = {h: i for i, h in enumerate(hours)}
    def hourly(x):
        acc = np.full(len(hours), np.nan)
        buckets: dict[int, list[float]] = {}
        for t, v in zip(times, x):
            if np.isfinite(v):
                buckets.setdefault(hidx[t.replace(minute=0)], []).append(v)
        for i, vs in buckets.items():
            if len(vs) >= 3:
                acc[i] = float(np.mean(vs))
        return acc
    cab = hourly(frame["cab"])
    hosts = tuple(spec)
    W = {h: hourly(s) for h, s in regressors(frame, spec).items()}
    px, ac = hourly(frame["proxy"]), hourly(frame["ac"])
    excl = hourly(frame["excluded"].astype(float))
    d = lambda x: np.concatenate([[np.nan], np.diff(x)])  # noqa: E731
    cols, names = [], []
    for h in hosts:
        dw = d(W[h]) / 100.0
        cols += [dw, _lag(dw, 1)]
        names += [f"w_{h}_lag0h", f"w_{h}_lag1h"]
    cols += [d(px), d(ac) / 100.0]
    names += ["d_central_ac_proxy_f", "d_ac_per100w"]
    for b in frame["breaks"]:
        step = np.array([1.0 if h >= b["at"].replace(minute=0) else 0.0 for h in hours])
        if 0 < step.sum() < len(step):
            cols.append(d(step))
            names.append("d_" + break_name(b))
    hod = np.array([h.hour for h in hours], dtype=float)
    for k in range(1, HARMONICS + 1):
        ang = 2 * math.pi * k * hod / 24.0
        cols += [np.sin(ang), np.cos(ang)]
        names += [f"hod_sin{k}", f"hod_cos{k}"]
    cols.append(np.ones(len(hours)))
    names.append("const")
    X = np.column_stack(cols)
    y = d(cab)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & ~(excl > 0) & ~(_lag(excl, 1) > 0)
    X, y = X[ok], y[ok]
    keep = [j for j, nm in enumerate(names) if nm == "const" or np.ptp(X[:, j]) > 0]
    X, names = X[:, keep], [names[j] for j in keep]
    day = np.array([h.date().toordinal() for h in hours])[ok]
    beta, resid = ols(X, y)
    contrasts = {h: np.array([1.0 if nm.startswith(f"w_{h}_lag") else 0.0 for nm in names]) for h in hosts}
    boots = block_bootstrap(X, y, day, contrasts, reps, seed)
    return {
        "label": "hourly_first_differences", "status": "ok", "regressor": spec, "n": int(len(y)),
        "hosts": {h: {"c_per_100w_step_2h": float(c @ beta),
                      "ci95_block_bootstrap": [float(v) for v in np.percentile(boots[h], [2.5, 97.5])]}
                  for h, c in contrasts.items()},
        "controls": {nm: float(beta[j]) for j, nm in enumerate(names) if nm.startswith("d_")},
    }


def host_diagnostics(frame: dict, spec: dict[str, str]) -> dict:
    out = {}
    hod = frame["hod"]
    for h, w in regressors(frame, spec).items():
        ok = np.isfinite(w)
        both = ok & np.isfinite(frame["outdoor"])
        fan = frame["fan"].get(h)
        fan_ok = ok & np.isfinite(frame["cab"]) & (np.isfinite(fan) if fan is not None else False)
        out[h] = {
            "regressor": spec[h],
            "fan_pct_corr_with_cabinet": float(np.corrcoef(fan[fan_ok], frame["cab"][fan_ok])[0, 1]) if fan is not None and fan_ok.sum() > 10 else None,
            "fan_pct_corr_with_regressor": float(np.corrcoef(fan[fan_ok], w[fan_ok])[0, 1]) if fan is not None and fan_ok.sum() > 10 else None,
            "bins_with_reading": int(ok.sum()),
            "bins_total": int(len(w)),
            "watts_p10": float(np.percentile(w[ok], 10)) if ok.any() else None,
            "watts_p50": float(np.percentile(w[ok], 50)) if ok.any() else None,
            "watts_p90": float(np.percentile(w[ok], 90)) if ok.any() else None,
            "watts_std": float(np.std(w[ok])) if ok.any() else None,
            "corr_with_outdoor": float(np.corrcoef(w[both], frame["outdoor"][both])[0, 1]) if both.sum() > 10 else None,
            "corr_with_hod_cos": float(np.corrcoef(w[ok], np.cos(2 * math.pi * hod[ok] / 24))[0, 1]) if ok.sum() > 10 else None,
        }
    ac = frame["ac"][np.isfinite(frame["ac"])]
    out["_ac"] = {"mean_w": float(ac.mean()), "std_w": float(ac.std()), "hours": int(len(ac) // 4)}
    o = frame["outdoor"][np.isfinite(frame["outdoor"])]
    out["_outdoor_c"] = {"min": float(o.min()), "max": float(o.max()), "std": float(o.std())}
    c = frame["cab"][np.isfinite(frame["cab"])]
    out["_cabinet_c"] = {"min": float(c.min()), "max": float(c.max()), "std": float(c.std())}
    return out


CORR_SQL = """
with b as (
  select date_trunc('hour', observed_at) h,
         avg((measurements_by_node->'athena'->>'cabinet_temp_c')::float) t,
         avg((measurements_by_node->'circe'->>'chassis_watts')::float)  cw,
         avg((measurements_by_node->'athena'->>'chassis_watts')::float) aw
  from orion_biometrics_cluster
  where observed_at >= timestamptz '{start} 00:00+00' and observed_at < timestamptz '{end} 00:00+00'
  group by 1),
e as (
  select hour_start h,
         max(case when source='circe'   then kwh*1000 end) ce,
         max(case when source='athena'  then kwh*1000 end) ae,
         max(case when source='cooling' then kwh*1000 end) co
  from orion_energy_hourly group by 1)
select json_build_object('n', count(*),
  'circe_chassis', corr(t, cw), 'circe_energy', corr(t, ce),
  'athena_chassis', corr(t, aw), 'athena_energy', corr(t, ae),
  'cooling_energy', corr(t, co))
from b join e on e.h = b.h + interval '{shift_h} hours';"""

CORR_WINDOWS = [
    ("2026-10-07", "2026-10-21", 0, "10-07..10-20 (the design's stated 14 days), aligned hours"),
    ("2026-10-09", "2026-10-23", 0, "10-09..10-22 (the last 14 days as of 10-23), aligned hours"),
    ("2026-10-07", "2026-10-21", 6, "10-07..10-20, energy hour joined 6 h late (UTC vs MDT mismatch)"),
]


def correlation_audit() -> list[dict]:
    rows = []
    for start, end, shift, label in CORR_WINDOWS:
        sql = CORR_SQL.format(start=start, end=end, shift_h=shift)
        res = data_mod.run_query(sql)
        rows.append({"label": label, "start": start, "end": end, "energy_hour_shift_h": shift,
                     "sql": sql.strip(), "result": res})
    return rows


def daily_levels(frame: dict) -> dict:
    """Daily means, and how the daily cabinet level tracks outdoor temperature and load."""
    days: dict[str, dict[str, list[float]]] = {}
    series = {"cabinet_c": frame["cab"], "outdoor_c": frame["outdoor"], "central_ac_proxy_f": frame["proxy"], "ac_w": frame["ac"],
              "circe_compute_w": frame["compute"]["circe"], "athena_compute_w": frame["compute"]["athena"]}
    for i, t in enumerate(frame["times"]):
        row = days.setdefault(t.date().isoformat(), {k: [] for k in series})
        for k, x in series.items():
            if np.isfinite(x[i]):
                row[k].append(float(x[i]))
    table = {d: {k: (float(np.mean(v)) if len(v) >= 48 else None) for k, v in row.items()} for d, row in days.items()}
    # House meter: household + central-AC kWh/h per day, all hours and local afternoon (12-20 MDT).
    rest = house_rest_hourly(frame["times"], frame.get("house_hourly") or (), frame["watts"], frame["ac"])
    for d in table:
        allh = [v for h, v in rest.items() if h.date().isoformat() == d]
        aft = [v for h, v in rest.items() if (h - timedelta(hours=6)).date().isoformat() == d
               and 12 <= (h - timedelta(hours=6)).hour <= 20]
        table[d]["house_rest_kwh"] = float(np.mean(allh)) if len(allh) >= 20 else None
        table[d]["house_rest_afternoon_kwh"] = float(np.mean(aft)) if len(aft) >= 8 else None
    core = ("cabinet_c", "outdoor_c", "central_ac_proxy_f", "ac_w", "circe_compute_w", "athena_compute_w")
    full = [r for r in table.values() if all(r.get(k) is not None for k in core)]
    corr = {}
    if len(full) >= 5:
        cab = np.array([r["cabinet_c"] for r in full])
        for k in ("outdoor_c", "central_ac_proxy_f", "ac_w", "circe_compute_w", "athena_compute_w"):
            corr[f"cabinet_vs_{k}"] = float(np.corrcoef(cab, [r[k] for r in full])[0, 1])
    return {"n_days": len(full), "daily_corr": corr, "table": table}


def _swing(d: dict) -> float:
    if d.get("watts_p90") is None or d.get("watts_p10") is None:
        return float("nan")
    return d["watts_p90"] - d["watts_p10"]


def headline(models: dict, diag: dict, decomp: dict | None = None) -> list[str]:
    lines = []
    for dk in ("primary", "proxy_offset_fixed_no_steps"):
        d = (decomp or {}).get(dk) or {}
        if d.get("status") != "ok":
            continue
        parts = ", ".join(f"{g} {v['degC']:+.2f} (CI {v['ci95'][0]:+.2f}..{v['ci95'][1]:+.2f})"
                          for g, v in sorted(d["contributions"].items(), key=lambda kv: -abs(kv[1]["degC"])))
        lines.append(f"[{dk}] cabinet rise {d['base'][0][:10]}..{d['base'][1][:10]} -> {d['target'][0][:10]}..{d['target'][1][:10]}: "
                     f"{d['rise_c']:+.2f} degC = {parts}; residual {d['residual_c']:+.2f}")
    prim = models["primary"]
    if prim.get("status") != "ok":
        return [f"primary fit did not run: {prim.get('status')} (n={prim.get('n')})"]
    for h, r in prim["hosts"].items():
        swing = _swing(diag[h])
        lo, hi = r["ci95_block_bootstrap"]
        lines.append(
            f"{h}: {r['c_per_100w']:+.3f} degC per 100 W held (95% CI {lo:+.3f}..{hi:+.3f}); "
            f"its p10->p90 swing ({swing:.0f} W) is worth {r['c_per_100w'] * swing / 100:+.2f} degC "
            f"(CI {lo * swing / 100:+.2f}..{hi * swing / 100:+.2f})"
        )
    fd = models.get("first_diff")
    if fd and fd.get("status") == "ok":
        for h, r in fd["hosts"].items():
            swing = _swing(diag[h])
            lo, hi = r["ci95_block_bootstrap"]
            lines.append(f"{h} short-run (hourly differences): {r['c_per_100w_step_2h']:+.3f} degC per 100 W step "
                         f"within 2 h (CI {lo:+.3f}..{hi:+.3f}); p10->p90 step = {r['c_per_100w_step_2h'] * swing / 100:+.2f} degC")
    return lines


def render_md(result: dict) -> str:
    L = ["# Heat sensitivity passive fit", "",
         f"Generated {result['generated_at']} over {result['window']['start']} .. {result['window']['end']}.", ""]
    L += ["## Headline", ""] + [f"- {x}" for x in result["headline"]] + [""]
    L += ["## Models (degC per 100 W held, sum of 0/15/30/60-min lags)", "",
          "| model | host | estimate | 95% CI (day-block bootstrap) | 95% CI (Newey-West) | n | R2 | resid lag-1 r |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in result["models"].items():
        if m.get("status") != "ok" or key.startswith("first_diff"):
            continue
        for h, r in m["hosts"].items():
            lo, hi = r["ci95_block_bootstrap"]
            hlo, hhi = r["ci95_hac"]
            L.append(f"| {key} | {h} | {r['c_per_100w']:+.3f} | {lo:+.3f}..{hi:+.3f} | {hlo:+.3f}..{hhi:+.3f} | "
                     f"{m['n']} | {m['r2']:.2f} | {m['resid_lag1_autocorr']:.2f} |")
    for key in ("first_diff", "first_diff_chassis"):
        fd = result["models"].get(key)
        if fd and fd.get("status") == "ok":
            L += ["", f"Hourly first differences, {key} (degC change over 2 h per 100 W step):", ""]
            for h, r in fd["hosts"].items():
                lo, hi = r["ci95_block_bootstrap"]
                L.append(f"- {h}: {r['c_per_100w_step_2h']:+.3f} (CI {lo:+.3f}..{hi:+.3f}), n={fd['n']}")
    L += ["", "## Controls (primary model)", ""]
    ci = result["models"]["primary"].get("controls_ci95") or {}
    for k, v in result["models"]["primary"]["controls"].items():
        extra = f" (CI {ci[k][0]:+.3f}..{ci[k][1]:+.3f})" if k in ci else ""
        L.append(f"- {k}: {v:+.4f}{extra}")
    for key, d in (result.get("decomposition") or {}).items():
        if d.get("status") != "ok":
            continue
        L += ["", f"## What explains the rise ({key})", "",
              f"Mean cabinet {d['base_mean_c']:.2f} C ({d['base'][0][:10]}..{d['base'][1][:10]}) -> "
              f"{d['target_mean_c']:.2f} C ({d['target'][0][:10]}..{d['target'][1][:10]}): {d['rise_c']:+.2f} C", "",
              "| term | degC | 95% CI |", "|---|---|---|"]
        for g, v in sorted(d["contributions"].items(), key=lambda kv: -abs(kv[1]["degC"])):
            L.append(f"| {g} | {v['degC']:+.2f} | {v['ci95'][0]:+.2f}..{v['ci95'][1]:+.2f} |")
        L.append(f"| residual (unexplained) | {d['residual_c']:+.2f} | |")
    dl = result.get("daily_levels") or {}
    if dl:
        L += ["", f"## Daily levels (n={dl['n_days']} full days)", ""]
        for k, v in dl["daily_corr"].items():
            L.append(f"- {k}: r = {v:+.2f}")
        L += ["", "| day | cabinet C | outdoor C | central-AC proxy F | portable AC W | circe compute W | athena compute W | house rest kWh/h | house rest afternoon (12-20 MDT) |", "|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda v: "" if v is None else f"{v:.1f}"  # noqa: E731
        fmt2 = lambda v: "" if v is None else f"{v:.2f}"  # noqa: E731
        for d, r in sorted(dl["table"].items()):
            L.append(f"| {d} | {fmt(r['cabinet_c'])} | {fmt(r['outdoor_c'])} | {fmt(r['central_ac_proxy_f'])} | {fmt(r['ac_w'])} | "
                     f"{fmt(r['circe_compute_w'])} | {fmt(r['athena_compute_w'])} | "
                     f"{fmt2(r.get('house_rest_kwh'))} | {fmt2(r.get('house_rest_afternoon_kwh'))} |")
    L += ["", "## Host diagnostics", "", "```json", json.dumps(result["diagnostics"], indent=1), "```"]
    if result.get("correlation_audit"):
        L += ["", "## Correlation audit (design Fact 2)", ""]
        for row in result["correlation_audit"]:
            r = row["result"]
            L.append(f"- {row['label']}: n={r['n']}, circe chassis {r['circe_chassis']:.3f} / energy {r['circe_energy']:.3f}; "
                     f"athena chassis {r['athena_chassis']:.3f} / energy {r['athena_energy']:.3f}; cooling {r['cooling_energy']:.3f}")
        L += ["", "Exact SQL (window and shift substituted per row):", "", "```sql", CORR_SQL.strip(), "```"]
    return "\n".join(L) + "\n"


def decompose(model: dict, frame: dict, *, base: tuple[datetime, datetime], target: tuple[datetime, datetime],
              reps: int, seed: int) -> dict:
    """How much of the change in mean cabinet temperature between two periods each term explains:
    contribution_k = coefficient_k x (target mean - base mean) of its columns, on the model's own
    rows. The residual is what no term explains. Intervals from the same day-block bootstrap."""
    f = model["_fit"]
    X, y, names, day, rows = f["X"], f["y"], f["names"], f["day"], f["rows"]
    times = [frame["times"][r] for r in rows]
    in_b = np.array([base[0] <= t < base[1] for t in times])
    in_t = np.array([target[0] <= t < target[1] for t in times])
    if in_b.sum() < 96 or in_t.sum() < 96:
        return {"status": "insufficient_rows", "base_rows": int(in_b.sum()), "target_rows": int(in_t.sum())}
    dx = X[in_t].mean(axis=0) - X[in_b].mean(axis=0)
    # Fixed measurement offsets were subtracted from y; add them back as their own fixed terms so
    # the decomposition is of the RAW reading's rise.
    fixed = {}
    for b in frame["breaks"]:
        off = (model.get("design") or {}).get("fixed_offsets", {}).get(b["title"])
        if off is not None:
            after = np.array([t >= b["at"] for t in times], dtype=float)
            fixed["fixed_offset:" + b["title"]] = off * float(after[in_t].mean() - after[in_b].mean())
    dy = float(y[in_t].mean() - y[in_b].mean()) + sum(fixed.values())
    raw = frame["cab"][rows]

    def group(nm: str) -> str:
        if nm.startswith("w_"):
            return "orion_load:" + nm.split("_")[1]
        if nm.startswith("hod_"):
            return "hour_of_day"
        if nm.startswith("outdoor_c"):
            return "outdoor_linear"
        if nm.startswith("day_"):
            return "day_dummies"
        return {"central_ac_proxy_f": "central_ac_proxy", "house_rest_kwh": "central_ac_house_meter",
                "ac_per100w": "portable_ac", "const": "const"}.get(nm, nm)

    groups: dict[str, np.ndarray] = {}
    for j, nm in enumerate(names):
        g = group(nm)
        if g == "const":
            continue
        groups.setdefault(g, np.zeros(len(names)))[j] = dx[j]
    boots = block_bootstrap(X, y, day, groups, reps, seed)
    beta = f["beta"]
    contrib = {g: {"degC": float(c @ beta), "ci95": [float(v) for v in np.percentile(boots[g], [2.5, 97.5])]}
               for g, c in groups.items()}
    for k, v in fixed.items():
        contrib[k] = {"degC": v, "ci95": [v, v], "fixed": True}
    explained = sum(v["degC"] for v in contrib.values())
    return {"status": "ok", "base": [base[0].isoformat(), base[1].isoformat()],
            "target": [target[0].isoformat(), target[1].isoformat()],
            "base_mean_c": float(raw[in_b].mean()), "target_mean_c": float(raw[in_t].mean()), "rise_c": dy,
            "rows": {"base": int(in_b.sum()), "target": int(in_t.sum())},
            "ci_note": "intervals cover coefficient uncertainty only; the input mean changes are held fixed",
            "contributions": contrib, "residual_c": dy - explained,
            "mean_change_of_inputs": {nm: float(dx[j]) for j, nm in enumerate(names) if not nm.startswith(("hod_", "day_"))}}


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


# #2780's change event: "cabinet reads ~2 C higher after" the 10-17 00:35 sensor-geometry break.
SENSOR_OFFSET_C = 2.0
DECOMP_BASE = (datetime(2026, 9, 26, tzinfo=timezone.utc), datetime(2026, 10, 11, tzinfo=timezone.utc))
DECOMP_TARGET = (datetime(2026, 10, 18, tzinfo=timezone.utc), datetime(2026, 10, 23, tzinfo=timezone.utc))


def run(extract: dict, *, reps: int, seed: int, with_corr: bool) -> dict:
    frame3 = build_frame(extract, data_mod.HOSTS)
    # Regressor is COMPUTE watts (CPU + GPU), not chassis watts: chassis watts include fans, and
    # fans spin up because the cabinet is hot (reverse causation). hecate reports only chassis watts.
    comp2 = {"athena": "compute", "circe": "compute"}
    chas2 = {"athena": "chassis", "circe": "chassis"}
    comp3 = {**comp2, "hecate": "chassis"}
    # Primary: orion-prediction's covariates and change events (#2829, #2780, #2716): the central-AC
    # proxy instead of a linear outdoor term, the portable AC, hour of day, one level-shift step per
    # change-event break, and the change-event exclusion windows masked. No day dummies: they would
    # absorb the central-AC effect, which acts at the day level.
    last_break = frame3["breaks"][-1]["at"] if frame3["breaks"] else None
    sensor_offset = {b["title"]: SENSOR_OFFSET_C for b in frame3["breaks"] if "sensor" in b["title"].lower()}
    models = {
        "primary": fit_model(frame3, comp2, label="compute watts + central-AC proxy + portable AC + hour; sensor break = fixed 2 C measurement offset; T10 break = free step", reps=reps, seed=seed, fixed_offsets=sensor_offset),
        "sensor_step_estimated": fit_model(frame3, comp2, label="as primary but the sensor break's size is estimated (free step)", reps=reps, seed=seed),
        "proxy_plus_linear_outdoor": fit_model(frame3, comp2, label="primary + linear outdoor temperature (0/1/3 h)", reps=reps, seed=seed, outdoor=True, fixed_offsets=sensor_offset),
        "house_meter_ac": fit_model(frame3, comp2, label="house-meter household+central kWh (4 h mean) instead of the proxy", reps=reps, seed=seed, proxy=False, house_ac=True),
        "house_meter_ac_offset_fixed": fit_model(frame3, comp2, label="house-meter AC, sensor offset fixed at the recorded value", reps=reps, seed=seed, proxy=False, house_ac=True, fixed_offsets=sensor_offset),
        "proxy_offset_fixed_no_steps": fit_model(frame3, comp2, label="primary, sensor offset fixed, no free break steps (T10 heat only through athena watts)", reps=reps, seed=seed, fixed_offsets=sensor_offset, breaks=False),
        "house_meter_ac_offset_fixed_no_steps": fit_model(frame3, comp2, label="house-meter AC, sensor offset fixed, no free break steps", reps=reps, seed=seed, proxy=False, house_ac=True, fixed_offsets=sensor_offset, breaks=False),
        "no_proxy": fit_model(frame3, comp2, label="primary without the central-AC proxy", reps=reps, seed=seed, proxy=False, fixed_offsets=sensor_offset),
        "primary_with_hecate": fit_model(frame3, comp3, label="primary + hecate chassis watts (data from 10-04)", reps=reps, seed=seed, fixed_offsets=sensor_offset),
        "day_fixed_effects": fit_model(frame3, comp2, label="primary + one dummy per day (absorbs day-level AC effect)", reps=reps, seed=seed, day_fe=True, fixed_offsets=sensor_offset),
        "chassis_watts": fit_model(frame3, chas2, label="primary with chassis watts (includes fans; reverse-causation risk)", reps=reps, seed=seed, fixed_offsets=sensor_offset),
        "first_diff": first_difference_hourly(frame3, comp2, reps=reps, seed=seed),
        "first_diff_chassis": first_difference_hourly(frame3, chas2, reps=reps, seed=seed),
    }
    if last_break is not None:
        models["before_last_break"] = fit_model(restrict(frame3, last_break), comp2,
                                                label=f"primary, only before {last_break.isoformat()}", reps=reps, seed=seed)
    decomp = {}
    for key in ("primary", "proxy_offset_fixed_no_steps", "sensor_step_estimated", "proxy_plus_linear_outdoor",
                "house_meter_ac", "house_meter_ac_offset_fixed", "house_meter_ac_offset_fixed_no_steps"):
        if models[key].get("status") == "ok":
            decomp[key] = decompose(models[key], frame3, base=DECOMP_BASE, target=DECOMP_TARGET, reps=reps, seed=seed + 7)
    diag = host_diagnostics(frame3, comp3)
    daily = daily_levels(frame3)
    diag["_chassis"] = host_diagnostics(frame3, chas2)
    hec = frame3["watts"]["hecate"]
    diag["hecate"]["awake_hours_over_150w"] = float(np.sum(np.isfinite(hec) & (hec > 150)) / 4.0)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": extract["window"],
        "method": {
            "bin_minutes": BIN_MIN, "watt_lags_min": [k * BIN_MIN for k in LAGS_BINS],
            "outdoor_lags_h": list(OUTDOOR_LAGS_H), "hour_harmonics": HARMONICS,
            "central_ac_proxy": frame3["proxy_spec"],
            "change_event_breaks": [{"at": b["at"].isoformat(), "title": b["title"], "notes": b["notes"]} for b in frame3["breaks"]],
            "change_event_excludes": [[a.isoformat(), b.isoformat()] for a, b in frame3["excludes"]],
            "sensitivity": "sum of a host's lag coefficients = degC per 100 W held steady",
            "ci": "day-block bootstrap percentile (whole days resampled) + Newey-West HAC lower bound (96 lags)",
            "outdoor_source": "prediction_weather_hourly (orion-prediction, Open-Meteo)",
        },
        "models": {k: _public(m) for k, m in models.items()},
        "decomposition": decomp,
        "diagnostics": diag,
        "daily_levels": daily,
        "headline": headline(models, diag, decomp),
        "replay_sensitivity": {},
        "queries": extract.get("queries", {}),
    }
    for h, r in models["primary"]["hosts"].items():
        lo, hi = r["ci95_block_bootstrap"]
        result["replay_sensitivity"][h] = {"point": r["c_per_100w"], "lo": lo, "hi": hi, "source": "primary"}
    wh = models["primary_with_hecate"]
    if wh.get("status") == "ok" and "hecate" in wh["hosts"]:
        r = wh["hosts"]["hecate"]
        lo, hi = r["ci95_block_bootstrap"]
        result["replay_sensitivity"]["hecate"] = {"point": r["c_per_100w"], "lo": lo, "hi": hi, "source": "primary_with_hecate"}
    if with_corr:
        result["correlation_audit"] = correlation_audit()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2026-09-26T00:00:00Z", help="weather history starts 09-26")
    ap.add_argument("--end", default="2026-10-23T00:00:00Z")
    ap.add_argument("--cache", default=None, help="read/write the raw extract here")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--no-correlation-audit", action="store_true")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    extract = data_mod.load_or_fetch(a.start, a.end, a.cache)
    result = run(extract, reps=a.reps, seed=a.seed, with_corr=not a.no_correlation_audit)
    Path(a.out_json).write_text(json.dumps(result, indent=1, default=str))
    Path(a.out_md).write_text(render_md(result))
    print("\n".join(result["headline"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
