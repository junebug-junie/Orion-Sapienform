"""Family C: "what do my own actions do?" -- the measured effect table and the efference copy.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md, "C. Causal effects".

The effect table
    One row per (actuator, target reading, window). ``window`` is minutes after the act starts:
    ``[burn_in_min, block_min)`` of the experiment that measured it (the part of each block the
    switchback analysis reads). Each row holds the ON-minus-OFF estimate, its 95 % interval and
    standard error, the blocks per arm, and the experiment ids it came from. Rows come ONLY from
    randomised switchback experiments (``app/causal.py`` level 3) that finished (status
    ``analyzed``); the targets are every raw reading Family A reads (``config/prediction/
    family_a.v1.yaml``). Nobody writes a row by hand: the table is recomputed from the logged blocks
    and the raw readings, so it grows only when Orion runs an experiment (lesson from orion-signals).

    ``choose()`` is the only way to read a row for acting or forecasting, and it refuses anything
    whose ``method`` is not ``randomised``. Observational and conditional results (levels 1 and 2)
    stay where they are, labelled, and can never reach it.

The efference copy
    Before an act: take the newest Family A forecast issued at or before the act started (so it has
    not seen the act) whose target falls in the row's window, and add the table's effect. That is
    the expected reading with the act. After the act: score the real reading against it.
    "I did X, expected Y, got Z" is a prediction error about Orion's own agency.

    Distribution: the forecast's own stored quantiles, every one moved by the effect and widened
    about the median for the effect's standard error (``shifted``); read as a CDF by ``pit`` (linear
    between quantiles, Gaussian tails). Two-sided p, surprisal ``-ln p``, excess ``-ln p - 1``
    (E[-ln U] = 1 for a uniform p, so a calibrated score rests at 0 by theory, the unit Family A uses).
    The same reading is also scored against the forecast alone (no effect). ``gain`` = log
    predictive density with the effect minus without it (a proper score): positive when knowing the
    act's effect made the forecast better. Not the surprisal difference: two-sided-p surprisal is
    lowered by any widening, so a table that only added uncertainty would have "won". ``gate()`` asks whether that gain is real across acts (gap, SE, prob_better; no alpha
    line).

Slice 1 is eval/replay only: no bus event, no substrate node, no consumer reads the miss score.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from app.causal import analyze_switchback, time_of_day_profile

METHOD_RANDOMISED = "randomised"
CHOOSABLE_METHODS = frozenset({METHOD_RANDOMISED})
ESTIMATOR = "family_c_effect_table"
PROFILE_DAYS = 14
N_PERM = 2000
Z80 = 1.2815515655446004            # standard normal 0.9 quantile: the 80 % range is +/- Z80 sigma
CALM_SURPRISAL = 1.0                # E[-ln U], U ~ Uniform(0, 1)
P_FLOOR = 1e-300


class NotChoosable(LookupError):
    """No randomised effect row answers this (actuator, target, offset)."""


class EfferenceRefused(ValueError):
    """The forecast or the act cannot make an efference copy (forecast saw the act, no range, ...)."""


# --------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectRow:
    actuator: str
    target: str
    unit: str | None
    window_start_min: int
    window_end_min: int
    method: str
    estimate: float | None
    ci_lo: float | None
    ci_hi: float | None
    se: float | None
    n_on: int
    n_off: int
    p_value: float | None
    experiment_ids: tuple[str, ...]
    reason: str | None = None
    profile_adjusted: bool = False

    @property
    def eff_se(self) -> float | None:
        """SE read off the row's own 95 % interval. With ~6 blocks per arm the Welch interval uses
        t(dof) ~ 2.2, so this is wider than the raw ``se`` -- the honest spread for widening a
        forecast and for pooling."""
        if self.ci_lo is None or self.ci_hi is None:
            return self.se
        return (self.ci_hi - self.ci_lo) / (2 * 1.96)

    def covers(self, offset_min: float) -> bool:
        return self.window_start_min <= offset_min < self.window_end_min

    def as_dict(self) -> dict:
        d = asdict(self)
        d["experiment_ids"] = list(self.experiment_ids)
        return d


def _profile_for(history: pd.Series, window: pd.Series, grain_min: int) -> pd.Series | None:
    """Time-of-day profile from the history before the experiment, or None when it does not cover
    every bucket the experiment ran in (a NaN profile bucket would silently drop that bucket)."""
    hist = history.dropna()
    if hist.empty:
        return None
    profile = time_of_day_profile(hist, grain_min)
    tod = (window.index.hour * 60 + window.index.minute) // grain_min * grain_min
    if profile.reindex(tod).isna().any():
        return None
    return profile


def rows_from_experiment(exp: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]], frame: pd.DataFrame,
                         units: Mapping[str, str], grain_min: int, n_perm: int = N_PERM) -> list[EffectRow]:
    """One row per frame column for one finished randomised experiment.

    ``blocks``: the experiment's design blocks as run ({index, arm, start, end}), already checked
    against the design (``Effects._match_design``). ``frame``: grain-bucket means on a regular UTC
    grid covering ``PROFILE_DAYS`` before the first block through the last block.

    The time-of-day profile is variance reduction only; randomisation is what keeps the estimate
    unbiased. Unlike ``Effects.analyze_experiment`` it does not cut the history at change events or
    mask registry exclusions (Family A's readings have no registry spec), so for the experiment's own
    target the interval can differ from the stored ``/effects`` row."""
    if not blocks:
        raise ValueError("no blocks")
    burn = int(exp["burn_in_min"])
    block_min = int(exp["design"]["block_min"])
    first = min(pd.Timestamp(b["start"]) for b in blocks)
    last = max(pd.Timestamp(b["end"]) for b in blocks)
    eid = str(exp["experiment_id"])
    rows = []
    for target in frame.columns:
        col = frame[target].astype(float)
        hist = col[(col.index >= first - pd.Timedelta(days=PROFILE_DAYS)) & (col.index < first)]
        win = col[(col.index >= first) & (col.index < last)]
        profile = _profile_for(hist, win, grain_min)
        est = analyze_switchback(win, list(blocks), burn, grain_min, profile, n_perm=n_perm, seed=0)
        d = est.details
        rows.append(EffectRow(
            actuator=str(exp["action"]), target=str(target), unit=units.get(target),
            window_start_min=burn, window_end_min=block_min, method=METHOD_RANDOMISED,
            estimate=est.effect, ci_lo=est.ci_lo, ci_hi=est.ci_hi, se=d.get("se"),
            n_on=int(d.get("n_on") or sum(1 for b in d["blocks"] if b["arm"] == "on" and b["mean"] is not None)),
            n_off=int(d.get("n_off") or sum(1 for b in d["blocks"] if b["arm"] == "off" and b["mean"] is not None)),
            p_value=est.p_value, experiment_ids=(eid,), reason=d.get("reason"),
            profile_adjusted=profile is not None))
    return rows


def pool_rows(rows: Iterable[EffectRow]) -> list[EffectRow]:
    """One row per (actuator, target, window, method): inverse-variance pool of the experiments that
    measured it. A single experiment's row passes through unchanged (its Welch interval kept)."""
    groups: dict[tuple, list[EffectRow]] = {}
    for r in rows:
        groups.setdefault((r.actuator, r.target, r.window_start_min, r.window_end_min, r.method), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        ids = tuple(sorted({e for r in rs for e in r.experiment_ids}))
        good = [r for r in rs if r.estimate is not None and r.eff_se is not None and r.eff_se > 0]
        if len(rs) == 1:
            out.append(rs[0])
            continue
        if not good:
            out.append(EffectRow(*key[:2], rs[0].unit, key[2], key[3], key[4], None, None, None, None,
                                 sum(r.n_on for r in rs), sum(r.n_off for r in rs), None, ids,
                                 reason="; ".join(sorted({r.reason or "no estimate" for r in rs}))))
            continue
        w = np.array([1.0 / r.eff_se ** 2 for r in good])
        est = float(np.sum(w * np.array([r.estimate for r in good])) / np.sum(w))
        se = float(math.sqrt(1.0 / np.sum(w)))
        p = math.erfc(abs(est / se) / math.sqrt(2)) if se > 0 else None
        out.append(EffectRow(key[0], key[1], good[0].unit, key[2], key[3], key[4], est, est - 1.96 * se,
                             est + 1.96 * se, se, sum(r.n_on for r in good), sum(r.n_off for r in good), p, ids,
                             reason=None if len(good) == len(rs) else f"{len(rs) - len(good)} experiment(s) gave no estimate",
                             profile_adjusted=all(r.profile_adjusted for r in good)))
    return out


def choose(table: Sequence[EffectRow], actuator: str, target: str, offset_min: float) -> EffectRow:
    """The row an act or a forecast may use. Refuses every method other than ``randomised``."""
    hits = [r for r in table if r.actuator == actuator and r.target == target and r.covers(offset_min)]
    if not hits:
        raise NotChoosable(f"no effect row for {actuator} -> {target} at +{offset_min:g} min")
    usable = [r for r in hits if r.method in CHOOSABLE_METHODS]
    if not usable:
        kinds = sorted({r.method for r in hits})
        raise NotChoosable(f"{actuator} -> {target}: only {kinds} rows; never used to choose an action")
    measured = [r for r in usable if r.estimate is not None and r.eff_se is not None]
    if not measured:
        raise NotChoosable(f"{actuator} -> {target}: randomised row has no estimate ({usable[0].reason})")
    return min(measured, key=lambda r: (r.eff_se, r.window_end_min - r.window_start_min))


# --------------------------------------------------------------------------------------------
# the efference copy
# --------------------------------------------------------------------------------------------

_N = NormalDist()


def _levels(quantiles: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(levels, values) sorted by level; refuses a range with no 80 % pair, a non-finite value, a
    decreasing value, or no spread."""
    pairs = []
    for k, v in quantiles.items():
        try:
            lv, val = float(k), float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 < lv < 1.0:
            pairs.append((lv, val))
    pairs.sort()
    lv = np.array([p[0] for p in pairs])
    val = np.array([p[1] for p in pairs])
    if not ({0.1, 0.9} <= {round(x, 6) for x in lv}):
        raise EfferenceRefused("forecast has no 80 % range (quantiles 0.1 and 0.9)")
    if not np.all(np.isfinite(val)) or np.any(np.diff(val) < 0) or val[-1] <= val[0]:
        raise EfferenceRefused("forecast range is not finite and increasing")
    # Equal neighbouring quantiles (a flat stretch) would make np.interp undefined: nudge them apart
    # by a negligible fraction of the range, keeping order.
    val = val + np.arange(len(val)) * (val[-1] - val[0]) * 1e-9
    return lv, val


def sigma_from_quantiles(quantiles: Mapping[str, float]) -> float:
    """Gaussian-equivalent spread of a forecast's 80 % range (keys as stored: '0.1', '0.9')."""
    lv, val = _levels(quantiles)
    lo, hi = float(np.interp(0.1, lv, val)), float(np.interp(0.9, lv, val))
    if hi <= lo:
        raise EfferenceRefused("forecast has no usable 80 % range")
    return (hi - lo) / (2.0 * Z80)


def pit(quantiles: Mapping[str, float], x: float) -> float:
    """P(reading <= x) under a stored quantile forecast: linear between the stored quantiles, and a
    Gaussian tail beyond the outermost pair, with the spread that pair implies (continuous at both
    ends). Live 2026-11-15 rest check: reading the forecast this way rests far nearer 0 than a single
    Gaussian read off the 80 % range (Family A's misses have heavier tails than a Gaussian)."""
    lv, val = _levels(quantiles)
    s = (val[-1] - val[0]) / (_N.inv_cdf(lv[-1]) - _N.inv_cdf(lv[0]))
    if x < val[0]:
        return _N.cdf((x - val[0]) / s + _N.inv_cdf(lv[0]))
    if x > val[-1]:
        return _N.cdf((x - val[-1]) / s + _N.inv_cdf(lv[-1]))
    return float(np.interp(x, val, lv))


def shifted(quantiles: Mapping[str, float], effect: float, effect_se: float) -> dict[str, float]:
    """The forecast with the act: every quantile moved by the effect, and the spread about the median
    widened by sqrt(1 + (se / sigma80)^2) for the effect's own uncertainty (exact for a Gaussian
    forecast, an approximation otherwise)."""
    lv, val = _levels(quantiles)
    med = float(np.interp(0.5, lv, val))
    k = math.sqrt(1.0 + (effect_se / sigma_from_quantiles(quantiles)) ** 2)
    return {f"{level:g}": med + effect + (v - med) * k for level, v in zip(lv, val)}


@dataclass(frozen=True)
class Efference:
    act_id: str
    actuator: str
    target: str
    act_start: datetime
    target_ts: datetime
    offset_min: float
    forecast_origin: datetime
    baseline_point: float
    baseline_quantiles: Mapping[str, float]
    effect: float
    effect_se: float
    expected: float
    expected_quantiles: Mapping[str, float]
    experiment_ids: tuple[str, ...]


def _utc(v: Any) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def efference_copy(act_id: str, actuator: str, act_start: datetime, act_end: datetime,
                   forecast: Mapping[str, Any], table: Sequence[EffectRow], grain_min: int = 5) -> Efference:
    """``forecast``: a Family A reading forecast {target, origin_ts, target_ts, point, quantiles}.

    Family A's ``origin_ts`` is the START of the last complete bucket, so the forecast saw data up to
    ``origin_ts + grain``: that must be at or before the act started. The effect was measured with the
    act held for the whole block, so the act must still be on at ``target_ts``."""
    origin, target_ts = _utc(forecast["origin_ts"]), _utc(forecast["target_ts"])
    start, end = _utc(act_start), _utc(act_end)
    if origin + pd.Timedelta(minutes=grain_min) > start:
        raise EfferenceRefused(f"forecast origin {origin.isoformat()} (+{grain_min} min of data) reaches past "
                               "the act start; it saw the act")
    if target_ts > end:
        raise EfferenceRefused(f"the act ended at {end.isoformat()}, before the target {target_ts.isoformat()}; "
                               "the table's effect is for an act held through the window")
    offset = (target_ts - start).total_seconds() / 60.0
    row = choose(table, actuator, str(forecast["target"]), offset)
    q = {str(k): float(v) for k, v in forecast["quantiles"].items()}
    expected_q = shifted(q, float(row.estimate), float(row.eff_se))
    point = float(forecast["point"])
    return Efference(act_id=act_id, actuator=actuator, target=str(forecast["target"]),
                     act_start=start.to_pydatetime(), target_ts=target_ts.to_pydatetime(), offset_min=offset,
                     forecast_origin=origin.to_pydatetime(), baseline_point=point, baseline_quantiles=q,
                     effect=float(row.estimate), effect_se=float(row.eff_se),
                     expected=float(np.interp(0.5, *_levels(expected_q))),
                     expected_quantiles=expected_q, experiment_ids=row.experiment_ids)


def null_efference(act_id: str, forecast: Mapping[str, Any]) -> Efference:
    """No act: the efference copy is the forecast itself (the score's rest point, for evals)."""
    q = {str(k): float(v) for k, v in forecast["quantiles"].items()}
    _levels(q)
    t = pd.Timestamp(forecast["origin_ts"]).to_pydatetime()
    tt = pd.Timestamp(forecast["target_ts"]).to_pydatetime()
    point = float(forecast["point"])
    return Efference(act_id, "none", str(forecast["target"]), t, tt, (tt - t).total_seconds() / 60.0, t, point, q,
                     0.0, 0.0, point, q, ())


def log_density(quantiles: Mapping[str, float], x: float) -> float:
    """ln of the density ``pit`` integrates: flat between stored quantiles, Gaussian beyond the outer
    pair. The log score is proper, so unlike two-sided-p surprisal it cannot be won by a forecast
    that is merely wider (that is why ``gain`` uses it)."""
    lv, val = _levels(quantiles)
    s = (val[-1] - val[0]) / (_N.inv_cdf(lv[-1]) - _N.inv_cdf(lv[0]))
    if x < val[0] or x > val[-1]:
        edge, z_edge = (val[0], _N.inv_cdf(lv[0])) if x < val[0] else (val[-1], _N.inv_cdf(lv[-1]))
        z = (x - edge) / s + z_edge
        return -0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(s)
    i = min(int(np.searchsorted(val, x, side="right")) - 1, len(val) - 2)
    return math.log((lv[i + 1] - lv[i]) / (val[i + 1] - val[i]))


def _p_two_sided(u: float) -> float:
    return min(1.0, max(2.0 * min(u, 1.0 - u), P_FLOOR))


def score(eff: Efference, actual: float | None) -> dict:
    """Score the realized reading. ``actual`` None -> status ``missing`` (scored, never dropped)."""
    out: dict[str, Any] = {"act_id": eff.act_id, "actuator": eff.actuator, "target": eff.target,
                           "target_ts": eff.target_ts.isoformat(), "offset_min": eff.offset_min,
                           "expected": eff.expected, "effect": eff.effect, "baseline_point": eff.baseline_point}
    if actual is None or not math.isfinite(float(actual)):
        return {**out, "status": "missing"}
    a = float(actual)
    u, ub = pit(eff.expected_quantiles, a), pit(eff.baseline_quantiles, a)
    p, pb = _p_two_sided(u), _p_two_sided(ub)
    s, sb = -math.log(p), -math.log(pb)
    ld, ldb = log_density(eff.expected_quantiles, a), log_density(eff.baseline_quantiles, a)
    return {**out, "status": "scored", "actual": a, "miss": a - eff.expected, "pit": u, "p": p, "surprisal": s,
            "excess": s - CALM_SURPRISAL, "baseline_pit": ub, "baseline_p": pb, "baseline_surprisal": sb,
            "log_density": ld, "baseline_log_density": ldb, "gain": ld - ldb}


# --------------------------------------------------------------------------------------------
# the model gate: does the efference copy beat the forecast alone?
# --------------------------------------------------------------------------------------------

def gate(scores: Sequence[Mapping[str, Any]]) -> dict:
    """Per act, mean gain over its scored readings (readings of one act are not independent); then the
    mean over acts with its standard error and prob_better = Phi(gap / SE). Numbers, no verdict line."""
    per_act: dict[str, list[float]] = {}
    excess = []
    p05 = 0
    n_scored = 0
    for s in scores:
        if s.get("status") != "scored":
            continue
        n_scored += 1
        per_act.setdefault(str(s["act_id"]), []).append(float(s["gain"]))
        excess.append(float(s["excess"]))
        p05 += s["p"] <= 0.05
    g = np.array([float(np.mean(v)) for v in per_act.values()])
    out: dict[str, Any] = {"n_acts": int(len(g)), "n_scored": n_scored,
                           "n_missing": sum(1 for s in scores if s.get("status") == "missing"),
                           "mean_excess": float(np.mean(excess)) if excess else None,
                           "share_p_le_0_05": (p05 / n_scored) if n_scored else None}
    if len(g) < 2:
        return {**out, "gap": float(g[0]) if len(g) else None, "se": None, "prob_better": None,
                "reason": "need >= 2 acts for a standard error"}
    gap, se = float(g.mean()), float(g.std(ddof=1) / math.sqrt(len(g)))
    prob = 0.5 * math.erfc(-(gap / se) / math.sqrt(2.0)) if se > 0 else (1.0 if gap > 0 else 0.0 if gap < 0 else 0.5)
    return {**out, "gap": gap, "se": se, "prob_better": prob}


# --------------------------------------------------------------------------------------------
# orchestration: the live table, read-only, recomputed from logged experiments
# --------------------------------------------------------------------------------------------

Loader = Callable[[datetime, datetime], "tuple[pd.DataFrame, dict]"]


@dataclass
class EffectTable:
    """Builds the table from every ``analyzed`` experiment. Read-only: nothing is written. A finished
    experiment's rows never change, so they are cached by experiment id."""
    store: Any
    loader: Loader
    units: Mapping[str, str]
    grain_min: int
    n_perm: int = N_PERM
    _cache: dict[str, list[EffectRow]] = field(default_factory=dict)

    def rows_for(self, experiment_id: str) -> list[EffectRow]:
        from app.causal import CausalError
        from app.effects import Effects

        hit = self._cache.get(experiment_id)
        if hit is not None:
            return hit
        exp = self.store.experiment(experiment_id)
        if exp is None:
            raise CausalError("unknown experiment")
        if exp["status"] != "analyzed":
            raise CausalError(f"experiment is {exp['status']}; only finished (analyzed) experiments enter the table")
        logged = [r for r in self.store.interventions(experiment_id=experiment_id, limit=1000) if r["arm"]]
        blocks = Effects._match_design(exp, logged)
        grain = f"{self.grain_min}min"
        # The loader's grid starts at ``start``; its SQL buckets sit on grain boundaries. An unaligned
        # start (logged blocks may be up to 2 min off the design) would leave every column NaN.
        first = (min(pd.Timestamp(b["start"]) for b in blocks) - pd.Timedelta(days=PROFILE_DAYS)).floor(grain)
        last = max(pd.Timestamp(b["end"]) for b in blocks).ceil(grain)
        frame, _ = self.loader(first.to_pydatetime(), last.to_pydatetime())
        rows = rows_from_experiment(exp, blocks, frame, self.units, self.grain_min, self.n_perm)
        self._cache[experiment_id] = rows
        return rows

    def build(self) -> dict:
        exps = self.store.experiments(status="analyzed")
        per, errors = [], {}
        for e in exps:
            eid = str(e["experiment_id"])
            try:
                per.extend(self.rows_for(eid))
            except Exception as exc:  # noqa: BLE001 -- one bad experiment is reported, not fatal
                errors[eid] = f"{type(exc).__name__}: {exc}"[:300]
        pooled = pool_rows(per)
        return {"method": METHOD_RANDOMISED, "estimator": ESTIMATOR, "experiments": [str(e["experiment_id"]) for e in exps],
                "errors": errors, "n_rows": len(pooled),
                "n_with_estimate": sum(1 for r in pooled if r.estimate is not None),
                "rows": [r.as_dict() for r in pooled],
                "note": ("Rows come only from randomised switchback experiments. One experiment measured many "
                         "readings: expect about 1 in 20 rows to show p <= 0.05 by chance alone.")}
