"""Surprise: how badly the world missed a forecast, in one unit with a calm point set by math.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md ("Definition").

A prediction error is valid only if the forecast was stored before its target time, it is scored
when the clock says so (a bucket with no data is ``missing``, never calm), and surprise is the
surprisal of the outcome under the forecast:

- ``p_value``: two-sided conformal p of the actual value under the forecast's stored quantiles,
  ``2 * min(u, 1 - u)`` where ``u`` is the forecast CDF at the actual value (linear between the
  stored quantile levels). Past the outermost stored quantile (the 95 % range) the tail shape is
  unknown, so ``p`` is censored at ``P_FLOOR`` and flagged -- never extrapolated.
- ``surprisal`` = ``-ln p``.
- ``CALM_SURPRISAL``: if the forecaster is calibrated, ``p`` is uniform on [0, 1]; with the floor,
  E[-ln max(p, P_FLOOR)] = 1 - P_FLOOR exactly (0.95). Calm rests here by theory, not by tuning.
  Known method bias: the CDF is linear between only 9 stored levels, so a perfectly calibrated
  normal forecaster reads about -0.033 (tests/test_surprise.py), not exactly 0.
- ``excess`` = surprisal - CALM_SURPRISAL. Mean excess near 0 = calm; a single large excess = a
  spike (one forecast missed badly); a sustained offset = drift (ranges too wide reads negative,
  too narrow reads positive -- the live 80 % coverage says which).

A horizon publishes (``gate``) only when its forecaster passed selection (the served model is not
a baseline), its live 80 % coverage sits in the selection band over enough scores, and its live
record is not worse than the baseline. Same bar the forecast itself must clear.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from app.backtest import COVERAGE_BAND
from app.models import BASELINES

# Outermost stored central interval is 95 % (quantiles 0.025 / 0.975), so p cannot be resolved
# below 0.05.
P_FLOOR = 0.05
CALM_SURPRISAL = 1.0 - P_FLOOR
MIN_GATE_SCORES = 200   # same count engine.MIN_LIVE_COVERAGE_N requires before judging coverage


def conformal_p_value(actual: float, quantiles: Mapping[str, float]) -> tuple[float, bool]:
    """(p, censored). ``censored`` = the actual fell outside every stored quantile, so the true p
    is somewhere at or below ``P_FLOOR``."""
    pts = sorted((float(k), float(v)) for k, v in quantiles.items())
    if not pts:
        raise ValueError("no quantiles")
    if actual < pts[0][1] or actual > pts[-1][1]:
        return P_FLOOR, True
    # Collapsed levels (equal values, e.g. a 0.025 and 0.05 both clipped to the pool's min) form one
    # CDF point at their mean level; between points the CDF is linear.
    groups: list[tuple[list[float], float]] = []
    for level, x in pts:
        if groups and x == groups[-1][1]:
            groups[-1][0].append(level)
        else:
            groups.append(([level], x))
    cdf = [(sum(ls) / len(ls), x) for ls, x in groups]
    u = cdf[0][0]
    for (l0, x0), (l1, x1) in zip(cdf, cdf[1:]):
        if x0 <= actual <= x1:
            u = l0 + (l1 - l0) * (actual - x0) / (x1 - x0)
            break
    p = 2.0 * min(u, 1.0 - u)
    return max(P_FLOOR, min(1.0, p)), False


def surprisal(p: float) -> float:
    return -math.log(max(P_FLOOR, min(1.0, p)))


def excess(p: float) -> float:
    return surprisal(p) - CALM_SURPRISAL


def horizon_gate(hm: int, horizon_live: Mapping | None, served_model: str | None,
                 live_verdict: Mapping | None) -> tuple[bool, str]:
    """(passes, reason) for one horizon, from the monitor's own live metrics."""
    if served_model is None:
        return False, "no_selection"
    if served_model in BASELINES:
        return False, "served_baseline"
    if live_verdict and live_verdict.get("verdict") == "worse":
        return False, "live_worse_than_baseline"
    if not horizon_live or (horizon_live.get("n") or 0) < MIN_GATE_SCORES:
        return False, "too_few_live_scores"
    c = horizon_live.get("coverage80")
    if c is None or not (COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1]):
        return False, "coverage_out_of_band"
    return True, "passed"


def summarize(rows: list[Mapping], grain_min: int) -> dict:
    """Per horizon over ``rows`` (scored champion rows with actual + quantiles): mean surprisal,
    mean excess with a 95 % null band from independent origins only (forecasts issued every grain
    overlap for h/grain steps), and the share of p at or below 0.05 and 0.2 (calibrated: 0.05, 0.2)."""
    by_h: dict[int, list[Mapping]] = {}
    for r in rows:
        by_h.setdefault(int(r["horizon_min"]), []).append(r)
    out: dict[str, dict] = {}
    for hm, rs in sorted(by_h.items()):
        rs = sorted(rs, key=lambda r: r["origin_ts"])
        ps = np.array([conformal_p_value(float(r["actual"]), r["quantiles"])[0] for r in rs])
        sur = -np.log(ps)
        thin = sur[:: max(1, hm // grain_min)]
        band = (float(1.96 * thin.std(ddof=1) / math.sqrt(len(thin)))
                if len(thin) >= 2 else None)
        out[str(hm)] = {
            "n": int(len(rs)),
            "n_independent": int(len(thin)),
            "mean_surprisal": float(sur.mean()),
            "mean_excess": float(sur.mean() - CALM_SURPRISAL),
            "null_band_95": band,
            "calm": (None if band is None else bool(abs(float(thin.mean() - CALM_SURPRISAL)) <= band)),
            "share_p_le_0_05": float(np.mean(ps <= P_FLOOR)),
            "share_p_le_0_2": float(np.mean(ps <= 0.2)),
        }
    return out
