"""Live verdicts: does a model beat the baseline on real readings, not just in the backtest?

Input: scored shadow forecasts paired with the baseline forecast issued at the same origin for the
same horizon. Forecasts issued every grain overlap for h/grain steps, so per (model, horizon) only the
first origin of each absolute h-long time block is kept (anchored to the epoch, so the kept set does not
shift as the window slides). With at least MIN_INDEPENDENT such pairs, the overlap-aware paired test
(backtest.paired_win_p) runs in both directions:

    better   -- the model's error is lower: Holm-adjusted p < LIVE_ALPHA across every (model, horizon)
                tested (with 4 contenders x 5 horizons, an unadjusted 0.05 read "better" for some null
                pair in ~1/3 of selections)
    worse    -- the baseline's error is lower: unadjusted p < LIVE_ALPHA (blocking errs on caution)
    unclear  -- enough data, neither
    pending  -- fewer than MIN_INDEPENDENT independent pairs

``window_for(h)``: a horizon needs MIN_INDEPENDENT non-overlapping forecasts, so its look-back is at
least MIN_INDEPENDENT + 5 horizons long (24 h -> 25 days); never shorter than MIN_WINDOW.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np

from app.backtest import paired_win_p

MIN_INDEPENDENT = 20
LIVE_ALPHA = 0.05
MIN_WINDOW = timedelta(days=7)
RETENTION = timedelta(days=35)


def window_for(horizon_min: int) -> timedelta:
    return max(MIN_WINDOW, timedelta(minutes=(MIN_INDEPENDENT + 5) * horizon_min))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def verdicts(pairs, now=None) -> dict[tuple[str, int], dict]:
    """``pairs``: rows {model, horizon_min, origin_ts, shadow_err, base_err}, any order. Rows older than
    ``window_for(h)`` before ``now`` are ignored (callers may pass a longer history)."""
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for r in pairs:
        e_s, e_b = r["shadow_err"], r["base_err"]
        if e_s is None or e_b is None or not (np.isfinite(e_s) and np.isfinite(e_b)):
            continue
        hm = int(r["horizon_min"])
        if now is not None and r["origin_ts"] < now - window_for(hm):
            continue
        groups[(r["model"], hm)].append(r)
    out, p_better = {}, {}
    for (model, hm), rows in groups.items():
        rows.sort(key=lambda r: r["origin_ts"])
        kept, block = [], None
        for r in rows:
            b = int(r["origin_ts"].timestamp() // 60) // hm
            if b != block:
                kept.append(r)
                block = b
        s = np.array([r["shadow_err"] for r in kept], dtype=float)
        b = np.array([r["base_err"] for r in kept], dtype=float)
        rel = float(s.mean() / b.mean()) if len(kept) and b.mean() > 0 else None
        v = {"n_scored": len(rows), "n_independent": len(kept), "rel_mae": rel,
             "p_better": None, "p_better_adjusted": None, "p_worse": None, "verdict": "pending"}
        if len(kept) >= MIN_INDEPENDENT:
            v["p_better"], _ = paired_win_p(s, b, lag=1)
            v["p_worse"], _ = paired_win_p(b, s, lag=1)
            p_better[(model, hm)] = v["p_better"]
        out[(model, hm)] = v
    for k, adj in holm(p_better).items():
        v = out[k]
        v["p_better_adjusted"] = adj
        v["verdict"] = ("better" if adj < LIVE_ALPHA else "worse" if v["p_worse"] < LIVE_ALPHA else "unclear")
    return out
