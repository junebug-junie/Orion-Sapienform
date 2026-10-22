"""Live verdicts: does a model beat the baseline on real readings, not just in the backtest?

Input: scored shadow forecasts paired with the baseline forecast issued at the same origin for the
same horizon. Forecasts issued every grain overlap for h/grain steps, so per (model, horizon) only
origins at least one horizon apart are kept (time-based thinning). With at least MIN_INDEPENDENT such
pairs, the overlap-aware paired test (backtest.paired_win_p) runs in both directions:

    better   -- the model's error is lower,  p < LIVE_ALPHA
    worse    -- the baseline's error is lower, p < LIVE_ALPHA
    unclear  -- enough data, neither significant
    pending  -- fewer than MIN_INDEPENDENT independent pairs
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np

from app.backtest import paired_win_p

MIN_INDEPENDENT = 20
LIVE_ALPHA = 0.05


def verdicts(pairs) -> dict[tuple[str, int], dict]:
    """``pairs``: rows {model, horizon_min, origin_ts, shadow_err, base_err}, any order."""
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for r in pairs:
        groups[(r["model"], int(r["horizon_min"]))].append(r)
    out = {}
    for (model, hm), rows in groups.items():
        rows.sort(key=lambda r: r["origin_ts"])
        kept, last = [], None
        for r in rows:
            if last is None or r["origin_ts"] - last >= timedelta(minutes=hm):
                kept.append(r)
                last = r["origin_ts"]
        s = np.array([r["shadow_err"] for r in kept], dtype=float)
        b = np.array([r["base_err"] for r in kept], dtype=float)
        rel = float(s.mean() / b.mean()) if len(kept) and b.mean() > 0 else None
        v = {"n_scored": len(rows), "n_independent": len(kept), "rel_mae": rel,
             "p_better": None, "p_worse": None, "verdict": "pending"}
        if len(kept) >= MIN_INDEPENDENT:
            v["p_better"], _ = paired_win_p(s, b, lag=1)
            v["p_worse"], _ = paired_win_p(b, s, lag=1)
            v["verdict"] = ("better" if v["p_better"] < LIVE_ALPHA else
                            "worse" if v["p_worse"] < LIVE_ALPHA else "unclear")
        out[(model, hm)] = v
    return out
