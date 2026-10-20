"""Grade threshold probabilities against what happened.

Brier score = mean (p - outcome)^2, 0 is perfect. The reference is the base rate: always forecasting
the observed share of windows that crossed. Skill = 1 - Brier / Brier_base_rate: above 0 means the
forecast knows *when* a crossing is coming, not just how often. Calibration bins compare the mean
probability in each bin with how often it actually happened.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

BINS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0000001)
MIN_GRADED = 50


def grade(probs: Sequence[float], hits: Sequence[bool]) -> dict:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(hits, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0, "brier": None, "base_rate": None, "brier_base_rate": None, "skill": None, "bins": []}
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((base - y) ** 2))
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            bins.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()), "mean_p": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    return {"n": n, "brier": brier, "base_rate": base, "brier_base_rate": brier_ref,
            "skill": (1 - brier / brier_ref) if brier_ref > 0 else None, "bins": bins,
            "enough": n >= MIN_GRADED}
