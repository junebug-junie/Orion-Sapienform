"""Grade threshold probabilities against what happened.

Brier score = mean (p - outcome)^2, 0 is perfect. The reference is the base rate: always forecasting
the observed share of windows that crossed. Skill = 1 - Brier / Brier_base_rate: above 0 means the
forecast knows *when* a crossing is coming, not just how often. Calibration bins compare the mean
probability in each bin with how often it actually happened.

``skill`` is None when the base rate is 0 or 1 (no crossing, or always crossing, in the window): the
reference then has zero error and there is nothing to beat. Read the Brier score and the bins then.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence

import numpy as np

BINS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0000001)
MIN_GRADED = 50


def grade(probs: Sequence[float], hits: Sequence[bool]) -> dict:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(hits, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0, "brier": None, "base_rate": None, "brier_base_rate": None, "skill": None, "bins": [],
                "enough": False}
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


def _avg_ranks(x: np.ndarray) -> tuple[np.ndarray, float]:
    """1-based ranks, ties averaged, and the tie term sum(t^3 - t) for the variance."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    for s0, c in zip(start, counts):
        ranks[order[s0:s0 + c]] = s0 + (c + 1) / 2
    return ranks, float(np.sum(counts.astype(float) ** 3 - counts))


def sorting_score(probs: Sequence[float], hits: Sequence[bool],
                  groups: Sequence[Hashable] | None = None) -> dict:
    """How well the probabilities put the turns where it happened above those where it did not.

    AUC: the share of (happened, did-not-happen) pairs where the happened one got the higher
    probability, ties counting half. 0.5 is a coin flip, 1.0 is perfect sorting. Unlike the Brier
    skill, it ignores the level of the probabilities, so a shift in how often the event happens
    (a moving base rate) cannot move it. With ``groups``, only pairs inside the same group count,
    so knowing which group a turn is in earns nothing.

    p_value: one-sided Mann-Whitney test of "no better than a coin flip", normal approximation
    with the tie correction, pooled over groups. None when there are no pairs.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(hits, dtype=bool)
    g = list(groups) if groups is not None else [None] * len(p)
    u = pairs = var = 0.0
    for key in dict.fromkeys(g):
        m = np.array([gi == key for gi in g], dtype=bool)
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        n = n1 + n0
        ranks, ties = _avg_ranks(p[m])
        u += float(ranks[y[m]].sum()) - n1 * (n1 + 1) / 2
        pairs += n1 * n0
        var += n1 * n0 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if not pairs:
        return {"auc": None, "pairs": 0, "p_value": None}
    z = (u - pairs / 2) / math.sqrt(var) if var > 0 else 0.0
    return {"auc": u / pairs, "pairs": int(pairs), "p_value": 0.5 * math.erfc(z / math.sqrt(2))}
