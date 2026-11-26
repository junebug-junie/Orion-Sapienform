"""One surprise library: how a settled prediction becomes surprise, and how a window of surprise becomes a
node value. Pure, stdlib only, so orion-prediction, substrate-runtime and the sql-writer can all import it.

Spec: docs/superpowers/specs/2026-11-25-shared-bet-primitives-design.md (Primitive 3, Patch A). Before
this module the same math lived in up to six places (the window fold four times, ``Z_FULL_SCALE`` six
times, the yes/no bet terms three times). Patch A moved it here without changing any number beyond 1e-9:
every node value is bit-identical; reachable's log loss moved from np.log to math.log (<= 1 ulp). The
golden tests (``orion/prediction/tests/test_surprisal_golden.py``) pin that on live stored rows.

Two kinds of stated forecast, one unit (nats), calm point set by math:

- **Yes/no bet** at probability q (``bet_terms``): s = -ln q(y), H = binary entropy (the mean of s if q is
  right), excess = s - H (mean 0 under calibration), calm_var = q (1 - q) logit(q)^2 (its variance).
- **Conformal p** of a range forecast (``conformal_terms``): s = -ln max(p, P), P the floor below which p
  cannot be resolved. For a calibrated forecaster p is uniform, so E[s] = 1 - P and
  Var[s] = ``calm_excess_variance(P)``.

A window of outcomes folds into one z (``fold``): z = sum(excess) / sqrt(sum(calm_var)), and the node
stores ``node_value(z)`` = clip(z / ``Z_FULL_SCALE``, 0, 1): a calm window reads exactly 0 about half the
time, 3 sigma reads 1.0. The exact calm tails of that stored value live in ``orion/substrate/calm_band.py``.

Which outcomes enter a window (the time window, one per target bucket, independent origins) stays with each
node's state, because it differs per source; the shared pieces for it are ``independent_origin`` and
``shortest_horizon_per_target``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence, TypeVar

# Every forecast-surprise node stores clip(z / Z_FULL_SCALE, 0, 1): 3 sigma = 1.0.
Z_FULL_SCALE = 3.0
# orion-prediction's conformal floor: the outermost stored central interval is 95 % (quantiles
# 0.025 / 0.975), so p cannot be resolved below 0.05.
P_FLOOR = 0.05


# --- yes/no bets ---------------------------------------------------------------------------------------


def clip_p(p: float, clip: float) -> float:
    """``p`` clamped to [clip, 1 - clip]."""
    return min(1.0 - clip, max(clip, float(p)))


def bernoulli_entropy(p: float, clip: float = 0.0) -> float:
    """Binary entropy in nats, -p ln p - (1 - p) ln(1 - p). With ``clip`` > 0, p is clamped first; with no
    clip, p at or outside 0 / 1 reads 0 (a sure thing carries no uncertainty)."""
    if clip > 0.0:
        p = clip_p(p, clip)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def log_loss(p: float, happened: bool, clip: float) -> float:
    """-ln p(outcome) for a yes/no forecast: -ln p if it happened, -ln(1 - p) if not, p clamped to
    [clip, 1 - clip] so a stated 0 or 1 costs a finite number."""
    q = clip_p(p, clip)
    return -math.log(q if happened else 1.0 - q)


@dataclass(frozen=True)
class BetTerms:
    """One settled yes/no bet's surprise. ``q`` is the clamped probability the terms were computed at."""

    q: float
    happened: bool
    surprisal: float     # s = -ln q(y)
    entropy: float       # H = E[s] if q is right
    excess: float        # s - H, mean 0 under calibration
    calm_var: float      # q (1 - q) logit(q)^2, Var[excess] under calibration (0 for a 50/50 bet)


def bet_terms(q: float, happened: bool, clip: float) -> BetTerms:
    """The surprise terms of one yes/no bet at probability ``q`` (clamped to [clip, 1 - clip])."""
    q = clip_p(q, clip)
    s = -math.log(q if happened else 1.0 - q)
    h = bernoulli_entropy(q)
    lg = math.log(q / (1.0 - q))
    return BetTerms(q=q, happened=bool(happened), surprisal=s, entropy=h, excess=s - h,
                    calm_var=q * (1.0 - q) * lg * lg)


# --- conformal p ---------------------------------------------------------------------------------------


def calm_excess_variance(p_floor: float = P_FLOOR) -> float:
    """Var[-ln max(p, P)] for p uniform on [0, 1] (a calibrated conformal p floored at P):
    2 - 2P(1 + ln(1/P)) - (1 - P)^2. 0.698 at P = 0.05 (sd 0.835)."""
    p = float(p_floor)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p floor {p_floor!r} is not in (0, 1)")
    return 2.0 - 2.0 * p * (1.0 + math.log(1.0 / p)) - (1.0 - p) ** 2


def calm_excess_variance_discrete(n_cal: int) -> float:
    """Var[-ln p] for p uniform on {1, ..., m} / m, m = n_cal + 1 (a calibrated conformal p from a pool of
    n_cal rows, never floored). -> 1.0 (Var of -ln U) as n_cal grows; 0.98-0.99 at ~1000 rows."""
    m = int(n_cal) + 1
    if m < 2:
        raise ValueError(f"n_cal {n_cal!r}: need at least one calibration row")
    logs = [math.log(k / m) for k in range(1, m + 1)]
    mean = sum(logs) / m
    return sum((x - mean) ** 2 for x in logs) / m


@dataclass(frozen=True)
class ConformalTerms:
    p: float             # max(P, min(1, p))
    surprisal: float     # -ln p
    calm_mean: float     # E[surprisal] under calibration = 1 - P
    excess: float        # surprisal - calm_mean
    calm_var: float      # Var[surprisal] under calibration


def conformal_terms(p: float, floor: float = P_FLOOR) -> ConformalTerms:
    """The surprise terms of one conformal p, floored at ``floor``."""
    q = max(floor, min(1.0, p))
    s = -math.log(q)
    calm = 1.0 - floor
    return ConformalTerms(p=q, surprisal=s, calm_mean=calm, excess=s - calm,
                          calm_var=calm_excess_variance(floor))


# --- which outcomes enter a window ---------------------------------------------------------------------


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def independent_origin(origin_ts: datetime, horizon_min: int) -> bool:
    """True when this origin starts a horizon-long block on the clock (one independent forecast per block).
    Every 5-min origin passes at h=5; one in three at h=15."""
    # Whole minutes since the epoch: tolerant of seconds on an origin (bucket starts are on the minute
    # today, live 2026-11-11), never silently dropping every sample if they drift.
    minute = int(_aware(origin_ts).timestamp()) // 60
    return minute % max(1, int(horizon_min)) == 0


_S = TypeVar("_S")


def shortest_horizon_per_target(samples: Iterable[_S]) -> list[_S]:
    """One outcome per target bucket, oldest target first. A 5- and a 15-min forecast of the same bucket
    share its actual (live correlation of their excess 0.25), so counting both overstates the evidence;
    the shortest horizon (the freshest forecast) wins, the first seen on a tie. Each sample needs
    ``target_ts`` and ``horizon_min``."""
    by_target: dict = {}
    for smp in samples:
        cur = by_target.get(smp.target_ts)
        if cur is None or smp.horizon_min < cur.horizon_min:
            by_target[smp.target_ts] = smp
    return sorted(by_target.values(), key=lambda s: s.target_ts)


# --- the window fold -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    total: float         # sum of excess, signed
    var: float           # sum of calm_var
    n: int               # outcomes folded
    z: float             # total / sqrt(var), signed

    @property
    def mean_excess(self) -> float:
        return self.total / self.n

    @property
    def value(self) -> float:
        return node_value(self.z)


def fold(terms: Sequence[tuple[float, float]], *, min_var: float | None = None) -> Fold | None:
    """Fold a window's (excess, calm_var) pairs into one z. None for an empty window, or, with ``min_var``,
    for a window whose summed calm variance is at or below it (only 50/50 bets: no information)."""
    if not terms:
        return None
    var = sum(v for _, v in terms)
    if min_var is not None and var <= min_var:
        return None
    total = sum(e for e, _ in terms)
    return Fold(total=total, var=var, n=len(terms), z=total / math.sqrt(var))


def node_value(z: float) -> float:
    """What a surprise node stores: clip(z / Z_FULL_SCALE, 0, 1). A calm (z <= 0) window is exactly 0."""
    return max(0.0, min(1.0, z / Z_FULL_SCALE))
