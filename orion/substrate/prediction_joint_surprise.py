"""Family A joint surprise -> ``node:prediction.body_joint`` / ``node:prediction.room_joint``. Pure, stdlib only.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md ("A. Neural forecaster",
"Retire order" step 6). Issue #3575; groups decided in #3580 (Juniper, 2026-11-17). Successor of
``node:substrate.biometrics`` (mean change of pressure hints / 0.30: not a forecast, retired here).

orion-prediction forecasts every raw body and room reading at once, 5 and 15 min ahead
(``services/orion-prediction/app/family_a_live.py``), and scores each bucket per joint group: the group's
readings' misses whitened by their residual covariance (one physical event that moves coupled readings is
one miss, not one per reading), an adaptive conformal p, and ``excess = -ln p - calm_point``. It publishes
one ``prediction.joint_surprise.v1`` per scored group row:

- ``body`` -> ``node:prediction.body_joint``: machines, cabinet, closet, AC, office temp. A body signal.
- ``room`` -> ``node:prediction.room_joint``: office eCO2, PM2.5, lux, humidity (sound joins after #3581). A WORLD signal
  (``orion/schemas/attention_candidate.py`` ``_WORLD_NODE_IDS``).

Aggregation (the node's value), same shape as ``node:prediction.room`` / ``.body``
(``orion/substrate/prediction_surprise.py``): keep the outcomes whose target bucket is in the last
``WINDOW`` (30 min), from independent origins only (one per horizon-long block on the clock), one per
target bucket (the shortest horizon wins). ``z = sum(excess) / sqrt(sum(calm_var))``, stored as
``clip(z / 3, 0, 1)``. In practice nearly every counted outcome is a 5-min one: a 15-min outcome counts
only when its bucket has no 5-min outcome (same rule as ``node:prediction.body``). ``calm_var`` is the variance of ``-ln p`` for a conformal p uniform on
``{1..n_cal+1} / (n_cal+1)`` (``orion.prediction.surprisal.calm_excess_variance_discrete``): 1.0 in the continuous limit, set by theory, not tuned.

- calm: z ~ N(0, 1) when calibrated -> exact 0 about half the time. Rest point 0.
- a coupled load spike (one big joint miss, p ~0.01, excess ~ +3.6) among 5 calm outcomes: z ~ 1.5 ->
  node ~ 0.5 for the 30 min it stays in the window.
- no events (prediction down, kill switch, refused buckets, no adaptive p yet): no writes; the node ages to
  stale (1800 s) and every reader treats it as absent, never calm. A target already outside the window is
  dropped, so a backlog never writes the past as now.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Mapping

from orion.prediction.surprisal import (  # noqa: F401  (Z_FULL_SCALE: re-exported for readers)
    Z_FULL_SCALE,
    calm_excess_variance_discrete,
    fold,
    independent_origin,
    shortest_horizon_per_target,
)

BODY_JOINT_NODE_ID = "node:prediction.body_joint"
ROOM_JOINT_NODE_ID = "node:prediction.room_joint"
GROUP_NODE_IDS: dict[str, str] = {"body": BODY_JOINT_NODE_ID, "room": ROOM_JOINT_NODE_ID}
PREDICTION_JOINT_SURPRISE_NODE_IDS: frozenset[str] = frozenset(GROUP_NODE_IDS.values())
WINDOW = timedelta(minutes=30)
REDUCER_KEY = "prediction_joint_surprise"

# Var[-ln p] for a calibrated conformal p from a pool of n_cal rows (orion.prediction.surprisal), cached:
# a live pool size repeats for hours.
_calm_var = lru_cache(maxsize=64)(calm_excess_variance_discrete)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class JointSample:
    joint_id: str
    horizon_min: int
    origin_ts: datetime
    target_ts: datetime
    excess: float
    calm_var: float
    top_reading: str | None


@dataclass(frozen=True)
class JointSurpriseReading:
    node_id: str
    mean_excess: float      # per outcome, signed
    z: float                # signed
    value: float            # clip(z / 3, 0, 1): what the node stores
    n: int                  # outcomes in the window (one per target bucket)
    newest_target_ts: datetime
    joint_ids: tuple[str, ...]
    top_reading: str | None  # the reading carrying most of the newest outcome's miss (a hint, not a cause)


@dataclass
class JointSurpriseState:
    """In-memory window per node. Lost on restart by design (the node then ages to stale). Thread-safe: the
    worker folds in a worker thread while /health reads on the event loop."""

    samples: dict[str, dict[tuple[int, datetime], JointSample]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def bump(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def counts_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)

    def _prune(self, node_id: str, now: datetime) -> None:
        cutoff = _aware(now) - WINDOW
        self.samples[node_id] = {k: s for k, s in (self.samples.get(node_id) or {}).items()
                                 if s.target_ts >= cutoff}

    def add(self, event: Mapping[str, Any] | Any, *, now: datetime) -> tuple[str | None, str]:
        """Fold one validated ``PredictionJointSurpriseV1`` (model or mapping). Returns (node_id or None,
        reason); only "accepted" changes a node. The same (horizon, origin) twice replaces itself."""
        get = event.get if isinstance(event, Mapping) else (lambda k: getattr(event, k, None))
        node_id = GROUP_NODE_IDS.get(str(get("group") or ""))
        if node_id is None:
            return None, "unknown_group"
        hm = int(get("horizon_min"))
        origin, target = _aware(get("origin_ts")), _aware(get("target_ts"))
        if not independent_origin(origin, hm):
            return node_id, "overlapping_origin"
        if target < _aware(now) - WINDOW:
            return node_id, "older_than_window"
        excess = float(get("excess"))
        if not math.isfinite(excess):
            return node_id, "non_finite_excess"
        try:
            var = _calm_var(int(get("n_cal")))
        except (TypeError, ValueError):
            return node_id, "no_calibration_size"
        shares = get("top_shares") or {}
        top = max(shares, key=lambda k: shares[k]) if shares else None
        with self._lock:
            self._prune(node_id, now)
            self.samples.setdefault(node_id, {})[(hm, origin)] = JointSample(
                joint_id=str(get("joint_id")), horizon_min=hm, origin_ts=origin, target_ts=target,
                excess=excess, calm_var=var, top_reading=top)
        return node_id, "accepted"

    def reading(self, node_id: str, *, now: datetime) -> JointSurpriseReading | None:
        """The node's window as a z-score of summed excess, or None when the window is empty."""
        with self._lock:
            self._prune(node_id, now)
            held = list((self.samples.get(node_id) or {}).values())
        # One outcome per target bucket, the shortest horizon wins (surprisal.shortest_horizon_per_target).
        rows = shortest_horizon_per_target(held)
        f = fold([(s.excess, s.calm_var) for s in rows])
        if f is None:
            return None
        return JointSurpriseReading(
            node_id=node_id, mean_excess=f.mean_excess, z=f.z, value=f.value,
            n=f.n, newest_target_ts=rows[-1].target_ts, joint_ids=tuple(s.joint_id for s in rows),
            top_reading=rows[-1].top_reading)
