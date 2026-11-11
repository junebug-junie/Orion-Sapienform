"""Forecast surprise -> two body nodes: ``node:prediction.room`` and ``node:prediction.body``. Pure.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md (build item 3).

orion-prediction publishes one ``prediction.surprise.v1`` per scored forecast that passed its
issue-time gate (champion, issued by a real model, adaptive range, real actual). Each carries
``excess`` = surprisal - 0.95: about 0 on average when the world matches the forecast, positive when
the temperature did something the forecast did not expect. substrate-runtime folds them here, one
node per series (spec: "one node per family x domain, never one per reading"):

- ``office.temp_c`` -> ``node:prediction.room``
- ``athena.cabinet_temp_c`` -> ``node:prediction.body``

Aggregation (the node's value). Per node, keep the samples whose target bucket is within the last
``WINDOW`` (30 min) of wall-clock time, from INDEPENDENT origins only: a forecast is issued every
5 minutes, so 15-min forecasts from consecutive origins overlap; keep one origin per horizon-long
block anchored to the clock (``origin_ts`` on a multiple of the horizon), the same thinning the
calm check uses (``services/orion-prediction/app/surprise.py summarize``). Up to ~6 five-minute and
2 fifteen-minute outcomes (live replay 2026-11-11: median 6 for the office, 2 for the cabinet while
its 5-min range is still warming up).

The value is the window's summed excess in units of its calm spread, a z-score:
``z = sum(excess) / sqrt(n * CALM_VAR)``, stored as ``clip(z / 3, 0, 1)`` (the node writer's 0..1
pressure channel; 3 sigma = 1.0, the same scale chat/execution use). Why not the plain mean: its
calm spread depends on how many outcomes the window holds (2 vs 6), so a calm cabinet with 2
samples crossed curiosity's 0.55 bar ~22 % of the time against ~9 % for the office. ``CALM_VAR``
is set by theory, not tuned: for a calibrated forecaster p is uniform, so with the floor P
``Var[-ln max(p, P)] = 2 - 2P(1 + ln(1/P)) - (1 - P)^2`` (0.698 at P = 0.05, sd 0.835). Live check:
the calm-check bands of 2026-11-11 imply sd 0.83-0.90 per horizon.

- calm: z ~ N(0, 1), so the node reads exact 0 about half the time and crosses 0.55 (z > 1.65)
  about 5 % of the time, whatever n is. Rest point 0.
- one bad miss (excess +2.05) among 6 calm samples: z ~ 1.0, node ~ 0.33 for the 30 minutes it
  stays in the window; a sustained run of misses drives it to 1.
- no events (prediction service down, kill switch, no adaptive range) writes nothing: the node's
  ``observed_at`` ages and every reader treats it as stale/absent, never as calm. Samples older
  than the window never revive the node: an event whose target is already outside the window is
  dropped, so a backlog scored after an outage does not write the past as now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

ROOM_NODE_ID = "node:prediction.room"
BODY_NODE_ID = "node:prediction.body"
SERIES_NODE_IDS: dict[str, str] = {
    "office.temp_c": ROOM_NODE_ID,
    "athena.cabinet_temp_c": BODY_NODE_ID,
}
PREDICTION_SURPRISE_NODE_IDS: frozenset[str] = frozenset(SERIES_NODE_IDS.values())
WINDOW = timedelta(minutes=30)
REDUCER_KEY = "prediction_surprise"
Z_FULL_SCALE = 3.0


def calm_excess_variance(calm_surprisal: float) -> float:
    """Variance of one outcome's excess under a calibrated forecaster, from the floor the producer
    used (``P = 1 - calm_surprisal``, since E[-ln max(p, P)] = 1 - P). 0.698 at P = 0.05."""
    p_floor = 1.0 - float(calm_surprisal)
    if not 0.0 < p_floor < 1.0:
        raise ValueError(f"calm_surprisal {calm_surprisal!r} implies no valid p floor")
    return 2.0 - 2.0 * p_floor * (1.0 + math.log(1.0 / p_floor)) - (1.0 - p_floor) ** 2


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def independent_origin(origin_ts: datetime, horizon_min: int) -> bool:
    """True when this origin starts a horizon-long block on the clock (one independent forecast per
    block). Every 5-min origin passes at h=5; one in three at h=15."""
    block = max(1, int(horizon_min)) * 60
    return int(_aware(origin_ts).timestamp()) % block == 0


@dataclass(frozen=True)
class SurpriseSample:
    forecast_id: str
    horizon_min: int
    origin_ts: datetime
    target_ts: datetime
    excess: float
    calm_var: float


@dataclass(frozen=True)
class SurpriseReading:
    """One node's current reading. ``mean_excess`` and ``z`` are signed (negative on a calm,
    slightly over-wide forecaster); ``value`` = clip(z / 3, 0, 1) is what the node stores."""

    node_id: str
    mean_excess: float
    z: float
    value: float
    n: int
    newest_target_ts: datetime
    forecast_ids: tuple[str, ...]


@dataclass
class PredictionSurpriseState:
    """In-memory per-node window. Lost on restart by design: the node then ages to stale until the
    next events arrive (absent, never calm)."""

    samples: dict[str, dict[tuple[int, datetime], SurpriseSample]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def _prune(self, node_id: str, now: datetime) -> None:
        cutoff = _aware(now) - WINDOW
        kept = {k: s for k, s in (self.samples.get(node_id) or {}).items() if s.target_ts >= cutoff}
        self.samples[node_id] = kept

    def add(self, event: Mapping[str, Any] | Any, *, now: datetime) -> tuple[str | None, str]:
        """Fold one validated ``PredictionSurpriseV1`` (model or mapping). Returns (node_id or None,
        reason). Only "accepted" changes a node."""
        get = event.get if isinstance(event, Mapping) else (lambda k: getattr(event, k, None))
        node_id = SERIES_NODE_IDS.get(str(get("series_id") or ""))
        if node_id is None:
            return None, "unknown_series"
        hm = int(get("horizon_min"))
        origin = _aware(get("origin_ts"))
        target = _aware(get("target_ts"))
        if not independent_origin(origin, hm):
            return node_id, "overlapping_origin"
        if target < _aware(now) - WINDOW:
            return node_id, "older_than_window"
        excess = float(get("excess"))
        calm_var = calm_excess_variance(float(get("calm_surprisal")))
        if not math.isfinite(excess):
            return node_id, "non_finite_excess"
        self._prune(node_id, now)
        self.samples.setdefault(node_id, {})[(hm, origin)] = SurpriseSample(
            forecast_id=str(get("forecast_id")), horizon_min=hm, origin_ts=origin, target_ts=target,
            excess=excess, calm_var=calm_var,
        )
        return node_id, "accepted"

    def reading(self, node_id: str, *, now: datetime) -> SurpriseReading | None:
        """The node's window as a z-score of summed excess, or None when the window is empty."""
        self._prune(node_id, now)
        rows = sorted((self.samples.get(node_id) or {}).values(), key=lambda s: s.target_ts)
        if not rows:
            return None
        total = sum(s.excess for s in rows)
        z = total / math.sqrt(sum(s.calm_var for s in rows))
        return SurpriseReading(
            node_id=node_id,
            mean_excess=total / len(rows),
            z=z,
            value=max(0.0, min(1.0, z / Z_FULL_SCALE)),
            n=len(rows),
            newest_target_ts=rows[-1].target_ts,
            forecast_ids=tuple(s.forecast_id for s in rows),
        )
