"""Organ 3 Appetite: wants that grow hungry (body economy Patch C1, SHADOW).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 3: Appetite".
The R2 drive template applied to every want family:

* **Level.** ``intensity = 1 - exp(-t / tau)``, where ``t`` is the time the family has gone
  unfunded: since its last act closed (the act is the discharge, so a close resets it to 0).
  ``tau`` is the family's own rhythm: the median gap between consecutive closes over 30 days
  (at least 5 gaps), else its ``deferrable_window_s``. At ``t = tau`` intensity is 0.63.
* **Seed from the body.** Waiting deferrable self-work beside a lane (Patch B1's
  ``deferrable_queued`` / ``deferrable_waiting_sec`` on ``lane:<role>``) is hunger that is
  already visible. A want whose acceptable lanes have queued work is at least that hungry.
* **Rest.** A close sets ``t`` to 0. With no want and no close, intensity rises toward 1 and
  stays there: a family starved for days reads as a drive (design: "drives are read, not declared").

**value_rank** is the market's one common unit: this want's ``intensity x outcome_value`` as a
percentile (mid-rank) of the SAME family's last 30 days of acts, each act's intensity
reconstructed from its own gap since the previous close. Families are never compared on their
raw outcome units (nats, P(upheld), ...).

**graded**: a family with at least 30 graded closes in 30 days. A graded close has an outcome
from its producer or the horizon resolver; an ``expired`` close (the act was never produced) is
not a grade. An ungraded family's outcome_value is 1, so its rank rests on intensity alone, and
it is labelled ``ungraded`` everywhere.

**Cost estimate**: per ``ActCostV2`` field, the want's own value, else the family's definitional
``fixed_cost``, else the 90th percentile of the family's history ("unknown fields priced at their
90th percentile"), else None (unpriced, and said so). Not 0.

Pure: the caller passes the act history and ``now``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.physiology.families import FamilyConfig

WINDOW = timedelta(days=30)
GRADED_MIN_CLOSES = 30
TAU_MIN_GAPS = 5
TAU_FLOOR_SEC = 60.0
GRADED_SOURCES = frozenset({"producer", "resolver"})
COST_QUANTILE = 0.9
SCALAR_COST_FIELDS = ("seconds", "gpu_seconds", "tokens", "watts", "juniper_msgs", "claude_limit_events")


@dataclass(frozen=True)
class ActRecord:
    """One ``grammar_acts`` row, as much as Appetite, weights and Choice read of it."""
    act_id: str
    family: str                                   # grammar_acts.actor
    opened_at: Optional[datetime]
    closed_at: Optional[datetime]
    outcome_value: Optional[float] = None
    outcome_source: Optional[str] = None         # producer | resolver | expired | None
    signal_id: Optional[str] = None
    cost: Mapping[str, Any] = field(default_factory=dict)   # cost_json (ActCostV2 shape) or {}

    @property
    def graded(self) -> bool:
        return (self.closed_at is not None and self.outcome_value is not None
                and self.outcome_source in GRADED_SOURCES and math.isfinite(self.outcome_value))


def intensity(unfunded_sec: Optional[float], tau_sec: float) -> float:
    """0 at a fresh close, 1 - 1/e at one tau, toward 1 after. Unknown time is 0, never 1."""
    if unfunded_sec is None or not math.isfinite(unfunded_sec) or unfunded_sec <= 0:
        return 0.0
    return 1.0 - math.exp(-unfunded_sec / max(tau_sec, TAU_FLOOR_SEC))


def _quantile(vals: Sequence[float], q: float) -> float:
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (pos - lo)) + v[hi] * (pos - lo)


def mid_rank(x: float, history: Sequence[float]) -> float:
    """Percentile of x within history, mid-rank (ties count half). Empty history: undefined."""
    if not history:
        raise ValueError("mid_rank needs a history")
    below = sum(1 for h in history if h < x)
    equal = sum(1 for h in history if h == x)
    return (below + 0.5 * equal) / len(history)


@dataclass(frozen=True)
class FamilyAppetite:
    family: str
    tau_sec: float
    tau_basis: str                     # rhythm:<n gaps> | deferrable_window
    last_close_at: Optional[datetime]
    n_closes: int
    n_graded: int
    graded: bool
    mean_outcome: Optional[float]      # over graded closes; None if none
    values: tuple[float, ...]          # intensity x outcome per historical act (the rank's reference)
    cost_q: Mapping[str, float] = field(default_factory=dict)         # scalar field -> p90
    lane_slot_s_q: Mapping[str, float] = field(default_factory=dict)  # lane -> p90 slot-s (acts that used it)
    total_slot_s_q: Optional[float] = None                            # p90 of per-act slot-s summed over lanes
    n_costed: int = 0


def build_family_appetite(cfg: FamilyConfig, acts: Iterable[ActRecord], now: datetime) -> FamilyAppetite:
    """The family's appetite state from its last 30 days of acts (acts of other families ignored)."""
    since = now - WINDOW
    mine = [a for a in acts if a.family == cfg.family]
    closes = sorted(a.closed_at for a in mine if a.closed_at is not None and a.closed_at <= now)
    last_close = closes[-1] if closes else None
    in_win = [a for a in mine if (a.opened_at or a.closed_at) is not None and (a.opened_at or a.closed_at) >= since
              and (a.opened_at or a.closed_at) <= now]
    win_closes = [c for c in closes if c >= since]
    gaps = [(b - a).total_seconds() for a, b in zip(win_closes, win_closes[1:]) if (b - a).total_seconds() > 0]
    if len(gaps) >= TAU_MIN_GAPS:
        tau, basis = max(_quantile(gaps, 0.5), TAU_FLOOR_SEC), f"rhythm:{len(gaps)}"
    else:
        tau, basis = max(cfg.deferrable_window_s, TAU_FLOOR_SEC), "deferrable_window"
    graded_acts = [a for a in in_win if a.graded]
    graded = len(graded_acts) >= GRADED_MIN_CLOSES
    mean_out = (sum(a.outcome_value for a in graded_acts) / len(graded_acts)) if graded_acts else None
    # Reference values: each act's intensity at its own open (its gap since the previous close).
    values: list[float] = []
    for a in sorted(in_win, key=lambda x: x.opened_at or x.closed_at):
        t0 = a.opened_at or a.closed_at
        prev = [c for c in closes if c < t0]
        if not prev:
            continue      # the family's first act: its hunger is unknown, not 0 or 1
        i = intensity((t0 - prev[-1]).total_seconds(), tau)
        if graded:
            if not a.graded:
                continue  # a graded family ranks graded acts only
            values.append(i * float(a.outcome_value))
        else:
            values.append(i)
    # Cost history (ActCostV2 fields; None means not measured and is skipped, never 0).
    scal: dict[str, list[float]] = {}
    lanes: dict[str, list[float]] = {}
    totals: list[float] = []
    n_costed = 0
    for a in in_win:
        c = a.cost or {}
        if not c:
            continue
        n_costed += 1
        for k in SCALAR_COST_FIELDS:
            v = c.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
                scal.setdefault(k, []).append(float(v))
        by_lane = c.get("gpu_slot_s_by_lane")
        if isinstance(by_lane, dict) and by_lane:
            tot = 0.0
            for lane, v in by_lane.items():
                if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
                    lanes.setdefault(str(lane), []).append(float(v))
                    tot += float(v)
            totals.append(tot)
    return FamilyAppetite(
        family=cfg.family, tau_sec=tau, tau_basis=basis, last_close_at=last_close, n_closes=len(win_closes),
        n_graded=len(graded_acts), graded=graded, mean_outcome=mean_out, values=tuple(values),
        cost_q={k: _quantile(v, COST_QUANTILE) for k, v in scal.items()},
        lane_slot_s_q={k: _quantile(v, COST_QUANTILE) for k, v in lanes.items()},
        total_slot_s_q=_quantile(totals, COST_QUANTILE) if totals else None, n_costed=n_costed)


@dataclass(frozen=True)
class Hunger:
    unfunded_sec: Optional[float]
    intensity: float
    basis: str           # since_last_close | since_opened | lane_seed:<lane> | unknown


def lane_hunger_seed(lane_details: Mapping[str, Mapping[str, Any]], lanes: Sequence[str]) -> tuple[Optional[float], Optional[str]]:
    """The longest wait of queued deferrable self-work beside any of ``lanes`` (Patch B1's
    ``deferrable_queued`` / ``deferrable_waiting_sec``), or (None, None) when none is queued or unread."""
    best: tuple[Optional[float], Optional[str]] = (None, None)
    for lane in lanes:
        d = lane_details.get(lane) or {}
        q, w = d.get("deferrable_queued"), d.get("deferrable_waiting_sec")
        if isinstance(q, int) and q > 0 and isinstance(w, (int, float)) and math.isfinite(w):
            if best[0] is None or w > best[0]:
                best = (float(w), lane)
    return best


def hunger(app: FamilyAppetite, now: datetime, *, opened_at: Optional[datetime] = None,
           seed_sec: Optional[float] = None, seed_lane: Optional[str] = None) -> Hunger:
    """Time unfunded and the intensity it gives. From the last close (it resets the drive); a
    family that never closed an act counts from the want's own ``opened_at``; the lane seed is a
    floor. Nothing known: intensity 0 (unknown is not "starving")."""
    t, basis = None, "unknown"
    if app.last_close_at is not None:
        t, basis = max(0.0, (now - app.last_close_at).total_seconds()), "since_last_close"
    elif opened_at is not None:
        t, basis = max(0.0, (now - opened_at).total_seconds()), "since_opened"
    if seed_sec is not None and (t is None or seed_sec > t):
        t, basis = seed_sec, f"lane_seed:{seed_lane}"
    return Hunger(unfunded_sec=t, intensity=intensity(t, app.tau_sec), basis=basis)


def rank_value(app: FamilyAppetite, want_intensity: float, outcome_value: Optional[float]
               ) -> tuple[float, str, float]:
    """(value_rank, value_basis, outcome used). Graded: rank of intensity x expected outcome (the
    producer's, else the family mean). Ungraded: outcome 1. No history: rank = intensity."""
    if app.graded:
        out = outcome_value if outcome_value is not None else (app.mean_outcome if app.mean_outcome is not None else 1.0)
        basis = "graded"
    else:
        out, basis = 1.0, "ungraded"
    x = want_intensity * out
    if not app.values:
        return min(1.0, max(0.0, want_intensity)), "no_history", out
    return mid_rank(x, app.values), basis, out


@dataclass(frozen=True)
class CostEstimate:
    scalars: Mapping[str, float]            # field -> amount (measured / fixed / p90)
    lane_slot_s: Mapping[str, float]        # fixed placement: lane -> slot-s
    one_of_slot_s: Optional[float]          # one_of placement: slot-s wherever it lands
    basis: Mapping[str, str]                # field -> want | fixed | p90 | unpriced
    unpriced: tuple[str, ...] = ()


def estimate_cost(want_cost: Optional[Mapping[str, Any]], cfg: FamilyConfig, app: FamilyAppetite) -> CostEstimate:
    w = dict(want_cost or {})
    scal: dict[str, float] = {}
    basis: dict[str, str] = {}
    unpriced: list[str] = []
    for k in SCALAR_COST_FIELDS:
        v = w.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
            scal[k], basis[k] = float(v), "want"
        elif k in cfg.fixed_cost:
            scal[k], basis[k] = float(cfg.fixed_cost[k]), "fixed"
        elif k in app.cost_q:
            scal[k], basis[k] = float(app.cost_q[k]), "p90"
        else:
            basis[k] = "unpriced"
            unpriced.append(k)
    lanes_w = w.get("gpu_slot_s_by_lane")
    lane_slot: dict[str, float] = {}
    one_of: Optional[float] = None
    if isinstance(lanes_w, dict) and lanes_w:
        vals = {str(k): float(v) for k, v in lanes_w.items() if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0}
        basis["gpu_slot_s_by_lane"] = "want"
        if cfg.placement == "fixed":
            lane_slot = vals
        else:
            one_of = sum(vals.values())
    elif cfg.lanes:
        if cfg.placement == "fixed":
            lane_slot = {ln: app.lane_slot_s_q[ln] for ln in cfg.lanes if ln in app.lane_slot_s_q}
            basis["gpu_slot_s_by_lane"] = "p90" if lane_slot else "unpriced"
        elif app.total_slot_s_q is not None:
            one_of, basis["gpu_slot_s_by_lane"] = app.total_slot_s_q, "p90"
        else:
            basis["gpu_slot_s_by_lane"] = "unpriced"
        if basis["gpu_slot_s_by_lane"] == "unpriced":
            unpriced.append("gpu_slot_s_by_lane")
    return CostEstimate(scalars=scal, lane_slot_s=lane_slot, one_of_slot_s=one_of, basis=basis,
                        unpriced=tuple(unpriced))


__all__ = [
    "ActRecord",
    "CostEstimate",
    "FamilyAppetite",
    "GRADED_MIN_CLOSES",
    "Hunger",
    "build_family_appetite",
    "estimate_cost",
    "hunger",
    "intensity",
    "lane_hunger_seed",
    "mid_rank",
    "rank_value",
]
