"""Organ 4 Choice: a want bids against posted prices (body economy Patch C1, SHADOW).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 4: Choice" and
Organ 2's "Reservations".

    net       = family_weight x value_rank - sum_k price_k x cost_k
    fund if net > 0, else defer with reason = the binding constraint
    placement = the acceptable lane with the minimum sum price x cost

Units. A posted price is want-value per unit of the constraint's BUDGET (dual ascent runs on the
normalized excess), so ``cost_k`` is the want's cost as a fraction of constraint k's budget.

What is charged in C1, and what is not (said in every decision's warnings, never silently 0):

* ``lane:<role>``: slot-seconds (``ActCostV2.gpu_slot_s_by_lane``, the family's p90) over the lane's
  capacity (slots x window). A swap seat unloaded right now (0 slots: diffusion while gpu2 serves
  agent-gpu2) is priced against one nominal slot and warned ``swap_seat_unloaded``: the pool would
  load it, and the eviction cost is placement's (Patch E), not priced here.
* ``attention:juniper``: messages (``juniper_msgs``) over the daily cap.
* ``heat``, ``energy``, ``claude``: per-act heat, watt-hours and Claude limit events are NOT
  measured yet (``wh_by_host``/``claude_limit_events`` are null on every act), so they are listed
  as ``unpriced:<constraint>`` instead of charged at 0. The spinal heat reflex still applies.

Reservations (Organ 2, replay R6). Each funded bid reserves its cost on the constraints it would
use. A later bid on a CAPACITY line (a lane, Juniper's cap) is admitted only if
``forecast use + reservations + its cost <= the line``: forecast use is the trailing measured use
(persistence, as the price itself). Heat, energy and claude are not capacity lines: they are never
reserved against, so a hot room cannot become a veto here (anti-windup).

Reservation lifetime (Patch C3a). In shadow a funded want's real act may never run (today's gate
vetoes it), and a reservation nobody uses biases every later bid toward deferral. So a reservation
lives only as long as the act it stands for:

* released when the act closes (``release``: durable runs at their terminal, cortex-orch when the
  dispatch returns);
* before the act opens, it expires at the earlier of one act-length (the family's expected seconds,
  120 s .. one lane window) and the END OF THE WANT'S WINDOW: no act by then, no reservation;
* when the producer reports the act opened (``mark_opened``: durable runs at their first pool
  grant), it is re-anchored to one act-length from the open, then released at the close.

Tier 1 reserved share: per lane, the p75 of Tier 1's own hourly use over 14 days, as a share of
the lane, capped at 50%. Tier 1 pays no lane price while its reservations fit inside that share;
Tier 2's capacity line on that lane is the rest.

Exploration floor, a weekly quota (Patch C3a). A Tier 2 family under 5% of Tier 2's FUNDED WANTS
over the rolling 7 days is funded anyway when only the PRICE said defer (never past a capacity
line, never with zero hunger). Funded wants are counted from ``want_ledger`` (``seed_funded``) plus
this process's own funded decisions the ledger has not caught up with, deduplicated by
``decision_id``. Each floor fund raises the family's share, so the floor stops by itself at 5%;
there is no per-window rate limit (C1's once-per-lane-window rule starved families: R7 failed).
No ledger read yet (``funded_seeded`` false): no floor.

Holdback stays the allocator's own; Choice does not randomize.

Pure apart from the ``ReservationBook`` it is handed (in-memory; durable-runs owns one).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from orion.physiology.appetite import (
    ActRecord,
    FamilyAppetite,
    estimate_cost,
    hunger,
    lane_hunger_seed,
    rank_value,
)
from orion.physiology.families import FamiliesConfig, FamilyConfig
from orion.physiology.weights import FamilyWeight
from orion.schemas.physiology import (
    ATTENTION_JUNIPER,
    CLAUDE,
    ENERGY,
    HEAT,
    LANE_PREFIX,
    PhysiologyConstraintReadingV1,
    PhysiologyStateV1,
)
from orion.schemas.physiology_want import WantDecisionV1, WantV1

STALE_AFTER_SEC = 360.0              # design: the price is stale after 3 ticks
UNAVAILABLE_RETRY_SEC = 120.0        # one physiology tick
EXPLORATION_FLOOR = 0.05
TIER1_SHARE_CAP = 0.5
TIER1_SHARE_Q = 0.75
TIER1_LOOKBACK = timedelta(days=14)
RESERVATION_MIN_SEC = 120.0
DISPLACED_LOOKBACK = timedelta(hours=1)
FLOOR_WINDOW = timedelta(days=7)
FLOOR_TIER = 2
MAX_LISTED = 10
CAPACITY_CONSTRAINTS_PREFIX = (LANE_PREFIX, ATTENTION_JUNIPER)
UNMEASURED = {HEAT: "wh_by_host", ENERGY: "wh_by_host", CLAUDE: "claude_limit_events"}


# --- reservations ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reservation:
    want_id: str
    family: str
    tier: int
    constraint: str
    amount: float               # in the constraint's own unit (slot-s, messages)
    expires_at: datetime
    net: float
    ttl_sec: float = RESERVATION_MIN_SEC   # one act-length: re-anchored from the open (mark_opened)
    opened: bool = False


@dataclass(frozen=True)
class FundedWant:
    """One funded decision, as the exploration floor counts it (a want_ledger row or this process's own)."""
    decision_id: str
    at: datetime
    family: str
    tier: int


class ReservationBook:
    """Funded bids' reserved costs, until close (``release``) or expiry. Plus the recent deferrals,
    so a funded bid can name what it displaced. In-memory: a restart releases everything (the
    reserved acts then show up in the measured use instead)."""

    def __init__(self) -> None:
        self._r: list[Reservation] = []
        self._deferred: deque[tuple[datetime, str, Optional[str], float]] = deque(maxlen=500)
        # decision_id -> funded want, the rolling FLOOR_WINDOW: want_ledger rows + this process's own.
        self._funded: dict[str, FundedWant] = {}
        self.funded_seeded = False

    def _prune(self, now: datetime) -> None:
        self._r = [r for r in self._r if r.expires_at > now]
        while self._deferred and now - self._deferred[0][0] > DISPLACED_LOOKBACK:
            self._deferred.popleft()

    def total(self, constraint: str, now: datetime, *, tier: Optional[int] = None) -> float:
        self._prune(now)
        return sum(r.amount for r in self._r if r.constraint == constraint and (tier is None or r.tier == tier))

    def holders(self, constraint: str, now: datetime) -> list[Reservation]:
        self._prune(now)
        return sorted((r for r in self._r if r.constraint == constraint), key=lambda r: r.expires_at)

    def add(self, r: Reservation) -> None:
        self._r.append(r)

    def release(self, want_id: str) -> int:
        """The want's act closed (or will never run): its reservations end now."""
        n = len(self._r)
        self._r = [r for r in self._r if r.want_id != want_id]
        return n - len(self._r)

    def mark_opened(self, want_id: str, at: datetime) -> int:
        """The want's act opened at ``at``: its reservations now run one act-length from the open
        (no longer capped by the want's window). A reservation that already lapsed stays gone."""
        self._prune(at)
        n = 0
        out = []
        for r in self._r:
            if r.want_id == want_id and not r.opened:
                r = Reservation(r.want_id, r.family, r.tier, r.constraint, r.amount,
                                at + timedelta(seconds=r.ttl_sec), r.net, r.ttl_sec, True)
                n += 1
            out.append(r)
        self._r = out
        return n

    # --- the exploration floor's weekly count ----------------------------------------------------
    def seed_funded(self, rows: "Sequence[FundedWant]", now: datetime) -> None:
        """Merge the ledger's funded wants of the rolling week. This process's own decisions the
        ledger has not written yet stay; the same decision is never counted twice (decision_id)."""
        for r in rows:
            self._funded[r.decision_id] = r
        self.funded_seeded = True
        self._prune_funded(now)

    def note_funded(self, decision_id: str, family: str, tier: int, at: datetime) -> None:
        self._funded[decision_id] = FundedWant(decision_id, at, family, tier)

    def _prune_funded(self, now: datetime) -> None:
        since = now - FLOOR_WINDOW
        for k in [k for k, r in self._funded.items() if r.at < since]:
            del self._funded[k]

    def funded_share(self, family: str, tier: int, now: datetime) -> Optional[float]:
        """``family``'s share of its tier's funded wants over the rolling week; None when the tier
        funded nothing (no share to defend: every hungry family is under the floor)."""
        self._prune_funded(now)
        in_tier = [r for r in self._funded.values() if r.tier == tier and r.at <= now]
        if not in_tier:
            return None
        return sum(1 for r in in_tier if r.family == family) / len(in_tier)

    def note_deferred(self, now: datetime, want_id: str, lane: Optional[str], net: float) -> None:
        self._deferred.append((now, want_id, lane, net))

    def displaced_on(self, lane: Optional[str], net: float, now: datetime) -> list[str]:
        self._prune(now)
        return [w for (_, w, ln, n) in reversed(self._deferred) if ln == lane and n < net][:MAX_LISTED]

    def snapshot(self, now: datetime) -> list[dict]:
        self._prune(now)
        return [{"want_id": r.want_id, "family": r.family, "tier": r.tier, "constraint": r.constraint,
                 "amount": round(r.amount, 3), "expires_at": r.expires_at.isoformat(), "opened": r.opened}
                for r in self._r]


# --- market context ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketContext:
    families: FamiliesConfig
    appetite: Mapping[str, FamilyAppetite]
    weights: Mapping[str, FamilyWeight]
    tier1_share: Mapping[str, float] = field(default_factory=dict)   # lane role -> reserved share 0..0.5
    # lane role -> Tier 1's own slot-seconds in the last window (acts closed inside it). Part of the
    # lane's measured level: Tier 2's line subtracts only the UNUSED rest of Tier 1's share.
    tier1_recent_use: Mapping[str, float] = field(default_factory=dict)
    lane_window_sec: float = 3600.0
    stale_after_sec: float = STALE_AFTER_SEC


def tier1_reserved_share(acts: Sequence[ActRecord], families: FamiliesConfig, lane: str, capacity_slot_s: float,
                         now: datetime, *, window_sec: float = 3600.0) -> float:
    """p75 of Tier 1's own hourly slot-seconds on ``lane`` over 14 days, as a share of the lane's
    capacity, capped at 50%. An act's slot-seconds land in the hour it closed. 0 when the capacity
    is unknown or 0 (nothing to reserve a share of)."""
    if not (capacity_slot_s and capacity_slot_s > 0):
        return 0.0
    since = now - TIER1_LOOKBACK
    n_hours = int(TIER1_LOOKBACK.total_seconds() // window_sec)
    hourly = [0.0] * n_hours
    for a in acts:
        cfg = families.get(a.family)
        if cfg is None or cfg.tier != 1 or a.closed_at is None or not (since <= a.closed_at < now):
            continue
        by_lane = (a.cost or {}).get("gpu_slot_s_by_lane")
        if not isinstance(by_lane, dict):
            continue
        v = by_lane.get(lane)
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            i = min(n_hours - 1, int((a.closed_at - since).total_seconds() // window_sec))
            hourly[i] += float(v)
    hourly.sort()
    pos = TIER1_SHARE_Q * (len(hourly) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(hourly) - 1)
    p75 = hourly[lo] * (1 - (pos - lo)) + hourly[hi] * (pos - lo)
    return min(TIER1_SHARE_CAP, p75 / capacity_slot_s)


# --- the bid ----------------------------------------------------------------------------------------

def build_context(families: FamiliesConfig, acts: Sequence[ActRecord], uses: Sequence[Any],
                  state: Optional[PhysiologyStateV1], now: datetime, *, lane_window_sec: float = 3600.0,
                  weights: Optional[Mapping[str, FamilyWeight]] = None) -> MarketContext:
    """Everything a bid reads besides the want, the prices and the book: per-family appetite,
    today's weights and Tier 1's reserved share per lane. (The floor's weekly funded shares live on
    the ReservationBook: they come from want_ledger, not act history.)"""
    from orion.physiology.appetite import build_family_appetite
    from orion.physiology.weights import weights_now

    app = {f: build_family_appetite(c, acts, now) for f, c in families.families.items() if c.bids}
    w = dict(weights) if weights is not None else weights_now(families, acts, uses, now)
    t1: dict[str, float] = {}
    t1_use: dict[str, float] = {}
    for a in acts:
        cfg = families.get(a.family)
        if (cfg is None or cfg.tier != 1 or a.closed_at is None
                or not (now - timedelta(seconds=lane_window_sec) <= a.closed_at <= now)):
            continue
        by_lane = (a.cost or {}).get("gpu_slot_s_by_lane")
        if isinstance(by_lane, dict):
            for ln, v in by_lane.items():
                if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
                    t1_use[str(ln)] = t1_use.get(str(ln), 0.0) + min(float(v), lane_window_sec)
    if state is not None:
        for r in state.constraints:
            if r.constraint.startswith(LANE_PREFIX) and r.budget:
                t1[r.constraint[len(LANE_PREFIX):]] = tier1_reserved_share(
                    acts, families, r.constraint[len(LANE_PREFIX):], r.budget, now, window_sec=lane_window_sec)
    return MarketContext(families=families, appetite=app, weights=w, tier1_share=t1, tier1_recent_use=t1_use,
                         lane_window_sec=lane_window_sec)



def _readings(state: PhysiologyStateV1) -> dict[str, PhysiologyConstraintReadingV1]:
    return {c.constraint: c for c in state.constraints}


def _price(r: Optional[PhysiologyConstraintReadingV1]) -> tuple[Optional[float], Optional[str]]:
    if r is None or r.price is None:
        return None, None
    return float(r.price.value), r.price.bind_state


def _decision(want: WantV1, now: datetime, *, tier: int, outcome: str, reason: str, **kw) -> WantDecisionV1:
    return WantDecisionV1(decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
                          producer=want.producer, verb=want.verb, object_ref=want.object_ref, tier=tier,
                          decided_at=now, outcome=outcome, reason=reason, want=want, **kw)


def unavailable(want: WantV1, now: datetime, *, why: str, state: Optional[PhysiologyStateV1] = None) -> WantDecisionV1:
    """The market itself could not be read (no families config, no act history): deferred,
    ``physiology_unavailable``. Never funded by default."""
    age = (now - state.generated_at).total_seconds() if state is not None else None
    return _decision(want, now, tier=want.tier, outcome="deferred", reason="physiology_unavailable",
                     binding_constraint="physiology_unavailable", retry_after_sec=UNAVAILABLE_RETRY_SEC,
                     state_tick_id=state.tick_id if state is not None else None,
                     state_age_sec=round(age, 1) if age is not None else None,
                     why=f"the market could not be read: {why}; the caller keeps its own gate")


def decide(want: WantV1, state: Optional[PhysiologyStateV1], ctx: MarketContext, book: ReservationBook,
           now: datetime) -> WantDecisionV1:
    """One shadow bid. Never raises on market data: every unknown is a named deferral or warning."""
    cfg: Optional[FamilyConfig] = ctx.families.get(want.family)
    if cfg is None:
        return _decision(want, now, tier=want.tier, outcome="deferred", reason="unknown_family",
                         binding_constraint="unknown_family",
                         why=f"family {want.family!r} is not in config/physiology/families.yaml")
    if not cfg.bids:
        return _decision(want, now, tier=cfg.tier, outcome="deferred", reason="family_never_bids",
                         binding_constraint="family_never_bids",
                         why=f"{want.family} is Tier 0 (Juniper's turns, perception): it never bids and is never deferred")
    warnings: list[str] = []
    if want.tier != cfg.tier:
        warnings.append(f"tier_mismatch:want={want.tier}:config={cfg.tier}")
    tier = cfg.tier
    age = (now - state.generated_at).total_seconds() if state is not None else None
    if state is None or age is None or age > ctx.stale_after_sec or age < -60.0 or not state.constraints:
        what = "absent" if state is None else (f"stale ({age:.0f} s old)" if age is not None and age > 0 else "unusable")
        return _decision(want, now, tier=tier, outcome="deferred", reason="physiology_unavailable",
                         binding_constraint="physiology_unavailable", retry_after_sec=UNAVAILABLE_RETRY_SEC,
                         state_tick_id=state.tick_id if state is not None else None,
                         state_age_sec=round(age, 1) if age is not None else None, warnings=warnings,
                         why=f"no fresh body state ({what}): the caller keeps its own gate; never funded by default")
    readings = _readings(state)
    prices_seen = {k: round(float(r.price.value), 6) for k, r in readings.items() if r.price is not None}
    app = ctx.appetite.get(want.family)
    if app is None:
        return _decision(want, now, tier=tier, outcome="deferred", reason="physiology_unavailable",
                         binding_constraint="physiology_unavailable", retry_after_sec=UNAVAILABLE_RETRY_SEC,
                         prices_seen=prices_seen, state_tick_id=state.tick_id, state_age_sec=round(age, 1),
                         warnings=warnings, why=f"no act history was read for {want.family}")

    # --- Appetite: intensity, value_rank -----------------------------------------------------------
    lanes = list(want.lanes or cfg.lanes)
    lane_details = {k[len(LANE_PREFIX):]: r.detail for k, r in readings.items() if k.startswith(LANE_PREFIX)}
    seed, seed_lane = lane_hunger_seed(lane_details, lanes)
    h = hunger(app, now, opened_at=want.opened_at, seed_sec=seed, seed_lane=seed_lane)
    inten = want.intensity if want.intensity is not None else h.intensity
    rank, vbasis, _out = rank_value(app, inten, want.outcome_value)
    if want.value_rank is not None:
        warnings.append("value_rank_ignored:market_fills_it")
    fw = ctx.weights.get(want.family)
    weight = fw.weight if fw is not None else cfg.prior_weight
    wbasis = fw.basis if fw is not None else "prior"
    value = weight * rank

    # --- Cost, placement, charge -------------------------------------------------------------------
    cost = estimate_cost(want.cost.model_dump() if want.cost is not None else None, cfg, app)
    for k in UNMEASURED:
        if k in readings:
            warnings.append(f"unpriced:{k}:{UNMEASURED[k]}_unmeasured")
    if cost.unpriced and "gpu_slot_s_by_lane" in cost.unpriced:
        warnings.append("unpriced:lanes:no_slot_history")

    unloaded: set[str] = set()

    def in_window(slot_s: float) -> float:
        # A lane's line is slots x ONE window (the trailing hour). An act holds at most one slot, so
        # inside any window it can use at most one window of slot-seconds: a 5,600 s curiosity hold
        # costs a 1-slot lane its whole hour, not 1.6 hours it does not have.
        return min(slot_s, ctx.lane_window_sec)

    def lane_frac(role: str, slot_s: float) -> tuple[Optional[float], float, Optional[str]]:
        """(cost as a share of the lane, the lane's line in slot-s, warning)."""
        r = readings.get(f"{LANE_PREFIX}{role}")
        if r is None:
            return None, 0.0, None
        budget = r.budget
        warn = None
        if budget is None or budget <= 0:
            # 0 live slots now: a swap seat the pool would load. One nominal slot, no measured use
            # held against it (what ran there ran while it was loaded).
            budget, warn = ctx.lane_window_sec, f"swap_seat_unloaded:{role}"
            unloaded.add(role)
        return in_window(slot_s) / budget, budget, warn

    def tier1_unused(k: str, budget: float) -> float:
        """The part of Tier 1's reserved share it is not using now (measured + reserved)."""
        if not k.startswith(LANE_PREFIX):
            return 0.0
        role = k[len(LANE_PREFIX):]
        share = ctx.tier1_share.get(role, 0.0) * budget
        return max(0.0, share - ctx.tier1_recent_use.get(role, 0.0) - book.total(k, now, tier=1))

    def tier_line(k: str, budget: float) -> float:
        # Tier 2 may use the lane minus what Tier 1's share still holds free; the measured level
        # already contains Tier 1's own use, so the share is not subtracted twice.
        return budget - tier1_unused(k, budget) if tier == 2 else budget

    def lane_level(role: str) -> Optional[float]:
        r = readings.get(f"{LANE_PREFIX}{role}")
        return float(r.level) if r is not None and r.level is not None else None

    def lane_unread(role: str) -> bool:
        # A lane with no reading (stale pool events): its use is unknown, so it is never "empty".
        return role not in unloaded and lane_level(role) is None

    # charges: constraint -> (cost_frac, amount in own unit, line in own unit or None)
    base: dict[str, tuple[float, float, Optional[float]]] = {}
    jmsgs = cost.scalars.get("juniper_msgs")
    jr = readings.get(ATTENTION_JUNIPER)
    if jmsgs and jr is not None and jr.budget:
        base[ATTENTION_JUNIPER] = (jmsgs / jr.budget, jmsgs, jr.budget)
    lane_warn: list[str] = []
    for role, slot_s in cost.lane_slot_s.items():                 # fixed placement: every lane charged
        frac, line, warn = lane_frac(role, slot_s)
        if frac is None:
            lane_warn.append(f"lane_not_in_body:{role}")
            continue
        if warn:
            lane_warn.append(warn)
        base[f"{LANE_PREFIX}{role}"] = (frac, in_window(slot_s), line)

    def charge_of(items: Mapping[str, tuple[float, float, Optional[float]]]) -> tuple[float, dict[str, float]]:
        total, parts = 0.0, {}
        for k, (frac, amount, _line) in items.items():
            p = prices_seen.get(k)
            if p is None:
                continue
            if tier == 1 and k.startswith(LANE_PREFIX):
                role = k[len(LANE_PREFIX):]
                share = ctx.tier1_share.get(role, 0.0)
                used = ctx.tier1_recent_use.get(role, 0.0) + book.total(k, now, tier=1)
                if share > 0 and used + amount <= share * (_line or 0.0):
                    parts[k] = 0.0      # inside Tier 1's reserved share: upkeep pays no lane price
                    continue
            parts[k] = p * frac
            total += p * frac
        return total, parts

    def capacity_overflow(items: Mapping[str, tuple[float, float, Optional[float]]]) -> Optional[tuple[str, float]]:
        worst: Optional[tuple[str, float]] = None
        for k, (_frac, amount, line) in items.items():
            if line is None or not k.startswith(CAPACITY_CONSTRAINTS_PREFIX):
                continue
            if k.startswith(LANE_PREFIX):
                role = k[len(LANE_PREFIX):]
                lv = lane_level(role)
                level = 0.0 if role in unloaded else (lv if lv is not None else float("inf"))
            else:
                level = float(jr.level) if jr is not None and jr.level is not None else 0.0
            over = level + book.total(k, now) + amount - tier_line(k, line)
            if over > 1e-9 and (worst is None or over / line > worst[1]) and math.isfinite(over):
                worst = (k, over / line)
        return worst

    chosen: Optional[str] = None
    candidates: list[tuple[float, str, dict]] = []
    one_of_slot = cost.one_of_slot_s
    if lanes and one_of_slot is None and not cost.lane_slot_s:
        # A GPU family with no measured slot time: never "free". Priced at its upper bound, one
        # whole window on whichever lane it lands (an act holds at most one slot), and placed.
        one_of_slot = ctx.lane_window_sec
        warnings.append("lane_cost_unmeasured:priced_at_one_window")
    if (cfg.placement == "one_of" or not cost.lane_slot_s) and lanes and one_of_slot is not None:
        for role in lanes:
            frac, line, warn = lane_frac(role, one_of_slot)
            if frac is None:
                continue
            if lane_unread(role):
                lane_warn.append(f"lane_no_reading:{role}")
                continue
            items = dict(base)
            items[f"{LANE_PREFIX}{role}"] = (frac, in_window(one_of_slot), line)
            ch, _ = charge_of(items)
            candidates.append((ch, role, {"items": items, "warn": warn}))
        if not candidates:
            if any(w.startswith("lane_no_reading:") for w in lane_warn):
                return _decision(want, now, tier=tier, outcome="deferred", reason="physiology_unavailable",
                                 binding_constraint="physiology_unavailable", prices_seen=prices_seen,
                                 state_tick_id=state.tick_id, state_age_sec=round(age, 1),
                                 retry_after_sec=UNAVAILABLE_RETRY_SEC, warnings=warnings + sorted(set(lane_warn)),
                                 why=f"none of {lanes} has a reading: their use is unknown, never empty")
            return _decision(want, now, tier=tier, outcome="deferred", reason="no_acceptable_lane",
                             binding_constraint="no_acceptable_lane", intensity=round(inten, 6), value_rank=round(rank, 6),
                             graded=app.graded, value_basis=vbasis, family_weight=round(weight, 6), weight_basis=wbasis,
                             prices_seen=prices_seen, state_tick_id=state.tick_id, state_age_sec=round(age, 1),
                             warnings=warnings + [f"lanes:{','.join(lanes)}"], retry_after_sec=UNAVAILABLE_RETRY_SEC,
                             why=f"none of {lanes} is a lane the body reads")
        # Placement: the cheapest lane that still has room; else the cheapest lane (deferred on it).
        # A seat with no live slot now (swap seat unloaded, hecate asleep) is a last resort: loading
        # or waking it has a cost this patch does not price (Patch E), so a live seat wins any tie.
        # An unloaded seat satisfies capacity only when it is the want's ONLY lane (its home seat,
        # e.g. painting on diffusion: the pool loads it for this work). Otherwise landing there
        # would evict the seat sharing its card, which nobody pays for yet (Patch E).
        only = len(candidates) == 1
        fits = [c for c in candidates if capacity_overflow(c[2]["items"]) is None
                and (only or c[1] not in unloaded)]
        live = [c for c in candidates if c[1] not in unloaded] or candidates
        pick = min(fits or live, key=lambda c: (c[1] in unloaded, c[0], lanes.index(c[1])))
        chosen = pick[1]
        items = pick[2]["items"]
        if pick[2]["warn"]:
            lane_warn.append(pick[2]["warn"])
    else:
        items = dict(base)
        unread = [k for k in items if k.startswith(LANE_PREFIX) and lane_unread(k[len(LANE_PREFIX):])]
        if unread:
            return _decision(want, now, tier=tier, outcome="deferred", reason="physiology_unavailable",
                             binding_constraint="physiology_unavailable", prices_seen=prices_seen,
                             state_tick_id=state.tick_id, state_age_sec=round(age, 1),
                             retry_after_sec=UNAVAILABLE_RETRY_SEC, warnings=warnings + [f"lane_no_reading:{k}" for k in unread],
                             why=f"{', '.join(unread)} has no reading: its use is unknown, never empty")
        if len(cost.lane_slot_s) == 1:
            chosen = next(iter(cost.lane_slot_s))
    warnings += sorted(set(lane_warn))
    charge, parts = charge_of(items)
    net = value - charge
    cost_frac = {k: round(v[0], 6) for k, v in items.items()}
    common = dict(intensity=round(inten, 6), value_rank=round(rank, 6), graded=app.graded, value_basis=vbasis,
                  family_weight=round(weight, 6), weight_basis=wbasis, prices_seen=prices_seen, cost_frac=cost_frac,
                  charge=round(charge, 6), net=round(net, 6), lane=chosen, state_tick_id=state.tick_id,
                  state_age_sec=round(age, 1))
    hunger_s = f"hunger {h.unfunded_sec or 0:.0f} s ({h.basis}, tau {app.tau_sec:.0f} s {app.tau_basis})"
    head = (f"{want.family}: weight {weight:.2f} ({wbasis}) x rank {rank:.2f} ({vbasis}; intensity {inten:.2f}, "
            f"{hunger_s}) - charge {charge:.3f} = net {net:+.3f}")

    over = capacity_overflow(items)
    if over is None and chosen in unloaded and cfg.placement == "one_of" and len(lanes) > 1:
        # Every live seat is full; the only room is a seat that would have to be loaded (evicting
        # its card-mate). Not funded on that alone until placement prices the swap (Patch E).
        over = (f"{LANE_PREFIX}{chosen}", 0.0)
        warnings.append(f"only_room_on_unloaded_seat:{chosen}")
    if over is not None:
        k, frac_over = over
        holders = book.holders(k, now)
        if holders:
            retry = max(0.0, (holders[0].expires_at - now).total_seconds())
        else:
            r = readings.get(k)
            slots = (r.detail.get("slots") if r is not None else None) or 1
            retry = frac_over * (items[k][2] or 0.0) / max(int(slots), 1) if k.startswith(LANE_PREFIX) else None
        book.note_deferred(now, want.want_id, chosen, net)
        return _decision(want, now, tier=tier, outcome="deferred", reason="capacity_reserved", binding_constraint=k,
                         displaced_by=[x.want_id for x in holders][:MAX_LISTED],
                         retry_after_sec=round(retry, 1) if retry is not None else None,
                         warnings=_window_warn(want, cfg, now, retry, warnings), **common,
                         why=f"{head}; {k} is full: forecast use + {len(holders)} reservation(s) + this cost exceed its line by {frac_over:.0%}")

    # The weekly quota: a Tier 2 family under 5% of Tier 2's funded wants over the rolling 7 days
    # (want_ledger + this process's own), counted only once the ledger was read...
    floor_share = book.funded_share(want.family, tier, now) if book.funded_seeded else None
    # ...and only for a want with some hunger: the floor guarantees a share to families that want,
    # it does not fund a family whose drive just discharged (intensity 0). Each floor fund raises
    # the family's share, so the quota limits itself (no per-window rate limit).
    under_floor = (tier == FLOOR_TIER and book.funded_seeded and inten > 0
                   and (floor_share is None or floor_share < EXPLORATION_FLOOR))
    if net > 0 or under_floor:
        reason = "net_positive" if net > 0 else "exploration_floor"
        if reason == "exploration_floor":
            fs = ("nothing funded in its tier this week" if floor_share is None
                  else f"{floor_share:.1%} of Tier {tier}'s funded wants over 7 days")
            head += f"; under the 5% weekly exploration quota ({fs})"
        secs = cost.scalars.get("seconds")
        ttl = min(ctx.lane_window_sec, max(RESERVATION_MIN_SEC, secs if secs else RESERVATION_MIN_SEC))
        expires = now + timedelta(seconds=ttl)
        # Before the act opens, a reservation lasts no longer than the want's own window.
        latest = _window_end(want, cfg)
        if latest is not None and latest < expires:
            expires = latest
        reserved: dict[str, float] = {}
        if expires > now:
            for k, (_frac, amount, line) in items.items():
                if line is not None and k.startswith(CAPACITY_CONSTRAINTS_PREFIX) and amount > 0:
                    book.add(Reservation(want.want_id, want.family, tier, k, amount, expires, net, ttl))
                    reserved[k] = round(amount, 3)
        else:
            warnings.append("window_passed:no_reservation")
        displaced = book.displaced_on(chosen, net, now)
        decision = _decision(want, now, tier=tier, outcome="funded", reason=reason, reserved=reserved,
                             reservation_expires_at=expires if reserved else None, displaced=displaced,
                             warnings=warnings, **common, why=head)
        book.note_funded(decision.decision_id, want.family, tier, now)
        return decision

    if charge <= 0.0:
        # Nothing to pay and nothing wanted: the family just acted (hunger ~0), so its value is 0.
        book.note_deferred(now, want.want_id, chosen, net)
        return _decision(want, now, tier=tier, outcome="deferred", reason="sated", binding_constraint=None,
                         retry_after_sec=None, warnings=warnings, **common,
                         why=f"{head}; nothing charged, but no hunger to fund (the family's last act just closed)")
    # Deferred on price: the constraint with the largest price x cost binds.
    k = max(parts, key=lambda x: parts[x]) if parts else None
    retry = None
    if k is not None and items[k][0] > 0:
        r = readings[k]
        p = r.price
        rest = charge - parts[k]
        needed = max(0.0, (value - rest) / items[k][0])
        # Only a slack price falls. Its guaranteed decay is the slack floor (ceiling / horizon), so
        # this is an upper bound on the wait. binding / exogenous_bind / stuck prices do not fall.
        rate = p.ceiling / p.horizon_sec if p.horizon_sec else None
        if rate and p.bind_state == "slack":
            retry = max(0.0, (p.value - needed) / rate)
    book.note_deferred(now, want.want_id, chosen, net)
    return _decision(want, now, tier=tier, outcome="deferred", reason="price", binding_constraint=k,
                     retry_after_sec=round(retry, 1) if retry is not None else None,
                     warnings=_window_warn(want, cfg, now, retry, warnings), **common,
                     why=f"{head}; {k or 'no constraint'} binds")


def _window_end(want: WantV1, cfg: FamilyConfig) -> Optional[datetime]:
    """The end of the want's useful window: its own ``window.latest``, else ``opened_at`` + the
    family's deferrable window, else unknown (None). A perishable family (window 0) without an
    explicit ``latest`` is treated as unknown here, not as already over."""
    if want.window.latest is not None:
        return want.window.latest
    if want.opened_at is not None and cfg.deferrable_window_s:
        return want.opened_at + timedelta(seconds=cfg.deferrable_window_s)
    return None


def _window_warn(want: WantV1, cfg: FamilyConfig, now: datetime, retry: Optional[float], warnings: list[str]) -> list[str]:
    latest = _window_end(want, cfg)
    if retry is not None and latest is not None and now + timedelta(seconds=retry) > latest:
        return warnings + ["expires_before_relief"]
    return warnings


__all__ = [
    "EXPLORATION_FLOOR",
    "FLOOR_TIER",
    "FLOOR_WINDOW",
    "FundedWant",
    "MarketContext",
    "Reservation",
    "ReservationBook",
    "STALE_AFTER_SEC",
    "build_context",
    "decide",
    "tier1_reserved_share",
    "unavailable",
]
