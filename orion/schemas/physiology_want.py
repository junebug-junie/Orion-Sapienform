"""Wants and bid decisions: Organ 3 Appetite and Organ 4 Choice, SHADOW (body economy Patch C1).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 3: Appetite",
"Organ 3b", "Organ 4: Choice" and the "Proposed schema / API / bus changes" table.

* ``WantV1``: "Orion would like to do this". A producer sends one to ``POST /physiology/bid`` on
  orion-durable-runs (``orion.physiology.client.bid``). The producer fills what it knows (family,
  verb, object, its own cost estimate if it has one); the market fills the rest from the family's
  act history (intensity, value_rank, graded, the cost estimate).
* ``WantDecisionV1``: what Choice decided, with the prices it saw, the net, the binding
  constraint and what it displaced. One per bid. Bus ``orion:physiology:want:decision``,
  persisted by orion-sql-writer to ``want_ledger``.

SHADOW: every decision carries ``shadow=True`` (``Literal[True]``), but two callers OBEY theirs:
Orion Prime (services/orion-mind/app/prime_loop.py) and, since Patch D (2026-10-30), curiosity
investigate durable runs (producer ``orion-durable-runs.admission.binding``,
services/orion-durable-runs/app/run_bids.py). Patch D cuts over one reader per PR. A caller that gets no decision (timeout, error, stale prices) keeps its
current gate. It never treats "no answer" as "funded".

Tier 0 (Juniper's turns, perception) NEVER bids: ``WantV1.tier`` is 1 or 2 only.

Cost unit. Prices on ``PhysiologyStateV1`` come from dual ascent on the NORMALIZED excess
``(use - budget) / budget`` (``orion/physiology/prices.py``), so a price is want-value per unit
of budget. ``WantDecisionV1.cost_frac`` is therefore each constraint's cost as a FRACTION of that
constraint's budget (a 1,800 slot-s hold on a 1-slot lane with a 3,600 s window is 0.5).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.schemas.grammar import ActCostV2

WANT_KIND = "physiology.want.v1"
WANT_DECISION_KIND = "physiology.want.decision.v1"
WANT_DECISION_CHANNEL = "orion:physiology:want:decision"

# Why a decision came out the way it did. Each is a branch in orion/physiology/choice.py.
DECISION_REASONS = (
    "net_positive",           # funded: family_weight x value_rank beat the charge
    "exploration_floor",      # funded: price said defer, but the family is under 5% of its tier's funded acts
    "price",                  # deferred: the charge (sum price x cost) beat the value
    "sated",                  # deferred: nothing charged, but value 0 (the family just acted: hunger ~0)
    "capacity_reserved",      # deferred: forecast use + reservations + this cost exceed a capacity line
    "no_acceptable_lane",     # deferred: none of the want's lanes is known to the body
    "physiology_unavailable", # deferred: no fresh body state (absent, malformed, older than the stale limit)
    "unknown_family",         # deferred: the family is not in config/physiology/families.yaml
    "family_never_bids",      # deferred: the family is Tier 0 (Juniper's turns, perception) and never bids
)


class WantWindowV1(BaseModel):
    """When the want is useful. ``latest`` None: deferrable without an end (the family's own
    ``deferrable_window_s`` then applies). ``earliest`` None: now."""

    model_config = ConfigDict(extra="forbid")

    earliest: Optional[AwareDatetime] = None
    latest: Optional[AwareDatetime] = None


class WantV1(BaseModel):
    """One want, as a producer sends it. Fields marked "market-filled" are ignored on input when
    None and filled from the family's history (``orion/physiology/appetite.py``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physiology.want.v1"] = WANT_KIND
    want_id: str = Field(min_length=1, max_length=200)
    # The want family: one per act-ledger actor (``grammar_acts.actor``), so its history joins.
    family: str = Field(min_length=1, max_length=120)
    producer: str = Field(min_length=1, max_length=120)
    verb: str = Field(min_length=1, max_length=120)          # the grammar verb (grammar_acts.verb)
    object_ref: Optional[str] = Field(default=None, max_length=500)
    # Tier 0 never bids (Juniper's turns, perception). The family config is authoritative; a
    # mismatch is decided at the config's tier and named in the decision's warnings.
    tier: Literal[1, 2]
    # When this want arose. Intensity grows from the family's last act close (or from here, if
    # the family never closed an act), never from the bid time.
    opened_at: Optional[AwareDatetime] = None
    # Market-filled when None: 0..1, rises with time unfunded at the family's rate.
    intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # The producer's own expected outcome in the family's own measure (never compared across
    # families). None: the family's mean graded outcome, or 1 while the family is ungraded.
    outcome_value: Optional[float] = Field(default=None, ge=0.0)
    # Market-filled: 0..1 percentile of intensity x outcome_value within the family's 30 days.
    value_rank: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    graded: Optional[bool] = None                              # market-filled
    # The producer's estimate. Fields left None are estimated from the family's history at their
    # 90th percentile (design: "unknown fields priced at their 90th percentile").
    cost: Optional[ActCostV2] = None
    window: WantWindowV1 = Field(default_factory=WantWindowV1)
    # Acceptable lanes (pool roles, without the "lane:" prefix). Empty: the family config's.
    lanes: List[str] = Field(default_factory=list)
    correlation_id: Optional[str] = Field(default=None, max_length=200)


class WantDecisionV1(BaseModel):
    """Choice's answer to one want. SHADOW: ``shadow`` is always True in Patch C; nobody obeys it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physiology.want.decision.v1"] = WANT_DECISION_KIND
    # Deterministic per decision: ``<want_id>@<decided_at iso>``. want_ledger's primary key.
    decision_id: str
    want_id: str
    family: str
    producer: str
    verb: str
    object_ref: Optional[str] = None
    tier: int = Field(ge=0, le=2)
    decided_at: AwareDatetime
    outcome: Literal["funded", "deferred"]
    shadow: Literal[True] = True
    reason: Literal[
        "net_positive", "exploration_floor", "price", "sated", "capacity_reserved", "no_acceptable_lane",
        "physiology_unavailable", "unknown_family", "family_never_bids",
    ]
    # The lane Choice placed it on (pool role), or None (no GPU lane, or not placed).
    lane: Optional[str] = None
    # The bid's inputs, as used.
    intensity: Optional[float] = None
    value_rank: Optional[float] = None
    graded: Optional[bool] = None
    # graded | ungraded (outcome_value 1) | no_history (rank = intensity: nothing to rank against)
    value_basis: Optional[str] = None
    family_weight: Optional[float] = None
    # prior (fewer than 10 evidenced acts) | evidence (shrunk toward the evidence rates)
    weight_basis: Optional[str] = None
    # Every constraint's posted price at the tick the decision read (want-value per budget unit).
    prices_seen: Dict[str, float] = Field(default_factory=dict)
    # Cost on each charged constraint as a fraction of that constraint's budget.
    cost_frac: Dict[str, float] = Field(default_factory=dict)
    charge: Optional[float] = None       # sum over constraints of price x cost_frac
    net: Optional[float] = None          # family_weight x value_rank - charge
    # The constraint that decided a deferral (largest price x cost, or the capacity line crossed),
    # or the bid's own refusal ("physiology_unavailable", ...). None when funded on its merits.
    binding_constraint: Optional[str] = None
    # Funded: recently deferred wants on the same lane with a lower net (who this pushed out).
    displaced: List[str] = Field(default_factory=list)
    # Deferred on capacity: the funded wants whose reservations hold that constraint.
    displaced_by: List[str] = Field(default_factory=list)
    # Funded: what this decision reserves until the act closes or the reservation expires.
    reserved: Dict[str, float] = Field(default_factory=dict)
    reservation_expires_at: Optional[AwareDatetime] = None
    # Deferred: seconds until the decision could plausibly flip (a reservation expires, or the
    # binding price decays far enough at its slack rate). None: nothing to wait for.
    retry_after_sec: Optional[float] = Field(default=None, ge=0.0)
    # The body state the prices came from.
    state_tick_id: Optional[str] = None
    state_age_sec: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    why: str = ""
    want: WantV1


__all__ = [
    "DECISION_REASONS",
    "WANT_DECISION_CHANNEL",
    "WANT_DECISION_KIND",
    "WANT_KIND",
    "WantDecisionV1",
    "WantV1",
    "WantWindowV1",
]
