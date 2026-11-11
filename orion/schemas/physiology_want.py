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

Event wants (attention as the want competition, docs/superpowers/specs/2026-11-04-attention-as-
want-competition-design.md, patch 1). A loop that sees something worth effort can attach the event:
``event_ref`` names the row it is about and ``salience`` says how unusual it is for its source. The
want's own family/verb/object_ref/cost already say which act it would run, so there is no separate
"affordance" field. Both fields are left out of the serialized want when None, so a want without
an event is byte-for-byte what it was before (consumers on ``extra=forbid`` keep parsing it while
they roll forward).

Cost unit. Prices on ``PhysiologyStateV1`` come from dual ascent on the NORMALIZED excess
``(use - budget) / budget`` (``orion/physiology/prices.py``), so a price is want-value per unit
of budget. ``WantDecisionV1.cost_frac`` is therefore each constraint's cost as a FRACTION of that
constraint's budget (a 1,800 slot-s hold on a 1-slot lane with a 3,600 s window is 0.5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, GetJsonSchemaHandler, SerializerFunctionWrapHandler,
                      model_serializer)

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
    "consent",                # deferred: the consent gate said refuse or needs_consent (never admitted)
)

# The act's reversibility class, as config/physiology/consent.yaml and the request name it.
REVERSIBILITY = ("reversible", "snapshot_restorable", "irreversible")
CONSENT_VERDICTS = ("allow", "refuse", "needs_consent")


class ConsentRequestV1(BaseModel):
    """What a want asks the consent gate: "may this act run, and who agreed?" (attention patch 3d).

    ``reversibility`` is the producer's own class for the act. The rule in config/physiology/consent.yaml
    is authoritative: a request that claims another class than its rule is refused. ``payload_digest``
    binds the verdict to the exact payload (a dream compaction delta's sha256), so a consumer can check
    that the verdict it got is for the thing it is about to change. ``attests`` are the named
    preconditions the producer says are true right now (``snapshot_dir_writable``, ``downscale_only``)."""

    model_config = ConfigDict(extra="forbid")

    want_id: str = Field(min_length=1, max_length=200)
    family: str = Field(min_length=1, max_length=120)
    verb: str = Field(min_length=1, max_length=120)
    object_ref: Optional[str] = Field(default=None, max_length=500)
    payload_digest: Optional[str] = Field(default=None, max_length=200)
    reversibility: Literal["reversible", "snapshot_restorable", "irreversible"]
    attests: List[str] = Field(default_factory=list, max_length=32)


class ConsentVerdictV1(BaseModel):
    """The consent gate's answer. ``rule_id``: the consent.yaml rule it used (None: no rule matched).
    ``consent_ref``: who agreed (``standing:<id>``), or None. ``reason``: the branch, readable."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["allow", "refuse", "needs_consent"]
    rule_id: Optional[str] = None
    reason: str = ""
    consent_ref: Optional[str] = None


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
    # The event this want is about, as ``<table>:<id>`` (``temporal_self_event:<id>``,
    # ``grammar_acts:<act_id>``, ...). None: a clock want (hunger only).
    event_ref: Optional[str] = Field(default=None, min_length=1, max_length=500)
    # How unusual the event is for its own source, 0..1 (the source's percentile, computed by the
    # source). Appetite reads intensity = max(hunger, salience): an unusual event can win before the
    # family's clock is due. None: no event, hunger alone.
    salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # The consent gate (patch 3d): present only for an act that changes something hard to put back.
    consent_request: Optional[ConsentRequestV1] = None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        # The wrap serializer below would blank the serialization JSON schema (FastAPI's OpenAPI for
        # /physiology/bid). Its output is the validation shape minus unset event fields: publish that.
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_unset_event(self, handler: SerializerFunctionWrapHandler) -> Any:
        out = handler(self)
        for key in _EVENT_FIELDS:
            if key in out and out[key] is None:
                del out[key]
        return out


# Left out of the serialized want when None (consumers on extra=forbid keep parsing it while they roll forward).
_EVENT_FIELDS = ("event_ref", "salience", "consent_request")


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
        "physiology_unavailable", "unknown_family", "family_never_bids", "consent",
    ]
    # The lane Choice placed it on (pool role), or None (no GPU lane, or not placed).
    lane: Optional[str] = None
    # The bid's inputs, as used.
    intensity: Optional[float] = None
    value_rank: Optional[float] = None
    graded: Optional[bool] = None
    # graded | ungraded (outcome_value 1) | no_history (rank = intensity: nothing to rank against);
    # an event want: event:graded | event:ungraded (ranked against past event wants) | event_cold:<basis>
    # (fewer than 10 past event wants: ranked as above)
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
    # The consent gate's verdict (patch 3d). Only on a decision whose want carried a consent request;
    # left out when None, so every other decision serializes exactly as before.
    consent: Optional[ConsentVerdictV1] = None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_unset_consent(self, handler: SerializerFunctionWrapHandler) -> Any:
        out = handler(self)
        if "consent" in out and out["consent"] is None:
            del out["consent"]
        return out


class FundedEventWantV1(BaseModel):
    """One event want whose latest decision is ``funded`` and whose event has not expired (patch 2).

    Served by orion-durable-runs ``GET /physiology/event-wants`` from ``want_ledger``; read by Orion
    Prime as its concern source (``PRIME_CONCERNS_FROM_EVENT_WANTS``). Not a bus payload."""

    model_config = ConfigDict(extra="forbid")

    want_id: str
    family: str
    event_ref: str
    salience: Optional[float] = None
    value_rank: Optional[float] = None
    net: Optional[float] = None
    decided_at: AwareDatetime
    # When the event stops being worth effort: the want's window end (its own, else opened_at + the
    # family's deferrable window).
    expires_at: Optional[AwareDatetime] = None
    # What the event is, in words, read from its source row (the read's title, the ask's question).
    label: str = ""


class FundedEventWantsV1(BaseModel):
    """``GET /physiology/event-wants``: the market's funded, unexpired event wants, highest net first."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physiology.funded_event_wants.v1"] = "physiology.funded_event_wants.v1"
    as_of: AwareDatetime
    # Event wants decided in the window, and how many of them are funded and unexpired (before ``limit``).
    considered: int = 0
    funded_open: int = 0
    wants: List[FundedEventWantV1] = Field(default_factory=list)


__all__ = [
    "CONSENT_VERDICTS",
    "ConsentRequestV1",
    "ConsentVerdictV1",
    "DECISION_REASONS",
    "REVERSIBILITY",
    "FundedEventWantV1",
    "FundedEventWantsV1",
    "WANT_DECISION_CHANNEL",
    "WANT_DECISION_KIND",
    "WANT_KIND",
    "WantDecisionV1",
    "WantV1",
    "WantWindowV1",
]
