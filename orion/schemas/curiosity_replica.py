"""Contracts of the curiosity replica: a small planner in orion-mind that writes the top of the offer order
for curiosity's replica-arm runs, and that can answer Prime (support, challenge or refuse).

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md (sections 2-5 and 9, "Proposed schema
/ API changes"). Patch 2 of its build order: the shapes only. The producer of the request and the consumer
of the result (Hub) and the replica instance that answers (orion-mind) land in patch 3; until then no run
is drawn onto the replica arm (Hub passes no replica propensity to orion/curiosity/value.py draw_offer_arm).

The RPC: Hub publishes ``CuriosityOfferPlanRequestV1`` on ``orion:curiosity:replica:plan:request`` and waits
at most ``PLAN_DEADLINE_SEC`` for ``CuriosityOfferPlanResultV1`` on
``orion:curiosity:replica:plan:result:<correlation_id>`` (the same request/reply shape as the introspect
focus RPC). Everything the replica writes stays on Hub's ``curiosity_offer_plans`` row
(services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql); its reasons (``why``, ``reading``,
stance reasons) never reach Orion's prompt.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

# The act-ledger actor and body-economy family (config/physiology/families.yaml, tier 2, lane chat).
REPLICA_ACTOR = "curiosity.replica"
REPLICA_FAMILY = "curiosity.replica"

PLAN_REQUEST_CHANNEL = "orion:curiosity:replica:plan:request"
PLAN_RESULT_CHANNEL_PREFIX = "orion:curiosity:replica:plan:result:"
PLAN_REQUEST_KIND = "curiosity.replica.plan.request.v1"
PLAN_RESULT_KIND = "curiosity.replica.plan.result.v1"

# Hub waits this long for a plan, then uses the uncertainty order (design section 7: covers Prime's p90
# thought act, 61 s including seat wait). The family's deferrable_window_s is the same number.
PLAN_DEADLINE_SEC = 90

CLAIM_MAX = 120  # a candidate's claim, as shown in the replica's frame
MAX_CANDIDATES = 60  # design section 2: 30 most uncertain + 20 highest expected nats + tested in 7 days + 10 rotated
PICKS_MIN = 3
PICKS_MAX = 5
WHY_MAX = 200  # one sentence per pick
STANCE_REASON_MAX = 200  # one sentence per support / challenge / refusal
READING_MAX = 600  # "one or two sentences"
# A parse bound, not the rule: a plan with more than PICKS_MAX picks parses, and Hub's validation
# rejects it as invalid (a reason recorded), rather than the answer reading as unparseable.
PICKS_PARSE_MAX = 20

Stance = Literal["support", "challenge", "refuse"]

# What orion-mind can answer. Hub adds two of its own on the plan row: PLAN_STATUS_TIMEOUT, and
# PLAN_STATUS_INVALID_PREFIX + <reason> when its validation rejects the plan.
ReplicaPlanStatus = Literal["ok", "abstain", "unfunded", "model_failed", "unparseable"]
REPLICA_PLAN_STATUSES: tuple[str, ...] = ("ok", "abstain", "unfunded", "model_failed", "unparseable")
PLAN_STATUS_TIMEOUT = "timeout"
PLAN_STATUS_INVALID_PREFIX = "invalid:"


def plan_row_status_allowed(status: str) -> bool:
    """True for every status a ``curiosity_offer_plans`` row may carry (the table's CHECK says the same)."""
    return status in REPLICA_PLAN_STATUSES or status == PLAN_STATUS_TIMEOUT or (
        status.startswith(PLAN_STATUS_INVALID_PREFIX) and len(status) > len(PLAN_STATUS_INVALID_PREFIX)
    )


class CuriosityOfferCandidateV1(BaseModel):
    """One fresh live prior Hub could offer this run: one line of the replica's frame."""

    model_config = ConfigDict(extra="forbid")

    prior_id: str = Field(min_length=1)
    claim: str = Field(min_length=1, max_length=CLAIM_MAX)
    # Orion's own confidence; None when the stored value is unusable (shown as a gap, never as 0.5).
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    times_tested: int = Field(default=0, ge=0)
    # Hours since its last test; None = never tested.
    last_tested_age_h: Optional[float] = Field(default=None, ge=0.0)
    # Confidence change its last test made (after - before); None = never tested, or unknown.
    last_move: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class CuriosityPendingFocusV1(BaseModel):
    """The focus waiting in the slot when Hub asked, so the replica can take a stance on it."""

    model_config = ConfigDict(extra="forbid")

    focus_id: str = Field(min_length=1)
    origin: Literal["self", "juniper"]
    question: str = Field(min_length=1)
    # True only for a focus Prime queued (orion/curiosity/focus.py is_prime_focus): the only kind the
    # replica may refuse. Juniper's focus, and one Orion chose in a conversation, can be challenged only.
    from_prime: bool = False

    @model_validator(mode="after")
    def _prime_focus_is_self_origin(self) -> "CuriosityPendingFocusV1":
        if self.from_prime and self.origin != "self":
            raise ValueError("a focus Prime queued has origin 'self'; Juniper's focus is never from_prime")
        return self


class CuriosityOfferPlanRequestV1(BaseModel):
    """Hub -> orion-mind: plan the top of this run's offer before ``deadline_at``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.replica.plan.request.v1"] = "curiosity.replica.plan.request.v1"
    run_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    deadline_at: AwareDatetime
    candidates: list[CuriosityOfferCandidateV1] = Field(default_factory=list, max_length=MAX_CANDIDATES)
    # The stale bucket's ids (tested 3+ times): shown to the run anyway, never pickable.
    stale_ids: list[str] = Field(default_factory=list)
    pending_focus: Optional[CuriosityPendingFocusV1] = None

    @model_validator(mode="after")
    def _distinct_candidates(self) -> "CuriosityOfferPlanRequestV1":
        ids = [c.prior_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate prior_ids must be distinct")
        if set(ids) & set(self.stale_ids):
            raise ValueError("a stale prior is never a candidate")
        return self


class CuriosityOfferPickV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prior_id: str = Field(min_length=1)
    why: str = Field(min_length=1, max_length=WHY_MAX)


class CuriosityFocusStanceV1(BaseModel):
    """The replica's answer to the pending focus. ``refuse`` applies only to a Prime focus
    (``CuriosityPendingFocusV1.from_prime``); on any other focus it is not applied (Hub's check, and
    orion/curiosity/focus.py refuse_focus refuses it a second time, the table a third)."""

    model_config = ConfigDict(extra="ignore")

    stance: Stance
    reason: str = Field(min_length=1, max_length=STANCE_REASON_MAX)


class CuriositySignalStanceV1(BaseModel):
    """The replica's answer to a live Prime regulate signal on durable.curiosity_investigate."""

    model_config = ConfigDict(extra="ignore")

    signal_id: str = Field(min_length=1)
    stance: Stance
    reason: str = Field(min_length=1, max_length=STANCE_REASON_MAX)


class CuriosityOfferPlanAnswerV1(BaseModel):
    """The replica model's JSON answer (design section 3). Lenient on extra keys, like Prime's answer;
    the counts (3-5 picks, no duplicates, every id a candidate) are Hub's validation, so a bad plan is
    recorded as ``invalid:<reason>`` rather than unparseable."""

    model_config = ConfigDict(extra="ignore")

    reading: str = Field(default="", max_length=READING_MAX)
    picks: list[CuriosityOfferPickV1] = Field(default_factory=list, max_length=PICKS_PARSE_MAX)
    focus_stance: Optional[CuriosityFocusStanceV1] = None
    signal_stance: Optional[CuriositySignalStanceV1] = None
    # "Nothing here is worth reordering." Valid, and counts as a fallback for the arm.
    abstain: bool = False

    @model_validator(mode="after")
    def _abstain_has_no_picks(self) -> "CuriosityOfferPlanAnswerV1":
        if self.abstain and self.picks:
            raise ValueError("an abstaining answer carries no picks")
        return self


class CuriosityOfferPlanResultV1(BaseModel):
    """orion-mind -> Hub on ``orion:curiosity:replica:plan:result:<correlation_id>``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.replica.plan.result.v1"] = "curiosity.replica.plan.result.v1"
    run_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    status: ReplicaPlanStatus
    picks: list[CuriosityOfferPickV1] = Field(default_factory=list, max_length=PICKS_PARSE_MAX)
    focus_stance: Optional[CuriosityFocusStanceV1] = None
    signal_stance: Optional[CuriositySignalStanceV1] = None
    # The plan's thought act (actor curiosity.replica, verb infer); None when no model call was made.
    act_id: Optional[str] = None
    latency_ms: int = Field(ge=0)
    frame_tokens: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _only_ok_carries_picks(self) -> "CuriosityOfferPlanResultV1":
        if self.status != "ok" and self.picks:
            raise ValueError(f"a {self.status!r} result carries no picks")
        if self.status == "ok" and not self.picks:
            raise ValueError("an 'ok' result carries its picks")
        return self


__all__ = [
    "CLAIM_MAX",
    "CuriosityFocusStanceV1",
    "CuriosityOfferCandidateV1",
    "CuriosityOfferPickV1",
    "CuriosityOfferPlanAnswerV1",
    "CuriosityOfferPlanRequestV1",
    "CuriosityOfferPlanResultV1",
    "CuriosityPendingFocusV1",
    "CuriositySignalStanceV1",
    "MAX_CANDIDATES",
    "PICKS_MAX",
    "PICKS_MIN",
    "PLAN_DEADLINE_SEC",
    "PLAN_REQUEST_CHANNEL",
    "PLAN_REQUEST_KIND",
    "PLAN_RESULT_CHANNEL_PREFIX",
    "PLAN_RESULT_KIND",
    "PLAN_STATUS_INVALID_PREFIX",
    "PLAN_STATUS_TIMEOUT",
    "REPLICA_ACTOR",
    "REPLICA_FAMILY",
    "REPLICA_PLAN_STATUSES",
    "STANCE_REASON_MAX",
    "Stance",
    "WHY_MAX",
    "plan_row_status_allowed",
]
