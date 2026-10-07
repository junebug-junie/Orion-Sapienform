from __future__ import annotations

from orion.schemas.reverie_visual import VisualBaselineEligibilityV1, VisualRunOutcome

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProposalCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_baseline: VisualBaselineEligibilityV1 | None = None

    proposal_id: str

    proposal_kind: Literal[
        "observe",
        "inspect",
        "summarize",
        "stabilize",
        "defer",
        "request_policy_review",
        "prepare_action",
        # 2026-08-12: the first MUTATING proposal kind. Everything above is
        # read-only by construction. `maintain` exists so a bounded, reversible
        # housekeeping action (today: Docker build-cache pruning) can be
        # proposed at all -- it is still gated three independent ways before
        # anything happens. See orion/execution_dispatch/builder.py's scope
        # check and config/execution_dispatch/execution_dispatch_policy.v1.yaml.
        "maintain",
        # 2026-08-30: the first OUTWARD kind. Every kind above either observes
        # Orion (inspect/summarize/observe) or tidies it (maintain) -- the whole
        # 17-template repertoire was introspection plus docker cleanup, which is
        # why a value-of-information allocator correctly refused all of it:
        # posterior variance 5.2e-06 over 7,685 observations on the busiest
        # action. `express` is for an action whose product exists outside Orion
        # and costs a physical resource to make. Closed Literal, same as
        # `maintain`: a new kind is a deliberate schema change, not a config typo.
        "express",
        # 2026-10-01 (attend-to-act loop A1): Orion's first WORLD-facing kind whose effect is
        # physical and self-reverting: it changes Orion's own resource use (GPU pool shed reason,
        # bounded TTL, pool-enforced caps) and is scored on a real sensor (cabinet air). Its own
        # kind so policy and dispatch can allow it independently of maintain/express.
        "self_regulate",
    ]

    title: str
    description: str

    target_id: str
    target_kind: Literal[
        "node",
        "capability",
        "field",
        "self_state",
        "service",
        "system",
    ]

    priority_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    reversibility_score: float = Field(ge=0.0, le=1.0)

    # 2026-08-21: the template's declared, falsifiable claim -- which
    # measured signal this proposal expects to move, and which way. Distinct
    # from `proposed_effect` directly below, which is a closed vocabulary of
    # intent labels ("increase_observability") with no signal, no magnitude,
    # and nothing that can ever be checked against the world. These two
    # fields are the checkable version; `proposed_effect` is left alone
    # because other consumers read it.
    expected_signal: str | None = None
    expected_direction: Literal["increase", "decrease", "no_change"] | None = None

    motivating_dimensions: dict[str, float] = Field(default_factory=dict)
    motivating_targets: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    proposed_effect: Literal[
        "increase_observability",
        "reduce_pressure",
        "preserve_stability",
        "increase_coherence",
        "defer_until_policy",
        "prepare_for_policy_gate",
        "no_effect",
    ]

    required_policy_gate: Literal[
        "none",
        "read_only",
        "operator_review",
        "autonomy_policy",
        "execution_policy",
        # 2026-10-01: a bounded, auto-reverting action on Orion's own resources. Allowed by
        # execution dispatch only while ORION_WORLD_ACTIONS_ENABLED and the template is named in
        # ORION_WORLD_ACTIONS_ALLOWED; no per-action operator approval (Juniper's approval is the flag).
        "self_reversible",
    ] = "read_only"

    execution_intent: dict[str, str] = Field(default_factory=dict)

    # Provenance for candidates injected from outside the deterministic builder
    # (Phase B: spontaneous-thought proposals). None for builder-native candidates.
    source: str | None = None
    thought_id: str | None = None

    # Provenance for template target_id/target_kind resolved from live attention
    # (P5: attention-bound proposal template) rather than the template's literal
    # fallback. Set to the recognized binding path string when resolution
    # succeeded; None when the candidate used the template's literal target.
    binding_resolved_from: str | None = None

    # 2026-10-01 (attend-to-act loop D1): the WORKSPACE winner this candidate is bound to
    # (``workspace.winner`` binding), or None for every other candidate. Additive on an
    # extra="forbid" model: a consumer-first rollout (readers deploy before this writer).
    attention_winner: "AttentionWinnerRefV1 | None" = None
    # What the builder saw when it judged a world action eligible (thermal verdict and reading age,
    # rise and threshold, hardware-watch health + open incidents, background leases, holdback
    # fraction). Control-arm rows carry the same snapshot so both arms are comparable.
    world_eligibility: dict[str, object] | None = None


class AttentionWinnerRefV1(BaseModel):
    """The workspace broadcast winner a proposal binds to (design D1).

    Bindable only when the projection is <= 90 s old, the coalition held >= 2 ticks and an action
    was selected. ``open_loop_id`` is the attention loop the action answers to; it is also the
    ``attention_loop_outcome.loop_id`` Orion's non-final ``acted`` verdict is written under."""

    model_config = ConfigDict(extra="forbid")

    broadcast_log_id: str
    open_loop_id: str
    node_id: str
    generated_at: datetime
    dwell_ticks: int = Field(ge=0)
    selected_action_type: str
    age_sec: float = Field(ge=0.0)


ProposalCandidateV1.model_rebuild()


class ProposalFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["proposal.frame.v1"] = "proposal.frame.v1"

    frame_id: str
    generated_at: datetime

    # source_self_state_id/source_self_state_generated_at removed 2026-07-22
    # (SelfStateV1 burn) -- redundant with source_field_tick_id/
    # source_attention_frame_id below, which already independently identified
    # the same tick without the self-state hop.
    source_field_tick_id: str
    source_field_generated_at: datetime

    source_attention_frame_id: str

    proposal_policy_id: str = "proposal_policy.v1"

    overall_action_pressure: float = Field(ge=0.0, le=1.0)
    overall_risk: float = Field(ge=0.0, le=1.0)
    policy_required: bool = True

    candidates: list[ProposalCandidateV1] = Field(default_factory=list)
    suppressed_candidates: list[ProposalCandidateV1] = Field(default_factory=list)

    dominant_motivations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # 2026-08-12: the tick-level gate. `action_warrant` (orion/field/
    # action_warrant.py) answers "does this tick's state warrant acting at
    # all", on a scale where 0.5 is a median normal day for this machine; the
    # per-template scoring below still decides WHICH candidates. Recorded on
    # the frame rather than only logged so a quiet tick is inspectable
    # evidence ("state was calm") instead of an absence.
    #
    # Optional for backward compatibility: frames persisted before this field
    # existed load unchanged, and `None` means "gate not evaluated", which is
    # deliberately distinguishable from a real 0.0 reading.
    action_warrant: float | None = Field(default=None, ge=0.0, le=1.0)
    # The dimensions that actually contributed. The score is only
    # interpretable alongside this count -- the statistic's null is
    # chi-square with 2N degrees of freedom, so a warrant of 0.9 over four
    # dimensions and one over a single dimension are different claims.
    action_warrant_dimensions: list[str] = Field(default_factory=list)
    # Why the gate opened or closed, e.g. "warranted", "below_threshold",
    # "no_live_dimensions". A frame with zero candidates must say which,
    # because "Orion was calm" and "the signal broke" look identical from the
    # outside otherwise (CLAUDE.md 0A, no empty-shell cognition).
    action_warrant_gate: str | None = None
