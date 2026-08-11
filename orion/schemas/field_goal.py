"""Field-native goal-provenance record (Sentience Striving Program §6 Objective 3).

Produced by orion-attention-runtime when a real FieldAttentionFrameV1's dominant_targets[0]
sustains competition dominance across consecutive real field ticks -- see
docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-design.md
for the full design. Replaces the deleted GoalProposalEngine's role: the channel this schema
publishes to (orion:memory:goals:proposed) has been producer-less since the 2026-07-30 drives
deletion (chore/delete-orion-drives, PR #1486).

Deliberately does not subclass orion.core.schemas.drives.GoalProposalV1: that schema's only
drive-specific field, drive_origin, is permanently None going forward, and formalizing a new
producer on a schema half of whose fields are dead weight repeats the "formalize before
validating" mistake orion/sentience_striving_program/README.md's own §6 re-sequencing note
names as the pattern to avoid. GraphReadyArtifact and ProposalStatus ARE reused from
drives.py -- both are genuinely generic, already-shared infrastructure (GraphReadyArtifact
is also the base class for concept_induction's TurnDossierV1, not a drives-specific type;
ProposalStatus is the exact literal orion/substrate/attention/goal_context.py's
_ACTIVE_STATES already checks against), not a coupling to the deleted system.

No drive_origin, no taxonomy label -- per Sentience Striving Program §5's O4, if named
motivational categories ever emerge they are a report on which field_target_ids recur across
real winning history, derived later from these records, not asserted here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.drives import GraphReadyArtifact, ProposalStatus


class FieldGoalProvenanceV1(GraphReadyArtifact):
    # Which real field competition winner triggered this -- e.g. "node:substrate.biometrics",
    # "capability:memory". Matches FieldAttentionTargetV1.target_id exactly, not a derived id.
    field_target_id: str
    # Mirrors FieldAttentionTargetV1.target_kind's real value ("node" | "capability" | "system"
    # for this producer's real emitters -- see that schema's own Literal for the full set).
    # Not re-declared as a Literal here to avoid two schemas needing to agree on membership.
    target_kind: str
    # The real dominant_targets[0].salience_score reading that triggered this record.
    salience_score: float = Field(ge=0.0, le=1.0)
    # FieldStateV1.tick_id and FieldAttentionFrameV1.frame_id this record is traceable to --
    # required for the O2 falsifiability trace (field tick -> salience winner -> goal record
    # -> capability decision -> dispatch), not optional metadata.
    source_field_tick_id: str
    source_attention_frame_id: str
    # Normalized salience_score, feeds GoalContext.priority directly (same [0,1] scale
    # top_down.py's relevance() already expects).
    priority: float = Field(ge=0.0, le=1.0)
    # Reuses goal_context.py's existing _ACTIVE_STATES vocabulary unchanged -- no new status
    # vocabulary for this producer.
    proposal_status: ProposalStatus = "proposed"


class DominanceStreakTickV1(BaseModel):
    """Debug-tier telemetry: the real DominanceStreak state after every real tick the
    goal-provenance producer runs -- not just qualifying emissions.

    Part H's "measure before minting" calibration step (docs/superpowers/specs/
    2026-07-30-goal-system-remaining-gaps-design.md, Missing Question 5) needs the true
    empirical streak-length distribution to judge whether ORION_GOAL_PROVENANCE_MIN_STREAK's
    placeholder value of 3 is right. FieldGoalProvenanceV1 alone cannot answer that: it only
    exists once a streak has *already* survived min_streak ticks -- a censored sample that
    can show a survival rate, never a rejection rate, since every 1-tick and 2-tick streak
    that got discarded left no trace anywhere (confirmed live 2026-08-11: no
    FieldAttentionFrameV1 history and no other durable per-tick record exists to replay
    after the fact). This schema is the raw, uncensored signal underneath it.

    Deliberately does NOT subclass GraphReadyArtifact: this is raw per-tick telemetry
    (~1 row per real field tick, matching substrate_attention_frames' volume), not a
    cognitive artifact meant for substrate/reasoning-graph materialization -- forcing
    GraphReadyArtifact's confidence/provenance/join_keys ceremony onto it would misrepresent
    what it is. Meant to be temporary: collect a few days, run
    scripts/analysis/measure_goal_provenance_streak_distribution.py, decide on
    ORION_GOAL_PROVENANCE_MIN_STREAK, then this table's growth is a disclosed, known
    follow-up (retire the channel/table, or add real retention) -- not solved in this patch.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tick_telemetry_id: str = Field(default_factory=lambda: f"streak-tick-{uuid4()}")
    # The real node-target winner this tick (FieldAttentionTargetV1.target_id), or None when
    # no target won (update_dominance_streak's own None-target_id reset case).
    target_id: str | None = None
    # DominanceStreak.count after this tick's update -- 0 when target_id is None, 1 on a
    # fresh streak start (target changed from the previous tick), incrementing otherwise. A
    # row's own count therefore always says whether it started a new streak (count <= 1) or
    # continued one, with no separate boolean needed.
    streak_count: int = Field(ge=0)
    # ORION_GOAL_PROVENANCE_MIN_STREAK's value at the moment this tick ran -- self-describing
    # per row since the setting could change across the observation window this telemetry is
    # meant to span.
    min_streak_at_tick: int = Field(ge=1)
    # Whether this tick's streak_count >= min_streak_at_tick -- i.e. whether this exact tick
    # is the one that would have triggered (or did trigger) a FieldGoalProvenanceV1 emission.
    qualified: bool
    source_field_tick_id: str
    source_attention_frame_id: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
