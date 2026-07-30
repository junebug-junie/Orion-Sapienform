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

from pydantic import Field

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
