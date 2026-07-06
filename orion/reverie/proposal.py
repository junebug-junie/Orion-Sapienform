"""Phase B — spontaneous thought → governed proposal candidate.

Maps a non-hollow `SpontaneousThoughtV1` into a `ProposalCandidateV1` that the
proposal-runtime builder can incorporate. The awake organ *proposes*; the
substrate *disposes* — so a reverie candidate always carries a policy gate
(`operator_review`) and never `none`/`read_only`. It can enter Layer 7→8, but it
can never auto-dispatch: an action only happens if policy (L8) approves and the
separate `ORION_REVERIE_AUTOACTION_ENABLED` dispatch gate is on.

Deterministic (§4): given a thought, the candidate is a pure function — no LLM.
This module must never import orion-actions (contract test enforces it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orion.schemas.proposal_frame import ProposalCandidateV1

if TYPE_CHECKING:
    from orion.schemas.reverie import SpontaneousThoughtV1

REVERIE_PROPOSAL_SOURCE = "reverie_thought"
# A thought below this salience is not worth a governed proposal.
DEFAULT_MIN_PROPOSE_SALIENCE = 0.5
_MAX_EVIDENCE_REFS = 200


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def spontaneous_thought_to_candidate(
    thought: "SpontaneousThoughtV1",
    *,
    self_state_id: str,
    min_salience: float = DEFAULT_MIN_PROPOSE_SALIENCE,
) -> ProposalCandidateV1 | None:
    """Convert a thought into a review-gated proposal candidate, or None.

    Returns None (degrades, never raises) when the thought is hollow, absent, or
    below the salience floor — so a weak reverie never manufactures a proposal.
    """
    if thought is None or thought.is_hollow():
        return None
    if thought.salience < min_salience:
        return None

    target_id = (thought.coalition.selected_open_loop_id if thought.coalition else None) or "self_state"
    evidence_refs = list(thought.evidence_refs)[:_MAX_EVIDENCE_REFS]

    return ProposalCandidateV1(
        proposal_id=f"proposal:reverie:{thought.thought_id}",
        proposal_kind="request_policy_review",
        title="Reverie proposal (policy review required)",
        description=thought.interpretation[:500],
        target_id=target_id,
        target_kind="self_state",
        priority_score=_clamp01(thought.salience),
        urgency_score=_clamp01(thought.salience * 0.5),
        confidence_score=_clamp01(thought.salience),
        # A spontaneous proposal is low-risk to *propose* and fully reversible —
        # the risk lives in the action, which policy must still gate.
        risk_score=0.3,
        reversibility_score=1.0,
        evidence_refs=evidence_refs,
        reasons=["spontaneous_thought", f"thought_id:{thought.thought_id}"],
        proposed_effect="prepare_for_policy_gate",
        required_policy_gate="operator_review",  # never none/read_only — cannot auto-dispatch
        source=REVERIE_PROPOSAL_SOURCE,
        thought_id=thought.thought_id,
    )
