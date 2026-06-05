from __future__ import annotations

from orion.consolidation.policy import ConsolidationPolicyV1
from orion.schemas.consolidation_frame import ExpectationV1, MotifObservationV1
from orion.schemas.feedback_frame import FeedbackFrameV1

_MOTIF_TO_EXPECTATION: dict[str, str] = {
    "loaded_but_reliable": "reliability_clear",
    "attention_saturated_execution": "execution_pressure_high",
    "read_only_policy_loop": "read_only_approved",
    "dry_run_feedback_loop": "dry_run_feedback",
    "blocked_review_loop": "policy_review_required",
    "stable_after_dry_run": "dry_run_feedback",
    "transport_contract_drift_loop": "contract_drift_persists",
    "transport_healthy_idle": "transport_stable",
}


def build_expectations_from_motifs(
    *,
    motifs: list[MotifObservationV1],
    feedback_frames: list[FeedbackFrameV1],
    policy: ConsolidationPolicyV1,
) -> list[ExpectationV1]:
    del feedback_frames, policy
    out: list[ExpectationV1] = []
    for motif in motifs:
        kind = _MOTIF_TO_EXPECTATION.get(motif.label)
        if kind is None:
            continue
        out.append(
            ExpectationV1(
                expectation_id=f"expectation:{motif.motif_id}:{kind}",
                trigger_motif_id=motif.motif_id,
                expected_outcome_kind=kind,  # type: ignore[arg-type]
                confidence_score=motif.confidence_score,
                support_count=motif.recurrence_count,
                evidence_refs=list(motif.evidence_frame_ids),
                reasons=[f"derived_from_motif:{motif.label}"],
            )
        )
    return out
