from __future__ import annotations

from orion.consolidation.policy import ConsolidationPolicyV1
from orion.schemas.consolidation_frame import ExpectationV1, MotifObservationV1, SchemaCandidateV1

_CANDIDATE_SPECS: dict[str, dict[str, object]] = {
    "loaded_but_reliable": {
        "candidate_kind": "prior_candidate",
        "label": "loaded_but_reliable_operating_mode",
        "proposed_schema": {
            "condition": "loaded",
            "reliability_pressure": "low",
            "execution_pressure": "high",
            "interpretation": "high load does not imply instability",
        },
        "reason": "derived_from_motif:loaded_but_reliable",
    },
    "dry_run_feedback_loop": {
        "candidate_kind": "habit_candidate",
        "label": "dry_run_feedback_review_loop",
        "proposed_schema": {
            "trigger": "approved_read_only dispatch envelope",
            "expected_feedback": "dry_run_only",
            "mutation": "none",
        },
        "reason": "derived_from_motif:dry_run_feedback_loop",
    },
    "read_only_policy_loop": {
        "candidate_kind": "policy_candidate",
        "label": "read_only_policy_approval_pattern",
        "proposed_schema": {
            "candidate_behavior": "inspect/summarize/observe",
            "gate": "read_only",
            "execution_allowed": False,
        },
        "reason": "derived_from_motif:read_only_policy_loop",
    },
}


def build_schema_candidates(
    *,
    motifs: list[MotifObservationV1],
    expectations: list[ExpectationV1],
    policy: ConsolidationPolicyV1,
) -> list[SchemaCandidateV1]:
    del policy
    expectations_by_motif = {
        expectation.trigger_motif_id: expectation.expectation_id for expectation in expectations
    }
    out: list[SchemaCandidateV1] = []
    for motif in motifs:
        spec = _CANDIDATE_SPECS.get(motif.label)
        if spec is None:
            continue
        candidate_kind = str(spec["candidate_kind"])
        out.append(
            SchemaCandidateV1(
                schema_candidate_id=f"schema_candidate:{candidate_kind}:{motif.motif_id}",
                candidate_kind=candidate_kind,  # type: ignore[arg-type]
                label=str(spec["label"]),
                source_motif_ids=[motif.motif_id],
                source_expectation_ids=[
                    expectations_by_motif[motif.motif_id]
                ]
                if motif.motif_id in expectations_by_motif
                else [],
                support_score=motif.support_score,
                confidence_score=motif.confidence_score,
                proposed_schema=dict(spec["proposed_schema"]),  # type: ignore[arg-type]
                promotion_status="candidate_only",
                reasons=[str(spec["reason"])],
                evidence_refs=list(motif.evidence_frame_ids),
            )
        )
    return out
