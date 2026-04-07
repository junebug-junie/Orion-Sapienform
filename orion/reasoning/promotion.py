from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.reasoning import (
    ConceptV1,
    ContradictionV1,
    MentorProposalV1,
    PromotionDecisionV1,
    ReasoningArtifactBaseV1,
    ReasoningStatus,
)
from orion.core.schemas.reasoning_policy import (
    ContradictionFindingV1,
    EntityLifecycleEvaluationRequestV1,
    PromotionEvaluationItemV1,
    PromotionEvaluationRequestV1,
    PromotionEvaluationResultV1,
)
from orion.reasoning.lifecycle import evaluate_entity_lifecycle
from orion.reasoning.repository import InMemoryReasoningRepository


_ALLOWED_TRANSITIONS: set[tuple[ReasoningStatus, ReasoningStatus]] = {
    ("proposed", "provisional"),
    ("provisional", "canonical"),
    ("canonical", "deprecated"),
    ("proposed", "rejected"),
    ("provisional", "rejected"),
    ("canonical", "rejected"),
    ("deprecated", "rejected"),
}


class PromotionEngine:
    def __init__(self, repository: InMemoryReasoningRepository, *, policy_version: str = "phase3.v1") -> None:
        self._repository = repository
        self._policy_version = policy_version

    def evaluate(self, request: PromotionEvaluationRequestV1) -> PromotionEvaluationResultV1:
        items: list[PromotionEvaluationItemV1] = []
        decisions: list[PromotionDecisionV1] = []

        for artifact_id in request.artifact_ids:
            artifact = self._repository.get_by_id(artifact_id)
            if artifact is None:
                items.append(
                    PromotionEvaluationItemV1(
                        artifact_id=artifact_id,
                        artifact_type="unknown",
                        current_status="proposed",
                        target_status=request.target_status,
                        outcome="blocked",
                        reasons=["artifact_not_found"],
                        risk_tier="high",
                        policy_version=self._policy_version,
                    )
                )
                continue

            contradictions = self._repository.list_contradictions_for(artifact_id)
            contradiction_findings = [
                ContradictionFindingV1(
                    contradiction_id=c.artifact_id,
                    severity=c.severity,
                    resolution_status=c.resolution_status,
                )
                for c in contradictions
            ]

            outcome, reasons, hitl, escalation_reason = self._evaluate_single(
                artifact=artifact,
                target_status=request.target_status,
                contradiction_findings=contradiction_findings,
            )

            lifecycle = evaluate_entity_lifecycle(
                EntityLifecycleEvaluationRequestV1(
                    anchor_scope=artifact.anchor_scope,
                    subject_ref=artifact.subject_ref,
                ),
                artifacts=self._repository.list_by_subject_ref(artifact.subject_ref),
            )

            item = PromotionEvaluationItemV1(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                current_status=artifact.status,
                target_status=request.target_status,
                outcome=outcome,
                reasons=reasons,
                risk_tier=artifact.risk_tier,
                contradiction_findings=contradiction_findings,
                human_review_required=hitl,
                escalation_reason=escalation_reason,
                lifecycle=lifecycle,
                policy_version=self._policy_version,
            )
            items.append(item)

            decision_action = "deferred"
            if outcome == "promoted":
                decision_action = "accepted"
            elif outcome == "blocked":
                decision_action = "deferred"
            elif outcome == "rejected":
                decision_action = "rejected"
            elif outcome == "deprecated":
                decision_action = "downgraded"
            elif outcome == "escalated_hitl":
                decision_action = "escalated_hitl"

            decisions.append(
                PromotionDecisionV1(
                    anchor_scope=artifact.anchor_scope,
                    subject_ref=artifact.subject_ref,
                    authority="local_inferred",
                    confidence=artifact.confidence,
                    salience=artifact.salience,
                    novelty=artifact.novelty,
                    risk_tier=artifact.risk_tier,
                    observed_at=datetime.now(timezone.utc),
                    provenance=artifact.provenance,
                    proposal_artifact_id=artifact.artifact_id,
                    action=decision_action,
                    rationale="; ".join(reasons),
                    decided_by=request.actor,
                    escalated_to_hitl=hitl,
                )
            )

            if outcome == "promoted":
                self._repository.update_status(artifact.artifact_id, request.target_status)

        accepted = sum(1 for i in items if i.outcome == "promoted")
        blocked = sum(1 for i in items if i.outcome == "blocked")
        escalated = sum(1 for i in items if i.human_review_required)
        rejected = sum(1 for i in items if i.outcome == "rejected")

        return PromotionEvaluationResultV1(
            request_id=request.request_id,
            policy_version=self._policy_version,
            evaluated_count=len(items),
            items=items,
            decisions=decisions,
            accepted_count=accepted,
            blocked_count=blocked,
            escalated_count=escalated,
            rejected_count=rejected,
        )

    def _evaluate_single(
        self,
        *,
        artifact: ReasoningArtifactBaseV1,
        target_status: ReasoningStatus,
        contradiction_findings: list[ContradictionFindingV1],
    ) -> tuple[str, list[str], bool, str | None]:
        reasons: list[str] = []
        hitl = False
        escalation_reason: str | None = None

        if artifact.status == target_status:
            return "no_change", ["already_in_target_status"], False, None

        if (artifact.status, target_status) not in _ALLOWED_TRANSITIONS:
            return "blocked", ["invalid_transition"], False, None

        if not artifact.provenance.source_channel or not artifact.provenance.source_kind or not artifact.provenance.producer:
            return "blocked", ["provenance_incomplete"], True, "provenance_incomplete"

        unresolved = [c for c in contradiction_findings if c.resolution_status != "resolved"]
        severe_unresolved = [c for c in unresolved if c.severity in {"high", "critical"}]
        medium_plus_unresolved = [c for c in unresolved if c.severity in {"medium", "high", "critical"}]

        if target_status == "canonical" and medium_plus_unresolved:
            return "blocked", ["unresolved_contradiction_blocks_canonical"], True, "contradiction_unresolved"

        if target_status == "provisional" and severe_unresolved:
            return "blocked", ["severe_contradiction_blocks_provisional"], True, "contradiction_severe"

        if artifact.artifact_type == "spark_state_snapshot" and target_status == "canonical":
            return "blocked", ["spark_snapshot_not_canonical_identity"], False, None

        if artifact.artifact_type == "claim":
            claim_kind = getattr(artifact, "claim_kind", "")
            if claim_kind in {"concept_item", "concept_delta"} and target_status == "canonical":
                return "blocked", ["concept_translation_conservative_policy"], False, None

            if claim_kind == "goal_proposal_headline" and target_status == "canonical":
                hitl = True
                escalation_reason = "autonomy_goal_requires_hitl"
                reasons.append("autonomy_goal_requires_hitl")

        if isinstance(artifact, ConceptV1):
            has_evidence = bool(artifact.provenance.evidence_refs)
            if target_status == "canonical" and (artifact.confidence < 0.75 or artifact.salience < 0.4 or not has_evidence):
                return "blocked", ["concept_semantics_insufficient_for_canonical"], False, None

        if isinstance(artifact, MentorProposalV1) and artifact.risk_tier == "high":
            hitl = True
            escalation_reason = "high_risk_mentor_proposal"
            reasons.append("high_risk_mentor_proposal")
        if isinstance(artifact, MentorProposalV1):
            contradiction_flags = artifact.suggested_payload.get("contradiction_flags") or []
            if target_status == "canonical":
                return "blocked", ["mentor_proposal_cannot_directly_canonicalize"], True, "mentor_advisory_only"
            if contradiction_flags:
                return "blocked", ["mentor_proposal_reports_contradiction"], True, "mentor_contradiction_flag"

        if target_status == "canonical" and artifact.anchor_scope in {"orion", "juniper", "relationship"}:
            hitl = True
            escalation_reason = escalation_reason or "identity_scope_requires_hitl"
            reasons.append("identity_scope_requires_hitl")

        if target_status == "rejected":
            return "rejected", reasons or ["explicit_rejection_transition"], hitl, escalation_reason

        if target_status == "deprecated":
            return "deprecated", reasons or ["explicit_deprecation_transition"], hitl, escalation_reason

        if hitl and target_status == "canonical":
            return "escalated_hitl", reasons or ["hitl_required"], True, escalation_reason

        return "promoted", reasons or ["policy_checks_passed"], hitl, escalation_reason
