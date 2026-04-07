from __future__ import annotations

from datetime import datetime

from orion.core.schemas.reasoning import ClaimV1, ConceptV1, ContradictionV1, MentorProposalV1, ReasoningSparkStateSnapshotV1
from orion.core.schemas.reasoning_summary import (
    ReasoningAutonomySummaryV1,
    ReasoningClaimDigestV1,
    ReasoningConceptDigestV1,
    ReasoningSparkSummaryV1,
    ReasoningSummaryDebugV1,
    ReasoningSummaryRequestV1,
    ReasoningSummaryV1,
)
from orion.reasoning.lifecycle import evaluate_entity_lifecycle
from orion.core.schemas.reasoning_policy import EntityLifecycleEvaluationRequestV1
from orion.reasoning.repository import InMemoryReasoningRepository


class ReasoningSummaryCompiler:
    """Deterministic compiler from reasoning artifacts to bounded turn-time summaries."""

    def __init__(self, repository: InMemoryReasoningRepository) -> None:
        self._repository = repository

    def compile(self, request: ReasoningSummaryRequestV1) -> ReasoningSummaryV1:
        artifacts = self._repository.list_latest(limit=250)
        debug = ReasoningSummaryDebugV1(compiler_ran=True)
        debug.considered_count = len(artifacts)

        unresolved_contradictions: list[ContradictionV1] = [
            a
            for a in artifacts
            if isinstance(a, ContradictionV1) and a.resolution_status != "resolved"
        ]
        unresolved_ids: set[str] = {
            artifact_id
            for contradiction in unresolved_contradictions
            for artifact_id in contradiction.involved_artifact_ids
        }

        active_claims: list[ReasoningClaimDigestV1] = []
        active_concepts: list[ReasoningConceptDigestV1] = []
        relationship_signals: list[str] = []
        autonomy_posture: list[str] = []
        autonomy_goals: list[str] = []
        hazards: list[str] = []

        latest_spark: ReasoningSparkStateSnapshotV1 | None = None

        for artifact in artifacts:
            if artifact.status in {"rejected", "deprecated"}:
                debug.suppressed_count += 1
                debug.suppressed_by_status += 1
                continue

            if artifact.artifact_id in unresolved_ids and not isinstance(artifact, ContradictionV1):
                debug.suppressed_count += 1
                debug.suppressed_by_contradiction += 1
                continue

            if isinstance(artifact, MentorProposalV1) and artifact.risk_tier == "high":
                hazards.append("high_risk_mentor_suppressed")
                debug.suppressed_count += 1
                continue

            if isinstance(artifact, ReasoningSparkStateSnapshotV1):
                if latest_spark is None or artifact.observed_at > latest_spark.observed_at:
                    latest_spark = artifact
                continue

            if isinstance(artifact, ConceptV1):
                if artifact.status == "proposed":
                    debug.suppressed_count += 1
                    debug.suppressed_by_status += 1
                    continue
                if artifact.status == "provisional" and artifact.confidence < 0.65:
                    debug.suppressed_count += 1
                    debug.suppressed_by_status += 1
                    continue
                if artifact.confidence < 0.55 or artifact.salience < 0.25:
                    debug.suppressed_count += 1
                    debug.suppressed_by_drift += 1
                    continue
                active_concepts.append(
                    ReasoningConceptDigestV1(
                        artifact_id=artifact.artifact_id,
                        concept_id=artifact.concept_id,
                        label=artifact.label,
                        concept_type=artifact.concept_type,
                        anchor_scope=artifact.anchor_scope,
                        subject_ref=artifact.subject_ref,
                        status=artifact.status,
                        confidence=artifact.confidence,
                        salience=artifact.salience,
                    )
                )
                continue

            if not isinstance(artifact, ClaimV1):
                continue

            if artifact.anchor_scope != request.anchor_scope and request.anchor_scope != "world":
                continue

            if request.subject_refs and artifact.subject_ref not in request.subject_refs:
                continue

            if artifact.status == "proposed":
                debug.suppressed_count += 1
                debug.suppressed_by_status += 1
                continue

            if artifact.status == "provisional" and (not request.include_provisional or artifact.confidence < 0.65):
                debug.suppressed_count += 1
                debug.suppressed_by_status += 1
                continue

            if artifact.claim_kind in {"concept_item", "concept_delta"}:
                debug.suppressed_count += 1
                debug.suppressed_by_drift += 1
                continue

            active_claims.append(
                ReasoningClaimDigestV1(
                    artifact_id=artifact.artifact_id,
                    anchor_scope=artifact.anchor_scope,
                    subject_ref=artifact.subject_ref,
                    status=artifact.status,
                    claim_kind=artifact.claim_kind,
                    claim_text=artifact.claim_text,
                    confidence=artifact.confidence,
                )
            )

            if artifact.anchor_scope == "relationship":
                relationship_signals.append(artifact.claim_text)

            if artifact.claim_kind == "autonomy_state_summary":
                autonomy_posture.extend(artifact.qualifiers.get("active_drives") or [])
                dominant = artifact.qualifiers.get("dominant_drive")
                if isinstance(dominant, str) and dominant.strip():
                    autonomy_posture.append(dominant)

            if artifact.claim_kind == "goal_proposal_headline":
                autonomy_goals.append(artifact.claim_text)

            if len(active_claims) >= request.max_claims:
                break

        tensions = [c.summary for c in unresolved_contradictions][:4]
        if unresolved_contradictions:
            hazards.append("unresolved_contradictions_present")

        active_subject_refs = self._active_subject_refs(request, active_claims, active_concepts)

        spark_summary = ReasoningSparkSummaryV1(present=False)
        if latest_spark is not None:
            spark_summary = ReasoningSparkSummaryV1(
                present=True,
                observed_at=latest_spark.observed_at,
                dimensions=dict(latest_spark.dimensions),
                tensions=list(latest_spark.tensions)[:4],
            )

        hazards = list(dict.fromkeys(hazards))
        included_count = len(active_claims)
        included_count += len(active_concepts)
        debug.included_count = included_count
        debug.compiler_succeeded = True
        debug.selected_anchor_scopes = sorted({c.anchor_scope for c in active_claims})
        debug.selected_anchor_scopes = sorted(set(debug.selected_anchor_scopes) | {c.anchor_scope for c in active_concepts})
        debug.selected_subject_refs = active_subject_refs

        confidence = 0.0 if debug.considered_count == 0 else min(1.0, included_count / max(1, debug.considered_count))
        completeness = 0.0
        if included_count:
            completeness += 0.5
        if spark_summary.present:
            completeness += 0.2
        if autonomy_posture or autonomy_goals:
            completeness += 0.2
        if tensions:
            completeness += 0.1

        fallback = included_count == 0 and not spark_summary.present
        debug.fallback_used = fallback

        return ReasoningSummaryV1(
            request_id=request.request_id,
            anchor_scope=request.anchor_scope,
            active_subject_refs=active_subject_refs,
            active_claims=active_claims,
            active_concepts=active_concepts[: request.max_claims],
            relationship_signals=relationship_signals[:4],
            tensions=tensions,
            hazards=hazards[:6],
            autonomy=ReasoningAutonomySummaryV1(
                present=bool(autonomy_posture or autonomy_goals),
                posture=list(dict.fromkeys(autonomy_posture))[:5],
                active_goals=autonomy_goals[:3],
                hazards=["autonomy_goals_are_proposals"] if autonomy_goals else [],
            ),
            spark=spark_summary,
            confidence=round(confidence, 3),
            completeness=round(completeness, 3),
            fallback_recommended=fallback,
            debug=debug,
        )

    def _active_subject_refs(
        self,
        request: ReasoningSummaryRequestV1,
        claims: list[ReasoningClaimDigestV1],
        concepts: list[ReasoningConceptDigestV1],
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        refs = [c.subject_ref for c in claims] + [c.subject_ref for c in concepts]
        for subject_ref in refs:
            if not subject_ref:
                continue
            if subject_ref in seen:
                continue
            history = self._repository.list_by_subject_ref(subject_ref)
            lifecycle = evaluate_entity_lifecycle(
                EntityLifecycleEvaluationRequestV1(anchor_scope=request.anchor_scope, subject_ref=subject_ref),
                artifacts=history,
            )
            if lifecycle.lifecycle_action in {"dormant", "decay", "retire"}:
                continue
            seen.add(subject_ref)
            out.append(subject_ref)
            if len(out) >= 6:
                break
        return out
