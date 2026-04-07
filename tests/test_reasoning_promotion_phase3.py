from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.reasoning import ClaimV1, ContradictionV1, MentorProposalV1, ReasoningSparkStateSnapshotV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_policy import EntityLifecycleEvaluationRequestV1, PromotionEvaluationRequestV1
from orion.reasoning.lifecycle import evaluate_entity_lifecycle
from orion.reasoning.promotion import PromotionEngine
from orion.reasoning.repository import InMemoryReasoningRepository


def _provenance() -> dict:
    return {
        "evidence_refs": ["evidence:1"],
        "source_channel": "orion:test",
        "source_kind": "unit",
        "producer": "pytest",
    }


def _claim(*, status: str = "proposed", anchor_scope: str = "world", subject_ref: str = "entity:alpha", claim_kind: str = "assertion", risk_tier: str = "low") -> ClaimV1:
    return ClaimV1(
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        status=status,
        authority="local_inferred",
        confidence=0.9,
        salience=0.7,
        novelty=0.4,
        risk_tier=risk_tier,
        observed_at=datetime.now(timezone.utc),
        provenance=_provenance(),
        claim_text="test",
        claim_kind=claim_kind,
    )


def _repo_with(*artifacts):
    repo = InMemoryReasoningRepository()
    request = ReasoningWriteRequestV1(
        context=ReasoningWriteContextV1(
            source_family="manual",
            source_kind="unit",
            source_channel="orion:test",
            producer="pytest",
        ),
        artifacts=list(artifacts),
    )
    repo.write_artifacts(request)
    return repo


def test_transition_proposed_to_provisional_allowed() -> None:
    claim = _claim(status="proposed", anchor_scope="world")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)

    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="provisional"))
    assert result.items[0].outcome == "promoted"


def test_transition_provisional_to_canonical_allowed_world_scope() -> None:
    claim = _claim(status="provisional", anchor_scope="world")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)

    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="canonical"))
    assert result.items[0].outcome == "promoted"
    assert result.items[0].human_review_required is False


def test_blocked_invalid_transition() -> None:
    claim = _claim(status="proposed")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)

    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="deprecated"))
    assert result.items[0].outcome == "blocked"


def test_rejected_transition() -> None:
    claim = _claim(status="proposed")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)
    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="rejected"))
    assert result.items[0].outcome == "rejected"


def test_deprecated_transition() -> None:
    claim = _claim(status="canonical")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)
    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="deprecated"))
    assert result.items[0].outcome == "deprecated"


def test_hitl_orion_self_scope_canonical() -> None:
    claim = _claim(status="provisional", anchor_scope="orion", subject_ref="self:orion")
    repo = _repo_with(claim)
    engine = PromotionEngine(repo)
    result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="canonical"))
    assert result.items[0].human_review_required is True
    assert result.items[0].outcome == "escalated_hitl"


def test_hitl_juniper_and_relationship_scope_canonical() -> None:
    juniper = _claim(status="provisional", anchor_scope="juniper", subject_ref="user:juniper")
    relationship = _claim(status="provisional", anchor_scope="relationship", subject_ref="relationship:orion|juniper")
    repo = _repo_with(juniper, relationship)
    engine = PromotionEngine(repo)

    result = engine.evaluate(
        PromotionEvaluationRequestV1(artifact_ids=[juniper.artifact_id, relationship.artifact_id], target_status="canonical")
    )
    assert all(item.human_review_required for item in result.items)
    assert all(item.outcome == "escalated_hitl" for item in result.items)


def test_hitl_autonomy_goal_and_high_risk_mentor() -> None:
    goal_claim = _claim(status="provisional", claim_kind="goal_proposal_headline", anchor_scope="world")
    mentor = MentorProposalV1(
        anchor_scope="world",
        subject_ref="concept:alpha",
        status="proposed",
        authority="mentor_inferred",
        confidence=0.6,
        salience=0.7,
        novelty=0.5,
        risk_tier="high",
        observed_at=datetime.now(timezone.utc),
        provenance=_provenance(),
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="ontology_cleanup",
        proposal_type="merge",
        rationale="r",
    )
    repo = _repo_with(goal_claim, mentor)
    engine = PromotionEngine(repo)

    goal_result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[goal_claim.artifact_id], target_status="canonical"))
    mentor_result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[mentor.artifact_id], target_status="provisional"))

    assert goal_result.items[0].human_review_required is True
    assert mentor_result.items[0].human_review_required is True


def test_contradiction_gating_unresolved_blocks_and_resolved_allows() -> None:
    claim = _claim(status="proposed", anchor_scope="world", subject_ref="entity:beta")
    unresolved = ContradictionV1(
        anchor_scope="world",
        subject_ref="entity:beta",
        status="proposed",
        authority="local_inferred",
        confidence=0.8,
        salience=0.7,
        novelty=0.1,
        risk_tier="medium",
        observed_at=datetime.now(timezone.utc),
        provenance=_provenance(),
        contradiction_type="evidence_conflict",
        severity="high",
        resolution_status="open",
        involved_artifact_ids=[claim.artifact_id, "artifact-other"],
        summary="conflict",
    )
    resolved = unresolved.model_copy(update={"artifact_id": "artifact-resolved", "resolution_status": "resolved", "severity": "medium"})
    repo = _repo_with(claim, unresolved, resolved)
    engine = PromotionEngine(repo)

    blocked = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="provisional"))
    assert blocked.items[0].outcome == "blocked"

    # Remove unresolved contradiction and retest.
    repo2 = _repo_with(claim, resolved)
    engine2 = PromotionEngine(repo2)
    allowed = engine2.evaluate(PromotionEvaluationRequestV1(artifact_ids=[claim.artifact_id], target_status="provisional"))
    assert allowed.items[0].outcome == "promoted"


def test_lifecycle_dormant_revive_decay_retire_and_anchor_not_retired() -> None:
    now = datetime.now(timezone.utc)
    old_claim = _claim(status="proposed", subject_ref="entity:lifecycle")
    old_claim.observed_at = now - timedelta(days=31)
    old_claim.salience = 0.1

    dormant = evaluate_entity_lifecycle(
        EntityLifecycleEvaluationRequestV1(anchor_scope="world", subject_ref="entity:lifecycle"),
        artifacts=[old_claim],
    )
    assert dormant.lifecycle_action == "retire"

    revived = evaluate_entity_lifecycle(
        EntityLifecycleEvaluationRequestV1(anchor_scope="world", subject_ref="entity:lifecycle", current_state="dormant"),
        artifacts=[_claim(subject_ref="entity:lifecycle"), _claim(subject_ref="entity:lifecycle")],
    )
    assert revived.lifecycle_action == "revive"

    anchor = evaluate_entity_lifecycle(
        EntityLifecycleEvaluationRequestV1(anchor_scope="orion", subject_ref=None, current_state="active"),
        artifacts=[_claim(anchor_scope="orion", subject_ref="self:orion")],
    )
    assert anchor.lifecycle_action == "none"


def test_drift_aware_policy_concept_and_spark_conservative() -> None:
    concept_claim = _claim(status="provisional", claim_kind="concept_item", anchor_scope="world")
    spark = ReasoningSparkStateSnapshotV1(
        anchor_scope="orion",
        subject_ref="node:atlas",
        status="provisional",
        authority="sensed",
        confidence=0.7,
        salience=0.4,
        novelty=0.5,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance=_provenance(),
        dimensions={"coherence": 0.8},
    )
    repo = _repo_with(concept_claim, spark)
    engine = PromotionEngine(repo)

    concept_result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[concept_claim.artifact_id], target_status="canonical"))
    spark_result = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[spark.artifact_id], target_status="canonical"))

    assert concept_result.items[0].outcome == "blocked"
    assert "concept_translation_conservative_policy" in concept_result.items[0].reasons
    assert spark_result.items[0].outcome == "blocked"
    assert "spark_snapshot_not_canonical_identity" in spark_result.items[0].reasons
