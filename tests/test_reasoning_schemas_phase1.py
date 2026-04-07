from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.reasoning import (
    ClaimV1,
    ContradictionV1,
    MentorProposalV1,
    PromotionDecisionV1,
    RelationV1,
    SparkStateSnapshotV1,
    VerbEvaluationV1,
)


def _base_kwargs() -> dict:
    return {
        "anchor_scope": "orion",
        "subject_ref": "project:orion_sapienform",
        "authority": "local_inferred",
        "observed_at": datetime.now(timezone.utc),
        "provenance": {
            "evidence_refs": ["chat:turn:1"],
            "source_channel": "orion:test",
            "source_kind": "unit-test",
            "producer": "pytest",
        },
    }


def test_claim_and_relation_validate() -> None:
    claim = ClaimV1(**_base_kwargs(), claim_text="Orion continuity pressure is elevated")
    relation = RelationV1(
        **_base_kwargs(),
        source_ref=claim.artifact_id,
        target_ref="concept:continuity_pressure",
        relation_type="supports",
    )

    assert claim.artifact_type == "claim"
    assert relation.artifact_type == "relation"


def test_contradiction_requires_multiple_artifacts() -> None:
    with pytest.raises(ValidationError):
        ContradictionV1(
            **_base_kwargs(),
            contradiction_type="logical_conflict",
            severity="high",
            summary="single reference should fail",
            involved_artifact_ids=["artifact-only-one"],
        )


def test_mentor_proposal_is_proposed_and_mentor_inferred() -> None:
    proposal = MentorProposalV1(
        **{**_base_kwargs(), "authority": "mentor_inferred"},
        mentor_provider="openai",
        mentor_model="gpt-x",
        task_type="ontology_cleanup",
        proposal_type="merge_duplicate_concepts",
        rationale="duplicate labels collide",
    )

    assert proposal.status == "proposed"
    assert proposal.authority == "mentor_inferred"


def test_promotion_decision_is_canonical() -> None:
    decision = PromotionDecisionV1(
        **_base_kwargs(),
        status="canonical",
        proposal_artifact_id="artifact-proposal-1",
        action="accepted",
        rationale="strong evidence",
        decided_by="human:juniper",
    )

    assert decision.status == "canonical"


def test_verb_eval_depth_bounds() -> None:
    with pytest.raises(ValidationError):
        VerbEvaluationV1(
            **_base_kwargs(),
            verb_name="reflect",
            execution_depth=4,
        )


def test_spark_state_snapshot_and_time_window_validation() -> None:
    snap = SparkStateSnapshotV1(
        **_base_kwargs(),
        dimensions={"coherence": 0.71, "novelty": 0.62},
        tensions=["continuity_drift"],
    )
    assert snap.artifact_type == "spark_state_snapshot"

    with pytest.raises(ValidationError):
        ClaimV1(
            **_base_kwargs(),
            claim_text="invalid time window",
            valid_from=datetime(2026, 1, 2, tzinfo=timezone.utc),
            valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
