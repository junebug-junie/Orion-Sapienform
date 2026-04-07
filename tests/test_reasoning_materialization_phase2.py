from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.core.schemas.concept_induction import ConceptCluster, ConceptItem, ConceptProfile, ConceptProfileDelta
from orion.core.schemas.reasoning import ClaimV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning.adapters.autonomy import map_autonomy_state_to_reasoning
from orion.reasoning.adapters.concept_induction import map_concept_delta_to_reasoning, map_concept_profile_to_reasoning
from orion.reasoning.materializer import ReasoningMaterializer
from orion.reasoning.repository import InMemoryReasoningRepository


def _claim() -> ClaimV1:
    return ClaimV1(
        anchor_scope="orion",
        subject_ref="concept:test",
        status="proposed",
        authority="local_inferred",
        confidence=0.7,
        salience=0.4,
        novelty=0.2,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["unit:test"],
            "source_channel": "orion:test",
            "source_kind": "unit",
            "producer": "pytest",
        },
        claim_text="test claim",
    )


def test_write_request_and_repo_roundtrip() -> None:
    repo = InMemoryReasoningRepository()
    mat = ReasoningMaterializer(repo)
    request = ReasoningWriteRequestV1(
        context=ReasoningWriteContextV1(
            source_family="manual",
            source_kind="unit",
            source_channel="orion:test",
            producer="pytest",
        ),
        artifacts=[_claim(), _claim()],
        idempotency_key="k1",
    )

    result = mat.materialize(request)
    assert result.accepted is True
    assert result.stored_count == 2
    assert len(repo.list_latest(limit=10)) == 2

    deduped = mat.materialize(request)
    assert deduped.deduped is True


def test_write_request_rejects_empty_artifacts() -> None:
    try:
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="unit",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=[],
        )
        raise AssertionError("Expected ValidationError")
    except ValidationError:
        pass


def test_concept_induction_adapter_conservative_status() -> None:
    profile = ConceptProfile(
        subject="orion",
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
        concepts=[
            ConceptItem(concept_id="c1", label="continuity", type="motif", salience=0.8, confidence=0.9),
        ],
        clusters=[ConceptCluster(cluster_id="cluster-1", label="core", concept_ids=["c1"], cohesion_score=0.7)],
    )
    artifacts = map_concept_profile_to_reasoning(profile)
    assert any(a.artifact_type == "claim" for a in artifacts)
    assert all(a.status in {"proposed", "provisional"} for a in artifacts)
    assert all(a.anchor_scope == "orion" for a in artifacts)

    delta = ConceptProfileDelta(profile_id=profile.profile_id, from_rev=1, to_rev=2, added=["c1"])
    delta_artifacts = map_concept_delta_to_reasoning(delta, subject="orion", observed_at=datetime.now(timezone.utc))
    assert delta_artifacts
    assert all(a.claim_kind == "concept_delta" for a in delta_artifacts)


def test_autonomy_adapter_goal_headlines_stay_proposed() -> None:
    state = AutonomyStateV1(
        subject="relationship",
        model_layer="relationship-model",
        entity_id="relationship:orion|juniper",
        identity_summary="Relationship trust is stable",
        dominant_drive="relational",
        source="unit",
        generated_at=datetime.now(timezone.utc),
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="a1",
                goal_statement="Ask a clarifying question",
                drive_origin="relational",
                priority=0.8,
                proposal_signature="sig-1",
            )
        ],
    )

    artifacts = map_autonomy_state_to_reasoning(state)
    assert artifacts[0].anchor_scope == "relationship"
    assert any(a.claim_kind == "goal_proposal_headline" and a.status == "proposed" for a in artifacts)
