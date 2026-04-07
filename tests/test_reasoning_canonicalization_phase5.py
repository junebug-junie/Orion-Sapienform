from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.concept_induction import ConceptCluster, ConceptItem, ConceptProfile
from orion.core.schemas.reasoning import ConceptV1, ReasoningSparkStateSnapshotV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_policy import PromotionEvaluationRequestV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryRequestV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.reasoning.adapters.concept_induction import map_concept_profile_to_reasoning
from orion.reasoning.adapters.spark_state import (
    map_canonical_spark_to_reasoning,
    map_spark_snapshot_to_reasoning,
    normalize_legacy_spark_snapshot,
)
from orion.reasoning.promotion import PromotionEngine
from orion.reasoning.repository import InMemoryReasoningRepository
from orion.reasoning.summary import ReasoningSummaryCompiler
from orion.schemas.telemetry.spark import SparkStateSnapshotV1


def _prov() -> dict:
    return {
        "evidence_refs": ["ev:1"],
        "source_channel": "orion:test",
        "source_kind": "unit",
        "producer": "pytest",
    }


def test_concept_v1_validation_and_adapter_emits_canonical_concepts() -> None:
    with pytest.raises(ValidationError):
        ConceptV1(
            anchor_scope="orion",
            authority="local_inferred",
            observed_at=datetime.now(timezone.utc),
            provenance=_prov(),
            concept_id="c1",
            label="",
        )

    profile = ConceptProfile(
        subject="orion",
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
        concepts=[ConceptItem(concept_id="c1", label="continuity", type="identity", salience=0.8, confidence=0.9)],
        clusters=[ConceptCluster(cluster_id="cluster-1", label="core", concept_ids=["c1"], cohesion_score=0.7)],
    )
    artifacts = map_concept_profile_to_reasoning(profile, include_legacy_claims=False)
    concepts = [a for a in artifacts if isinstance(a, ConceptV1)]
    assert concepts
    assert concepts[0].label == "continuity"
    assert concepts[0].cluster_refs == ["cluster-1"]


def test_spark_canonicalization_normalizes_legacy_and_embedded_shapes() -> None:
    legacy = SparkStateSnapshotV1(
        source_service="spark",
        source_node="atlas",
        producer_boot_id="boot",
        seq=10,
        snapshot_ts=datetime.now(timezone.utc),
        phi={"coherence": 0.8},
    )
    canonical = normalize_legacy_spark_snapshot(legacy)
    assert isinstance(canonical, SparkSourceSnapshotV1)
    assert canonical.dimensions["coherence"] == 0.8

    reasoned = map_spark_snapshot_to_reasoning(legacy)
    assert isinstance(reasoned, ReasoningSparkStateSnapshotV1)

    reasoned2 = map_canonical_spark_to_reasoning(canonical)
    assert reasoned2.observed_at == canonical.snapshot_ts


def test_materialization_accepts_canonical_concept_and_spark_with_compat() -> None:
    concept = ConceptV1(
        anchor_scope="orion",
        subject_ref="concept:c1",
        status="provisional",
        authority="local_inferred",
        confidence=0.9,
        salience=0.8,
        novelty=0.3,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance=_prov(),
        concept_id="c1",
        label="continuity",
        concept_type="identity",
    )
    spark = map_canonical_spark_to_reasoning(
        SparkSourceSnapshotV1(
            source_service="spark",
            source_node="atlas",
            snapshot_ts=datetime.now(timezone.utc),
            source_snapshot_id="id:1",
            dimensions={"coherence": 0.9},
        )
    )
    repo = InMemoryReasoningRepository()
    result = repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(source_family="manual", source_kind="unit", source_channel="orion:test", producer="pytest"),
            artifacts=[concept, spark],
        )
    )
    assert result.stored_count == 2


def test_promotion_and_summary_use_canonical_concept_semantics() -> None:
    concept = ConceptV1(
        anchor_scope="world",
        subject_ref="concept:c1",
        status="provisional",
        authority="local_inferred",
        confidence=0.85,
        salience=0.75,
        novelty=0.2,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance=_prov(),
        concept_id="c1",
        label="continuity",
        concept_type="identity",
    )
    weak_concept = concept.model_copy(update={"artifact_id": "concept-weak", "confidence": 0.6, "salience": 0.2, "label": "weak"})

    repo = InMemoryReasoningRepository()
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(source_family="manual", source_kind="unit", source_channel="orion:test", producer="pytest"),
            artifacts=[concept, weak_concept],
        )
    )

    engine = PromotionEngine(repo)
    promoted = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[concept.artifact_id], target_status="canonical"))
    blocked = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[weak_concept.artifact_id], target_status="canonical"))
    assert promoted.items[0].outcome == "promoted"
    assert blocked.items[0].outcome == "blocked"

    summary = ReasoningSummaryCompiler(repo).compile(ReasoningSummaryRequestV1(anchor_scope="world"))
    labels = [c.label for c in summary.active_concepts]
    assert "continuity" in labels
    assert "weak" not in labels
