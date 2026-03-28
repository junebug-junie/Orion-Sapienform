from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.concept_induction import (
    ConceptCluster,
    ConceptItem,
    ConceptProfile,
    StateEstimate,
)
from orion.spark.concept_induction.rdf_materialization import build_concept_profile_rdf_request


def _fixture_profile() -> ConceptProfile:
    return ConceptProfile(
        profile_id="profile-abc",
        subject="orion",
        revision=7,
        created_at=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        window_start=datetime(2026, 3, 24, 10, 15, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        concepts=[
            ConceptItem(
                concept_id="concept-1",
                label="coherence",
                type="motif",
                salience=0.9,
                confidence=0.8,
                aliases=["consistency"],
                metadata={"source": "test"},
            )
        ],
        clusters=[
            ConceptCluster(
                cluster_id="cluster-1",
                label="core",
                summary="Core concepts",
                concept_ids=["concept-1"],
                cohesion_score=0.75,
                metadata={"size": "1"},
            )
        ],
        state_estimate=StateEstimate(
            dimensions={"novelty": 0.2},
            trend={"novelty": -0.1},
            confidence=0.6,
            window_start=datetime(2026, 3, 24, 10, 15, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 25, 10, 15, tzinfo=timezone.utc),
        ),
        metadata={"algorithm": "concept_induction.v1"},
    )


def test_build_concept_profile_rdf_request_includes_queryable_structure() -> None:
    req = build_concept_profile_rdf_request(
        profile=_fixture_profile(),
        correlation_id="corr-123",
        writer_service="orion-spark-concept-induction",
        writer_version="0.1.0",
    )

    assert req.kind == "spark.concept_profile.graph.v1"
    assert req.graph == "http://conjourney.net/graph/spark/concept-profile"
    assert req.payload["source_kind"] == "spark_concept_profile"
    assert "SparkConceptProfile" in req.triples
    assert "hasConcept" in req.triples
    assert "SparkConceptCluster" in req.triples
    assert "hasStateEstimate" in req.triples
    assert "MaterializationProvenance" in req.triples
    assert "concept-1" in req.triples


def test_build_concept_profile_rdf_request_provenance_payload_fields() -> None:
    req = build_concept_profile_rdf_request(
        profile=_fixture_profile(),
        correlation_id="corr-777",
        writer_service="orion-spark-concept-induction",
        writer_version="0.1.1",
    )

    assert req.payload["subject"] == "orion"
    assert req.payload["revision"] == 7
    assert req.payload["concept_count"] == 1
    assert req.payload["cluster_count"] == 1
    assert req.payload["state_estimate_present"] is True
    assert req.payload["schema_kind"] == "spark.concept_profile.graph.v1"
    assert req.payload["correlation_id"] == "corr-777"
    assert req.payload["writer_service"] == "orion-spark-concept-induction"
