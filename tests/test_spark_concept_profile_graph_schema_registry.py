from __future__ import annotations

from orion.schemas.registry import _REGISTRY
from orion.schemas.spark_concept_graph import SparkConceptProfileGraphMaterializationV1


def test_spark_concept_profile_graph_schema_registered() -> None:
    assert _REGISTRY["SparkConceptProfileGraphMaterializationV1"] is SparkConceptProfileGraphMaterializationV1


def test_spark_concept_profile_graph_schema_validates_payload_shape() -> None:
    model = SparkConceptProfileGraphMaterializationV1.model_validate(
        {
            "profile_id": "profile-1",
            "subject": "orion",
            "revision": 1,
            "produced_at": "2026-03-26T10:00:00+00:00",
            "window_start": "2026-03-26T09:00:00+00:00",
            "window_end": "2026-03-26T10:00:00+00:00",
            "concept_count": 3,
            "cluster_count": 1,
            "state_estimate_present": True,
            "correlation_id": "corr-1",
            "writer_service": "orion-spark-concept-induction",
            "writer_version": "0.1.0",
        }
    )
    assert model.schema_kind == "spark.concept_profile.graph.v1"
    assert model.source_kind == "spark_concept_profile"
