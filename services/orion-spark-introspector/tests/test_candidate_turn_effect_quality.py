from __future__ import annotations

import pytest

from orion.schemas.telemetry.spark_candidate import SparkCandidateV1

from app.worker import _candidate_quality


@pytest.mark.parametrize(
    "spark_meta,expected",
    [
        ({}, 0),
        ({"mode": "brain"}, 0),
        ({"phi_before": {"coherence": 0.5}}, 1),
        ({"turn_effect": {"turn": {"novelty": 0.1}}}, 1),
        ({"turn_effect_evidence": {"phi_before": {"coherence": 0.4}}}, 1),
    ],
)
def test_candidate_quality_treats_turn_effect_as_rich(spark_meta, expected) -> None:
    assert _candidate_quality(spark_meta) == expected


def test_spark_candidate_payload_accepts_turn_effect_only_meta() -> None:
    candidate = SparkCandidateV1.model_validate(
        {
            "trace_id": "trace-1",
            "source": "hub_http",
            "prompt": "hello",
            "response": "hi",
            "spark_meta": {
                "turn_effect": {"turn": {"coherence": -0.05}},
                "turn_effect_status": "present",
            },
        }
    )
    assert candidate.spark_meta["turn_effect"]["turn"]["coherence"] == -0.05
