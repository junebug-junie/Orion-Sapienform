from __future__ import annotations

from datetime import datetime, timezone

from orion.reasoning.adapters.spark_state import map_spark_snapshot_to_reasoning, map_spark_telemetry_to_reasoning
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload


def test_map_spark_snapshot_to_reasoning() -> None:
    snap = SparkStateSnapshotV1(
        source_service="orion-spark-introspector",
        source_node="atlas",
        producer_boot_id="boot-1",
        seq=7,
        snapshot_ts=datetime.now(timezone.utc),
        correlation_id="corr-1",
        phi={"coherence": 0.8, "novelty": 0.4},
        valence=0.7,
        arousal=0.5,
        dominance=0.6,
        metadata={"tensions": ["continuity_drift"], "attention_targets": ["project:orion_sapienform"]},
    )
    reasoning = map_spark_snapshot_to_reasoning(snap)
    assert reasoning.anchor_scope == "orion"
    assert reasoning.status == "provisional"
    assert reasoning.dimensions["coherence"] == 0.8
    assert reasoning.tensions == ["continuity_drift"]


def test_map_spark_telemetry_to_reasoning_from_embedded_snapshot() -> None:
    payload = SparkTelemetryPayload(
        correlation_id="corr-2",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "spark_state_snapshot": {
                "source_service": "orion-spark-introspector",
                "producer_boot_id": "boot-2",
                "seq": 8,
                "snapshot_ts": datetime.now(timezone.utc).isoformat(),
                "phi": {"coherence": 0.9},
            }
        },
    )
    mapped = map_spark_telemetry_to_reasoning(payload)
    assert mapped is not None
    assert mapped.dimensions["coherence"] == 0.9
