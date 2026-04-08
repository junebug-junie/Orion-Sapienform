from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.reasoning import ReasoningProvenanceV1, ReasoningSparkStateSnapshotV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload


def _ensure_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def normalize_legacy_spark_snapshot(snapshot: SparkStateSnapshotV1) -> SparkSourceSnapshotV1:
    dimensions = dict(snapshot.phi)
    dimensions.setdefault("valence", snapshot.valence)
    dimensions.setdefault("arousal", snapshot.arousal)
    dimensions.setdefault("dominance", snapshot.dominance)

    metadata = snapshot.metadata or {}
    tensions = [str(t) for t in metadata.get("tensions", []) if str(t).strip()]
    attention_targets = [str(t) for t in metadata.get("attention_targets", []) if str(t).strip()]

    return SparkSourceSnapshotV1(
        source_service=snapshot.source_service,
        source_node=snapshot.source_node,
        snapshot_ts=_ensure_datetime(snapshot.snapshot_ts),
        source_snapshot_id=snapshot.idempotency_key,
        correlation_id=snapshot.correlation_id,
        source_kind="spark.legacy.snapshot.v1",
        dimensions=dimensions,
        tensions=tensions,
        attention_targets=attention_targets,
        metadata=metadata,
    )


def map_canonical_spark_to_reasoning(snapshot: SparkSourceSnapshotV1, *, producer: str = "spark_adapter") -> ReasoningSparkStateSnapshotV1:
    return ReasoningSparkStateSnapshotV1(
        anchor_scope="orion",
        subject_ref=f"node:{snapshot.source_node}" if snapshot.source_node else "node:unknown",
        status="provisional",
        authority="sensed",
        confidence=0.8,
        salience=0.6,
        novelty=0.3,
        risk_tier="low",
        observed_at=snapshot.snapshot_ts,
        provenance=ReasoningProvenanceV1(
            evidence_refs=[snapshot.source_snapshot_id],
            source_channel="orion:spark:state:snapshot",
            source_kind=snapshot.source_kind,
            producer=producer,
            correlation_id=snapshot.correlation_id,
        ),
        dimensions=dict(snapshot.dimensions),
        tensions=list(snapshot.tensions),
        attention_targets=list(snapshot.attention_targets),
        trend_window_s=max(0, int((snapshot.metadata or {}).get("valid_for_ms", 15000) / 1000)),
    )


def map_spark_snapshot_to_reasoning(snapshot: SparkStateSnapshotV1 | SparkSourceSnapshotV1, *, producer: str = "spark_adapter") -> ReasoningSparkStateSnapshotV1:
    if isinstance(snapshot, SparkSourceSnapshotV1):
        canonical = snapshot
    else:
        canonical = normalize_legacy_spark_snapshot(snapshot)
    return map_canonical_spark_to_reasoning(canonical, producer=producer)


def map_spark_telemetry_to_reasoning(payload: SparkTelemetryPayload, *, producer: str = "spark_adapter") -> ReasoningSparkStateSnapshotV1 | None:
    if payload.state_snapshot is not None:
        return map_spark_snapshot_to_reasoning(payload.state_snapshot, producer=producer)

    raw = payload.metadata.get("spark_state_snapshot") if isinstance(payload.metadata, dict) else None
    if isinstance(raw, dict):
        snapshot = SparkStateSnapshotV1.model_validate(raw)
        return map_spark_snapshot_to_reasoning(snapshot, producer=producer)
    return None
