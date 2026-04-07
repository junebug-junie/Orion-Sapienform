from __future__ import annotations

from typing import Any, Mapping

from orion.core.schemas.cognitive_substrate import EventNodeV1, NodeRefV1, StateSnapshotNodeV1, SubstrateEdgeV1, SubstrateGraphRecordV1, TensionNodeV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1

from ._common import make_provenance, make_temporal


def map_spark_source_snapshot_to_substrate(
    *,
    snapshot: SparkSourceSnapshotV1,
    anchor_scope: str = "orion",
    subject_ref: str | None = None,
) -> SubstrateGraphRecordV1:
    resolved_subject_ref = subject_ref or str(snapshot.metadata.get("subject_ref") or "orion")
    state_node_id = f"sub-state-spark-source-{snapshot.source_snapshot_id}"
    nodes = [
        StateSnapshotNodeV1(
            node_id=state_node_id,
            anchor_scope=anchor_scope,
            subject_ref=resolved_subject_ref,
            temporal=make_temporal(observed_at=snapshot.snapshot_ts),
            provenance=make_provenance(
                source_kind="spark.source_snapshot",
                source_channel=snapshot.source_service,
                producer="spark_adapter",
                correlation_id=snapshot.correlation_id,
            ),
            dimensions={k: float(v) for k, v in snapshot.dimensions.items()},
            snapshot_source="spark_source_snapshot",
            metadata={"attention_targets": list(snapshot.attention_targets), "source_snapshot_id": snapshot.source_snapshot_id},
        )
    ]
    edges = []

    # Conservative tension mapping: only explicit tension labels map to Tension nodes.
    for tension in snapshot.tensions:
        tension_node_id = f"sub-tension-spark-{snapshot.source_snapshot_id}-{tension}"
        nodes.append(
            TensionNodeV1(
                node_id=tension_node_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=snapshot.snapshot_ts),
                provenance=make_provenance(
                    source_kind="spark.source_snapshot.tension",
                    source_channel=snapshot.source_service,
                    producer="spark_adapter",
                    correlation_id=snapshot.correlation_id,
                ),
                tension_kind=tension,
                intensity=0.5,
                signals={"confidence": 0.5, "salience": 0.5},
            )
        )
        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(node_id=tension_node_id, node_kind="tension"),
                target=NodeRefV1(node_id=state_node_id, node_kind="state_snapshot"),
                predicate="associated_with",
                temporal=make_temporal(observed_at=snapshot.snapshot_ts),
                provenance=make_provenance(
                    source_kind="spark.source_snapshot.tension",
                    source_channel=snapshot.source_service,
                    producer="spark_adapter",
                ),
                confidence=0.5,
                salience=0.5,
            )
        )

    return SubstrateGraphRecordV1(
        graph_id=f"sub-graph-spark-source-{snapshot.source_snapshot_id}",
        anchor_scope=anchor_scope,
        subject_ref=resolved_subject_ref,
        nodes=nodes,
        edges=edges,
        created_at=snapshot.snapshot_ts,
    )


def map_spark_state_snapshot_to_substrate(
    *,
    snapshot: SparkStateSnapshotV1,
    anchor_scope: str = "orion",
    subject_ref: str | None = None,
) -> SubstrateGraphRecordV1:
    metadata: Mapping[str, Any] = snapshot.metadata or {}
    resolved_subject_ref = subject_ref or str(metadata.get("subject_ref") or "orion")
    node_id = f"sub-state-spark-{snapshot.producer_boot_id}-{snapshot.seq}"
    nodes = [
        StateSnapshotNodeV1(
            node_id=node_id,
            anchor_scope=anchor_scope,
            subject_ref=resolved_subject_ref,
            temporal=make_temporal(observed_at=snapshot.snapshot_ts),
            provenance=make_provenance(
                source_kind="spark.state_snapshot",
                source_channel=snapshot.source_service,
                producer="spark_adapter",
                correlation_id=snapshot.correlation_id,
            ),
            dimensions={
                "valence": float(snapshot.valence),
                "arousal": float(snapshot.arousal),
                "dominance": float(snapshot.dominance),
                **{f"phi.{k}": float(v) for k, v in snapshot.phi.items()},
            },
            snapshot_source="spark_state_snapshot",
            signals={"confidence": 0.7, "salience": max(float(snapshot.arousal), float(snapshot.valence))},
            metadata={"producer_boot_id": snapshot.producer_boot_id, "seq": snapshot.seq, "trace_mode": snapshot.trace_mode, "trace_verb": snapshot.trace_verb},
        )
    ]
    edges = []

    transition = metadata.get("transition_event")
    if isinstance(transition, str) and transition.strip():
        event_id = f"sub-event-spark-{snapshot.producer_boot_id}-{snapshot.seq}"
        nodes.append(
            EventNodeV1(
                node_id=event_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=snapshot.snapshot_ts),
                provenance=make_provenance(
                    source_kind="spark.state_snapshot.transition",
                    source_channel=snapshot.source_service,
                    producer="spark_adapter",
                    correlation_id=snapshot.correlation_id,
                ),
                event_type="spark_transition",
                summary=transition,
                signals={"confidence": 0.6, "salience": 0.6},
            )
        )
        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(node_id=event_id, node_kind="event"),
                target=NodeRefV1(node_id=node_id, node_kind="state_snapshot"),
                predicate="observed_in",
                temporal=make_temporal(observed_at=snapshot.snapshot_ts),
                provenance=make_provenance(
                    source_kind="spark.state_snapshot.transition",
                    source_channel=snapshot.source_service,
                    producer="spark_adapter",
                ),
                confidence=0.6,
                salience=0.6,
            )
        )

    return SubstrateGraphRecordV1(
        graph_id=f"sub-graph-spark-state-{snapshot.producer_boot_id}-{snapshot.seq}",
        anchor_scope=anchor_scope,
        subject_ref=resolved_subject_ref,
        nodes=nodes,
        edges=edges,
        created_at=snapshot.snapshot_ts,
    )
