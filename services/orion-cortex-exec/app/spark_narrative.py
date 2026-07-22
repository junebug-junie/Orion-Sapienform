from __future__ import annotations

# services/orion-cortex-exec/app/spark_narrative.py
from __future__ import annotations

from orion.schemas.telemetry.spark import SparkStateSnapshotV1


def spark_embodiment_hint(snapshot: SparkStateSnapshotV1) -> dict[str, str]:
    """
    Compact hint naming which real hardware node is most salient this tick
    (2026-07-12, inner-state unification Phase 3). Sourced from
    PhiIntrinsicRewardV1.dominant_node/dominant_node_reason (Phase 2), which
    is itself filtered to real hardware nodes only -- excludes synthetic
    pseudo-nodes and non-"node" attention target kinds (system/capability),
    both confirmed live to otherwise pollute this signal.

    "none" when no qualifying node is present this tick -- never fabricated.
    """
    return {
        "dominant_node": snapshot.dominant_node or "none",
        "dominant_node_reason": snapshot.dominant_node_reason or "no node currently salient",
    }


def spark_embodiment_narrative(snapshot: SparkStateSnapshotV1) -> str:
    node = snapshot.dominant_node
    reason = snapshot.dominant_node_reason

    if node is None:
        return (
            "No single hardware node is currently the most salient part of my body -- "
            "attention is either quiet or spread across the mesh."
        )

    node_label = node.removeprefix("node:")
    reason_clause = f" ({reason})" if reason else ""
    return (
        f"Right now, {node_label} is the most salient part of my body{reason_clause}. "
        "This names a real machine in the mesh (Atlas, Circe, Athena, or Prometheus), "
        "not a mood -- use it to ground any claim about where load or attention currently sits."
    )
