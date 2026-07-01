"""Execution-trajectory adapter — binds the substrate's execution "felt state".

Maps ``ExecutionTrajectoryProjectionV1`` (per-run trace state: verb, status,
step counts, and the pressure hints the execution lane derives — execution_load,
execution_friction, failure_pressure, reasoning_load) into substrate belief nodes
anchored to Orion, so the unified belief set contains beliefs about Orion's own
in-flight execution.

ctx-sourced, pure (no network, no DB): reads
``ctx['execution_trajectory_projection']`` as a model, dict, or JSON string, and
degrades to ``None`` when the key is absent or unparseable — never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.execution_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived in-flight execution state
_MAX_NODES = 20


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="execution_trajectory",
        source_channel="substrate.execution",
        producer="execution_lane_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> ExecutionTrajectoryProjectionV1 | None:
    try:
        if isinstance(raw, ExecutionTrajectoryProjectionV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return ExecutionTrajectoryProjectionV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return ExecutionTrajectoryProjectionV1.model_validate(raw)
    except Exception as exc:
        logger.debug("execution_ctx_adapter_parse_failed error=%s", exc)
    return None


def map_execution_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['execution_trajectory_projection']`` → execution belief nodes."""
    ctx = ctx if isinstance(ctx, dict) else {}
    projection = _coerce(ctx.get("execution_trajectory_projection"))
    if projection is None:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()

    runs = sorted(
        projection.runs.values(),
        key=lambda r: r.last_updated_at,
        reverse=True,
    )[:_MAX_NODES]

    nodes: list[Any] = []
    for run in runs:
        hints = dict(run.pressure_hints or {})
        salience = _clamp(max(hints.values(), default=0.0))
        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"execution:{run.verb}:{run.trace_id[:8]}",
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.7, salience=salience),
                metadata={
                    "source_kind": "execution_trajectory",
                    "trace_id": run.trace_id,
                    "verb": run.verb,
                    "status": run.status,
                    "step_count": run.step_count,
                    "pressure_hints": hints,
                },
            )
        )

    if not nodes:
        return None

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)
