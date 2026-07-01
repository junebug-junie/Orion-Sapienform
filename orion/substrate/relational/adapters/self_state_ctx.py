"""Self-state adapter — binds Orion's self-model into the unification layer.

Higher-order rung: maps ``SelfStateV1`` (Orion's model of its own condition —
coherence, uncertainty, agency-readiness, the various pressure dimensions) into
substrate belief nodes so the unified belief set contains beliefs *about itself*,
not only about the world.

It also carries each dimension's standing ``prediction_error`` (how wrong the
self-model's last one-step prediction was) into node metadata. When the
unification layer materializes this producer's record into the durable substrate
store, those nodes feed ``prediction_error_pressure`` in the dynamics engine —
closing the self-modeling feedback loop on durable self-model nodes.

ctx-sourced, no network: reads ``ctx['self_state']`` as a ``SelfStateV1``, a dict,
or a JSON string.
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
from orion.schemas.self_state import SelfStateV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.self_state_ctx")

_TIER_RANK = 2  # graphdb_durable-equivalent: self-model is a trusted internal lane


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="self_state",
        source_channel="orion:self_state",
        producer="self_state_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> SelfStateV1 | None:
    try:
        if isinstance(raw, SelfStateV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return SelfStateV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return SelfStateV1.model_validate(raw)
    except Exception as exc:
        logger.debug("self_state_adapter_parse_failed error=%s", exc)
    return None


def map_self_state_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['self_state']`` → substrate self-model belief nodes (anchor=orion)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    state = _coerce(ctx.get("self_state") or ctx.get("self_state_json"))
    if state is None:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()
    nodes: list[Any] = []

    for dim_id, dim in state.dimensions.items():
        prediction_error = float(state.prediction_error_scores.get(dim_id, 0.0) or 0.0)
        trajectory = float(state.dimension_trajectory.get(dim_id, 0.0) or 0.0)
        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"self:{dim_id}",
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(
                    confidence=max(0.0, min(1.0, dim.confidence)),
                    salience=max(0.0, min(1.0, dim.score)),
                ),
                metadata={
                    "source_kind": "self_state",
                    "self_dimension_id": dim_id,
                    "score": round(max(0.0, min(1.0, dim.score)), 6),
                    "trajectory": round(trajectory, 6),
                    # standing surprise on this self-dimension; seeds dynamics pressure
                    "prediction_error": round(max(0.0, min(1.0, prediction_error)), 6),
                },
            )
        )

    if not nodes:
        return None

    # An anchor node summarizing overall condition, carrying max surprise so the
    # self as a whole gains pressure when any dimension is badly mispredicted.
    overall_error = float(getattr(state, "overall_surprise", 0.0) or 0.0)
    nodes.append(
        ConceptNodeV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            label="self:overall_condition",
            temporal=temporal,
            provenance=prov,
            signals=SubstrateSignalBundleV1(
                confidence=max(0.0, min(1.0, state.overall_confidence)),
                salience=max(0.0, min(1.0, state.overall_intensity)),
            ),
            metadata={
                "source_kind": "self_state",
                "overall_condition": state.overall_condition,
                "trajectory_condition": state.trajectory_condition,
                "prediction_error": round(max(0.0, min(1.0, overall_error)), 6),
            },
        )
    )

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)
