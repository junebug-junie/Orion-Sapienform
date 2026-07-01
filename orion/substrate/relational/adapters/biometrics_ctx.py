"""Biometrics-pressure adapter — binds the substrate's biometric "felt state".

Maps ``ActiveNodePressureProjectionV1`` (per-node availability plus the active /
suppressed pressures and capability impacts the biometrics lane derives) into
substrate belief nodes anchored to Orion, so the unified belief set contains
beliefs about which parts of Orion's own body are under load.

Only nodes that are actually *felt* (pressure_score > 0 or non-empty
active_pressures) are emitted.

ctx-sourced, pure (no network, no DB): reads
``ctx['active_node_pressure_projection']`` as a model, dict, or JSON string, and
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
from orion.schemas.biometrics_projection import ActiveNodePressureProjectionV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.biometrics_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived biometric pressure state
_MAX_NODES = 20


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="biometrics_pressure",
        source_channel="substrate.biometrics",
        producer="biometrics_lane_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> ActiveNodePressureProjectionV1 | None:
    try:
        if isinstance(raw, ActiveNodePressureProjectionV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return ActiveNodePressureProjectionV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return ActiveNodePressureProjectionV1.model_validate(raw)
    except Exception as exc:
        logger.debug("biometrics_ctx_adapter_parse_failed error=%s", exc)
    return None


def map_biometrics_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['active_node_pressure_projection']`` → biometric belief nodes."""
    ctx = ctx if isinstance(ctx, dict) else {}
    projection = _coerce(ctx.get("active_node_pressure_projection"))
    if projection is None:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()

    felt = [
        node
        for node in projection.nodes.values()
        if node.pressure_score > 0.0 or node.active_pressures
    ]
    felt = sorted(felt, key=lambda n: n.pressure_score, reverse=True)[:_MAX_NODES]

    nodes: list[Any] = []
    for node in felt:
        salience = _clamp(node.pressure_score)
        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"biometrics:{node.node_id}",
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.7, salience=salience),
                metadata={
                    "source_kind": "biometrics_pressure",
                    "node_id": node.node_id,
                    "availability_status": node.availability_status,
                    "pressure_score": round(node.pressure_score, 6),
                    "active_pressures": list(node.active_pressures[:20]),
                    "capability_impacts": list(node.capability_impacts[:20]),
                },
            )
        )

    if not nodes:
        return None

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)
