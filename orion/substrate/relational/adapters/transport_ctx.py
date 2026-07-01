"""Transport-bus adapter — binds the substrate's transport "felt state".

Maps ``TransportBusProjectionV1`` (per-bus health, delivery confidence, and the
family of transport pressures — transport_pressure, backpressure, contract,
reliability, stream-depth) into substrate belief nodes anchored to Orion, so the
unified belief set contains beliefs about how Orion's own message bus is faring.

ctx-sourced, pure (no network, no DB): reads
``ctx['transport_bus_projection']`` as a model, dict, or JSON string, and
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
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.transport_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived transport bus state
_MAX_NODES = 20


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="transport_bus",
        source_channel="substrate.transport",
        producer="transport_lane_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> TransportBusProjectionV1 | None:
    try:
        if isinstance(raw, TransportBusProjectionV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return TransportBusProjectionV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return TransportBusProjectionV1.model_validate(raw)
    except Exception as exc:
        logger.debug("transport_ctx_adapter_parse_failed error=%s", exc)
    return None


def map_transport_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['transport_bus_projection']`` → transport belief nodes."""
    ctx = ctx if isinstance(ctx, dict) else {}
    projection = _coerce(ctx.get("transport_bus_projection"))
    if projection is None:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()

    buses = sorted(
        projection.buses.values(),
        key=lambda b: b.transport_pressure,
        reverse=True,
    )[:_MAX_NODES]

    nodes: list[Any] = []
    for bus in buses:
        salience = _clamp(
            max(
                bus.transport_pressure,
                bus.backpressure,
                bus.reliability_pressure,
                bus.contract_pressure,
                bus.stream_depth_pressure,
            )
        )
        confidence = _clamp(bus.delivery_confidence) or 0.7
        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                subject_ref="entity:orion",
                label=f"transport:{bus.node_id or bus.target_id}",
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=confidence, salience=salience),
                metadata={
                    "source_kind": "transport_bus",
                    "target_id": bus.target_id,
                    "node_id": bus.node_id,
                    "bus_health": round(bus.bus_health, 6),
                    "delivery_confidence": round(bus.delivery_confidence, 6),
                    "transport_pressure": round(bus.transport_pressure, 6),
                },
            )
        )

    if not nodes:
        return None

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)
