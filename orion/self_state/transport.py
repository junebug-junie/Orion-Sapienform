from __future__ import annotations

from orion.schemas.field_state import FieldStateV1
from orion.self_state.scoring import clamp01


def transport_channel_hints(field: FieldStateV1) -> dict[str, float]:
    node = field.node_vectors.get("node:athena") or {}
    cap = field.capability_vectors.get("capability:transport") or {}
    return {
        "bus_health": clamp01(float(node.get("bus_health", cap.get("available_capacity", 0.5)))),
        "delivery_confidence": clamp01(float(node.get("delivery_confidence", cap.get("confidence", 0.5)))),
        "transport_pressure": clamp01(
            max(float(node.get("transport_pressure", 0.0)), float(cap.get("pressure", 0.0)))
        ),
        "contract_pressure": clamp01(
            max(float(node.get("contract_pressure", 0.0)), float(cap.get("contract_pressure", 0.0)))
        ),
        "reliability_pressure": clamp01(
            max(float(node.get("reliability_pressure", 0.0)), float(cap.get("reliability_pressure", 0.0)))
        ),
    }


def transport_integrity_score(hints: dict[str, float]) -> float:
    return clamp01(
        min(
            hints.get("bus_health", 0.5),
            hints.get("delivery_confidence", 0.5),
            1.0 - hints.get("transport_pressure", 0.0),
            1.0 - hints.get("reliability_pressure", 0.0),
            1.0 - hints.get("contract_pressure", 0.0) * 0.5,
        )
    )


def transport_summary_labels(hints: dict[str, float], integrity: float) -> list[str]:
    labels: list[str] = []
    if hints.get("bus_health", 0.0) >= 0.9 and hints.get("transport_pressure", 0.0) < 0.2:
        labels.append("transport_healthy")
    if hints.get("contract_pressure", 0.0) >= 0.7:
        labels.append("transport_contract_drift")
    if hints.get("transport_pressure", 0.0) >= 0.6:
        labels.append("transport_backpressured")
    if integrity < 0.4:
        labels.append("transport_degraded")
    return labels
