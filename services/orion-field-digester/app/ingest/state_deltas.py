from __future__ import annotations

import math
from dataclasses import dataclass

from orion.schemas.state_delta import StateDeltaV1

GPU_NODES = {"atlas", "circe"}


@dataclass(frozen=True)
class Perturbation:
    node_id: str
    channel: str
    intensity: float
    label: str
    mode: str = "add"  # "add" accumulates; "replace" sets the channel to exactly intensity


def _node_key(raw: str) -> str:
    nid = raw.strip().lower()
    return nid if nid.startswith("node:") else f"node:{nid}"


def delta_to_perturbations(delta: StateDeltaV1) -> list[Perturbation]:
    if delta.operation == "noop":
        return []
    after = delta.after or {}
    node_id = _node_key(str(after.get("node_id") or delta.target_id))
    out: list[Perturbation] = []

    if delta.target_kind == "active_node_pressure":
        score = float(after.get("pressure_score", 0.0))
        pressures = list(after.get("active_pressures") or [])
        if "strain" in pressures:
            channel = "gpu_pressure" if node_id.replace("node:", "") in GPU_NODES else "cpu_pressure"
            out.append(Perturbation(node_id=node_id, channel=channel, intensity=score, label=delta.delta_id))
        if "availability" in pressures:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="availability",
                    intensity=max(0.0, 1.0 - min(1.0, score + 0.2)),
                    label=delta.delta_id,
                )
            )
        if delta.operation == "suppress":
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "node_biometrics":
        hints = dict(after.get("pressure_hints") or {})
        if "gpu" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="gpu_pressure",
                    intensity=float(hints["gpu"]),
                    label=delta.delta_id,
                )
            )
        if "strain" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="cpu_pressure",
                    intensity=float(hints["strain"]),
                    label=delta.delta_id,
                )
            )
        status = str(after.get("availability_status") or "")
        if status == "stale":
            out.append(Perturbation(node_id=node_id, channel="staleness", intensity=0.5, label=delta.delta_id))
        if after.get("expected_online") is False:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "execution_run":
        hints = dict(after.get("pressure_hints") or {})
        node_key = _node_key(str(after.get("node_id") or delta.target_id))
        for channel, key in (
            ("execution_load", "execution_load"),
            ("execution_friction", "execution_friction"),
            ("reasoning_load", "reasoning_load"),
            ("failure_pressure", "failure_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_key,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                        mode="replace",
                    )
                )
        try:
            egress_raw = float(hints["egress_confidence"]) if "egress_confidence" in hints else None
        except (TypeError, ValueError):
            egress_raw = None
        if egress_raw is not None and math.isfinite(egress_raw):
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="egress_confidence_deficit",
                    intensity=max(0.0, min(1.0, 1.0 - egress_raw)),
                    label=delta.delta_id,
                    mode="replace",
                )
            )

    if delta.target_kind == "chat_turn":
        hints = dict(after.get("pressure_hints") or {})
        for channel, key in (
            ("conversation_load", "conversation_load"),
            ("repair_pressure", "repair_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_id,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                        mode="replace",
                    )
                )

    if delta.target_kind == "transport_bus":
        hints = dict((delta.after or {}).get("pressure_hints") or {})
        node_key = _node_key(str((delta.after or {}).get("node_id") or delta.target_id.replace("bus:", "")))
        for channel, key in (
            ("bus_health", "bus_health"),
            ("delivery_confidence", "delivery_confidence"),
            ("transport_pressure", "transport_pressure"),
            ("catalog_drift_pressure", "catalog_drift_pressure"),
            ("observer_failure_pressure", "observer_failure_pressure"),
            ("reliability_pressure", "reliability_pressure"),
            ("contract_pressure", "contract_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_key,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                    )
                )
        if "stream_depth_pressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="transport_pressure",
                    intensity=float(hints["stream_depth_pressure"]),
                    label=delta.delta_id,
                )
            )
        if "backpressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="transport_pressure",
                    intensity=float(hints["backpressure"]),
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "prediction_signal":
        hints = dict((delta.after or {}).get("pressure_hints") or {})
        node_key = _node_key(str((delta.after or {}).get("node_id") or delta.target_id))
        if "prediction_error" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="prediction_error",
                    intensity=max(0.0, min(1.0, float(hints["prediction_error"]))),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
    return out
