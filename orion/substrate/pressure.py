from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1


@dataclass(frozen=True)
class PressureConfig:
    drive_base: float = 0.5
    drive_propagation_attenuation: float = 0.7
    contradiction_neighbor_attenuation: float = 0.5
    max_hops: int = 2
    max_pressure: float = 1.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _goal_modifier(goal: BaseSubstrateNodeV1) -> float:
    status = str(goal.metadata.get("goal_status") or "active").lower()
    if status == "blocked":
        return 1.3
    if status == "satisfied":
        return 0.5
    return 1.0


def drive_seed_pressure(node: BaseSubstrateNodeV1, config: PressureConfig) -> float:
    if node.node_kind != "drive":
        return 0.0
    drive_state = str(node.metadata.get("drive_status") or "active").lower()
    if drive_state != "active":
        return 0.0
    declared = float(node.metadata.get("pressure") or 0.0)
    return _clamp(max(config.drive_base * node.signals.salience, declared))


def contradiction_amplification(node: BaseSubstrateNodeV1, *, now: datetime) -> tuple[float, list[str]]:
    if node.node_kind != "contradiction":
        return 0.0, []
    if bool(node.metadata.get("resolved", False)):
        return 0.0, []
    severity = _clamp(float(node.metadata.get("severity") or 0.5))
    observed = node.temporal.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    persistence_days = max(0.0, (now - observed).total_seconds() / 86400.0)
    persistence_score = _clamp(persistence_days / 7.0)
    amplification = _clamp(severity * (0.5 + (0.5 * persistence_score)))
    involved = [str(item) for item in node.metadata.get("involved_node_ids") or []]
    if not involved and hasattr(node, "involved_node_ids"):
        involved = [str(item) for item in getattr(node, "involved_node_ids", [])]
    return amplification, involved


def pressure_edge_multiplier(edge: SubstrateEdgeV1, target: BaseSubstrateNodeV1) -> float:
    if edge.predicate == "blocks":
        return 1.2
    if edge.predicate == "satisfies":
        return 0.8
    if target.node_kind == "goal":
        return _goal_modifier(target)
    return 1.0
