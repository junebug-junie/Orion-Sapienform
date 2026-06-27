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
    # Prediction-error feedback: surprise (how wrong the model just was) seeds
    # pressure on the implicated node and decays with age so Orion preferentially
    # attends to the parts of its self-model that are currently wrong, then lets go
    # as prediction improves. weight<=0 disables the loop.
    prediction_error_weight: float = 0.6
    prediction_error_propagation_attenuation: float = 0.6
    prediction_error_decay_horizon_seconds: int = 1800


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


def prediction_error_pressure(node: BaseSubstrateNodeV1, config: PressureConfig, *, now: datetime) -> float:
    """Seed pressure from a node's standing prediction error (0..1 surprise score).

    The surprise is read from ``metadata['prediction_error']`` (written by the
    substrate runtime's prediction-error reducers) and decays linearly toward zero
    over ``prediction_error_decay_horizon_seconds`` measured from the node's
    ``observed_at``. A node that keeps being surprising is re-observed (its
    ``observed_at`` advances), so sustained error holds pressure; a node whose
    prediction improves stops refreshing and the seed fades on its own.
    """
    if config.prediction_error_weight <= 0:
        return 0.0
    raw = float(node.metadata.get("prediction_error") or 0.0)
    if raw <= 0.0:
        return 0.0
    observed = node.temporal.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    horizon = config.prediction_error_decay_horizon_seconds
    if horizon <= 0:
        decay = 1.0
    else:
        age_seconds = max(0.0, (now - observed).total_seconds())
        decay = max(0.0, 1.0 - (age_seconds / horizon))
    return _clamp(raw * config.prediction_error_weight * decay)


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
