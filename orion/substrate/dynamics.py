from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1

from .activation import ActivationConfig, decay_activation, seed_activation
from .pressure import PressureConfig, contradiction_amplification, drive_seed_pressure, pressure_edge_multiplier
from .store import InMemorySubstrateGraphStore


@dataclass(frozen=True)
class ActivationUpdateV1:
    node_id: str
    previous_activation: float
    new_activation: float
    reason: str


@dataclass(frozen=True)
class PressureUpdateV1:
    node_id: str
    previous_pressure: float
    new_pressure: float
    reason: str


@dataclass(frozen=True)
class DormancyTransitionV1:
    node_id: str
    from_state: str
    to_state: str
    reason: str


@dataclass(frozen=True)
class SubstrateDynamicsResultV1:
    tick_at: datetime
    activation_updates: list[ActivationUpdateV1]
    pressure_updates: list[PressureUpdateV1]
    dormancy_transitions: list[DormancyTransitionV1]


class SubstrateDynamicsEngine:
    """Deterministic bounded dynamics operating on the materialized substrate graph."""

    def __init__(
        self,
        *,
        store: InMemorySubstrateGraphStore,
        activation_config: ActivationConfig | None = None,
        pressure_config: PressureConfig | None = None,
        dormancy_threshold: float = 0.08,
        revival_threshold: float = 0.2,
    ) -> None:
        self._store = store
        self._activation_config = activation_config or ActivationConfig()
        self._pressure_config = pressure_config or PressureConfig()
        self._dormancy_threshold = dormancy_threshold
        self._revival_threshold = revival_threshold

    def tick(self, *, now: datetime | None = None) -> SubstrateDynamicsResultV1:
        tick_at = now or datetime.now(timezone.utc)
        if tick_at.tzinfo is None:
            tick_at = tick_at.replace(tzinfo=timezone.utc)
        state = self._store.snapshot()
        if not state.nodes:
            return SubstrateDynamicsResultV1(tick_at=tick_at, activation_updates=[], pressure_updates=[], dormancy_transitions=[])
        identity_by_node_id = {node_id: identity for identity, node_id in state.node_identity_index.items()}

        outgoing, incoming = self._adjacency(state.edges)
        pressures, pressure_reasons = self._compute_pressures(state.nodes, outgoing, tick_at)
        pressure_updates: list[PressureUpdateV1] = []
        updated_nodes: dict[str, BaseSubstrateNodeV1] = {}

        for node_id, node in state.nodes.items():
            prev_pressure = float(node.metadata.get("dynamic_pressure") or 0.0)
            new_pressure = pressures.get(node_id, 0.0)
            if abs(new_pressure - prev_pressure) < 1e-6:
                updated_nodes[node_id] = node
                continue
            metadata = dict(node.metadata)
            metadata["dynamic_pressure"] = round(new_pressure, 6)
            updated = node.model_copy(update={"metadata": metadata})
            updated_nodes[node_id] = updated
            pressure_updates.append(
                PressureUpdateV1(
                    node_id=node_id,
                    previous_pressure=prev_pressure,
                    new_pressure=new_pressure,
                    reason=pressure_reasons.get(node_id, "pressure_update"),
                )
            )

        activations = self._compute_activations(updated_nodes, outgoing, pressures, tick_at)
        activation_updates: list[ActivationUpdateV1] = []
        dormancy_transitions: list[DormancyTransitionV1] = []

        for node_id, node in updated_nodes.items():
            prev_activation = node.signals.activation.activation
            new_activation = activations.get(node_id, prev_activation)
            elapsed_seconds = max(0.0, (tick_at - node.temporal.observed_at).total_seconds())
            new_activation = decay_activation(
                current=new_activation,
                elapsed_seconds=elapsed_seconds,
                half_life_seconds=node.signals.activation.decay_half_life_seconds,
                floor=node.signals.activation.decay_floor,
            )
            metadata = dict(node.metadata)
            dormant_prev = bool(metadata.get("dormant", False))
            dormant_new = dormant_prev

            if new_activation <= self._dormancy_threshold and node.signals.activation.recency_score <= self._dormancy_threshold:
                dormant_new = True
            elif new_activation >= self._revival_threshold:
                dormant_new = False

            if dormant_new != dormant_prev:
                metadata["dormant"] = dormant_new
                metadata["dormancy_updated_at"] = tick_at.isoformat()
                dormancy_transitions.append(
                    DormancyTransitionV1(
                        node_id=node_id,
                        from_state="dormant" if dormant_prev else "active",
                        to_state="dormant" if dormant_new else "active",
                        reason="activation_threshold_crossed",
                    )
                )

            activation_bundle = node.signals.activation.model_copy(
                update={
                    "activation": round(new_activation, 6),
                    "recency_score": round(activations.get(f"{node_id}:recency", node.signals.activation.recency_score), 6),
                }
            )
            updated_signal = node.signals.model_copy(update={"activation": activation_bundle})
            updated_node = node.model_copy(update={"signals": updated_signal, "metadata": metadata})
            self._store.upsert_node(identity_key=identity_by_node_id.get(node_id), node=updated_node)

            if abs(new_activation - prev_activation) >= 1e-6:
                activation_updates.append(
                    ActivationUpdateV1(
                        node_id=node_id,
                        previous_activation=prev_activation,
                        new_activation=new_activation,
                        reason="seed_and_propagation",
                    )
                )

        return SubstrateDynamicsResultV1(
            tick_at=tick_at,
            activation_updates=activation_updates,
            pressure_updates=pressure_updates,
            dormancy_transitions=dormancy_transitions,
        )

    @staticmethod
    def _adjacency(edges: dict[str, SubstrateEdgeV1]) -> tuple[dict[str, list[SubstrateEdgeV1]], dict[str, list[SubstrateEdgeV1]]]:
        outgoing: dict[str, list[SubstrateEdgeV1]] = defaultdict(list)
        incoming: dict[str, list[SubstrateEdgeV1]] = defaultdict(list)
        for edge in edges.values():
            outgoing[edge.source.node_id].append(edge)
            incoming[edge.target.node_id].append(edge)
        return outgoing, incoming

    def _compute_pressures(
        self,
        nodes: dict[str, BaseSubstrateNodeV1],
        outgoing: dict[str, list[SubstrateEdgeV1]],
        now: datetime,
    ) -> tuple[dict[str, float], dict[str, str]]:
        pressure: dict[str, float] = defaultdict(float)
        reasons: dict[str, str] = {}

        for node in nodes.values():
            seed = drive_seed_pressure(node, self._pressure_config)
            if seed <= 0:
                continue
            pressure[node.node_id] = max(pressure[node.node_id], seed)
            reasons[node.node_id] = "drive_seed"
            frontier: list[tuple[str, float, int]] = [(node.node_id, seed, 0)]
            visited: set[tuple[str, int]] = set()
            while frontier:
                current_id, current_pressure, depth = frontier.pop(0)
                if depth >= self._pressure_config.max_hops:
                    continue
                for edge in outgoing.get(current_id, []):
                    target_id = edge.target.node_id
                    target = nodes.get(target_id)
                    if not target:
                        continue
                    attenuated = current_pressure * self._pressure_config.drive_propagation_attenuation
                    attenuated *= pressure_edge_multiplier(edge, target)
                    attenuated = max(0.0, min(self._pressure_config.max_pressure, attenuated))
                    if attenuated <= pressure[target_id] + 1e-6:
                        continue
                    pressure[target_id] = attenuated
                    reasons[target_id] = f"drive_propagation:{edge.predicate}"
                    key = (target_id, depth + 1)
                    if key in visited:
                        continue
                    visited.add(key)
                    frontier.append((target_id, attenuated, depth + 1))

        for node in nodes.values():
            amp, involved = contradiction_amplification(node, now=now)
            if amp <= 0:
                continue
            pressure[node.node_id] = max(pressure[node.node_id], amp)
            reasons[node.node_id] = "contradiction_unresolved"
            for involved_id in involved:
                propagated = max(0.0, min(self._pressure_config.max_pressure, amp * self._pressure_config.contradiction_neighbor_attenuation))
                if propagated > pressure[involved_id]:
                    pressure[involved_id] = propagated
                    reasons[involved_id] = "contradiction_involved"
        return dict(pressure), reasons

    def _compute_activations(
        self,
        nodes: dict[str, BaseSubstrateNodeV1],
        outgoing: dict[str, list[SubstrateEdgeV1]],
        pressures: dict[str, float],
        now: datetime,
    ) -> dict[str, float]:
        activations: dict[str, float] = {}
        recency_scores: dict[str, float] = {}
        for node in nodes.values():
            contradiction_boost = 0.0
            if node.node_kind == "contradiction" and not bool(node.metadata.get("resolved", False)):
                contradiction_boost = float(node.metadata.get("severity") or 0.5) * 0.2
            base = seed_activation(
                node,
                now=now,
                config=self._activation_config,
                pressure=pressures.get(node.node_id, 0.0),
                contradiction_boost=contradiction_boost,
            )
            recency_scores[node.node_id] = max(0.0, min(1.0, 1.0 - max(0.0, (now - node.temporal.observed_at).total_seconds()) / self._activation_config.recency_horizon_seconds))
            activations[node.node_id] = max(base, node.signals.activation.activation)

        frontier = [(node_id, value, 0) for node_id, value in activations.items() if value >= self._activation_config.min_delta]
        while frontier:
            node_id, signal, depth = frontier.pop(0)
            if depth >= self._activation_config.max_hops:
                continue
            for edge in outgoing.get(node_id, []):
                if edge.predicate not in self._activation_config.allowed_predicates:
                    continue
                propagated = signal * self._activation_config.attenuation * edge.confidence
                if propagated < self._activation_config.min_delta:
                    continue
                target_id = edge.target.node_id
                if propagated <= activations.get(target_id, 0.0) + 1e-6:
                    continue
                activations[target_id] = max(0.0, min(1.0, propagated))
                frontier.append((target_id, propagated, depth + 1))

        out: dict[str, float] = {}
        for node_id, value in activations.items():
            out[node_id] = max(0.0, min(1.0, value))
            out[f"{node_id}:recency"] = recency_scores.get(node_id, 0.0)
        return out
