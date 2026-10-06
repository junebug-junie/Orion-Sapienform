from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1

from .activation import (
    ACTIVATION_DECAYED_AT_KEY,
    DECAY_MODE_LEGACY,
    ActivationConfig,
    activation_decay_anchor,
    decay_activation,
    normalize_decay_mode,
    seed_activation,
)
from .eligibility import cognitive_view
from .falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS
from .pressure import (
    PressureConfig,
    contradiction_amplification,
    prediction_error_pressure,
    pressure_edge_multiplier,
)
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
        decay_mode: str = "since_last",
    ) -> None:
        self._store = store
        # since_last (default): decay the stored activation only by the time
        # since it was last decayed (metadata[activation_decayed_at]).
        # legacy: decay by the full time since observed_at every tick, which
        # compounds. Rollback only; see orion/substrate/activation.py.
        self._decay_mode = normalize_decay_mode(decay_mode)
        self._activation_config = activation_config or ActivationConfig()
        self._pressure_config = pressure_config or PressureConfig()
        self._dormancy_threshold = dormancy_threshold
        self._revival_threshold = revival_threshold

    def tick(self, *, now: datetime | None = None) -> SubstrateDynamicsResultV1:
        tick_at = now or datetime.now(timezone.utc)
        if tick_at.tzinfo is None:
            tick_at = tick_at.replace(tzinfo=timezone.utc)
        state = self._store.snapshot()
        # Cognitive subgraph only (orion/substrate/eligibility.py, #2497 rule 8):
        # assertions, fenced memory referents/evidence and role-bearing edges are
        # neither decayed, pressured, propagated through, nor re-persisted here.
        state_nodes, state_edges = cognitive_view(state.nodes, state.edges)
        if not state_nodes:
            return SubstrateDynamicsResultV1(tick_at=tick_at, activation_updates=[], pressure_updates=[], dormancy_transitions=[])
        identity_by_node_id = {node_id: identity for identity, node_id in state.node_identity_index.items()}

        outgoing, incoming = self._adjacency(state_edges)
        pressures, pressure_reasons = self._compute_pressures(state_nodes, outgoing, tick_at)
        pressure_updates: list[PressureUpdateV1] = []
        updated_nodes: dict[str, BaseSubstrateNodeV1] = {}

        for node_id, node in state_nodes.items():
            prev_pressure = float(node.metadata.get("dynamic_pressure") or 0.0)
            new_pressure = pressures.get(node_id, 0.0)
            if abs(new_pressure - prev_pressure) < 1e-6:
                updated_nodes[node_id] = node
                continue
            metadata = dict(node.metadata)
            metadata["dynamic_pressure"] = round(new_pressure, 6)
            # Persist which source actually drove this tick's pressure value
            # (drive_seed, prediction_error_seed/propagation:*, contradiction_*,
            # or "none" if this node has no active driver) so downstream
            # consumers -- attention_broadcast._node_salience() in particular --
            # can type a node by what is CURRENTLY moving its pressure, instead
            # of a raw metadata field that never decays and never clears once
            # set (see orion/substrate/attention_broadcast.py::_node_salience).
            metadata["dynamic_pressure_reason"] = pressure_reasons.get(node_id, "none")
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
        pressure_changed_ids = {update.node_id for update in pressure_updates}

        legacy_decay = self._decay_mode == DECAY_MODE_LEGACY
        for node_id, node in updated_nodes.items():
            prev_activation = node.signals.activation.activation
            combined_activation = activations.get(node_id, prev_activation)
            half_life = node.signals.activation.decay_half_life_seconds
            floor = node.signals.activation.decay_floor
            observed_at = node.temporal.observed_at
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (tick_at - observed_at).total_seconds())
            new_decay_stamp: datetime | None = None
            anchor = observed_at
            if legacy_decay:
                # Pre-2026-10-06: the stored (already-decayed) value is folded
                # into combined_activation and decayed again by the node's full
                # age, so the loss compounds tick over tick.
                new_activation = decay_activation(
                    current=combined_activation,
                    elapsed_seconds=age_seconds,
                    half_life_seconds=half_life,
                    floor=floor,
                )
            else:
                # Two terms, each decayed exactly once per unit of real time:
                # - fresh input (seed from recency/salience/pressure, plus
                #   propagation) is recomputed from node state every tick, so
                #   decaying it by full age is a closed form, not a compound;
                # - the stored value is already decayed up to its anchor, so it
                #   decays only by the time since then.
                # max() of the two equals the legacy max(seed, stored) whenever
                # the anchor is observed_at (no stamp yet).
                anchor = activation_decay_anchor(node)
                since_last_seconds = max(0.0, (tick_at - anchor).total_seconds())
                fresh_activation = activations.get(f"{node_id}:fresh", combined_activation)
                new_activation = max(
                    decay_activation(
                        current=fresh_activation,
                        elapsed_seconds=age_seconds,
                        half_life_seconds=half_life,
                        floor=floor,
                    ),
                    decay_activation(
                        current=prev_activation,
                        elapsed_seconds=since_last_seconds,
                        half_life_seconds=half_life,
                        floor=floor,
                    ),
                )
                # Never move the stamp backwards (clock skew, or an
                # observed_at in the future relative to this tick).
                new_decay_stamp = max(anchor, tick_at)
            metadata = dict(node.metadata)
            dormant_prev = bool(metadata.get("dormant", False))
            dormant_new = dormant_prev

            # Use the freshly-computed recency for this tick's dormancy decision, not
            # node.signals.activation.recency_score (the value from the top-of-tick
            # store snapshot). Recency decays continuously every tick regardless of
            # whether this node's activation/pressure changed, so once the write guard
            # below can skip persisting a node, the stored recency_score stops being
            # refreshed -- a dormancy check reading the stale stored value would then
            # never see a node's recency cross the threshold on its own, leaving it
            # stuck non-dormant indefinitely even as real time keeps passing.
            fresh_recency = activations.get(f"{node_id}:recency", node.signals.activation.recency_score)

            if new_activation <= self._dormancy_threshold and fresh_recency <= self._dormancy_threshold:
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

            activation_changed = abs(new_activation - prev_activation) >= 1e-6

            # Guard: only persist a node this tick if something about it actually moved.
            # upsert_node() is a full DELETE+INSERT SPARQL transaction against the store;
            # calling it unconditionally for every node on every tick (previously the
            # case here) churns TDB2's journal at the tick cadence regardless of whether
            # any node changed, which is indistinguishable on disk from real growth and
            # forces far more frequent compaction than actual data volume warrants.
            if activation_changed or dormant_new != dormant_prev or node_id in pressure_changed_ids:
                if legacy_decay:
                    persisted_activation = round(new_activation, 6)
                    # legacy never maintains the stamp; drop it so the store
                    # re-stamps with observed_at. Carrying a stale stamp
                    # forward would make a later roll-forward to since_last
                    # re-apply all the decay legacy applied since that stamp.
                    metadata.pop(ACTIVATION_DECAYED_AT_KEY, None)
                elif activation_changed:
                    # Unrounded on purpose: the stamp says "this exact value is
                    # valid as of tick_at". Rounding to 6 places would drop up
                    # to 5e-7 per write, a systematic bias for slow-decaying
                    # nodes that only cross the 1e-6 write threshold every few
                    # ticks.
                    persisted_activation = new_activation
                    metadata[ACTIVATION_DECAYED_AT_KEY] = new_decay_stamp.isoformat()
                else:
                    # Written only for pressure/dormancy. Keep the stored value
                    # AND its stamp together: persisting a sub-threshold decay
                    # with a fresh stamp would round it away and restart the
                    # clock every tick, freezing decay for any node whose
                    # pressure moves every tick. The stamp is written
                    # explicitly (not left to whatever is durable) so the
                    # value and its stamp always travel as a pair: writing
                    # prev_activation without its own stamp could pair it with
                    # a different stamp already durable and mis-time the next
                    # decay. (This tick is the only decay writer since
                    # 2026-10-06; the Hub's second one was removed.)
                    persisted_activation = prev_activation
                    metadata[ACTIVATION_DECAYED_AT_KEY] = anchor.isoformat()
                activation_bundle = node.signals.activation.model_copy(
                    update={
                        "activation": persisted_activation,
                        "recency_score": round(fresh_recency, 6),
                    }
                )
                updated_signal = node.signals.model_copy(update={"activation": activation_bundle})
                updated_node = node.model_copy(update={"signals": updated_signal, "metadata": metadata})
                # skip_metadata_keys: this engine reads prediction_error purely
                # to SEED pressure (prediction_error_pressure() above) -- it
                # never computes prediction_error/contributing_turn_ids
                # itself and must not re-persist whatever (possibly stale)
                # copy happened to be in this tick's start-of-loop snapshot.
                # Confirmed live 2026-07-29: this write guard fires on
                # activation decay alone, which changes on essentially every
                # node every tick, far more often than an external writer
                # like bus_synaptic's 30s cadence -- without this guard,
                # this engine's own frequent re-writes durably clobbered a
                # real writer's fresh values, freezing node:substrate.
                # bus_synaptic's prediction_error at a stale 1.0 for 3+
                # hours and causing real false "Bus Anomaly Detected"
                # alerts downstream. See falkor_codec.
                # EXTERNALLY_OWNED_METADATA_KEYS's docstring for the full trace.
                self._store.upsert_node(
                    identity_key=identity_by_node_id.get(node_id),
                    node=updated_node,
                    skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS,
                )

            if activation_changed:
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
            seed = prediction_error_pressure(node, self._pressure_config, now=now)
            if seed <= 0:
                continue
            if seed > pressure[node.node_id]:
                pressure[node.node_id] = seed
                reasons[node.node_id] = "prediction_error_seed"
            frontier = [(node.node_id, seed, 0)]
            visited = set()
            while frontier:
                current_id, current_pressure, depth = frontier.pop(0)
                if depth >= self._pressure_config.max_hops:
                    continue
                for edge in outgoing.get(current_id, []):
                    target_id = edge.target.node_id
                    target = nodes.get(target_id)
                    if not target:
                        continue
                    attenuated = current_pressure * self._pressure_config.prediction_error_propagation_attenuation
                    attenuated *= pressure_edge_multiplier(edge, target)
                    attenuated = max(0.0, min(self._pressure_config.max_pressure, attenuated))
                    if attenuated <= pressure[target_id] + 1e-6:
                        continue
                    pressure[target_id] = attenuated
                    reasons[target_id] = f"prediction_error_propagation:{edge.predicate}"
                    key = (target_id, depth + 1)
                    if key in visited:
                        continue
                    visited.add(key)
                    frontier.append((target_id, attenuated, depth + 1))

        for node in nodes.values():
            amp, involved = contradiction_amplification(node, now=now)
            if amp <= 0:
                continue
            if amp > pressure[node.node_id]:
                pressure[node.node_id] = amp
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
        # Seed input plus propagation, excluding this node's OWN stored value.
        # tick() decays this by full age and the stored value by time since
        # last decay; see the since_last branch there. Note propagation is
        # still sourced from a neighbor's combined (stored-including) value,
        # so a propagated term carries the neighbor's stored activation and is
        # decayed by this node's age -- same as legacy, not a new behavior.
        fresh: dict[str, float] = {}
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
            fresh[node.node_id] = base
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
                fresh[target_id] = max(fresh.get(target_id, 0.0), activations[target_id])
                frontier.append((target_id, propagated, depth + 1))

        out: dict[str, float] = {}
        for node_id, value in activations.items():
            out[node_id] = max(0.0, min(1.0, value))
            out[f"{node_id}:recency"] = recency_scores.get(node_id, 0.0)
            out[f"{node_id}:fresh"] = max(0.0, min(1.0, fresh.get(node_id, value)))
        return out
