from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.store import InMemorySubstrateGraphStore

_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _steady_node(node_id: str = "sub-node-1", *, activation: float = 0.3, pressure: float = 0.0) -> BaseSubstrateNodeV1:
    return BaseSubstrateNodeV1(
        node_id=node_id,
        node_kind="concept",
        anchor_scope="orion",
        temporal={"observed_at": _NOW},
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=0.4,
            activation={
                "activation": activation,
                "recency_score": 0.5,
                "decay_half_life_seconds": None,
                "decay_floor": 0.0,
            },
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test",
        ),
        metadata={"dynamic_pressure": pressure},
    )


class _CountingStore(InMemorySubstrateGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.upsert_node_calls: list[str] = []

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        self.upsert_node_calls.append(node.node_id)
        super().upsert_node(identity_key=identity_key, node=node)


def _engine_with(store: InMemorySubstrateGraphStore, *, activations: dict[str, float], pressures: dict[str, float]) -> SubstrateDynamicsEngine:
    engine = SubstrateDynamicsEngine(store=store)
    engine._compute_pressures = lambda nodes, outgoing, tick_at: (pressures, {})  # type: ignore[method-assign]
    engine._compute_activations = lambda updated_nodes, outgoing, pressures, tick_at: activations  # type: ignore[method-assign]
    return engine


def test_tick_does_not_rewrite_unchanged_node() -> None:
    store = _CountingStore()
    node = _steady_node(activation=0.3, pressure=0.0)
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()

    engine = _engine_with(store, activations={node.node_id: 0.3}, pressures={node.node_id: 0.0})
    result = engine.tick(now=_NOW)

    assert store.upsert_node_calls == []
    assert result.activation_updates == []
    assert result.pressure_updates == []


def test_tick_rewrites_node_when_activation_changes() -> None:
    store = _CountingStore()
    node = _steady_node(activation=0.3, pressure=0.0)
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()

    engine = _engine_with(store, activations={node.node_id: 0.5}, pressures={node.node_id: 0.0})
    result = engine.tick(now=_NOW)

    assert store.upsert_node_calls == [node.node_id]
    assert len(result.activation_updates) == 1


def test_tick_rewrites_node_when_pressure_changes_even_if_activation_steady() -> None:
    store = _CountingStore()
    node = _steady_node(activation=0.3, pressure=0.0)
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()

    engine = _engine_with(store, activations={node.node_id: 0.3}, pressures={node.node_id: 0.8})
    result = engine.tick(now=_NOW)

    assert store.upsert_node_calls == [node.node_id]
    assert result.activation_updates == []
    assert len(result.pressure_updates) == 1


def test_dormancy_uses_fresh_recency_not_stale_stored_value() -> None:
    """Regression: the dormancy check must read this tick's freshly computed
    recency (activations[f"{node_id}:recency"]), not node.signals.activation
    .recency_score off the top-of-tick store snapshot. Activation and pressure
    stay identical across both ticks here -- only recency moves -- so a version
    that reads the stale stored value would never see recency cross the
    dormancy threshold and would never transition the node to dormant.
    """
    store = _CountingStore()
    node = _steady_node(activation=0.05, pressure=0.0)  # at/below dormancy_threshold (0.08 default)
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()

    engine = SubstrateDynamicsEngine(store=store)
    engine._compute_pressures = lambda nodes, outgoing, tick_at: ({node.node_id: 0.0}, {})  # type: ignore[method-assign]

    recency_by_tick = iter([0.5, 0.05])  # tick 1: above threshold; tick 2: at/below threshold

    def fake_compute_activations(updated_nodes, outgoing, pressures, tick_at):
        return {node.node_id: 0.05, f"{node.node_id}:recency": next(recency_by_tick)}

    engine._compute_activations = fake_compute_activations  # type: ignore[method-assign]

    result1 = engine.tick(now=_NOW)
    assert store.upsert_node_calls == []
    assert result1.dormancy_transitions == []

    result2 = engine.tick(now=_NOW)
    assert store.upsert_node_calls == [node.node_id]
    assert len(result2.dormancy_transitions) == 1
    assert result2.dormancy_transitions[0].to_state == "dormant"
