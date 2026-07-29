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
        self.skip_metadata_keys_calls: list[frozenset[str] | None] = []

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        self.upsert_node_calls.append(node.node_id)
        self.skip_metadata_keys_calls.append(skip_metadata_keys)
        super().upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)


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


def test_tick_passes_externally_owned_keys_to_skip_metadata_keys() -> None:
    """Regression test for the live bug confirmed 2026-07-29: tick()'s write
    guard fires on activation decay alone (essentially every node, every
    tick), and must never re-persist prediction_error/contributing_turn_ids
    -- fields it only ever READS (to seed pressure), never computes. See
    falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring for the full
    incident (node:substrate.bus_synaptic frozen at a stale prediction_error
    for 3+ hours, causing real false "Bus Anomaly Detected" alerts)."""
    from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS

    store = _CountingStore()
    node = _steady_node(activation=0.3, pressure=0.0)
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()
    store.skip_metadata_keys_calls.clear()

    engine = _engine_with(store, activations={node.node_id: 0.5}, pressures={node.node_id: 0.0})
    engine.tick(now=_NOW)

    assert store.upsert_node_calls == [node.node_id]
    assert store.skip_metadata_keys_calls == [EXTERNALLY_OWNED_METADATA_KEYS]


def test_tick_does_not_clobber_externally_owned_prediction_error() -> None:
    """The actual end-to-end regression, reproducing the real race: a node's
    prediction_error is updated by a genuinely EXTERNAL writer (bus_synaptic's
    own tick, in the real incident) WHILE this engine's tick() is in flight --
    i.e. after this engine's own snapshot() was taken (top of tick()) but
    before its own upsert_node() call runs. The external writer's fresher
    value must survive; it must NOT be reverted to whatever stale copy was in
    this engine's own start-of-tick snapshot. InMemorySubstrateGraphStore has
    no caching layer (snapshot() always reflects the current store instantly),
    so the race is simulated by injecting the "external write" as a side
    effect of the already-test-only-stubbed _compute_activations call, which
    real tick() invokes strictly after snapshot() and strictly before its own
    upsert_node() call -- the exact window the real bug lived in.
    """
    store = _CountingStore()
    node = _steady_node(activation=0.3, pressure=0.0)
    node = node.model_copy(update={"metadata": {**node.metadata, "prediction_error": 0.9}})
    store.upsert_node(identity_key="identity-1", node=node)
    store.upsert_node_calls.clear()

    engine = SubstrateDynamicsEngine(store=store)
    engine._compute_pressures = lambda nodes, outgoing, tick_at: ({node.node_id: 0.0}, {})  # type: ignore[method-assign]

    def _compute_activations_with_concurrent_external_write(updated_nodes, outgoing, pressures, tick_at):
        fresher = store.get_node_by_id(node.node_id)
        fresher = fresher.model_copy(update={"metadata": {**fresher.metadata, "prediction_error": 0.42}})
        store.upsert_node(identity_key="identity-1", node=fresher)
        return {node.node_id: 0.5}

    engine._compute_activations = _compute_activations_with_concurrent_external_write  # type: ignore[method-assign]

    engine.tick(now=_NOW)

    assert store.upsert_node_calls == [node.node_id, node.node_id], "expected the external write plus this engine's own"
    stored = store.get_node_by_id(node.node_id)
    assert stored.metadata["prediction_error"] == 0.42, (
        "tick() clobbered a fresher externally-written prediction_error with its own "
        "stale snapshot-time copy -- this is the exact live bug this test guards against"
    )


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
