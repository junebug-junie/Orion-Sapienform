"""Regression: the dynamics tick must not revert writes it raced with.

Live incident 2026-10-11 (wp-read timestamp repair, PR #2608): the tick read
every node, decayed it, then re-upserted the WHOLE node from that read. A
write to any other property in between was reverted -- 238 corrected
``observed_at`` values within two minutes. The tick now writes only the
fields it owns, and writes activation compare-and-set on what it read.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.activation import ACTIVATION_DECAYED_AT_KEY
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.routed_store import RoutedSubstrateGraphStore
from orion.substrate.store import (
    InMemorySubstrateGraphStore,
    NodeDynamicsWriteV1,
    node_dynamics_writer,
)

T0 = datetime(2026, 10, 11, 1, 0, 0, tzinfo=timezone.utc)
HL = 3600
OBSERVED = T0 - timedelta(hours=2)
STAMP = (T0 - timedelta(minutes=10)).isoformat()


def _concept(node_id: str, *, activation: float = 0.8, observed_at: datetime = OBSERVED,
             stamp: str | None = STAMP, label: str | None = None) -> ConceptNodeV1:
    metadata = {ACTIVATION_DECAYED_AT_KEY: stamp} if stamp is not None else {}
    return ConceptNodeV1(
        node_id=node_id,
        label=label or node_id,
        anchor_scope="orion",
        temporal={"observed_at": observed_at},
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=0.0,
            activation=SubstrateActivationV1(
                activation=activation, recency_score=0.0, decay_half_life_seconds=HL, decay_floor=0.0
            ),
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="test", producer="test"
        ),
        metadata=metadata,
    )


def _race_engine(store, concurrent_write) -> SubstrateDynamicsEngine:
    """Engine whose concurrent_write runs after the tick's snapshot and before
    its writes -- the exact window the live bug lived in."""
    engine = SubstrateDynamicsEngine(store=store)
    real = engine._compute_activations

    def _with_race(nodes, outgoing, pressures, now):
        out = real(nodes, outgoing, pressures, now)
        concurrent_write()
        return out

    engine._compute_activations = _with_race  # type: ignore[method-assign]
    return engine


def test_tick_keeps_concurrent_observed_at_and_fresh_activation_in_memory() -> None:
    store = InMemorySubstrateGraphStore()
    for nid in ("a-unowned", "b-owned", "c-quiet"):
        store.upsert_node(identity_key=f"id:{nid}", node=_concept(nid))
    corrected_observed = OBSERVED - timedelta(minutes=30)
    fresh_stamp = (T0 - timedelta(seconds=5)).isoformat()

    def concurrent_write():
        # a: someone corrects observed_at only (the repair script's write).
        a = store.get_node_by_id("a-unowned")
        store.upsert_node(identity_key="id:a-unowned",
                          node=a.model_copy(update={"temporal": a.temporal.model_copy(update={"observed_at": corrected_observed})}))
        # b: a producer writes a fresh activation with its own stamp.
        store.upsert_node(identity_key="id:b-owned", node=_concept("b-owned", activation=0.95, stamp=fresh_stamp))

    result = _race_engine(store, concurrent_write).tick(now=T0)

    a = store.get_node_by_id("a-unowned")
    b = store.get_node_by_id("b-owned")
    c = store.get_node_by_id("c-quiet")
    # The unowned property survives; a's activation still decayed (its
    # activation/stamp were not touched, so the compare-and-set held).
    assert a.temporal.observed_at == corrected_observed
    expected = 0.8 * 0.5 ** (600 / HL)
    assert a.signals.activation.activation == pytest.approx(expected, abs=1e-12)
    # The fresh owned value is not clobbered by a decay of the stale one.
    assert b.signals.activation.activation == 0.95
    assert b.metadata[ACTIVATION_DECAYED_AT_KEY] == fresh_stamp
    # Decay math unchanged for an uncontended node.
    assert c.signals.activation.activation == pytest.approx(expected, abs=1e-12)
    assert c.metadata[ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()
    assert result.activation_cas_skips == 1


def test_in_memory_targeted_write_never_creates_a_node() -> None:
    store = InMemorySubstrateGraphStore()
    applied = store.update_node_dynamics(NodeDynamicsWriteV1(
        node_id="gone", node_kind="concept", expected_activation=0.5, expected_decay_stamp=None,
        activation=0.4, recency_score=0.0, activation_decayed_at=T0.isoformat(), dynamic_pressure=0.2,
    ))
    assert applied is False
    assert store.get_node_by_id("gone") is None


class _GraphDoubleClient:
    """Records Cypher; answers the targeted dynamics write as applied or not."""

    def __init__(self, *, applied: bool) -> None:
        self.applied = applied
        self.calls: list[tuple[str, dict]] = []

    def graph_query(self, cypher, params=None):  # noqa: ANN001
        self.calls.append((cypher, params or {}))
        if "applied_node_id" in cypher and self.applied:
            return [{"applied_node_id": params["node_id"]}]
        return []


def _falkor(applied: bool) -> tuple[FalkorSubstrateStore, _GraphDoubleClient]:
    client = _GraphDoubleClient(applied=applied)
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="t"), client=client, hydrate=False
    )
    store.upsert_node(identity_key="id:n", node=_concept("n"))
    return store, client


def _write(**kw) -> NodeDynamicsWriteV1:
    base = dict(node_id="n", node_kind="concept", expected_activation=0.8, expected_decay_stamp=STAMP,
                activation=0.7, recency_score=0.1, activation_decayed_at=T0.isoformat(),
                dynamic_pressure=0.3, dynamic_pressure_reason="prediction_error_seed")
    base.update(kw)
    return NodeDynamicsWriteV1(**base)


def test_falkor_targeted_write_sets_only_owned_properties_compare_and_set() -> None:
    store, client = _falkor(applied=True)
    client.calls.clear()
    assert store.update_node_dynamics(_write(dormant=True, dormancy_updated_at=T0.isoformat())) is True
    (cypher, params), = client.calls
    assert cypher.startswith("MATCH (n:SubstrateNode {node_id: $node_id})")
    assert "MERGE" not in cypher
    assert "WHERE abs(coalesce(n.activation, 0.0) - $expected_activation) <= $activation_tolerance" in cypher
    assert "n.activation_decayed_at = $expected_decay_stamp" in cypher
    for unowned in ("observed_at", "label", "provenance", "prediction_error", "identity_key"):
        assert unowned not in cypher
    assert params["expected_activation"] == 0.8 and params["expected_decay_stamp"] == STAMP
    cached = store.get_node_by_id("n")
    assert cached.signals.activation.activation == 0.7
    assert cached.metadata["dormant"] is True


def test_falkor_cas_miss_keeps_cache_activation_applies_pressure_and_forces_reread() -> None:
    store, client = _falkor(applied=False)
    gen = store._write_generation
    assert store.update_node_dynamics(_write()) is False
    cached = store.get_node_by_id("n")
    assert cached.signals.activation.activation == 0.8  # the decayed value never landed
    assert cached.metadata[ACTIVATION_DECAYED_AT_KEY] == STAMP
    assert cached.metadata["dynamic_pressure"] == 0.3  # pressure is unconditional
    assert store._write_generation == gen + 1  # next snapshot re-reads the graph


def test_falkor_non_concept_write_skips_concept_only_properties() -> None:
    store, client = _falkor(applied=True)
    client.calls.clear()
    store.update_node_dynamics(_write(node_kind="entity", dormant=True, dormancy_updated_at=T0.isoformat()))
    (cypher, _), = client.calls
    assert "dynamic_pressure" not in cypher and "dormant" not in cypher


def test_writer_lookup_falls_back_only_for_stores_without_targeted_path() -> None:
    from orion.substrate.graphdb_store import GraphDBSubstrateStore

    mem = InMemorySubstrateGraphStore()
    assert node_dynamics_writer(mem) is not None
    assert node_dynamics_writer(RoutedSubstrateGraphStore(primary=mem)) is not None
    sparql = GraphDBSubstrateStore.__new__(GraphDBSubstrateStore)
    assert node_dynamics_writer(sparql) is None
    assert node_dynamics_writer(RoutedSubstrateGraphStore(primary=sparql)) is None


# --- real FalkorDB (opt-in) -------------------------------------------------
# Proves the Cypher itself (compare-and-set via WITH ... WHERE, null stamp
# handling, float equality round trip) against a real engine, on a throwaway
# graph. Set ORION_TEST_FALKOR_URI (e.g. redis://localhost:6380) to run.

_LIVE_URI = os.environ.get("ORION_TEST_FALKOR_URI")


@pytest.mark.skipif(not _LIVE_URI, reason="ORION_TEST_FALKOR_URI not set")
def test_real_falkor_tick_does_not_revert_concurrent_writes() -> None:
    import redis

    graph = f"orion_test_decay_cas_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=_LIVE_URI, graph_name=graph, ensure_indexes=False)
    seeder = FalkorSubstrateStore(cfg, hydrate=False)
    raw = seeder._client
    try:
        seeder.upsert_node(identity_key="id:a", node=_concept("a", activation=0.8 * 0.5 ** (600 / HL)))
        seeder.upsert_node(identity_key="id:b", node=_concept("b"))
        seeder.upsert_node(identity_key="id:c", node=_concept("c"))
        # c: a pre-stamp node (no activation_decayed_at property at all).
        raw.graph_query("MATCH (n {node_id: 'c'}) REMOVE n.activation_decayed_at")
        # The tick's store hydrates from the graph, as in production.
        store = FalkorSubstrateStore(cfg, hydrate=True)
        corrected = (OBSERVED - timedelta(minutes=30)).isoformat()
        fresh_stamp = (T0 - timedelta(seconds=5)).isoformat()

        def concurrent_write():
            # Another process's direct writes: invisible to this store's cache.
            raw.graph_query("MATCH (n {node_id: 'a'}) SET n.observed_at = $o, n.sentinel = 'keep'", {"o": corrected})
            raw.graph_query("MATCH (n {node_id: 'b'}) SET n.activation = 0.95, n.activation_decayed_at = $s",
                            {"s": fresh_stamp})

        result = _race_engine(store, concurrent_write).tick(now=T0)
        rows = raw.graph_query(
            "MATCH (n:SubstrateNode) RETURN n.node_id AS id, n.observed_at AS o, n.sentinel AS s, "
            "n.activation AS act, n.activation_decayed_at AS st ORDER BY id"
        )
        by_id = {r["id"]: r for r in rows}
        assert by_id["a"]["o"] == corrected and by_id["a"]["s"] == "keep"
        assert by_id["a"]["act"] == pytest.approx(0.8 * 0.5 ** (1200 / HL), abs=1e-12)
        assert by_id["b"]["act"] == 0.95 and by_id["b"]["st"] == fresh_stamp
        # c had no stamp property (IS NULL branch): decayed from observed_at.
        assert by_id["c"]["st"] == T0.isoformat()
        assert by_id["c"]["act"] == pytest.approx(0.8 * 0.5 ** (7200 / HL), abs=1e-12)
        assert result.activation_cas_skips == 1
    finally:
        redis.Redis.from_url(_LIVE_URI).execute_command("GRAPH.DELETE", graph)


@pytest.mark.skipif(not _LIVE_URI, reason="ORION_TEST_FALKOR_URI not set")
def test_real_falkor_decay_keeps_moving_after_reload() -> None:
    """Review finding 2026-10-11: FalkorDB returns ~15 significant digits, so
    an equality CAS on the read-back activation missed on every node after a
    reload and froze decay. Seed real decay outputs (17 digits), reload, tick
    three times: every tick must apply and the value must keep falling."""
    import redis

    graph = f"orion_test_decay_cas_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=_LIVE_URI, graph_name=graph, ensure_indexes=False)
    try:
        seeder = FalkorSubstrateStore(cfg, hydrate=False)
        values = [0.8 * 0.5 ** (k / 7.0) for k in range(1, 6)]
        for k, v in enumerate(values):
            seeder.upsert_node(identity_key=f"id:r{k}", node=_concept(f"r{k}", activation=v))
        prev = None
        for t in range(1, 4):
            store = FalkorSubstrateStore(cfg, hydrate=True)  # reload every tick
            result = SubstrateDynamicsEngine(store=store).tick(now=T0 + timedelta(seconds=30 * t))
            assert result.activation_cas_skips == 0
            rows = seeder._client.graph_query("MATCH (n:SubstrateNode) RETURN n.node_id AS id, n.activation AS a")
            now_vals = {r["id"]: r["a"] for r in rows}
            if prev is not None:
                assert all(now_vals[k] < prev[k] for k in now_vals)
            prev = now_vals
    finally:
        redis.Redis.from_url(_LIVE_URI).execute_command("GRAPH.DELETE", graph)
