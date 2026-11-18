"""Regression: the prediction-error writer and the dynamics tick must not
revert each other's fields.

Live 2026-10-12, after PR #2617 fixed the tick's side: every
``_write_prediction_error_node`` call re-upserted the WHOLE node from a fresh
``ConceptNodeV1``. Pressure/dormancy were "carried forward" from a read taken a
moment earlier (lost any tick write in between), and activation/recency were
written as the model's validator defaults (activation = salience, recency 0.0)
paired with the tick's old decay stamp -- node:substrate.bus_synaptic read
activation 0.0118343 == its prediction_error 0.011834, recency 0, every 30 s.

Ownership now: the PE writer owns prediction_error, its channels/evidence/
contributing ids and its own stamps (observed_at, salience, label,
provenance); the tick owns pressure, dormancy, activation, recency and the
decay stamp. Both interleavings are pinned below, on the in-memory store and
(opt-in) a real FalkorDB.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.worker import BiometricsSubstrateWorker, _PREDICTION_ERROR_NODE_FLAG  # noqa: E402
from orion.core.schemas.cognitive_substrate import (  # noqa: E402
    ConceptNodeV1,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.graph.falkor_client import RecordingFalkorClient  # noqa: E402
from orion.substrate.activation import ACTIVATION_DECAYED_AT_KEY  # noqa: E402
from orion.substrate.dynamics import SubstrateDynamicsEngine  # noqa: E402
from orion.substrate.falkor_codec import TICK_OWNED_NODE_PROPERTIES  # noqa: E402
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402

NODE = "node:substrate.bus_synaptic"
IDENTITY = f"substrate_prediction_error|{NODE}"
T0 = datetime(2026, 10, 12, 1, 0, 0, tzinfo=timezone.utc)
SEED_OBSERVED = T0 - timedelta(hours=1)
SEED_STAMP = (T0 - timedelta(minutes=5)).isoformat()


def _seed_node(*, dormant: bool = False, activation: float = 0.6) -> ConceptNodeV1:
    """The node as the tick last left it: real pressure, dormancy, a decayed
    activation and its stamp, plus an earlier PE write's own fields."""
    return ConceptNodeV1(
        node_id=NODE,
        label="bus synaptic",
        anchor_scope="orion",
        subject_ref="entity:orion",
        temporal={"observed_at": SEED_OBSERVED},
        signals=SubstrateSignalBundleV1(
            confidence=1.0,
            salience=0.3,
            activation=SubstrateActivationV1(
                activation=activation, recency_score=0.4, decay_half_life_seconds=3600, decay_floor=0.0
            ),
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="substrate_prediction_error",
            source_channel="substrate.runtime",
            producer="substrate_runtime_worker",
        ),
        metadata={
            "source_kind": "substrate_prediction_error",
            "prediction_error": 0.3,
            "dynamic_pressure": 0.21,
            "dynamic_pressure_reason": "prediction_error_seed",
            "dormant": dormant,
            "dormancy_updated_at": (T0 - timedelta(hours=2)).isoformat(),
            ACTIVATION_DECAYED_AT_KEY: SEED_STAMP,
            "contributing_turn_ids": ["turn-0"],
        },
    )


def _worker(store) -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._substrate_graph_store = store
    return worker


def _pe_write(worker, *, error: float, now: datetime, turn: str) -> None:
    worker._write_prediction_error_node(
        node_id=NODE, error=error, now=now, reducer_key="bus_synaptic", contributing_id=turn
    )


def _tick_with_race(store, concurrent_write) -> SubstrateDynamicsEngine:
    """Engine whose concurrent_write runs after the tick's snapshot read and
    before its writes (same seam as orion/substrate/tests/
    test_dynamics_lost_update.py)."""
    engine = SubstrateDynamicsEngine(store=store)
    real = engine._compute_activations

    def _with_race(nodes, outgoing, pressures, now):
        out = real(nodes, outgoing, pressures, now)
        concurrent_write()
        return out

    engine._compute_activations = _with_race  # type: ignore[method-assign]
    return engine


class _PeReadThenTick:
    """Wraps a store so the PE writer's own read (get_node_by_id) is followed
    by one full tick before the PE writer's write -- the window the old
    read-and-carry-forward lost tick writes in."""

    def __init__(self, store, *, tick_at: datetime) -> None:
        self._store = store
        self._tick_at = tick_at
        self.tick_result = None

    def get_node_by_id(self, node_id):
        node = self._store.get_node_by_id(node_id)
        if self.tick_result is None:
            self.tick_result = SubstrateDynamicsEngine(store=self._store).tick(now=self._tick_at)
        return node

    def __getattr__(self, name):
        return getattr(self._store, name)


def _tick_owned(node) -> dict:
    meta = node.metadata or {}
    return {
        "activation": node.signals.activation.activation,
        "recency_score": node.signals.activation.recency_score,
        ACTIVATION_DECAYED_AT_KEY: meta.get(ACTIVATION_DECAYED_AT_KEY),
        "dynamic_pressure": meta.get("dynamic_pressure"),
        "dynamic_pressure_reason": meta.get("dynamic_pressure_reason"),
        "dormant": meta.get("dormant"),
        "dormancy_updated_at": meta.get("dormancy_updated_at"),
    }


def _assert_pe_fields(node, *, error: float, now: datetime, turns: list[str]) -> None:
    assert node.metadata["prediction_error"] == error
    assert node.signals.salience == error
    assert node.temporal.observed_at == now
    assert node.metadata["contributing_turn_ids"] == turns


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")


# --- interleaving 1: tick read -> PE write -> tick write ----------------------


def test_tick_read_pe_write_tick_write_keeps_both_in_memory() -> None:
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key=IDENTITY, node=_seed_node())
    worker = _worker(store)
    pe_at = T0 - timedelta(seconds=2)

    result = _tick_with_race(store, lambda: _pe_write(worker, error=0.9, now=pe_at, turn="turn-1")).tick(now=T0)

    node = store.get_node_by_id(NODE)
    # The PE write landed and the tick did not revert it.
    _assert_pe_fields(node, error=0.9, now=pe_at, turns=["turn-0", "turn-1"])
    # The tick's write applied: the PE write left activation + stamp alone, so
    # the compare-and-set held (the old writer zeroed recency and reseeded
    # activation, which made every in-flight tick miss).
    assert result.activation_cas_skips == 0
    assert node.metadata[ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()
    # Pressure is the tick's own computed value, not the PE writer's copy.
    (update,) = [u for u in result.pressure_updates if u.node_id == NODE]
    assert node.metadata["dynamic_pressure"] == pytest.approx(update.new_pressure, abs=1e-6)


# --- interleaving 2: PE read -> tick write -> PE write ------------------------


def test_pe_read_tick_write_pe_write_keeps_both_in_memory() -> None:
    inner = InMemorySubstrateGraphStore()
    inner.upsert_node(identity_key=IDENTITY, node=_seed_node())
    racing = _PeReadThenTick(inner, tick_at=T0)
    pe_at = T0 + timedelta(seconds=1)

    _pe_write(_worker(racing), error=0.9, now=pe_at, turn="turn-1")

    assert racing.tick_result is not None and racing.tick_result.activation_updates
    node = inner.get_node_by_id(NODE)
    _assert_pe_fields(node, error=0.9, now=pe_at, turns=["turn-0", "turn-1"])
    # Everything the tick wrote between the PE read and the PE write survives.
    after_tick = _tick_owned_after_tick_only()
    assert _tick_owned(node) == after_tick


def _tick_owned_after_tick_only() -> dict:
    """What the tick alone writes from the seed state at T0 (no PE write)."""
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key=IDENTITY, node=_seed_node())
    SubstrateDynamicsEngine(store=store).tick(now=T0)
    return _tick_owned(store.get_node_by_id(NODE))


def test_tick_alone_actually_moves_every_field_the_interleaving_test_pins() -> None:
    """Guard against test 2 passing vacuously: the tick must change pressure,
    activation, recency and the stamp from the seed, or "survived" proves
    nothing."""
    before = _tick_owned(_seed_node())
    after = _tick_owned_after_tick_only()
    for key in ("activation", "recency_score", ACTIVATION_DECAYED_AT_KEY, "dynamic_pressure"):
        assert after[key] != before[key], key


# --- waking a dormant node ---------------------------------------------------


def test_fresh_prediction_error_wakes_a_dormant_node_through_the_tick() -> None:
    """The PE writer no longer writes dormancy; a fresh surprise must still
    reactivate the node, via the tick reading the new observed_at/salience."""
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key=IDENTITY, node=_seed_node(dormant=True, activation=0.01))
    _pe_write(_worker(store), error=0.9, now=T0, turn="turn-1")
    assert store.get_node_by_id(NODE).metadata["dormant"] is True  # PE writer left it

    result = SubstrateDynamicsEngine(store=store).tick(now=T0 + timedelta(seconds=30))

    assert store.get_node_by_id(NODE).metadata["dormant"] is False
    assert any(t.node_id == NODE and t.to_state == "active" for t in result.dormancy_transitions)


def test_first_pe_write_still_creates_a_complete_node() -> None:
    store = InMemorySubstrateGraphStore()
    _pe_write(_worker(store), error=0.5, now=T0, turn="turn-1")
    node = store.get_node_by_id(NODE)
    assert node is not None
    assert node.metadata["prediction_error"] == 0.5
    assert node.signals.activation.activation == 0.5  # validator seed, on create only


# --- Falkor: the Cypher itself ----------------------------------------------


def test_falkor_pe_write_never_sets_tick_owned_properties_on_match() -> None:
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="t"), client=client, hydrate=False
    )
    store.upsert_node(identity_key=IDENTITY, node=_seed_node())
    client.calls.clear()

    _pe_write(_worker(store), error=0.9, now=T0, turn="turn-1")

    (cypher, params), = [(c, p) for c, p in client.calls if c.startswith("MERGE")]
    on_create, on_match = cypher.split("ON CREATE SET", 1)[1].split("ON MATCH SET", 1)
    for prop in sorted(TICK_OWNED_NODE_PROPERTIES | {"activation", "recency_score", ACTIVATION_DECAYED_AT_KEY}):
        assert f"n.{prop} = ${prop}" not in on_match, prop
        assert f"n.{prop} = ${prop}" in on_create, prop
    for owned in ("prediction_error", "observed_at", "salience", "contributing_turn_ids_json"):
        assert f"n.{owned} = ${owned}" in on_match, owned
    # The write-through cache mirrors the graph: tick-owned values kept.
    cached = store.get_node_by_id(NODE)
    assert _tick_owned(cached) == _tick_owned(_seed_node())
    _assert_pe_fields(cached, error=0.9, now=T0, turns=["turn-0", "turn-1"])


# --- real FalkorDB (opt-in) --------------------------------------------------

_LIVE_URI = os.environ.get("ORION_TEST_FALKOR_URI")


def _row(raw, node_id: str) -> dict:
    (row,) = raw.graph_query(
        "MATCH (n:SubstrateNode {node_id: $id}) RETURN n.activation AS activation, "
        "n.recency_score AS recency_score, n.activation_decayed_at AS stamp, "
        "n.dynamic_pressure AS dynamic_pressure, n.dynamic_pressure_reason AS reason, n.dormant AS dormant, "
        "n.dormancy_updated_at AS dormancy_updated_at, n.prediction_error AS prediction_error, "
        "n.observed_at AS observed_at, n.contributing_turn_ids_json AS turns",
        {"id": node_id},
    )
    return row


@pytest.mark.skipif(not _LIVE_URI, reason="ORION_TEST_FALKOR_URI not set")
def test_real_falkor_both_interleavings_keep_both_writers_fields() -> None:
    import redis

    graph = f"orion_test_pe_partial_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=_LIVE_URI, graph_name=graph, ensure_indexes=False)
    try:
        seeder = FalkorSubstrateStore(cfg, hydrate=False)
        raw = seeder._client
        seeder.upsert_node(identity_key=IDENTITY, node=_seed_node())

        # 1) tick read -> PE write (another process: its own store) -> tick write.
        tick_store = FalkorSubstrateStore(cfg, hydrate=True)
        pe_store = FalkorSubstrateStore(cfg, hydrate=False)
        pe_at = T0 - timedelta(seconds=2)
        result = _tick_with_race(
            tick_store, lambda: _pe_write(_worker(pe_store), error=0.9, now=pe_at, turn="turn-1")
        ).tick(now=T0)
        row = _row(raw, NODE)
        assert row["prediction_error"] == pytest.approx(0.9)
        assert row["observed_at"] == pe_at.isoformat()
        assert result.activation_cas_skips == 0
        assert row["stamp"] == T0.isoformat()
        (update,) = [u for u in result.pressure_updates if u.node_id == NODE]
        assert row["dynamic_pressure"] == pytest.approx(update.new_pressure, abs=1e-6)

        # 2) PE read -> tick write -> PE write, on one shared store (production
        # shares one store between the tick and the PE writer).
        shared = FalkorSubstrateStore(cfg, hydrate=True)
        tick2_at = T0 + timedelta(minutes=10)
        racing = _PeReadThenTick(shared, tick_at=tick2_at)
        pe2_at = tick2_at + timedelta(seconds=1)
        _pe_write(_worker(racing), error=0.4, now=pe2_at, turn="turn-2")
        assert racing.tick_result is not None and racing.tick_result.activation_updates
        row2 = _row(raw, NODE)
        assert row2["prediction_error"] == pytest.approx(0.4)
        assert row2["observed_at"] == pe2_at.isoformat()
        assert '"turn-2"' in row2["turns"]
        # The tick's decay landed and was not reverted by the PE write.
        assert row2["stamp"] == tick2_at.isoformat()
        assert row2["recency_score"] != 0.0
        cached = shared.get_node_by_id(NODE)
        assert cached.metadata[ACTIVATION_DECAYED_AT_KEY] == tick2_at.isoformat()
        assert cached.signals.activation.activation == pytest.approx(row2["activation"], abs=1e-12)
        assert cached.metadata["dynamic_pressure"] == pytest.approx(row2["dynamic_pressure"], abs=1e-9)
    finally:
        redis.Redis.from_url(_LIVE_URI).execute_command("GRAPH.DELETE", graph)


@pytest.mark.skipif(not _LIVE_URI, reason="ORION_TEST_FALKOR_URI not set")
def test_real_falkor_cache_miss_caches_what_the_graph_kept() -> None:
    """Review finding 2026-10-12: the node exists in the graph (written by
    another process) but not in this store's cache. ON MATCH keeps the tick's
    values in the graph; the cache must hold those, not the PE writer's model
    defaults, or the tick's next compare-and-set expects an activation the
    graph does not hold."""
    import redis

    graph = f"orion_test_pe_partial_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=_LIVE_URI, graph_name=graph, ensure_indexes=False)
    try:
        FalkorSubstrateStore(cfg, hydrate=False).upsert_node(identity_key=IDENTITY, node=_seed_node())
        cold = FalkorSubstrateStore(cfg, hydrate=False)  # empty cache
        assert cold._cache.get_node_by_id(NODE) is None
        _pe_write(_worker(cold), error=0.9, now=T0, turn="turn-1")
        cached = cold._cache.get_node_by_id(NODE)
        row = _row(cold._client, NODE)
        assert cached.signals.activation.activation == pytest.approx(row["activation"], abs=1e-12) == 0.6
        assert cached.metadata[ACTIVATION_DECAYED_AT_KEY] == row["stamp"] == SEED_STAMP
        assert cached.metadata["dynamic_pressure"] == pytest.approx(0.21)
        assert cached.metadata["prediction_error"] == pytest.approx(0.9)
    finally:
        redis.Redis.from_url(_LIVE_URI).execute_command("GRAPH.DELETE", graph)
