"""Propagation decay mode (2026-11-03): a propagated term is decayed once, not twice.

Real SubstrateDynamicsEngine over an in-memory store. Before this fix the tick
decayed a neighbour's (already decayed) activation again by the RECEIVER's full
age, so an old node reached by a fresh neighbour was muted -- a walkway day node
took a 0.61 sighting cluster as ~0.04.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.activation import normalize_propagation_decay_mode
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.store import InMemorySubstrateGraphStore

T0 = datetime(2026, 11, 2, 17, 22, tzinfo=timezone.utc)
SIX_HOURS = 6 * 3600
DORMANT = 0.08
_PROV = SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="test", producer="test")


def _node(node_id: str, *, observed_at: datetime, activation: float = 0.0, half_life: int | None = SIX_HOURS,
          salience: float = 0.0, kind: str = "concept") -> BaseSubstrateNodeV1:
    return BaseSubstrateNodeV1(
        node_id=node_id, node_kind=kind, anchor_scope="world", temporal={"observed_at": observed_at},
        signals=SubstrateSignalBundleV1(salience=salience, activation={
            "activation": activation, "decay_half_life_seconds": half_life, "decay_floor": 0.0}),
        provenance=_PROV)


def _edge(source: str, target: str, *, confidence: float = 0.8, predicate: str = "observed_in") -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        edge_id=f"e-{source}-{target}", source=NodeRefV1(node_id=source, node_kind="concept"),
        target=NodeRefV1(node_id=target, node_kind="concept"), predicate=predicate,
        temporal={"observed_at": T0 - timedelta(days=1)}, confidence=confidence, provenance=_PROV)


def _store(nodes, edges=()) -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    for n in nodes:
        store.upsert_node(identity_key=None, node=n)
    for e in edges:
        store.upsert_edge(identity_key=e.edge_id, edge=e)
    return store


def _a(store, node_id: str) -> float:
    return store.get_node_by_id(node_id).signals.activation.activation


def _day_world() -> InMemorySubstrateGraphStore:
    # A cluster just sighted (fresh, 0.61) and a day node observed at the start of the
    # local day (11 h before the sighting), joined by observed_in at confidence 0.8.
    cluster = _node("cluster", observed_at=T0, activation=0.61)
    day = _node("day", observed_at=T0.replace(hour=6, minute=0), salience=0.3)
    return _store([cluster, day], [_edge("cluster", "day")])


def test_day_node_reads_cluster_times_edge_attenuation_then_rests():
    store = _day_world()
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=T0 + timedelta(minutes=1))
    cluster = _a(store, "cluster")
    assert abs(_a(store, "day") - cluster * 0.6 * 0.8) < 1e-3
    assert _a(store, "day") > 0.25
    readings = []
    t = T0 + timedelta(minutes=10)
    while t <= T0 + timedelta(hours=24):
        engine.tick(now=t)
        readings.append(_a(store, "day"))
        t += timedelta(minutes=10)
    # Tracks the quieting cluster down (never rises with no new sighting) and reaches rest.
    assert all(b <= a + 1e-9 for a, b in zip(readings, readings[1:]))
    assert readings[-1] < DORMANT
    assert store.get_node_by_id("day").metadata.get("dormant") is True


def test_rollback_mode_reproduces_the_muted_day_node():
    store = _day_world()
    SubstrateDynamicsEngine(store=store, propagation_decay_mode="receiver_age").tick(now=T0 + timedelta(minutes=1))
    assert _a(store, "day") < 0.1  # ~0.079: muted by its 11 h age


def _random_world(seed: int, *, with_edges: bool, decaying_sources: bool):
    rng = random.Random(seed)
    nodes = []
    for i in range(40):
        decays = decaying_sources or i % 2 == 0
        nodes.append(_node(
            f"node-{i}", observed_at=T0 - timedelta(hours=rng.uniform(0, 400)), activation=rng.random(),
            half_life=rng.choice([3600, SIX_HOURS, 30 * 86400]) if decays else None, salience=rng.random()))
    edges = []
    if with_edges:
        for i in range(1, 40, 2):  # only odd ids point out (no half-life unless decaying_sources)
            for j in rng.sample(range(40), 3):
                if j != i:
                    edges.append(_edge(f"node-{i}", f"node-{j}", confidence=rng.random()))
    return nodes, edges


def _trajectory(nodes, edges, mode: str, ticks: int = 30) -> list[dict[str, float]]:
    store = _store(nodes, edges)
    engine = SubstrateDynamicsEngine(store=store, propagation_decay_mode=mode)
    out = []
    for k in range(ticks):
        engine.tick(now=T0 + timedelta(minutes=10 * k))
        out.append({n.node_id: _a(store, n.node_id) for n in nodes})
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_node_with_no_neighbours_behaves_exactly_as_before(seed):
    nodes, _ = _random_world(seed, with_edges=False, decaying_sources=True)
    assert _trajectory(nodes, [], "source") == _trajectory(nodes, [], "receiver_age")


@pytest.mark.parametrize("seed", [4, 5])
def test_one_hop_from_a_never_decaying_source_behaves_exactly_as_before(seed):
    # A source with no half-life carries no decay, so the receiver's age still decays
    # it once -- the receiver_age rule. Only odd (non-decaying) nodes have out-edges.
    nodes, edges = _random_world(seed, with_edges=True, decaying_sources=False)
    old = _trajectory(nodes, edges, "receiver_age", ticks=10)
    new = _trajectory(nodes, edges, "source", ticks=10)
    for k, (o, n) in enumerate(zip(old, new)):
        for nid in o:
            assert n[nid] == pytest.approx(o[nid], abs=1e-9), (k, nid)


@pytest.mark.parametrize("seed", [6, 7])
def test_legacy_decay_mode_ignores_the_propagation_mode(seed):
    nodes, edges = _random_world(seed, with_edges=True, decaying_sources=True)

    def run(mode):
        store = _store(nodes, edges)
        engine = SubstrateDynamicsEngine(store=store, decay_mode="legacy", propagation_decay_mode=mode)
        for k in range(10):
            engine.tick(now=T0 + timedelta(minutes=10 * k))
        return {n.node_id: _a(store, n.node_id) for n in nodes}

    assert run("source") == run("receiver_age")


def test_a_busy_hub_lifts_neighbours_to_at_most_hub_x_attenuation_and_lets_them_rest():
    hub = _node("hub", observed_at=T0, activation=1.0, half_life=SIX_HOURS)
    spokes = [_node(f"spoke-{i}", observed_at=T0 - timedelta(days=20), half_life=30 * 86400) for i in range(200)]
    second = [_node(f"second-{i}", observed_at=T0 - timedelta(days=20), half_life=30 * 86400) for i in range(200)]
    edges = [_edge("hub", f"spoke-{i}", confidence=1.0, predicate="associated_with") for i in range(200)]
    edges += [_edge(f"spoke-{i}", f"second-{i}", confidence=1.0, predicate="associated_with") for i in range(200)]
    store = _store([hub, *spokes, *second], edges)
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=T0)
    hub_now = _a(store, "hub")
    # Max (not sum) over paths and 0.6 per hop: 200 spokes at <= 0.6, two hops at <= 0.36.
    assert max(_a(store, f"spoke-{i}") for i in range(200)) <= hub_now * 0.6 + 1e-9
    assert max(_a(store, f"second-{i}") for i in range(200)) <= hub_now * 0.36 + 1e-9
    # Hub goes quiet (6 h half-life). The spokes hold their latched value on their own
    # 30-day clock, so "rest" is bounded by the slowest half-life on the path -- the same
    # as before this fix for any spoke young enough not to have been muted.
    t = T0
    while t <= T0 + timedelta(days=1):
        t += timedelta(hours=1)
        engine.tick(now=t)
    assert _a(store, "hub") < DORMANT
    assert max(_a(store, f"spoke-{i}") for i in range(200)) < 0.6


def test_no_loop_sustains_itself():
    # Two nodes pointing at each other: 0.6 x 0.6 < 1 per round trip, so with the
    # sources quiet both reach rest on their own clocks.
    a = _node("node-a", observed_at=T0, activation=0.9, half_life=3600)
    b = _node("node-b", observed_at=T0 - timedelta(days=3), half_life=3600)
    store = _store([a, b], [_edge("node-a", "node-b", confidence=1.0), _edge("node-b", "node-a", confidence=1.0)])
    engine = SubstrateDynamicsEngine(store=store)
    for k in range(1, 48):
        engine.tick(now=T0 + timedelta(minutes=30 * k))
    assert _a(store, "node-a") < DORMANT and _a(store, "node-b") < DORMANT


def test_unknown_propagation_mode_raises():
    with pytest.raises(ValueError):
        normalize_propagation_decay_mode("reciever_age")
    with pytest.raises(ValueError):
        SubstrateDynamicsEngine(store=InMemorySubstrateGraphStore(), propagation_decay_mode="nope")
    assert normalize_propagation_decay_mode(None) == "source"
