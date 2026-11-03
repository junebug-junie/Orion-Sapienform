"""Propagation decay eval (2026-11-03): a propagated activation is decayed once.

Three fixtures on the real SubstrateDynamicsEngine, ticked every 10 minutes for
24 h, each in both propagation decay modes:

1. day node: a just-sighted cluster (0.61, 6 h half-life) -> observed_in (0.8) ->
   a day node observed 11 h earlier. ``source`` must read ~cluster x 0.48 right
   after the sighting (``receiver_age`` read ~0.08) and both must reach rest.
2. hub: one fresh node (6 h half-life) -> 300 neighbours observed 20 days ago
   (6 h half-life). No neighbour may read above hub x 0.6, and all must rest.
3. fast hub, slow neighbours: the same hub -> 300 neighbours with a 30-day
   half-life. ``source`` must equal ``receiver_age`` here (a 6 h signal persisted
   on a 30-day clock would hold ~0.6 for weeks), checked over 14 days.
4. never-decaying source: a node with no half-life (0.52, like live topic-foundry
   evidence) -> a concept observed 35 days ago (30-day half-life). ``source``
   must equal ``receiver_age`` here: a source that carries no decay is decayed
   once by the receiver's age, so it cannot pin its neighbour forever.

Run: python -m orion.substrate.evals.run_propagation_decay_eval
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.store import InMemorySubstrateGraphStore

T0 = datetime(2026, 11, 2, 17, 22, tzinfo=timezone.utc)
SIX_H = 6 * 3600
THIRTY_D = 30 * 86400
DORMANT = 0.08
PROV = SubstrateProvenanceV1(authority="local_inferred", source_kind="eval", source_channel="eval", producer="eval")


def _node(node_id, observed_at, activation=0.0, half_life=SIX_H, salience=0.0):
    return BaseSubstrateNodeV1(
        node_id=node_id, node_kind="concept", anchor_scope="world", temporal={"observed_at": observed_at},
        signals=SubstrateSignalBundleV1(salience=salience, activation={
            "activation": activation, "decay_half_life_seconds": half_life, "decay_floor": 0.0}),
        provenance=PROV)


def _edge(source, target, confidence):
    return SubstrateEdgeV1(
        edge_id=f"e-{source}-{target}", source=NodeRefV1(node_id=source, node_kind="concept"),
        target=NodeRefV1(node_id=target, node_kind="concept"), predicate="observed_in",
        temporal={"observed_at": T0 - timedelta(days=40)}, confidence=confidence, provenance=PROV)


def _world(nodes, edges):
    store = InMemorySubstrateGraphStore()
    for n in nodes:
        store.upsert_node(identity_key=None, node=n)
    for e in edges:
        store.upsert_edge(identity_key=e.edge_id, edge=e)
    return store


def _run(build, mode, watch):
    store = build()
    engine = SubstrateDynamicsEngine(store=store, propagation_decay_mode=mode)
    first = end = None
    t = T0 + timedelta(minutes=1)
    while t <= T0 + timedelta(hours=24):
        engine.tick(now=t)
        values = {nid: store.get_node_by_id(nid).signals.activation.activation for nid in watch}
        first = first or values
        end = values
        t += timedelta(minutes=10)
    return first, end


def _day():
    return _world([_node("cluster", T0, 0.61), _node("day-node", T0.replace(hour=6, minute=0), salience=0.3)],
                  [_edge("cluster", "day-node", 0.8)])


HUB_N = 300


def _hub():
    spokes = [_node(f"spoke-{i}", T0 - timedelta(days=20)) for i in range(HUB_N)]
    return _world([_node("hub-node", T0, 1.0), *spokes], [_edge("hub-node", f"spoke-{i}", 1.0) for i in range(HUB_N)])


def _slow_hub():
    spokes = [_node(f"spoke-{i}", T0 - timedelta(days=20), half_life=THIRTY_D) for i in range(HUB_N)]
    return _world([_node("hub-node", T0, 1.0), *spokes], [_edge("hub-node", f"spoke-{i}", 1.0) for i in range(HUB_N)])


def _slow_curve(mode):
    store = _slow_hub()
    engine = SubstrateDynamicsEngine(store=store, propagation_decay_mode=mode)
    out = []
    for hours in range(0, 14 * 24 + 1, 12):
        engine.tick(now=T0 + timedelta(hours=hours))
        out.append(max(store.get_node_by_id(f"spoke-{i}").signals.activation.activation for i in range(HUB_N)))
    return out


def _frozen():
    return _world([_node("evidence", T0 - timedelta(days=40), 0.52, half_life=None),
                   _node("old-concept", T0 - timedelta(days=35), 0.05, half_life=THIRTY_D, salience=0.2)],
                  [_edge("evidence", "old-concept", 1.0)])


def run() -> dict:
    report: dict = {}
    for mode in ("source", "receiver_age"):
        d_first, d_end = _run(_day, mode, ["cluster", "day-node"])
        watch = ["hub-node"] + [f"spoke-{i}" for i in range(HUB_N)]
        h_first, h_end = _run(_hub, mode, watch)
        f_first, f_end = _run(_frozen, mode, ["old-concept"])
        report[mode] = {
            "day_node_after_sighting": round(d_first["day-node"], 4),
            "cluster_after_sighting": round(d_first["cluster"], 4),
            "day_node_24h": round(d_end["day-node"], 4),
            "hub_first": round(h_first["hub-node"], 4),
            "spoke_max_first": round(max(h_first[k] for k in watch[1:]), 4),
            "spoke_max_24h": round(max(h_end[k] for k in watch[1:]), 4),
            "frozen_source_neighbour_first": round(f_first["old-concept"], 6),
            "frozen_source_neighbour_24h": round(f_end["old-concept"], 6),
        }
    slow_source, slow_rollback = _slow_curve("source"), _slow_curve("receiver_age")
    report["slow_spoke_max_first_and_14d"] = {
        "source": [round(slow_source[0], 4), round(slow_source[-1], 4)],
        "receiver_age": [round(slow_rollback[0], 4), round(slow_rollback[-1], 4)],
    }
    s, r = report["source"], report["receiver_age"]
    assert abs(s["day_node_after_sighting"] - s["cluster_after_sighting"] * 0.48) < 2e-3, report
    assert r["day_node_after_sighting"] < 0.1, report
    assert s["day_node_24h"] < DORMANT and r["day_node_24h"] < DORMANT, report
    assert s["spoke_max_first"] <= s["hub_first"] * 0.6 + 1e-6, report
    assert s["spoke_max_24h"] < DORMANT, report
    assert s["frozen_source_neighbour_first"] == r["frozen_source_neighbour_first"], report
    assert s["frozen_source_neighbour_24h"] == r["frozen_source_neighbour_24h"], report
    assert max(abs(a - b) for a, b in zip(slow_source, slow_rollback)) < 1e-9, report
    report["passed"] = True
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
