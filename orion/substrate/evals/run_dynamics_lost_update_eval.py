"""Lost-update eval (2026-10-11): 40 concept nodes, 60 ticks. Before each
tick's writes land, a concurrent writer corrects observed_at on 5 random
nodes and writes a fresh activation on 3 others. Counts how many of those
writes the tick reverted. The targeted compare-and-set write must revert 0;
the old full-node re-upsert (forced here for comparison) reverts most.

Mirrors the live incident: 238/244 corrected observed_at values reverted
within two minutes (PR #2608).

Run: python -m orion.substrate.evals.run_dynamics_lost_update_eval
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from unittest import mock

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate import dynamics
from orion.substrate.activation import ACTIVATION_DECAYED_AT_KEY
from orion.substrate.store import InMemorySubstrateGraphStore

T0 = datetime(2026, 10, 11, 0, 0, 0, tzinfo=timezone.utc)
NODES = 40
TICKS = 60
TICK_S = 30


def _node(i: int) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=f"n{i}",
        label=f"n{i}",
        anchor_scope="orion",
        temporal={"observed_at": T0 - timedelta(hours=3)},
        signals=SubstrateSignalBundleV1(
            salience=0.0,
            activation=SubstrateActivationV1(activation=0.9, decay_half_life_seconds=3600, decay_floor=0.0),
        ),
        provenance=SubstrateProvenanceV1(authority="local_inferred", source_kind="eval", source_channel="eval", producer="eval"),
        metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()},
    )


def _run(targeted: bool) -> dict:
    rng = random.Random(7)
    store = InMemorySubstrateGraphStore()
    for i in range(NODES):
        store.upsert_node(identity_key=f"id:n{i}", node=_node(i))
    engine = dynamics.SubstrateDynamicsEngine(store=store)
    real = engine._compute_activations
    reverted = {"observed_at": 0, "fresh_activation": 0}
    total = {"observed_at": 0, "fresh_activation": 0}

    for t in range(1, TICKS + 1):
        now = T0 + timedelta(seconds=TICK_S * t)
        pending: list[tuple[str, str, object]] = []

        def _race(nodes, outgoing, pressures, tick_at, _now=now, _pending=pending):
            out = real(nodes, outgoing, pressures, tick_at)
            ids = rng.sample(range(NODES), 8)
            for i in ids[:5]:
                n = store.get_node_by_id(f"n{i}")
                new_obs = n.temporal.observed_at - timedelta(minutes=1)
                store.upsert_node(identity_key=f"id:n{i}",
                                  node=n.model_copy(update={"temporal": n.temporal.model_copy(update={"observed_at": new_obs})}))
                _pending.append(("observed_at", f"n{i}", new_obs))
            for i in ids[5:]:
                n = store.get_node_by_id(f"n{i}")
                bundle = n.signals.activation.model_copy(update={"activation": 0.97})
                store.upsert_node(identity_key=f"id:n{i}", node=n.model_copy(update={
                    "signals": n.signals.model_copy(update={"activation": bundle}),
                    "metadata": {**n.metadata, ACTIVATION_DECAYED_AT_KEY: (_now - timedelta(seconds=1)).isoformat()},
                }))
                _pending.append(("fresh_activation", f"n{i}", 0.97))
            return out

        engine._compute_activations = _race  # type: ignore[method-assign]
        if targeted:
            engine.tick(now=now)
        else:
            with mock.patch.object(dynamics, "node_dynamics_writer", lambda _s: None):
                engine.tick(now=now)
        for kind, nid, value in pending:
            total[kind] += 1
            n = store.get_node_by_id(nid)
            got = n.temporal.observed_at if kind == "observed_at" else n.signals.activation.activation
            if got != value:
                reverted[kind] += 1
    return {"reverted": reverted, "total": total}


def run() -> dict:
    targeted = _run(targeted=True)
    full_node = _run(targeted=False)
    report = {"nodes": NODES, "ticks": TICKS, "targeted": targeted, "full_node_reupsert": full_node}
    assert sum(targeted["reverted"].values()) == 0, report
    # The comparison path must actually reproduce the bug, or this eval proves nothing.
    assert full_node["reverted"]["observed_at"] > 0.9 * full_node["total"]["observed_at"], report
    report["passed"] = True
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
