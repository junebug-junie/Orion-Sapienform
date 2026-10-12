"""PE-writer vs dynamics-tick lost-update eval (2026-10-12).

6 prediction-error nodes, 60 ticks of 30 s. Every tick, each node gets one
prediction-error write, landing at random in one of three places:

- ``before``: before the tick reads (no race),
- ``tick_race``: after the tick's read, before its write (tick read -> PE write
  -> tick write),
- ``pe_race``: the PE writer reads, the tick runs, then the PE writer writes
  (PE read -> tick write -> PE write).

Counts (a) PE fields the tick reverted, (b) tick-owned fields the PE write
reverted, (c) tick activation compare-and-set misses caused by the PE write.
The fixed writer must score 0 on all three. The comparison run forces the
old whole-node behaviour (no tick-owned skip) and must reproduce the bug, or
this eval proves nothing.

Mirrors the live symptom: node:substrate.bus_synaptic's activation read
0.0118343 == its prediction_error and recency_score 0 after every 30 s write.

Run: PYTHONPATH=. python services/orion-substrate-runtime/evals/run_pe_writer_tick_lost_update_eval.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, SERVICE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import app.worker as worker_mod  # noqa: E402
from orion.substrate.activation import ACTIVATION_DECAYED_AT_KEY  # noqa: E402
from orion.substrate.dynamics import SubstrateDynamicsEngine  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402

T0 = datetime(2026, 10, 12, 0, 0, 0, tzinfo=timezone.utc)
NODES = [f"node:substrate.eval{i}" for i in range(6)]
TICKS = 60
TICK_S = 30
TICK_KEYS = ("activation", "recency_score", ACTIVATION_DECAYED_AT_KEY, "dynamic_pressure", "dormant")


def _tick_state(node) -> tuple:
    meta = node.metadata or {}
    return (
        node.signals.activation.activation,
        node.signals.activation.recency_score,
        meta.get(ACTIVATION_DECAYED_AT_KEY),
        meta.get("dynamic_pressure"),
        meta.get("dormant"),
    )


class _ReadHook:
    """Store wrapper: runs ``hook`` once right after the PE writer's read."""

    def __init__(self, store, hook) -> None:
        self._store, self._hook, self.fired = store, hook, False

    def get_node_by_id(self, node_id):
        node = self._store.get_node_by_id(node_id)
        if not self.fired:
            self.fired = True
            self._hook()
        return node

    def __getattr__(self, name):
        return getattr(self._store, name)


def _run(fixed: bool) -> dict:
    os.environ[worker_mod._PREDICTION_ERROR_NODE_FLAG] = "true"
    rng = random.Random(11)
    store = InMemorySubstrateGraphStore()
    worker = worker_mod.BiometricsSubstrateWorker.__new__(worker_mod.BiometricsSubstrateWorker)
    worker._substrate_graph_store = store
    skips = worker_mod._PE_WRITER_SKIPS_TICK_OWNED if fixed else frozenset()
    counts = {"pe_fields_reverted": 0, "tick_fields_reverted": 0, "tick_cas_misses": 0}
    totals = {"before": 0, "tick_race": 0, "pe_race": 0}

    def pe_write(store_for_write, nid, now, error, turn):
        worker._substrate_graph_store = store_for_write
        worker._write_prediction_error_node(node_id=nid, error=error, now=now, reducer_key="eval", contributing_id=turn)
        worker._substrate_graph_store = store

    with mock.patch.object(worker_mod, "_PE_WRITER_SKIPS_TICK_OWNED", skips):
        for nid in NODES:  # create, then settle one tick
            pe_write(store, nid, T0 - timedelta(minutes=5), 0.3, "seed")
        SubstrateDynamicsEngine(store=store).tick(now=T0)

        for t in range(1, TICKS + 1):
            now = T0 + timedelta(seconds=TICK_S * t)
            placement = {nid: rng.choice(("before", "tick_race", "pe_race")) for nid in NODES}
            errors = {nid: round(rng.uniform(0.05, 0.95), 6) for nid in NODES}
            for nid, where in placement.items():
                totals[where] += 1
                if where == "before":
                    pe_write(store, nid, now - timedelta(seconds=5), errors[nid], f"t{t}")

            engine = SubstrateDynamicsEngine(store=store)
            real = engine._compute_activations

            def _race(nodes, outgoing, pressures, tick_at, _now=now):
                out = real(nodes, outgoing, pressures, tick_at)
                for nid, where in placement.items():
                    if where == "tick_race":
                        pe_write(store, nid, _now - timedelta(seconds=2), errors[nid], f"t{t}")
                return out

            engine._compute_activations = _race  # type: ignore[method-assign]
            result = engine.tick(now=now)
            counts["tick_cas_misses"] += result.activation_cas_skips
            for nid, where in placement.items():
                if where == "tick_race" and store.get_node_by_id(nid).metadata.get("prediction_error") != errors[nid]:
                    counts["pe_fields_reverted"] += 1

            for nid, where in placement.items():
                if where != "pe_race":
                    continue
                tick_at = now + timedelta(seconds=10)
                after_tick: dict = {}

                def _tick_between(_nid=nid, _at=tick_at, _out=after_tick):
                    SubstrateDynamicsEngine(store=store).tick(now=_at)
                    _out["state"] = _tick_state(store.get_node_by_id(_nid))

                pe_write(_ReadHook(store, _tick_between), nid, tick_at + timedelta(seconds=1), errors[nid], f"t{t}")
                if _tick_state(store.get_node_by_id(nid)) != after_tick["state"]:
                    counts["tick_fields_reverted"] += 1
    return {"counts": counts, "writes": totals}


def run() -> dict:
    fixed = _run(fixed=True)
    old = _run(fixed=False)
    report = {"nodes": len(NODES), "ticks": TICKS, "fixed": fixed, "old_whole_node_write": old}
    assert all(v == 0 for v in fixed["counts"].values()), report
    assert old["counts"]["tick_fields_reverted"] > 0.9 * old["writes"]["pe_race"], report
    report["passed"] = True
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
