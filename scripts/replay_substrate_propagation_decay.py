"""Read-only replay: old vs new propagation decay on a snapshot of the live substrate graph.

Hydrates the Falkor substrate graph through a ``GRAPH.RO_QUERY`` client (no write
can reach Falkor), copies the in-memory cache twice, and runs the REAL
``SubstrateDynamicsEngine.tick`` on each copy: once with propagation decay mode
``receiver_age`` (before 2026-11-03) and once with ``source``. Nothing is written
anywhere except an optional local pickle of the snapshot (``--save``) so a run can
be repeated on the same graph (``--load``).

Two horizons, both with no new producer writes (the graph is frozen at snapshot):

- short: ``--ticks`` ticks at ``--step-sec`` (default 60 x 30 s = 30 min, the live
  cadence) -- "what changes right after deploy";
- long: ``--long-hours`` at ``--long-step-min`` steps (default 48 h x 60 min) -- "does everything return to rest
  when nothing new arrives" (a hub must not hold its neighbours up by itself).

Run:
  python scripts/replay_substrate_propagation_decay.py --uri redis://127.0.0.1:6380
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.dynamics import SubstrateDynamicsEngine  # noqa: E402
from orion.substrate.eligibility import cognitive_view  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402

MODES = ("receiver_age", "source")


def hydrate(uri: str, graph: str) -> InMemorySubstrateGraphStore:
    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    client = RedisGraphQueryClient(uri=uri, graph_name=graph, read_only=True,
                                   socket_timeout=60, socket_connect_timeout=5)
    store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=uri, graph_name=graph), client=client)
    receipt = store.last_scan_receipt
    if receipt is None or not receipt.complete:
        raise SystemExit(f"incomplete read-only scan: {receipt}")
    return copy.deepcopy(store._cache)  # noqa: SLF001 - read-only copy of the hydrated cache


def _values(store: InMemorySubstrateGraphStore) -> dict[str, float]:
    state = store.snapshot()
    nodes, _ = cognitive_view(state.nodes, state.edges)
    return {nid: float(n.signals.activation.activation) for nid, n in nodes.items()}


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return round(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))], 4)


def _dist(vals: dict[str, float]) -> dict[str, float]:
    s = sorted(vals.values())
    return {"n": len(s), "p50": _pct(s, 0.5), "p90": _pct(s, 0.9), "p99": _pct(s, 0.99),
            "max": round(s[-1], 4) if s else 0.0,
            "above_0.5": sum(v > 0.5 for v in s), "above_0.2": sum(v > 0.2 for v in s),
            "above_dormancy_0.08": sum(v > 0.08 for v in s)}


def run(base: InMemorySubstrateGraphStore, mode: str, *, start: datetime, steps: int,
        step: timedelta) -> dict[str, float]:
    store = copy.deepcopy(base)
    engine = SubstrateDynamicsEngine(store=store, propagation_decay_mode=mode)
    t = start
    for _ in range(steps):
        engine.tick(now=t)
        t += step
    return _values(store)


def compare(base, *, start, steps, step, kinds, labels, in_degree) -> dict:
    old = run(base, "receiver_age", start=start, steps=steps, step=step)
    new = run(base, "source", start=start, steps=steps, step=step)
    delta = {nid: new[nid] - old.get(nid, 0.0) for nid in new}
    changed = {nid: d for nid, d in delta.items() if abs(d) > 1e-3}
    rises = sorted(changed.items(), key=lambda kv: -kv[1])[:20]
    falls = sorted(changed.items(), key=lambda kv: kv[1])[:5]
    top_new = sorted(new.items(), key=lambda kv: -kv[1])[:10]

    def row(nid, d=None):
        out = {"node_id": nid, "kind": kinds.get(nid), "label": (labels.get(nid) or "")[:60],
               "old": round(old.get(nid, 0.0), 4), "new": round(new[nid], 4), "in_degree": in_degree.get(nid, 0)}
        if d is not None:
            out["delta"] = round(d, 4)
        return out

    return {
        "horizon": f"{steps} ticks x {int(step.total_seconds())} s",
        "nodes": len(new),
        "changed_gt_0.001": len(changed),
        "rose": sum(d > 0 for d in changed.values()),
        "fell": sum(d < 0 for d in changed.values()),
        "changed_by_kind": dict(Counter(kinds.get(n) for n in changed).most_common()),
        "dist_old": _dist(old),
        "dist_new": _dist(new),
        "top20_rises": [row(n, d) for n, d in rises],
        "top5_falls": [row(n, d) for n, d in falls],
        "top10_new": [row(n) for n, _ in top_new],
        "node_athena": {k: round(v.get("node:athena", 0.0), 4) for k, v in (("old", old), ("new", new))},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--uri", default="")
    ap.add_argument("--graph", default="orion_substrate")
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--step-sec", type=float, default=30.0)
    ap.add_argument("--long-hours", type=float, default=48.0)
    ap.add_argument("--long-step-min", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.load:
        with open(args.load, "rb") as fh:
            snap_at, base = pickle.load(fh)
    else:
        if not args.uri:
            print("--uri or --load required", file=sys.stderr)
            return 2
        snap_at = datetime.now(timezone.utc)
        base = hydrate(args.uri, args.graph)
        if args.save:
            with open(args.save, "wb") as fh:
                pickle.dump((snap_at, base), fh)
    state = base.snapshot()
    nodes, edges = cognitive_view(state.nodes, state.edges)
    kinds = {nid: str(n.node_kind) for nid, n in nodes.items()}
    labels = {nid: str(getattr(n, "label", "") or "") for nid, n in nodes.items()}
    in_degree = Counter(e.target.node_id for e in edges.values())
    out = {
        "snapshot_at": snap_at.isoformat(),
        "graph_nodes": len(state.nodes), "cognitive_nodes": len(nodes), "cognitive_edges": len(edges),
        "short": compare(base, start=snap_at, steps=args.ticks, step=timedelta(seconds=args.step_sec),
                         kinds=kinds, labels=labels, in_degree=in_degree),
        "long": compare(base, start=snap_at, steps=int(args.long_hours * 60 / args.long_step_min),
                        step=timedelta(minutes=args.long_step_min),
                        kinds=kinds, labels=labels, in_degree=in_degree),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
