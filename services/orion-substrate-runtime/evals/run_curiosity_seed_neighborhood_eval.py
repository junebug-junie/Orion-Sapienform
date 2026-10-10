#!/usr/bin/env python3
"""Eval: do stored curiosity seeds pick up accepted reading links, and only those?

Fixture mode (default, CI): a graph with planted links -- one accepted reading claim,
one rejected, one deprecated, one stale projection (assertion moved past the edge's
revision), one legacy unreviewed edge between canonical concepts. Replays:
- the live seed focal sets sampled 2026-10-10 (fixtures/, organ nodes and gev_ ids),
- seeds on each planted endpoint, and a link-accepted seed per planted claim.
Reports how many seeds gained links, that only the accepted projection ever appears,
and read timing. Exit 1 on any violation.

Live mode (``--live``, read-only): the last N stored candidate sets from Postgres,
attached against production Falkor through a GRAPH.RO_QUERY client, plus the
accepted-links journal query. Prints counts/timing; never writes.

    python services/orion-substrate-runtime/evals/run_curiosity_seed_neighborhood_eval.py
    python services/orion-substrate-runtime/evals/run_curiosity_seed_neighborhood_eval.py --live \
        --pg "postgresql://...@localhost:55432/conjourney" --falkor redis://localhost:6380
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.schemas.cognitive_substrate import AssertionNodeV1  # noqa: E402
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1  # noqa: E402
from orion.substrate.curiosity_seed_neighborhood import attach_seed_neighborhoods  # noqa: E402
from orion.substrate.evals.neighborhood_fixture import PROVENANCE, TEMPORAL, concept, edge, graph  # noqa: E402
from orion.substrate.link_accepted_seeds import (  # noqa: E402
    ACCEPTED_UNSEEDED_LINKS_SQL,
    AcceptedLinkV1,
    accepted_links_params,
    link_accepted_seed,
    projection_edge_id,
)

FIXTURE = Path(__file__).with_name("fixtures") / "live_seed_focal_sets_2026-10-10.json"
# In-memory reads; a regression that makes one read slow (e.g. an unbounded scan) shows here.
FIXTURE_P95_BOUND_MS = 50.0

# claim id -> (subject, object, claim state, claim revision, edge's projected revision)
PLANTED = {
    "assertion-accepted": ("read-a", "read-b", "provisional", 1, 1),
    "assertion-rejected": ("read-c", "read-d", "rejected", 1, 1),
    "assertion-deprecated": ("read-e", "read-f", "deprecated", 2, 2),
    "assertion-stale": ("read-g", "read-h", "provisional", 2, 1),
}


def _seed(focal: list[str], note: str) -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="curiosity_candidate", anchor_scope="orion", subject_ref="entity:orion",
        target_zone="concept_graph", task_type_candidate="evidence_gap_scan",
        focal_node_refs=focal[:32], signal_strength=0.7, confidence=0.7,
        notes=["endogenous_seed", f"source:{note}"])


def _fixture_graph():
    nodes, edges = [concept("canon-1"), concept("canon-2")], [edge("legacy-1", "canon-1", "canon-2")]
    for claim_id, (sub, obj, state, revision, edge_revision) in PLANTED.items():
        nodes += [concept(n).model_copy(update={"promotion_state": "proposed"}) for n in (sub, obj)]
        nodes.append(AssertionNodeV1(
            node_id=claim_id, anchor_scope="world", promotion_state=state, temporal=TEMPORAL,
            provenance=PROVENANCE, predicate="associated_with", statement_key=f"k|{claim_id}",
            statement_text="planted", revision=revision))
        edges.append(edge(projection_edge_id(claim_id), sub, obj, edge_role="semantic_projection",
                          assertion_id=claim_id, assertion_revision=edge_revision))
    return graph(nodes, edges)


def _report(signals, stored, receipt, read_ms):
    linked = [s for s in stored if s.focal_edge_refs or s.boundary_edge_refs]
    edges = {e for s in stored for e in (*s.focal_edge_refs, *s.boundary_edge_refs)}
    read_ms = sorted(read_ms)
    p95 = read_ms[max(0, int(round(0.95 * len(read_ms))) - 1)] if read_ms else 0.0
    return {
        "seeds": len(signals), "seeds_with_accepted_link": len(linked),
        "link_seeds_with_internal_edge": sum(
            1 for s in stored if "source:reading_link_accepted" in s.notes and s.focal_edge_refs),
        "distinct_edges": sorted(edges), "receipt": receipt,
        "p95_read_ms": round(p95, 3), "max_read_ms": round(read_ms[-1], 3) if read_ms else 0.0,
    }


def _attach_each(signals, store):
    """One attach per seed so per-read timing is visible (the tick does them together)."""
    stored, read_ms, total = [], [], None
    for sig in signals:
        started = perf_counter()
        out, receipt = attach_seed_neighborhoods([sig], store=store)
        read_ms.append((perf_counter() - started) * 1000.0)
        stored.extend(out)
        if total is None:
            total = receipt
        else:
            for key in ("reads", "nonempty", "internal_edges", "boundary_edges", "projection_endpoints",
                        "legacy_edges_excluded", "truncated"):
                total[key] += receipt[key]
            for reason, n in receipt["degraded_reasons"].items():
                total["degraded_reasons"][reason] = total["degraded_reasons"].get(reason, 0) + n
    return stored, total or {}, read_ms


def run_fixture() -> tuple[dict, list[str]]:
    store = _fixture_graph()
    live_sets = json.loads(FIXTURE.read_text())["focal_sets"]
    signals = [_seed(f, "live_replay") for f in live_sets]
    for sub, obj, *_ in PLANTED.values():
        signals += [_seed([sub], "planted_endpoint"), _seed([obj], "planted_endpoint")]
    signals.append(_seed(["canon-1"], "planted_legacy"))
    for claim_id, (sub, obj, _state, revision, _edge_rev) in PLANTED.items():
        signals.append(link_accepted_seed(AcceptedLinkV1(
            assertion_id=claim_id, revision=revision, subject_node_id=sub, object_node_id=obj,
            predicate="associated_with", statement_text="planted", projection_edge_id=projection_edge_id(claim_id))))
    stored, receipt, read_ms = _attach_each(signals, store)
    report = _report(signals, stored, receipt, read_ms)
    report["live_replay_seeds"] = len(live_sets)
    report["live_replay_seeds_with_link"] = sum(
        1 for s in stored if "source:live_replay" in s.notes and (s.focal_edge_refs or s.boundary_edge_refs))

    accepted = projection_edge_id("assertion-accepted")
    failures = []
    if report["distinct_edges"] != [accepted]:
        failures.append(f"only the accepted projection may appear, got {report['distinct_edges']}")
    if report["seeds_with_accepted_link"] != 3:  # read-a seed, read-b seed, accepted link seed
        failures.append(f"expected 3 seeds with the accepted link, got {report['seeds_with_accepted_link']}")
    if report["link_seeds_with_internal_edge"] != 1:
        failures.append("the accepted link seed must hold the link as an internal edge")
    if report["live_replay_seeds_with_link"] != 0:
        failures.append("live replay seeds name edgeless organ nodes; they must stay empty")
    if receipt.get("legacy_edges_excluded", 0) < 1:
        failures.append("the planted legacy edge must be read and excluded")
    if report["p95_read_ms"] > FIXTURE_P95_BOUND_MS:
        failures.append(f"p95 read {report['p95_read_ms']} ms > {FIXTURE_P95_BOUND_MS}")
    return report, failures


def run_live(pg: str, falkor: str, graph_name: str, limit: int) -> dict:
    from sqlalchemy import create_engine, text

    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    engine = create_engine(pg)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = conn.execute(text(
            "SELECT candidates_json FROM substrate_endogenous_curiosity_candidates "
            "ORDER BY generated_at DESC LIMIT :n"), {"n": limit}).all()
        started = perf_counter()
        links = conn.execute(text(ACCEPTED_UNSEEDED_LINKS_SQL),
                             accepted_links_params(lookback_hours=168.0, limit=10)).mappings().all()
        links_ms = (perf_counter() - started) * 1000.0
    signals = []
    for (candidates,) in rows:
        for item in candidates or []:
            try:
                sig = FrontierInvocationSignalV1.model_validate(item)
            except Exception:  # noqa: BLE001 - old rows; counted below
                continue
            if "endogenous_seed" in sig.notes:
                signals.append(sig)
    client = RedisGraphQueryClient(uri=falkor, graph_name=graph_name, read_only=True,
                                   socket_timeout=10, socket_connect_timeout=5)
    assert client.read_only
    store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=falkor, graph_name=graph_name),
                                 client=client, hydrate=False)
    stored, receipt, read_ms = _attach_each(signals, store)
    report = _report(signals, stored, receipt, read_ms)
    report.update({"candidate_sets": len(rows), "unseeded_accepted_reading_links": len(links),
                   "accepted_links_sql_ms": round(links_ms, 1), "read_only_client": client.read_only})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--pg", default="")
    parser.add_argument("--falkor", default="redis://localhost:6380")
    parser.add_argument("--graph", default="orion_substrate")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)
    if args.live:
        report = run_live(args.pg, args.falkor, args.graph, args.limit)
        print(json.dumps(report, indent=2, default=str))
        return 0
    report, failures = run_fixture()
    print(json.dumps(report, indent=2, default=str))
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
