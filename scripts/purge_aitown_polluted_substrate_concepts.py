#!/usr/bin/env python3
"""One-off cleanup: purge AI-Town-polluted concept/evidence nodes from the
live ``orion_substrate`` FalkorDB graph (Track A item #2 of
docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md
-- "the already-ingested god nodes stay AI-Town-derived until this runs").

Why this run, precisely: PR #1721 (2026-08-18) added an AI-Town platform
filter to topic-foundry's dataset, but nothing retroactively cleaned up
concepts already ingested before that filter existed. Traced the actual
live graph, not guessed:

    node_id prefix                                   dataset       where_sql  min_cluster_size  doc_count  created_at
    sub-concept-topicfoundry-87e6539e-...             (unfiltered)  None       15                 843        2026-08-16
    sub-concept-topicfoundry-ece65e49-...              -v2 (filtered) real       8                  62         2026-08-19
    sub-concept-topicfoundry-2032434f-...              -v2 (filtered) real       8                  62         2026-08-19

Run 87e6539e-0962-4ef3-8dc1-568866c4c57d is the one and only polluted run
-- confirmed live via GET /runs/{run_id} against orion-topic-foundry
(dataset "orion-hub-autonomous-dataset", the old unfiltered name,
where_sql=None, min_cluster_size=15 -- the exact broken default PR #1726
fixed). Its 22 concept labels ("Electrical Testing", "storm and memory",
"Lighting and storytelling", "Soldering Techniques", "Glass and steam
narrative", ...) are literally the AI-Town NPC-roleplay topics named in
the original bug report. The other two runs (ece65e49, 2032434f) are
confirmed clean -- both trained against the real "-v2" filtered dataset
with the actual AI-Town where_sql applied, both created 2026-08-19 after
the fix shipped -- and are explicitly NOT touched by this script.

Scope, live-verified before writing this script (not assumed): exactly 22
Concept nodes + 22 Evidence nodes (44 total) + their supports/co_occurs_with
edges, well under AGENTS.md section 14's 100k-row/100MB stop-and-ask
threshold.

Snapshot-first per AGENTS.md section 14: full node+edge dump written to
/tmp/aitown_substrate_concept_purge/snapshot_before.json before any
mutation.

Usage:
    python scripts/purge_aitown_polluted_substrate_concepts.py --dry-run
    python scripts/purge_aitown_polluted_substrate_concepts.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import redis  # noqa: E402

DEFAULT_HOST = os.environ.get("FALKORDB_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("FALKORDB_PORT", "6380"))
GRAPH_NAME = "orion_substrate"
JOB_DIR = Path("/tmp/aitown_substrate_concept_purge")

# The single confirmed-polluted run. Deliberately a hardcoded, explicit
# allowlist of ONE run_id, not a pattern/heuristic -- this script is a
# one-off for a specific, already-diagnosed incident, not a general
# "detect and purge AI-Town content" tool. A different polluted run
# discovered later needs its own provenance trace, not a rerun of this
# script with a loosened filter.
_POLLUTED_RUN_ID = "87e6539e-0962-4ef3-8dc1-568866c4c57d"
_NODE_ID_PREFIX = f"sub-concept-topicfoundry-{_POLLUTED_RUN_ID}-"
_EVIDENCE_ID_PREFIX = f"sub-evidence-topicfoundry-{_POLLUTED_RUN_ID}-"

_MATCH_WHERE = (
    f"n.node_id STARTS WITH '{_NODE_ID_PREFIX}' "
    f"OR n.node_id STARTS WITH '{_EVIDENCE_ID_PREFIX}'"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _query(r: redis.Redis, cypher: str) -> tuple[list[str], list[list[Any]]]:
    result = r.execute_command("GRAPH.QUERY", GRAPH_NAME, cypher)
    header = [h[1] if isinstance(h, list) else h for h in result[0]] if result[0] else []
    rows = result[1] if len(result) > 1 else []
    return header, rows


def _snapshot(r: redis.Redis, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    _, node_rows = _query(
        r,
        f"MATCH (n) WHERE {_MATCH_WHERE} "
        "RETURN labels(n), n.node_id, n.label, n.anchor_scope, n.metadata, n.evidence_type",
    )
    _, edge_rows = _query(
        r,
        f"MATCH (n)-[e]->(m) WHERE {_MATCH_WHERE} "
        "RETURN n.node_id, type(e), m.node_id",
    )
    # Parens around _MATCH_WHERE are load-bearing: it's an OR expression,
    # and Cypher's AND binds tighter than OR, so combining it with "AND NOT
    # (...)" unparenthesized silently mis-scopes to
    # "X OR (Y AND NOT (...))" instead of "(X OR Y) AND NOT (...)" --
    # caught live before this script's first real run: the unparenthesized
    # version returned edges *between two polluted nodes* under a query
    # named "edges from kept nodes", which is definitionally impossible for
    # a correct query.
    _, edge_rows_in = _query(
        r,
        f"MATCH (m)-[e]->(n) WHERE ({_MATCH_WHERE}) AND NOT ("
        f"m.node_id STARTS WITH '{_NODE_ID_PREFIX}' OR m.node_id STARTS WITH '{_EVIDENCE_ID_PREFIX}') "
        "RETURN m.node_id, type(e), n.node_id",
    )

    snapshot = {
        "snapshot_taken_at": datetime.now(timezone.utc).isoformat(),
        "graph": GRAPH_NAME,
        "polluted_run_id": _POLLUTED_RUN_ID,
        "nodes": node_rows,
        "outgoing_edges": edge_rows,
        "incoming_edges_from_kept_nodes": edge_rows_in,
    }
    out_path = out_dir / "snapshot_before.json"
    with open(out_path, "w") as fh:
        json.dump(snapshot, fh, default=_json_default, indent=2)
    return {"path": out_path, "node_count": len(node_rows), "edge_count": len(edge_rows) + len(edge_rows_in)}


def _counts(r: redis.Redis) -> dict[str, int]:
    _, total = _query(r, "MATCH (n) RETURN count(n)")
    _, matching = _query(r, f"MATCH (n) WHERE {_MATCH_WHERE} RETURN count(n)")
    return {
        "graph_total_nodes": total[0][0] if total else 0,
        "matching_polluted_nodes": matching[0][0] if matching else 0,
    }


def _purge(r: redis.Redis) -> dict:
    """DETACH DELETE the matched nodes -- removes the nodes and every edge
    touching them (both directions) in one atomic Cypher statement."""
    _, result = _query(r, f"MATCH (n) WHERE {_MATCH_WHERE} DETACH DELETE n")
    return {"deleted": True}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge AI-Town-polluted concept/evidence nodes from orion_substrate"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dry-run", action="store_true", help="snapshot + report only, no mutation")
    args = parser.parse_args()

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)

    before = _counts(r)
    print(f"before: {json.dumps(before)}")

    snap = _snapshot(r, JOB_DIR)
    print(f"snapshot: {snap['node_count']} nodes, {snap['edge_count']} edges -> {snap['path']}")

    if before["matching_polluted_nodes"] != snap["node_count"]:
        print("ERROR: snapshot node count does not match the live match count -- aborting before any mutation.")
        return 1

    if snap["node_count"] == 0:
        print("Nothing to purge -- 0 matching polluted nodes. Done.")
        return 0

    if args.dry_run:
        print(f"DRY RUN: would delete {snap['node_count']} nodes. No changes made.")
        return 0

    _purge(r)
    after = _counts(r)
    print(f"after: {json.dumps(after)}")

    verdict = "ok" if after["matching_polluted_nodes"] == 0 else "needs_review"
    report = {
        "job": "purge_aitown_polluted_substrate_concepts",
        "polluted_run_id": _POLLUTED_RUN_ID,
        "before": before,
        "after": after,
        "snapshot_path": str(snap["path"]),
        "verdict": verdict,
    }
    with open(JOB_DIR / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"report: {JOB_DIR / 'report.json'}")
    print(f"verdict: {verdict}")
    return 0 if verdict == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
