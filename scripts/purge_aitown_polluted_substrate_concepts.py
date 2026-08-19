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
Concept nodes + 22 Evidence nodes (44 total) + 231 outgoing edges, zero
edges from any KEPT node into the polluted subgraph (confirmed via a
direct query before the first real run) -- well under AGENTS.md section
14's 100k-row/100MB stop-and-ask threshold, and fully self-contained.

Uses ``orion.graph.falkor_client.RedisGraphQueryClient`` (code review
2026-08-19: the first version of this script hand-rolled a raw
``redis.Redis`` + ``GRAPH.QUERY`` wrapper instead of reusing the repo's
existing, tested client -- every sibling Falkor backfill/cleanup script
already uses this one).

Snapshot-first per AGENTS.md section 14: full node+edge dump written to
/tmp/aitown_substrate_concept_purge/snapshot_before.json before any
mutation. On completion, writes report.json, report.md, and
before_after.csv (code review: the first version only wrote report.json).

Usage:
    python scripts/purge_aitown_polluted_substrate_concepts.py --dry-run
    python scripts/purge_aitown_polluted_substrate_concepts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: E402

DEFAULT_FALKORDB_URI = os.environ.get("FALKORDB_URI", "redis://localhost:6380")
DEFAULT_GRAPH_NAME = os.environ.get("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")
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

# Parens here are load-bearing: this is an OR expression, and Cypher's AND
# binds tighter than OR, so any caller combining this with "AND NOT (...)"
# unparenthesized would silently mis-scope to "X OR (Y AND NOT (...))"
# instead of "(X OR Y) AND NOT (...)" -- caught live before this script's
# first real run (see _snapshot's incoming-edge query for why it matters).
_MATCH_WHERE = (
    f"(n.node_id STARTS WITH '{_NODE_ID_PREFIX}' OR n.node_id STARTS WITH '{_EVIDENCE_ID_PREFIX}')"
)


def _log_progress(message: str) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line)
    with open(JOB_DIR / "progress.log", "a") as fh:
        fh.write(line + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _snapshot(client: RedisGraphQueryClient, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = client.graph_query(
        f"MATCH (n) WHERE {_MATCH_WHERE} "
        "RETURN labels(n) AS labels, n.node_id AS node_id, n.label AS label, "
        "n.anchor_scope AS anchor_scope, n.evidence_type AS evidence_type"
    )
    outgoing_edges = client.graph_query(
        f"MATCH (n)-[e]->(m) WHERE {_MATCH_WHERE} "
        "RETURN n.node_id AS src, type(e) AS rel, m.node_id AS dst"
    )
    incoming_from_kept = client.graph_query(
        f"MATCH (m)-[e]->(n) WHERE {_MATCH_WHERE} AND NOT "
        f"(m.node_id STARTS WITH '{_NODE_ID_PREFIX}' OR m.node_id STARTS WITH '{_EVIDENCE_ID_PREFIX}') "
        "RETURN m.node_id AS src, type(e) AS rel, n.node_id AS dst"
    )

    snapshot = {
        "snapshot_taken_at": datetime.now(timezone.utc).isoformat(),
        "graph": DEFAULT_GRAPH_NAME,
        "polluted_run_id": _POLLUTED_RUN_ID,
        "nodes": node_rows,
        "outgoing_edges": outgoing_edges,
        "incoming_edges_from_kept_nodes": incoming_from_kept,
    }
    out_path = out_dir / "snapshot_before.json"
    with open(out_path, "w") as fh:
        json.dump(snapshot, fh, default=_json_default, indent=2)
    return {
        "path": out_path,
        "node_count": len(node_rows),
        "edge_count": len(outgoing_edges),
        "incoming_from_kept_count": len(incoming_from_kept),
        "nodes": node_rows,
    }


def _counts(client: RedisGraphQueryClient) -> dict[str, int]:
    total = client.graph_query("MATCH (n) RETURN count(n) AS c")
    matching = client.graph_query(f"MATCH (n) WHERE {_MATCH_WHERE} RETURN count(n) AS c")
    return {
        "graph_total_nodes": total[0]["c"] if total else 0,
        "matching_polluted_nodes": matching[0]["c"] if matching else 0,
    }


def _purge(client: RedisGraphQueryClient) -> None:
    """DETACH DELETE the matched nodes -- removes the nodes and every edge
    touching them (both directions) in one atomic Cypher statement."""
    client.graph_query(f"MATCH (n) WHERE {_MATCH_WHERE} DETACH DELETE n")


def _write_before_after_csv(out_dir: Path, nodes: list[dict]) -> Path:
    out_path = out_dir / "before_after.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "label", "anchor_scope", "before", "after"])
        for row in nodes:
            writer.writerow([row.get("node_id"), row.get("label"), row.get("anchor_scope"), "present", "deleted"])
    return out_path


def _write_report_md(
    out_dir: Path, *, before: dict, after: dict, snapshot_path: Path, csv_path: Path, verdict: str,
) -> Path:
    out_path = out_dir / "report.md"
    lines = [
        "# AI-Town-polluted substrate concept purge -- report",
        "",
        f"- verdict: **{verdict}**",
        f"- polluted run_id: `{_POLLUTED_RUN_ID}`",
        f"- nodes deleted: {before['matching_polluted_nodes']}",
        "",
        "## Before",
        f"- graph total nodes: {before['graph_total_nodes']}",
        f"- matching polluted nodes: {before['matching_polluted_nodes']}",
        "",
        "## After",
        f"- graph total nodes: {after['graph_total_nodes']}",
        f"- matching polluted nodes (must be 0): {after['matching_polluted_nodes']}",
        "",
        "## Artifacts",
        f"- snapshot (full node+edge dump, pre-mutation): `{snapshot_path}`",
        f"- before/after per-node audit: `{csv_path}`",
        f"- progress log: `{out_dir / 'progress.log'}`",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge AI-Town-polluted concept/evidence nodes from orion_substrate"
    )
    parser.add_argument("--uri", default=DEFAULT_FALKORDB_URI)
    parser.add_argument("--graph", default=DEFAULT_GRAPH_NAME)
    parser.add_argument("--dry-run", action="store_true", help="snapshot + report only, no mutation")
    args = parser.parse_args()

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    client = RedisGraphQueryClient(uri=args.uri, graph_name=args.graph)

    before = _counts(client)
    _log_progress(f"before: {json.dumps(before)}")

    snap = _snapshot(client, JOB_DIR)
    _log_progress(f"snapshot: {snap['node_count']} nodes, {snap['edge_count']} outgoing edges -> {snap['path']}")
    _log_progress(f"incoming edges from kept nodes into the polluted subgraph: {snap['incoming_from_kept_count']}")

    if before["matching_polluted_nodes"] != snap["node_count"]:
        _log_progress("ERROR: snapshot node count does not match the live match count -- aborting before any mutation.")
        return 1

    if snap["node_count"] == 0:
        _log_progress("Nothing to purge -- 0 matching polluted nodes. Done.")
        return 0

    if args.dry_run:
        _log_progress(f"DRY RUN: would delete {snap['node_count']} nodes. No changes made.")
        return 0

    _purge(client)
    after = _counts(client)
    _log_progress(f"after: {json.dumps(after)}")

    verdict = "ok" if after["matching_polluted_nodes"] == 0 else "needs_review"
    csv_path = _write_before_after_csv(JOB_DIR, snap["nodes"])
    report_md_path = _write_report_md(
        JOB_DIR, before=before, after=after, snapshot_path=snap["path"], csv_path=csv_path, verdict=verdict,
    )
    report = {
        "job": "purge_aitown_polluted_substrate_concepts",
        "polluted_run_id": _POLLUTED_RUN_ID,
        "before": before,
        "after": after,
        "snapshot_path": str(snap["path"]),
        "before_after_csv_path": str(csv_path),
        "verdict": verdict,
    }
    with open(JOB_DIR / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    _log_progress(f"report: {JOB_DIR / 'report.json'}, {report_md_path}, {csv_path}")
    _log_progress(f"verdict: {verdict}")
    return 0 if verdict == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
