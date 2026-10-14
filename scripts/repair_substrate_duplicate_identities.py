#!/usr/bin/env python3
"""Merge SubstrateNodes that share one identity_key (dry-run by default).

Why: until 2026-10-14 FalkorSubstrateStore resolved identity against its
in-process cache only. Any incomplete cache (FalkorDB's 10k RESULTSET_SIZE cap
before the 2026-10-06 paged hydrate, then a hydrate that failed on the first
duplicate) made the topic-foundry ingest mint a fresh run-scoped node for an
identity that already existed. orion_substrate_aitown and orion_substrate_self
accumulated thousands of duplicate concept/entity nodes.

Merge rule (one rule everywhere -- hydrate and get_node_id_by_identity use it):
  * keeper = the lexically smallest node_id in the identity group
  * every edge on a duplicate is re-pointed to its keeper (both endpoints are
    mapped, so an edge between two duplicates lands keeper->keeper)
  * a re-pointed edge that becomes a self-loop is dropped
  * a re-pointed edge whose (type, source, target) the keeper already has is
    dropped -- the keeper's own edge wins, properties untouched
  * the edge's identity_key has its node_id tokens mapped the same way
  * keeper node properties win, except evidence_refs_json (union) and
    observed_at (latest)
  * duplicates are then DETACH DELETEd

Backfill protocol (CLAUDE.md section 14): the dry run writes, per graph,
  /tmp/<job>/<graph>/snapshot_nodes.jsonl   every node in a duplicate group
  /tmp/<job>/<graph>/snapshot_edges.jsonl   every edge touching a duplicate
  /tmp/<job>/<graph>/plan.json              counts + samples
  /tmp/<job>/report.md, before_after.csv
and refuses to write a snapshot over 100k rows (ask Juniper first; the nightly
FalkorDB RDB backup is the alternative). --apply needs --approved and an
existing snapshot for the graph, and logs progress to /tmp/<job>/progress.log.

Usage:
  python scripts/repair_substrate_duplicate_identities.py \\
      --graph orion_substrate_self --graph orion_substrate_aitown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT_ROW_LIMIT = 100_000
DEFAULT_JOB = "substrate-dedupe-identity"
PAGE = 500


# ── pure planning (unit-tested) ─────────────────────────────────────────────


@dataclass
class EdgeRow:
    edge_id: str
    rel_type: str
    source: str
    target: str
    props: dict[str, Any]


@dataclass
class RepairPlan:
    groups: dict[str, list[str]]  # identity -> sorted node_ids
    keeper_of: dict[str, str]  # duplicate node_id -> keeper node_id
    rewire: list[tuple[EdgeRow, str, str, dict[str, Any]]] = field(default_factory=list)
    drop_self_loops: list[EdgeRow] = field(default_factory=list)
    drop_parallel: list[EdgeRow] = field(default_factory=list)
    node_updates: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def nodes_to_delete(self) -> list[str]:
        return sorted(self.keeper_of)

    def counts(self) -> dict[str, int]:
        return {
            "duplicate_identities": len(self.groups),
            "nodes_in_groups": sum(len(v) for v in self.groups.values()),
            "nodes_to_delete": len(self.keeper_of),
            "edges_rewired": len(self.rewire),
            "edges_dropped_self_loop": len(self.drop_self_loops),
            "edges_dropped_parallel": len(self.drop_parallel),
            "keepers_updated": len(self.node_updates),
        }


def group_duplicates(nodes: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    by_identity: dict[str, list[str]] = defaultdict(list)
    for row in nodes:
        identity = row.get("identity_key")
        if identity:
            by_identity[str(identity)].append(str(row["node_id"]))
    return {k: sorted(v) for k, v in by_identity.items() if len(v) > 1}


def _map_identity(identity: Any, keeper_of: dict[str, str]) -> Any:
    if not isinstance(identity, str) or not identity:
        return identity
    return "|".join(keeper_of.get(part, part) for part in identity.split("|"))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def build_plan(
    nodes: list[dict[str, Any]],
    edges: list[EdgeRow],
    keeper_edge_keys: set[tuple[str, str, str]],
) -> RepairPlan:
    """``nodes``: every node in a duplicate group (node_id, identity_key, and
    the mergeable properties). ``edges``: every edge touching a duplicate
    (non-keeper). ``keeper_edge_keys``: (type, source, target) for the edges
    keepers already hold -- a re-pointed edge colliding with one is dropped."""
    groups = group_duplicates(nodes)
    keeper_of = {dup: ids[0] for ids in groups.values() for dup in ids[1:]}
    plan = RepairPlan(groups=groups, keeper_of=keeper_of)

    taken = set(keeper_edge_keys)
    seen_edges: set[str] = set()
    for edge in sorted(edges, key=lambda e: e.edge_id):
        if edge.edge_id in seen_edges:
            continue  # an edge between two duplicates is listed from both ends
        seen_edges.add(edge.edge_id)
        source = keeper_of.get(edge.source, edge.source)
        target = keeper_of.get(edge.target, edge.target)
        if source == edge.source and target == edge.target:
            continue
        if source == target:
            plan.drop_self_loops.append(edge)
            continue
        key = (edge.rel_type, source, target)
        if key in taken:
            plan.drop_parallel.append(edge)
            continue
        taken.add(key)
        props = dict(edge.props)
        props["identity_key"] = _map_identity(props.get("identity_key"), keeper_of)
        plan.rewire.append((edge, source, target, props))

    by_id = {str(row["node_id"]): row for row in nodes}
    for ids in groups.values():
        keeper = by_id[ids[0]]
        refs: list[Any] = []
        for node_id in ids:
            for ref in _json_list(by_id[node_id].get("evidence_refs_json")):
                if ref not in refs:
                    refs.append(ref)
        observed = max((str(by_id[n].get("observed_at") or "") for n in ids), default="")
        update: dict[str, Any] = {}
        if refs != _json_list(keeper.get("evidence_refs_json")):
            update["evidence_refs_json"] = json.dumps(refs)
        if observed and observed != str(keeper.get("observed_at") or ""):
            update["observed_at"] = observed
        if update:
            plan.node_updates[ids[0]] = update
    return plan


# ── live I/O ────────────────────────────────────────────────────────────────


def _client(uri: str, graph: str, *, read_only: bool):
    from orion.graph.falkor_client import RedisGraphQueryClient

    return RedisGraphQueryClient(uri=uri, graph_name=graph, read_only=read_only, socket_timeout=120.0)


# FalkorDB's RESULTSET_SIZE (10000 on this deployment) silently truncates a
# result -- the first dry run of this script reported exactly 10000 nodes in
# groups on a graph holding 14022. Every read below is chunked, and a chunk
# that comes back at the cap is split and re-read, never trusted.
SERVER_RESULT_CAP = 10_000


def _query_chunked(client, cypher: str, items: list[Any], chunk: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [items[i:i + chunk] for i in range(0, len(items), chunk)][::-1]
    while stack:
        part = stack.pop()
        rows = list(client.graph_query(cypher, {"items": part}) or [])
        if len(rows) >= SERVER_RESULT_CAP:
            if len(part) == 1:
                raise RuntimeError(f"one item returned >= {SERVER_RESULT_CAP} rows; cannot page it: {part[0]!r}")
            mid = len(part) // 2
            stack.extend([part[mid:], part[:mid]])
            continue
        out.extend(rows)
    return out


def fetch_group_nodes(client) -> list[dict[str, Any]]:
    """Keyset-page every identity-bearing node (id(n) cursor, same as the
    store's hydrate), then keep only duplicate groups client-side. A per-key
    UNWIND lookup scans the graph per key without an identity_key index and
    hits the server query timeout."""
    rows: list[dict[str, Any]] = []
    cursor = -1
    while True:
        page = list(client.graph_query(
            "MATCH (n:SubstrateNode) WHERE n.identity_key IS NOT NULL AND n.identity_key <> '' "
            "AND n.payload_json IS NULL AND id(n) > $after "
            "RETURN id(n) AS object_id, n.node_id AS node_id, n.identity_key AS identity_key, "
            "n.node_kind AS node_kind, n.label AS label, n.observed_at AS observed_at, "
            "n.evidence_refs_json AS evidence_refs_json, n.provenance_producer AS producer, "
            "properties(n) AS props ORDER BY object_id LIMIT 2000",
            {"after": cursor},
        ) or [])
        if not page:
            break
        cursor = int(page[-1]["object_id"])
        rows.extend(page)
    groups = group_duplicates(rows)
    members = {node_id for ids in groups.values() for node_id in ids}
    return [row for row in rows if str(row["node_id"]) in members]


def fetch_edges(client, node_ids: list[str], *, keepers: bool) -> list[EdgeRow]:
    # UNWIND + node_id-indexed MATCH per id: an `s.node_id IN $ids OR
    # t.node_id IN $ids` filter scans every edge and hits the server query
    # timeout on the 140k-edge AI Town graph.
    rows = _query_chunked(
        client,
        "UNWIND $items AS id MATCH (n:SubstrateNode {node_id: id})-[e]-(o:SubstrateNode) "
        "RETURN e.edge_id AS edge_id, type(e) AS rel_type, "
        "startNode(e).node_id AS source, endNode(e).node_id AS target"
        + ("" if keepers else ", properties(e) AS props"),
        node_ids, PAGE,
    )
    return [
        EdgeRow(
            edge_id=str(row.get("edge_id") or ""), rel_type=str(row["rel_type"]),
            source=str(row["source"]), target=str(row["target"]),
            props=dict(row.get("props") or {}),
        )
        for row in rows
    ]


def _write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


def dry_run(uri: str, graph: str, out_dir: Path) -> tuple[RepairPlan, dict[str, Any]]:
    client = _client(uri, graph, read_only=True)
    nodes = fetch_group_nodes(client)
    groups = group_duplicates(nodes)
    keepers = sorted({ids[0] for ids in groups.values()})
    dups = sorted({d for ids in groups.values() for d in ids[1:]})
    dup_edges = fetch_edges(client, dups, keepers=False)
    keeper_edges = fetch_edges(client, keepers, keepers=True)
    dup_set = set(dups)
    keeper_keys = {(e.rel_type, e.source, e.target) for e in keeper_edges
                   if e.source not in dup_set and e.target not in dup_set}
    plan = build_plan(nodes, dup_edges, keeper_keys)

    graph_dir = out_dir / graph
    graph_dir.mkdir(parents=True, exist_ok=True)
    unique_dup_edges = {e.edge_id: e for e in dup_edges}
    snapshot_rows = len(nodes) + len(unique_dup_edges)
    snapshot_written = snapshot_rows <= SNAPSHOT_ROW_LIMIT
    for name in ("snapshot_nodes.jsonl", "snapshot_edges.jsonl"):
        (graph_dir / name).unlink(missing_ok=True)  # never leave an older run's snapshot behind
    if snapshot_written:
        _write_jsonl(graph_dir / "snapshot_nodes.jsonl", nodes)
        _write_jsonl(graph_dir / "snapshot_edges.jsonl", (e.__dict__ for e in unique_dup_edges.values()))
    by_kind: dict[str, int] = defaultdict(int)
    by_producer: dict[str, int] = defaultdict(int)
    for row in nodes:
        if row["node_id"] in plan.keeper_of:
            by_kind[str(row.get("node_kind"))] += 1
            by_producer[str(row.get("producer"))] += 1
    biggest = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
    summary = {
        "graph": graph,
        **plan.counts(),
        "nodes_to_delete_by_kind": dict(by_kind),
        "nodes_to_delete_by_producer": dict(by_producer),
        "snapshot_rows": snapshot_rows,
        "snapshot_written": snapshot_written,
        "largest_groups": [{"identity_key": k, "copies": len(v), "keeper": v[0]} for k, v in biggest],
        "sample_rewire": [
            {"edge_id": e.edge_id, "type": e.rel_type, "before": [e.source, e.target], "after": [s, t]}
            for e, s, t, _ in plan.rewire[:5]
        ],
    }
    (graph_dir / "plan.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return plan, summary


def apply_plan(uri: str, graph: str, plan: RepairPlan, out_dir: Path) -> dict[str, int]:
    """Writes. Only reached with --apply --approved and a snapshot on disk."""
    client = _client(uri, graph, read_only=False)
    log = (out_dir / "progress.log").open("a", encoding="utf-8")
    started = time.monotonic()
    errors = 0
    total = len(plan.rewire) + len(plan.node_updates) + len(plan.keeper_of)
    done = 0

    def progress(stage: str) -> None:
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("nan")
        log.write(
            f"{graph} {stage} {100.0 * done / max(total, 1):.1f}% eta_s={eta:.0f} "
            f"{done}/{total} rate={rate:.1f}/s errors={errors}\n"
        )
        log.flush()

    for edge, source, target, props in plan.rewire:
        try:
            client.graph_query(
                "MATCH (s:SubstrateNode {node_id: $s}), (t:SubstrateNode {node_id: $t}) "
                f"CREATE (s)-[e:`{edge.rel_type}`]->(t) SET e = $props",
                {"s": source, "t": target, "props": props},
            )
        except Exception as exc:  # noqa: BLE001 - counted and reported
            errors += 1
            log.write(f"{graph} rewire_error edge_id={edge.edge_id} error={exc}\n")
        done += 1
        if done % 500 == 0:
            progress("rewire")
    for node_id, update in plan.node_updates.items():
        try:
            sets = ", ".join(f"n.{k} = ${k}" for k in update)
            client.graph_query(f"MATCH (n:SubstrateNode {{node_id: $node_id}}) SET {sets}",
                               {"node_id": node_id, **update})
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.write(f"{graph} keeper_update_error node_id={node_id} error={exc}\n")
        done += 1
    progress("keepers")
    dups = plan.nodes_to_delete
    for i in range(0, len(dups), 500):
        chunk = dups[i:i + 500]
        try:
            client.graph_query("MATCH (n:SubstrateNode) WHERE n.node_id IN $ids DETACH DELETE n", {"ids": chunk})
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.write(f"{graph} delete_error first={chunk[0]} error={exc}\n")
        done += len(chunk)
        progress("delete")
    log.close()
    return {"errors": errors, "done": done, "total": total}


def write_report(out_dir: Path, summaries: list[dict[str, Any]], applied: bool) -> None:
    with (out_dir / "before_after.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["graph", "identity_key", "copies_before", "copies_after", "keeper"])
        for s in summaries:
            for g in s["largest_groups"]:
                writer.writerow([s["graph"], g["identity_key"], g["copies"], 1, g["keeper"]])
    lines = [f"# Substrate duplicate-identity repair ({'APPLIED' if applied else 'DRY RUN -- nothing written'})", ""]
    for s in summaries:
        lines += [
            f"## {s['graph']}", "",
            f"- duplicate identities: {s['duplicate_identities']}",
            f"- nodes in duplicate groups: {s['nodes_in_groups']}; to delete: {s['nodes_to_delete']}"
            f" (by kind {s['nodes_to_delete_by_kind']}; by producer {s['nodes_to_delete_by_producer']})",
            f"- edges re-pointed to keeper: {s['edges_rewired']}",
            f"- edges dropped (would become self-loops): {s['edges_dropped_self_loop']}",
            f"- edges dropped (keeper already has the same type+endpoints): {s['edges_dropped_parallel']}",
            f"- keeper nodes updated (evidence_refs union / latest observed_at): {s['keepers_updated']}",
            f"- snapshot rows: {s['snapshot_rows']} -- "
            + ("written" if s["snapshot_written"] else
               f"NOT written (over {SNAPSHOT_ROW_LIMIT}); needs Juniper's call before --apply"),
            "",
        ]
    lines += ["Verdict: needs approval before --apply. Re-run the dry run immediately before applying;",
              "counts grow until the writer fix is deployed.", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", action="append", required=True)
    parser.add_argument("--uri", default=os.getenv("FALKORDB_URI", "redis://127.0.0.1:6380"))
    parser.add_argument("--job", default=DEFAULT_JOB)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved", action="store_true", help="Juniper approved this exact apply")
    args = parser.parse_args(argv)

    out_dir = Path("/tmp") / args.job
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.apply and not args.approved:
        print("refusing --apply without --approved (CLAUDE.md section 14)", file=sys.stderr)
        return 2
    summaries = []
    for graph in args.graph:
        plan, summary = dry_run(args.uri, graph, out_dir)
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("largest_groups", "sample_rewire")}))
        if args.apply:
            if not summary["snapshot_written"]:
                print(f"refusing --apply on {graph}: no snapshot written (too large)", file=sys.stderr)
                return 3
            result = apply_plan(args.uri, graph, plan, out_dir)
            print(json.dumps({"graph": graph, **result}))
    write_report(out_dir, summaries, applied=args.apply)
    print(f"report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
