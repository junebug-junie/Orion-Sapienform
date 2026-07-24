#!/usr/bin/env python3
"""Delete stale pre-normalization Channel nodes from the live bus synaptic
graph (``orion_bus_synapse``).

Idea F of docs/superpowers/specs/2026-07-24-bus-synaptic-graph-wider-net-brainstorm.md.
PR #1337 added ``orion.bus.census.normalize_channel_name()`` to collapse
dynamic per-request reply channels (e.g.
``orion:exec:result:LLMGatewayService:<uuid>``) to their wildcard catalog
entry -- but that fix only applies at write time, going forward. Every
literal per-UUID Channel node created *before* the fix shipped is still
sitting in the graph, uncollapsed, forever (``MERGE`` matches on exact
string; nothing ever deletes the old node once a new normalized write starts
landing on the wildcard node instead).

Live-verified before writing this script (2026-07-24): the graph had 9,436
Channel nodes against a 264-entry real catalog. 8,622 had exactly one
PUBLISHES observation (``count=1``). The most recently-touched such literal
node was ~4.5 hours stale while its corresponding wildcard node (e.g.
``orion:vision:reply:*``) was being updated every few seconds -- confirming
the fix works correctly for new traffic and this is purely historical debt,
not an active leak.

Uses the real, live ``normalize_channel_name()`` function itself as the
source of truth for "is this node stale": a Channel node is a deletion
candidate if and only if running today's production normalization logic on
its name produces a *different* string. If it does, that node's traffic
history is already correctly represented under the wildcard node it would
now normalize to -- this node is pure duplicate weight, not lost data.

Safe by construction: never deletes a node whose name already survives
normalization unchanged (real catalog entries, or currently-relevant
wildcard nodes themselves are always no-ops under normalize_channel_name).

Usage (dry run is the default -- nothing is deleted without --execute):

    docker exec orion-athena-recall python3 \
        services/orion-bus-mirror/scripts/stale_channel_node_cleanup.py

    docker exec orion-athena-recall python3 \
        services/orion-bus-mirror/scripts/stale_channel_node_cleanup.py --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from orion.bus.census import load_channel_catalog_names, normalize_channel_name
    from orion.graph.falkor_client import RedisGraphQueryClient
except ImportError:
    # Repo checkout layout (services/orion-bus-mirror/scripts/<this file>) --
    # orion/ isn't importable from cwd. Container layouts (e.g. copied
    # standalone into another service's image, which already has orion/ on
    # its default path) hit the try branch above and never reach this.
    _file_parents = Path(__file__).resolve().parents
    _repo_root = _file_parents[3] if len(_file_parents) > 3 else _file_parents[-1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from orion.bus.census import load_channel_catalog_names, normalize_channel_name
    from orion.graph.falkor_client import RedisGraphQueryClient


def compute_stale_channel_candidates(
    channel_rows: list[dict], catalog_names: set[str]
) -> list[dict]:
    """Pure function: given every Channel node's name + its aggregate
    PUBLISHES count, return the subset that today's ``normalize_channel_name``
    would collapse to something else -- i.e. nodes whose data is already
    duplicated, correctly, under a different (wildcard) node.
    """
    candidates = []
    for row in channel_rows:
        channel = row["channel"]
        normalized = normalize_channel_name(channel, catalog_names)
        if normalized != channel:
            candidates.append({**row, "normalizes_to": normalized})
    return candidates


def _fetch_channel_rows(client: RedisGraphQueryClient) -> list[dict]:
    rows = client.graph_query(
        """
        MATCH (ch:Channel)
        OPTIONAL MATCH ()-[e:PUBLISHES]->(ch)
        RETURN ch.channel AS channel, count(e) AS edge_count
        """
    )
    return [{"channel": r["channel"], "edge_count": int(r["edge_count"])} for r in rows]


def _delete_channels(client: RedisGraphQueryClient, channel_names: list[str], *, batch_size: int) -> int:
    deleted = 0
    for i in range(0, len(channel_names), batch_size):
        batch = channel_names[i : i + batch_size]
        client.graph_query(
            """
            UNWIND $names AS name
            MATCH (ch:Channel {channel: name})
            DETACH DELETE ch
            """,
            {"names": batch},
        )
        deleted += len(batch)
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually delete stale nodes (default: dry run)")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    uri = os.getenv("FALKORDB_URI", "redis://orion-athena-falkordb:6379")
    graph_name = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    client = RedisGraphQueryClient(uri=uri, graph_name=graph_name)

    catalog_names = load_channel_catalog_names()
    print(f"catalog channel names loaded: {len(catalog_names)}")

    channel_rows = _fetch_channel_rows(client)
    print(f"total Channel nodes: {len(channel_rows)}")

    candidates = compute_stale_channel_candidates(channel_rows, catalog_names)
    print(f"stale candidates (would normalize to a different name today): {len(candidates)}")

    if candidates:
        total_edges_on_candidates = sum(c["edge_count"] for c in candidates)
        print(f"total PUBLISHES edges on candidates: {total_edges_on_candidates}")
        print("sample candidates (up to 10):")
        for c in candidates[:10]:
            print(f"  {c['channel']!r} -> {c['normalizes_to']!r} (edge_count={c['edge_count']})")

    if not args.execute:
        print("\nDRY RUN -- no nodes deleted. Re-run with --execute to actually delete the above candidates.")
        return

    if not candidates:
        print("\nNothing to delete.")
        return

    names = [c["channel"] for c in candidates]
    deleted = _delete_channels(client, names, batch_size=args.batch_size)
    print(f"\nDeleted {deleted} stale Channel nodes.")


if __name__ == "__main__":
    main()
