#!/usr/bin/env python3
"""One-off migration: copy the old ``EXECUTES_VERB`` edges' ``last_node``
property onto the new ``node`` property Idea 2.1 introduces
(docs/superpowers/specs/2026-07-25-bus-synaptic-graph-verb-latency-and-causal-gaps.md).

Why this is needed: as of that patch, ``BusSynapticGraphWriter.record_verb_step``
matches/creates edges via ``MERGE (o)-[e:EXECUTES_VERB {node: $node}]->(v)`` --
``node`` is now part of the relationship's identity, not a mutable ``SET``
property. Edges written by the *old* code (which only ever set ``e.last_node``,
never ``e.node``) would never match that new pattern again -- they'd sit
orphaned forever with a stale, blended baseline, exactly the same class of
problem ``stale_channel_node_cleanup.py`` fixed for pre-normalization Channel
nodes. Unlike that fix, this one is purely additive (copies a value onto a new
property, deletes nothing) -- safe to run without a separate dry-run/execute
split.

Already run live 2026-07-24 against the real ``orion_bus_synapse`` graph (11
edges, all migrated cleanly, verified via direct query before and after) as
part of shipping this PR -- this script exists so the migration is reproducible
and auditable, not just a one-off interactive command.

Usage:

    docker exec orion-athena-recall python3 \
        services/orion-bus-mirror/scripts/migrate_execute_verb_node_property.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from orion.graph.falkor_client import RedisGraphQueryClient
except ImportError:
    _file_parents = Path(__file__).resolve().parents
    _repo_root = _file_parents[3] if len(_file_parents) > 3 else _file_parents[-1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from orion.graph.falkor_client import RedisGraphQueryClient


def main() -> None:
    uri = os.getenv("FALKORDB_URI", "redis://orion-athena-falkordb:6379")
    graph_name = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    client = RedisGraphQueryClient(uri=uri, graph_name=graph_name)

    result = client.graph_query(
        """
        MATCH (:Organ)-[e:EXECUTES_VERB]->(:Verb)
        WHERE e.node IS NULL
        SET e.node = coalesce(e.last_node, "unknown")
        RETURN count(e) AS migrated
        """
    )
    migrated = result[0]["migrated"] if result else 0
    print(f"migrated {migrated} EXECUTES_VERB edges (last_node -> node)")

    verify = client.graph_query(
        """
        MATCH (o:Organ)-[e:EXECUTES_VERB]->(v:Verb)
        RETURN o.organ_id AS organ_id, v.verb_name AS verb_name, e.node AS node, e.count AS count
        """
    )
    for row in verify:
        print(row)


if __name__ == "__main__":
    main()
