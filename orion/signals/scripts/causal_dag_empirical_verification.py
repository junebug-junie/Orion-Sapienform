#!/usr/bin/env python3
"""Read-only diff: ORGAN_REGISTRY's hand-authored causal_parent_organs edges
vs. real observed CAUSALLY_FOLLOWED_BY edges on the live bus synaptic graph.

Idea 2.3 of docs/superpowers/specs/2026-07-25-bus-synaptic-graph-verb-latency-
and-causal-gaps.md. registry.py's own trailing comment admits these edges are
"first-pass structural approximations derived from code inspection" that need
verification against real traffic -- this is that verification, using data
that didn't exist when the registry was written.

Key subtlety this script does NOT ignore: several ORGAN_REGISTRY entries
(e.g. "graph_cognition", "autonomy", "cortex_exec") all map to the same
physical service (orion-cortex-exec), because the registry models internal
reasoning stages, not just bus-visible services. A passive wiretap only ever
sees envelope.source.name -- one value per physical service -- so any
registry edge where both ends resolve to the same live organ_id is
structurally unobservable, not "missing." Reporting those as false negatives
would be dishonest; they're tracked in their own bucket instead.

Live organ_id is derived from each registry entry's `service` field
(stripping the "orion-" prefix, e.g. "orion-cortex-exec" -> "cortex-exec") --
this repo's own live-organ-naming convention, not a guess. Entries whose
`service` doesn't look like a real bus service name (e.g. "orion library
(orion/journaler/)") are reported as unmapped, not silently skipped or
force-matched.

Usage:

    docker exec orion-athena-recall python3 \
        orion/signals/scripts/causal_dag_empirical_verification.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.signals.registry import ORGAN_REGISTRY
except ImportError:
    _file_parents = Path(__file__).resolve().parents
    _repo_root = _file_parents[3] if len(_file_parents) > 3 else _file_parents[-1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.signals.registry import ORGAN_REGISTRY


def _live_organ_id_for_service(service: str) -> str | None:
    """"orion-cortex-exec" -> "cortex-exec". Returns None for a service
    string that doesn't look like a real bus service name (e.g. a free-text
    library reference) -- must not be force-matched."""
    service = (service or "").strip()
    if not service.startswith("orion-"):
        return None
    live_id = service[len("orion-") :].strip()
    return live_id or None


def compute_causal_dag_diff(
    registry: dict[str, Any], live_edges: list[tuple[str, str]]
) -> dict[str, Any]:
    """Pure diff, no I/O. ``registry`` maps organ_key -> entry-like object
    with ``.service`` and ``.causal_parent_organs`` attributes (or a dict
    with those keys -- both accepted so tests don't need the real pydantic
    model). ``live_edges`` is the full (source_organ, target_organ) edge
    list from CAUSALLY_FOLLOWED_BY.
    """

    def _get(entry: Any, key: str) -> Any:
        return entry.get(key) if isinstance(entry, dict) else getattr(entry, key)

    organ_key_to_live_id: dict[str, str | None] = {
        key: _live_organ_id_for_service(_get(entry, "service")) for key, entry in registry.items()
    }

    unmapped_entries = sorted(key for key, live_id in organ_key_to_live_id.items() if live_id is None)

    registry_edges: set[tuple[str, str]] = set()
    same_service_internal: set[tuple[str, str]] = set()
    for child_key, entry in registry.items():
        child_live = organ_key_to_live_id.get(child_key)
        if child_live is None:
            continue
        for parent_key in _get(entry, "causal_parent_organs") or []:
            parent_live = organ_key_to_live_id.get(parent_key)
            if parent_live is None:
                continue
            edge = (parent_live, child_live)
            if parent_live == child_live:
                same_service_internal.add(edge)
            else:
                registry_edges.add(edge)

    live_edge_set = set(live_edges)

    confirmed = sorted(registry_edges & live_edge_set)
    missing_from_live = sorted(registry_edges - live_edge_set)
    observed_not_in_registry = sorted(live_edge_set - registry_edges)

    return {
        "unmapped_registry_entries": unmapped_entries,
        "same_service_internal_edges": sorted(same_service_internal),
        "confirmed": confirmed,
        "missing_from_live": missing_from_live,
        "observed_not_in_registry": observed_not_in_registry,
    }


def main() -> None:
    uri = os.getenv("FALKORDB_URI", "redis://orion-athena-falkordb:6379")
    graph_name = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    client = RedisGraphQueryClient(uri=uri, graph_name=graph_name)

    rows = client.graph_query(
        """
        MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        RETURN a.organ_id AS source_organ, b.organ_id AS target_organ
        """
    )
    live_edges = [(r["source_organ"], r["target_organ"]) for r in rows]

    result = compute_causal_dag_diff(ORGAN_REGISTRY, live_edges)

    print(f"live CAUSALLY_FOLLOWED_BY edges: {len(live_edges)}")
    print(f"registry entries: {len(ORGAN_REGISTRY)}")
    print(f"unmapped registry entries (no clear live organ_id): {result['unmapped_registry_entries']}")
    print(f"same-service internal edges (structurally unobservable): {len(result['same_service_internal_edges'])}")
    for edge in result["same_service_internal_edges"]:
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"\nconfirmed (registry edge, real live evidence): {len(result['confirmed'])}")
    for edge in result["confirmed"]:
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"\nmissing from live (registry claims it, no live evidence): {len(result['missing_from_live'])}")
    for edge in result["missing_from_live"]:
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"\nobserved, not in registry (real edge the hand-authored DAG never named): {len(result['observed_not_in_registry'])}")
    for edge in result["observed_not_in_registry"]:
        print(f"  {edge[0]} -> {edge[1]}")


if __name__ == "__main__":
    main()
