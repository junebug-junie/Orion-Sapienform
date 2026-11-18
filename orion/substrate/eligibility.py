"""Which substrate nodes and edges take part in cognition (dynamics, attention, brain frame).

#2497 rule 8 ("prevent provenance from changing cognition by accident"): source,
evidence and assertion structure must not enter pressure/activation paths, and
introducing them must not change existing eligible topology. So the rule is
deliberately the old graph exactly:

- a node is cognitive unless it is an Assertion or was written by an
  identity-fenced producer (memory referents, their evidence nodes);
- an edge is cognitive only if it is ``legacy_unreviewed`` (every edge that
  existed before edge roles) and touches no excluded node. An edge whose
  endpoint is missing from the snapshot keeps today's treatment.

New ``semantic_projection`` edges are walkable by the neighborhood read but do not
propagate pressure or activation yet. Admitting them is a separate, measured
decision, not a side effect of this patch.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1

from .reconcile import is_identity_fenced

COGNITIVE_EDGE_ROLES: frozenset[str] = frozenset({"legacy_unreviewed"})


def is_cognitive_node(node: Any) -> bool:
    if not isinstance(node, BaseSubstrateNodeV1):
        # Duck-typed callers (attention tests, brain-frame fixtures) pass plain
        # objects; only real substrate nodes can be fenced or assertions.
        return getattr(node, "node_kind", None) != "assertion"
    return not is_identity_fenced(node)


def is_cognitive_edge(edge: SubstrateEdgeV1, excluded_node_ids: set[str] | frozenset[str]) -> bool:
    return (
        getattr(edge, "edge_role", "legacy_unreviewed") in COGNITIVE_EDGE_ROLES
        and edge.source.node_id not in excluded_node_ids
        and edge.target.node_id not in excluded_node_ids
    )


def cognitive_nodes(nodes: Iterable[Any]) -> list[Any]:
    return [node for node in nodes if is_cognitive_node(node)]


def cognitive_view(
    nodes: Mapping[str, BaseSubstrateNodeV1], edges: Mapping[str, SubstrateEdgeV1]
) -> tuple[dict[str, BaseSubstrateNodeV1], dict[str, SubstrateEdgeV1]]:
    """The cognitive subgraph of a snapshot: same dict shapes, same insertion order."""
    kept_nodes = {node_id: node for node_id, node in nodes.items() if is_cognitive_node(node)}
    excluded = frozenset(nodes) - frozenset(kept_nodes)
    kept_edges = {edge_id: edge for edge_id, edge in edges.items() if is_cognitive_edge(edge, excluded)}
    return kept_nodes, kept_edges
