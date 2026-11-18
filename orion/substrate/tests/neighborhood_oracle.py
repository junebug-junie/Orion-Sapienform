"""Test oracle: read_neighborhood as merged in #2497 (origin/main 029322db2),
before memory Stage 2 PR E batched the neighbor fetch. Frozen verbatim except
for the function name. Do not edit to match the live code: its only job is to
prove the live algorithm returns the same receipts.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, get_args

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1, SubstrateEdgePredicateV1, SubstrateEdgeV1,
)
from orion.substrate.neighborhood import (
    Group, NeighborhoodRequestV1, NeighborhoodResultV1, _unique_nodes, now,
)


def read_neighborhood_per_neighbor(
    request: NeighborhoodRequestV1, *, source_kind: str,
    nodes: Callable[[list[str]], list[BaseSubstrateNodeV1]],
    groups: Callable[[list[str]], list[Group]],
    edges: Callable[[list[str], Group | None, str, int], list[SubstrateEdgeV1]],
) -> NeighborhoodResultV1:
    """One algorithm for all backends; callbacks must filter BEFORE limiting.

    Read at most budget+1 edges per group, paging even through short pages.
    Missing/invalid data fails closed, without cache hydration or writeback.
    """
    started = now()
    if request.continuation is not None:
        return NeighborhoodResultV1(source_kind=source_kind, read_started_at=started,
                                    reason="restart_required:continuation_unsupported", degraded=True)
    try:
        requested = sorted(set(request.focal_node_ids))
        focal = _unique_nodes(nodes(requested)) if requested else {}
        if not set(focal) <= set(requested):
            raise ValueError("unexpected_focal_node")
        focal = {key: node for key, node in focal.items() if request.eligible(node)}
        missing = tuple(sorted(set(requested) - set(focal)))
        ids = sorted(focal)
        if not ids:
            return NeighborhoodResultV1(source_kind=source_kind, read_started_at=started,
                complete_for_request=not missing, missing_focal_node_ids=missing,
                degraded=bool(missing), reason="focal_unavailable_or_filtered" if missing else None)

        def bounded(group: Group | None, limit: int) -> list[SubstrateEdgeV1]:
            result: list[SubstrateEdgeV1] = []
            after = ""
            while len(result) <= limit:
                page = edges(ids, group, after, min(64, limit + 1 - len(result)))
                if not page:
                    break
                if len(page) > min(64, limit + 1 - len(result)):
                    raise ValueError("edge_page_exceeds_limit")
                for edge in page:
                    if edge.edge_id <= after:
                        raise ValueError("nonadvancing_or_duplicate_edge_id")
                    after = edge.edge_id
                    src, dst = edge.source.node_id, edge.target.node_id
                    if group is None:
                        valid = src in focal and dst in focal
                    else:
                        focus, direction, predicate = group
                        valid = (edge.predicate == predicate and
                            ((direction == "incoming" and dst == focus and src not in focal) or
                             (direction == "outgoing" and src == focus and dst not in focal)))
                    if not valid:
                        raise ValueError("edge_outside_requested_group")
                    result.append(edge)
            return result

        internal = bounded(None, request.internal_edge_limit)
        truncated = len(internal) > request.internal_edge_limit
        internal = internal[:request.internal_edge_limit]
        boundary_groups = sorted(set(groups(ids)))
        for focus, direction, predicate in boundary_groups:
            if (focus not in focal or direction not in {"incoming", "outgoing"}
                    or predicate not in get_args(SubstrateEdgePredicateV1)):
                raise ValueError("invalid_boundary_group")
        boundary_groups = [g for g in boundary_groups if request.direction in {"both", g[1]}]
        candidates = {g: deque(bounded(g, request.boundary_edge_limit)) for g in boundary_groups}
        # Focal-first nested rounds: one focal gets one turn, rotating its
        # directions and predicates. Having both directions or many predicates
        # must not buy a hub extra turns ahead of a smaller focal node.
        predicates: dict[tuple[str, str], deque[Group]] = {}
        directions: dict[str, deque[str]] = {}
        for group in boundary_groups:
            key = group[:2]
            if key not in predicates:
                predicates[key] = deque()
                directions.setdefault(group[0], deque()).append(group[1])
            predicates[key].append(group)
        ordered: list[SubstrateEdgeV1] = []
        while directions:
            for focal_id in list(directions):
                direction_queue = directions[focal_id]
                while direction_queue:
                    direction = direction_queue.popleft()
                    queue = predicates[(focal_id, direction)]
                    while queue and not candidates[queue[0]]:
                        queue.popleft()
                    if not queue:
                        continue
                    group = queue.popleft()
                    ordered.append(candidates[group].popleft())
                    queue.append(group)
                    direction_queue.append(direction)
                    break
                if not direction_queue:
                    del directions[focal_id]

        neighbors: dict[str, BaseSubstrateNodeV1] = {}
        boundary: list[SubstrateEdgeV1] = []
        seen_edges = {edge.edge_id for edge in internal}
        for edge in ordered:
            if edge.edge_id in seen_edges:
                raise ValueError("duplicate_edge_id")
            seen_edges.add(edge.edge_id)
            outside = edge.target.node_id if edge.source.node_id in focal else edge.source.node_id
            if (len(boundary) >= request.boundary_edge_limit or
                    (outside not in neighbors and len(neighbors) >= request.neighbor_node_limit)):
                truncated = True
                continue
            if outside not in neighbors:
                found = _unique_nodes(nodes([outside]))
                if set(found) != {outside} or not request.eligible(found[outside]):
                    raise ValueError("endpoint_changed_or_unavailable")
                neighbors[outside] = found[outside]
            boundary.append(edge)
        all_nodes = {**focal, **neighbors}
        for edge in internal + boundary:
            for ref in (edge.source, edge.target):
                if all_nodes[ref.node_id].node_kind != ref.node_kind:
                    raise ValueError("endpoint_kind_mismatch")
        return NeighborhoodResultV1(
            focal_nodes=[focal[key] for key in ids], neighbor_nodes=list(neighbors.values()),
            internal_edges=internal, boundary_edges=boundary, source_kind=source_kind,
            read_started_at=started, read_finished_at=now(), truncated=truncated,
            complete_for_request=not truncated and not missing, degraded=bool(missing),
            missing_focal_node_ids=missing,
            reason="focal_unavailable_or_filtered" if missing else "budget_exhausted" if truncated else None,
        )
    except Exception as exc:
        # No stale cache substitution or partially successful result. Error text
        # can contain backend credentials; expose only its type, not its message.
        return NeighborhoodResultV1(source_kind=source_kind, read_started_at=started,
            read_finished_at=now(), degraded=True, reason=f"unavailable:{type(exc).__name__}")
