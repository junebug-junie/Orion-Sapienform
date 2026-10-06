"""Bounded semantic reads. These receipts are not transactional snapshots or ACLs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1, SubstrateAnchorScopeV1, SubstrateEdgePredicateV1,
    SubstrateEdgeV1, SubstratePromotionStateV1,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NeighborhoodRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    focal_node_ids: tuple[str, ...] = Field(max_length=16)
    direction: Literal["both", "incoming", "outgoing"] = "both"
    semantic_states: tuple[SubstratePromotionStateV1, ...] = ("provisional", "canonical")
    anchor_scopes: tuple[SubstrateAnchorScopeV1, ...] = get_args(SubstrateAnchorScopeV1)
    internal_edge_limit: int = Field(default=12, ge=0, le=256)
    boundary_edge_limit: int = Field(default=16, ge=0, le=256)
    neighbor_node_limit: int = Field(default=16, ge=0, le=256)
    # No stable snapshot token exists yet. Never silently reuse a mutable cursor.
    continuation: str | None = None

    def eligible(self, node: BaseSubstrateNodeV1) -> bool:
        return (node.node_kind in {"concept", "entity"}
                and node.promotion_state in self.semantic_states
                and node.anchor_scope in self.anchor_scopes)


@dataclass(frozen=True)
class NeighborhoodResultV1:
    focal_nodes: list[BaseSubstrateNodeV1] = field(default_factory=list)
    neighbor_nodes: list[BaseSubstrateNodeV1] = field(default_factory=list)
    internal_edges: list[SubstrateEdgeV1] = field(default_factory=list)
    boundary_edges: list[SubstrateEdgeV1] = field(default_factory=list)
    source_kind: str = "cache"
    read_started_at: str = field(default_factory=now)
    read_finished_at: str = field(default_factory=now)
    complete_for_request: bool = False
    truncated: bool = False
    degraded: bool = False
    reason: str | None = None
    missing_focal_node_ids: tuple[str, ...] = ()
    continuations: tuple[str, ...] = ()
    consistency: str = "best_effort_non_atomic"


# A boundary group has one focal node, one direction, and one predicate.
Group = tuple[str, str, str]


def read_neighborhood(
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


def _unique_nodes(values: list[BaseSubstrateNodeV1]) -> dict[str, BaseSubstrateNodeV1]:
    result = {node.node_id: node for node in values}
    if len(result) != len(values):
        raise ValueError("duplicate_node_id")
    return result


def read_memory_neighborhood(store, request: NeighborhoodRequestV1) -> NeighborhoodResultV1:
    # Local fixture store already owns the whole graph. Never call snapshot(),
    # which could trigger durable hydration on a different backend.
    def eligible_edges(ids):
        for edge in store._edges.values():
            src, dst = store._nodes.get(edge.source.node_id), store._nodes.get(edge.target.node_id)
            if src and dst and request.eligible(src) and request.eligible(dst):
                if src.node_id in ids or dst.node_id in ids:
                    yield edge

    def groups(ids):
        result = set()
        for edge in eligible_edges(ids):
            src, dst = edge.source.node_id, edge.target.node_id
            if src in ids and dst not in ids:
                result.add((src, "outgoing", edge.predicate))
            if dst in ids and src not in ids:
                result.add((dst, "incoming", edge.predicate))
        return sorted(result)

    def edges(ids, group, after, limit):
        values = []
        for edge in eligible_edges(ids):
            src, dst = edge.source.node_id, edge.target.node_id
            match = src in ids and dst in ids if group is None else (
                edge.predicate == group[2] and (
                    (group[1] == "incoming" and dst == group[0] and src not in ids) or
                    (group[1] == "outgoing" and src == group[0] and dst not in ids)))
            if match and edge.edge_id > after:
                values.append(edge)
        return sorted(values, key=lambda edge: edge.edge_id)[:limit]

    return read_neighborhood(request, source_kind="cache",
        nodes=lambda ids: [store._nodes[key] for key in ids if key in store._nodes],
        groups=groups, edges=edges)
