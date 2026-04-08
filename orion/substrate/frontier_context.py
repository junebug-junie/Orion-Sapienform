from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1
from orion.core.schemas.frontier_expansion import FrontierExpansionRequestV1
from orion.substrate.store import MaterializedSubstrateGraphState


@dataclass(frozen=True)
class FrontierContextPackV1:
    request_id: str
    target_zone: str
    focal_node_ids: tuple[str, ...]
    focal_edge_ids: tuple[str, ...]
    hotspot_node_ids: tuple[str, ...]
    contradiction_node_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    truncated: bool


class FrontierContextPackBuilder:
    def __init__(self, *, max_nodes: int = 48, max_edges: int = 96, hotspot_threshold: float = 0.6) -> None:
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._hotspot_threshold = hotspot_threshold

    def build(
        self,
        *,
        state: MaterializedSubstrateGraphState,
        request: FrontierExpansionRequestV1,
        now: datetime | None = None,
    ) -> FrontierContextPackV1:
        tick = now or datetime.now(timezone.utc)
        if tick.tzinfo is None:
            tick = tick.replace(tzinfo=timezone.utc)

        allowed_nodes = [
            node
            for node in state.nodes.values()
            if node.anchor_scope == request.anchor_scope
            and (request.subject_ref is None or node.subject_ref in (None, request.subject_ref))
        ]
        if request.target_zone == "self_relationship_graph":
            allowed_nodes = [
                node
                for node in allowed_nodes
                if node.node_kind in {"hypothesis", "tension", "contradiction", "goal", "state_snapshot"}
            ]

        focal_ids = set(request.graph_region.focal_node_ids)
        if not focal_ids:
            focal_ids = {node.node_id for node in allowed_nodes}

        scored_nodes: list[tuple[float, BaseSubstrateNodeV1]] = []
        for node in allowed_nodes:
            if node.node_id not in focal_ids:
                continue
            observed = node.temporal.observed_at if node.temporal.observed_at.tzinfo else node.temporal.observed_at.replace(tzinfo=timezone.utc)
            recency = max(0.0, 1.0 - ((tick - observed).total_seconds() / 86400.0))
            score = (node.signals.activation.activation * 0.45) + (float(node.metadata.get("dynamic_pressure") or 0.0) * 0.35) + (recency * 0.2)
            scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        truncated = len(scored_nodes) > self._max_nodes
        selected_nodes = [node for _, node in scored_nodes[: self._max_nodes]]
        selected_ids = {node.node_id for node in selected_nodes}

        edge_ids: list[str] = []
        for edge_id, edge in state.edges.items():
            if edge.source.node_id in selected_ids and edge.target.node_id in selected_ids:
                edge_ids.append(edge_id)
        edge_truncated = len(edge_ids) > self._max_edges
        edge_ids = edge_ids[: self._max_edges]

        hotspot_ids = [
            node.node_id
            for node in selected_nodes
            if node.signals.activation.activation >= self._hotspot_threshold or float(node.metadata.get("dynamic_pressure") or 0.0) >= self._hotspot_threshold
        ]
        contradiction_ids = [node.node_id for node in selected_nodes if node.node_kind == "contradiction" and not bool(node.metadata.get("resolved", False))]
        evidence_refs = sorted({ref for node in selected_nodes for ref in node.provenance.evidence_refs})

        return FrontierContextPackV1(
            request_id=request.request_id,
            target_zone=request.target_zone,
            focal_node_ids=tuple(sorted(selected_ids)),
            focal_edge_ids=tuple(edge_ids),
            hotspot_node_ids=tuple(sorted(hotspot_ids)),
            contradiction_node_ids=tuple(sorted(contradiction_ids)),
            evidence_refs=tuple(evidence_refs[:128]),
            truncated=truncated or edge_truncated,
        )
