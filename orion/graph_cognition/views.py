from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.substrate.store import MaterializedSubstrateGraphState


@dataclass(frozen=True)
class SubgraphViewV1:
    name: str
    scope: str
    subject_ref: str | None
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    truncated: bool
    max_nodes: int
    max_edges: int
    time_window_seconds: int


@dataclass(frozen=True)
class SemanticGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class EpisodicGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class SelfGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class SocialGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class ExecutiveGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class ConceptGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class ContradictionGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class TemporalDeltaGraphViewV1(SubgraphViewV1):
    pass


@dataclass(frozen=True)
class GraphViewBundleV1:
    semantic: SemanticGraphViewV1
    episodic: EpisodicGraphViewV1
    self_view: SelfGraphViewV1
    social: SocialGraphViewV1
    executive: ExecutiveGraphViewV1
    concept: ConceptGraphViewV1
    contradiction: ContradictionGraphViewV1
    temporal_delta: TemporalDeltaGraphViewV1


def _score(node: BaseSubstrateNodeV1) -> float:
    pressure = float(node.metadata.get("dynamic_pressure") or 0.0)
    activation = node.signals.activation.activation
    return (activation * 0.5) + (pressure * 0.3) + (node.signals.salience * 0.2)


def _within_window(node: BaseSubstrateNodeV1, *, now: datetime, window_seconds: int) -> bool:
    observed = node.temporal.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed >= now - timedelta(seconds=window_seconds)


def _select_nodes(
    nodes: Iterable[BaseSubstrateNodeV1],
    *,
    now: datetime,
    window_seconds: int,
    max_nodes: int,
) -> tuple[list[BaseSubstrateNodeV1], bool]:
    candidates = [node for node in nodes if _within_window(node, now=now, window_seconds=window_seconds)]
    candidates.sort(key=_score, reverse=True)
    truncated = len(candidates) > max_nodes
    return candidates[:max_nodes], truncated


def _edges_for_nodes(edges: dict[str, SubstrateEdgeV1], node_ids: set[str], *, max_edges: int) -> tuple[list[str], bool]:
    ids = [
        edge_id
        for edge_id, edge in edges.items()
        if edge.source.node_id in node_ids and edge.target.node_id in node_ids
    ]
    truncated = len(ids) > max_edges
    return ids[:max_edges], truncated


def _build_view(
    *,
    state: MaterializedSubstrateGraphState,
    name: str,
    node_predicate,
    now: datetime,
    scope: str,
    subject_ref: str | None,
    max_nodes: int,
    max_edges: int,
    time_window_seconds: int,
    cls,
) -> SubgraphViewV1:
    scoped_nodes = [
        node
        for node in state.nodes.values()
        if node.anchor_scope == scope and (subject_ref is None or node.subject_ref in (None, subject_ref)) and node_predicate(node)
    ]
    selected, node_truncated = _select_nodes(scoped_nodes, now=now, window_seconds=time_window_seconds, max_nodes=max_nodes)
    node_ids = {item.node_id for item in selected}
    edge_ids, edge_truncated = _edges_for_nodes(state.edges, node_ids, max_edges=max_edges)
    return cls(
        name=name,
        scope=scope,
        subject_ref=subject_ref,
        node_ids=tuple(sorted(node_ids)),
        edge_ids=tuple(edge_ids),
        truncated=node_truncated or edge_truncated,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
    )


def build_graph_views(
    *,
    state: MaterializedSubstrateGraphState,
    now: datetime,
    scope: str,
    subject_ref: str | None,
    max_nodes: int = 64,
    max_edges: int = 128,
    time_window_seconds: int = 86400,
) -> GraphViewBundleV1:
    semantic = _build_view(
        state=state,
        name="semantic",
        node_predicate=lambda n: n.node_kind in {"entity", "concept", "evidence", "hypothesis", "ontology_branch", "event"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=SemanticGraphViewV1,
    )
    episodic = _build_view(
        state=state,
        name="episodic",
        node_predicate=lambda n: n.node_kind in {"event", "state_snapshot", "evidence"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=EpisodicGraphViewV1,
    )
    self_view = _build_view(
        state=state,
        name="self",
        node_predicate=lambda n: n.subject_ref == subject_ref,
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=SelfGraphViewV1,
    )
    social = _build_view(
        state=state,
        name="social",
        node_predicate=lambda n: n.anchor_scope == "relationship" or bool(n.metadata.get("social")),
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=SocialGraphViewV1,
    )
    executive = _build_view(
        state=state,
        name="executive",
        node_predicate=lambda n: n.node_kind in {"drive", "goal", "tension", "state_snapshot"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=ExecutiveGraphViewV1,
    )
    concept = _build_view(
        state=state,
        name="concept",
        node_predicate=lambda n: n.node_kind in {"concept", "hypothesis", "ontology_branch"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=ConceptGraphViewV1,
    )
    contradiction = _build_view(
        state=state,
        name="contradiction",
        node_predicate=lambda n: n.node_kind in {"contradiction", "tension", "goal"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=ContradictionGraphViewV1,
    )
    temporal_delta = _build_view(
        state=state,
        name="temporal_delta",
        node_predicate=lambda n: bool(n.metadata.get("materialization_lineage")) or n.node_kind in {"event", "state_snapshot"},
        now=now,
        scope=scope,
        subject_ref=subject_ref,
        max_nodes=max_nodes,
        max_edges=max_edges,
        time_window_seconds=time_window_seconds,
        cls=TemporalDeltaGraphViewV1,
    )
    return GraphViewBundleV1(
        semantic=semantic,
        episodic=episodic,
        self_view=self_view,
        social=social,
        executive=executive,
        concept=concept,
        contradiction=contradiction,
        temporal_delta=temporal_delta,
    )
