from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.graph_cognition.views import GraphViewBundleV1
from orion.substrate.store import MaterializedSubstrateGraphState


@dataclass(frozen=True)
class GraphFeatureSetV1:
    structural: dict[str, float]
    temporal: dict[str, float]
    semantic: dict[str, float]
    dynamic: dict[str, float]
    social_executive: dict[str, float]
    degraded: bool
    notes: tuple[str, ...]


def _component_count(node_ids: set[str], edges: dict[str, SubstrateEdgeV1]) -> int:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge.source.node_id in node_ids and edge.target.node_id in node_ids:
            graph[edge.source.node_id].add(edge.target.node_id)
            graph[edge.target.node_id].add(edge.source.node_id)
    seen: set[str] = set()
    components = 0
    for node in node_ids:
        if node in seen:
            continue
        components += 1
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(graph.get(current, set()) - seen)
    return components


def extract_graph_features(
    *,
    state: MaterializedSubstrateGraphState,
    views: GraphViewBundleV1,
    now: datetime,
    lookback_seconds: int = 86400,
) -> GraphFeatureSetV1:
    notes: list[str] = []
    nodes = state.nodes
    edges = state.edges
    if not nodes:
        return GraphFeatureSetV1({}, {}, {}, {}, {}, degraded=True, notes=("empty_substrate",))

    node_ids = set(views.self_view.node_ids or nodes.keys())
    active_nodes = [nodes[nid] for nid in node_ids if nid in nodes]
    edge_count = sum(1 for edge in edges.values() if edge.source.node_id in node_ids and edge.target.node_id in node_ids)
    component_count = _component_count(node_ids, edges)
    avg_degree = (2.0 * edge_count / len(node_ids)) if node_ids else 0.0

    degrees: dict[str, int] = defaultdict(int)
    for edge in edges.values():
        if edge.source.node_id in node_ids:
            degrees[edge.source.node_id] += 1
        if edge.target.node_id in node_ids:
            degrees[edge.target.node_id] += 1
    bridge_nodes = float(sum(1 for value in degrees.values() if value >= 3))

    cutoff = now - timedelta(seconds=lookback_seconds)
    recent_nodes = [n for n in active_nodes if (n.temporal.observed_at if n.temporal.observed_at.tzinfo else n.temporal.observed_at.replace(tzinfo=timezone.utc)) >= cutoff]
    recent_density = float(len(recent_nodes)) / float(max(1, len(active_nodes)))
    inactivity_seconds = max((now - n.temporal.observed_at).total_seconds() for n in active_nodes)
    resurfacing = sum(1 for n in active_nodes if n.metadata.get("dormancy_updated_at"))

    support_count = sum(1 for edge in edges.values() if edge.predicate == "supports")
    conflict_count = sum(1 for edge in edges.values() if edge.predicate in {"contradicts", "blocks"})
    contradiction_nodes = [n for n in nodes.values() if n.node_kind == "contradiction" and not bool(n.metadata.get("resolved", False))]
    provenance_diversity = len({n.provenance.source_channel for n in active_nodes})

    activation_values = [n.signals.activation.activation for n in active_nodes]
    pressure_values = [float(n.metadata.get("dynamic_pressure") or 0.0) for n in active_nodes]
    dormant_count = sum(1 for n in active_nodes if bool(n.metadata.get("dormant", False)))
    hotspot_count = sum(1 for value in activation_values if value >= 0.6)
    tension_accumulation = sum(float(getattr(n, "intensity", 0.0)) for n in active_nodes if n.node_kind == "tension")

    goals = [n for n in active_nodes if n.node_kind == "goal"]
    stalled_goals = sum(1 for g in goals if str(g.metadata.get("goal_status") or "active").lower() == "blocked")
    unresolved_commitments = sum(1 for g in goals if str(g.metadata.get("goal_status") or "active").lower() != "satisfied")
    retries = sum(float(g.metadata.get("retry_count") or 0.0) for g in goals)

    seekers: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge.predicate == "seeks" and edge.target.node_id in node_ids:
            seekers[edge.target.node_id].add(edge.source.node_id)
    competition_density = float(sum(1 for srcs in seekers.values() if len(srcs) > 1)) / float(max(1, len(goals)))

    degraded = False
    if not views.social.node_ids:
        notes.append("social_view_sparse")
        degraded = True

    return GraphFeatureSetV1(
        structural={
            "node_count": float(len(node_ids)),
            "edge_count": float(edge_count),
            "avg_degree": avg_degree,
            "component_count": float(component_count),
            "bridge_nodes": bridge_nodes,
            "fragmentation": float(max(0, component_count - 1)),
        },
        temporal={
            "recent_change_density": recent_density,
            "inactivity_duration_seconds": inactivity_seconds,
            "resurfacing_count": float(resurfacing),
            "node_churn": float(abs(len(recent_nodes) - (len(active_nodes) - len(recent_nodes)))),
            "escalation_rate": float(len(contradiction_nodes)) / float(max(1, len(active_nodes))),
            "persistence_duration_seconds": float(min((now - n.temporal.observed_at).total_seconds() for n in active_nodes)),
        },
        semantic={
            "support_count": float(support_count),
            "conflict_count": float(conflict_count),
            "contradiction_density": float(len(contradiction_nodes)) / float(max(1, len(active_nodes))),
            "provenance_diversity": float(provenance_diversity),
            "concept_stability": 1.0 - min(1.0, float(len(contradiction_nodes)) / float(max(1, len(goals) + 1))),
        },
        dynamic={
            "mean_activation": sum(activation_values) / float(max(1, len(activation_values))),
            "max_activation": max(activation_values) if activation_values else 0.0,
            "mean_pressure": sum(pressure_values) / float(max(1, len(pressure_values))),
            "max_pressure": max(pressure_values) if pressure_values else 0.0,
            "hotspot_count": float(hotspot_count),
            "dormant_count": float(dormant_count),
            "tension_accumulation": float(tension_accumulation),
            "coherence_trend": max(0.0, 1.0 - (float(len(contradiction_nodes)) / float(max(1, len(active_nodes))))),
        },
        social_executive={
            "unresolved_commitments": float(unresolved_commitments),
            "stalled_goals": float(stalled_goals),
            "repeated_retries": retries,
            "goal_competition_density": competition_density,
            "dependency_blockage": float(conflict_count),
            "reciprocity_balance": 1.0 if views.social.node_ids else 0.0,
        },
        degraded=degraded,
        notes=tuple(notes),
    )
