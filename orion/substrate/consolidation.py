from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from orion.core.schemas.substrate_consolidation import (
    GraphConsolidationDecisionV1,
    GraphConsolidationRequestV1,
    GraphConsolidationResultV1,
    GraphReviewCycleRecordV1,
    GraphStateDeltaDigestV1,
)
from orion.graph_cognition.interpreters import GraphCognitionReportV1
from orion.substrate.frontier_landing import FrontierLandingExecutionResultV1
from orion.substrate.query_planning import SubstrateQueryExecutionMetaV1, SubstrateQueryPlanner, SubstrateSemanticReadCoordinator
from orion.substrate.store import SubstrateGraphStore, SubstrateQueryResultV1


@dataclass(frozen=True)
class GraphConsolidationExecutionV1:
    result: GraphConsolidationResultV1
    cycle_record: GraphReviewCycleRecordV1
    semantic_source: str
    semantic_degraded: bool
    semantic_plan_kind: str
    semantic_reused_cache: bool


class GraphConsolidationEvaluator:
    """Deterministic bounded consolidation evaluator over substrate graph regions."""

    def __init__(self, *, store: SubstrateGraphStore, max_region_nodes: int = 32, max_region_edges: int = 64) -> None:
        self._store = store
        self._max_region_nodes = max_region_nodes
        self._max_region_edges = max_region_edges

    def consolidate(
        self,
        *,
        request: GraphConsolidationRequestV1,
        prior_cycle: GraphReviewCycleRecordV1 | None = None,
        cognition_report: GraphCognitionReportV1 | None = None,
        landing_result: FrontierLandingExecutionResultV1 | None = None,
        max_region_nodes: int | None = None,
        max_region_edges: int | None = None,
        query_cache_enabled: bool = True,
    ) -> GraphConsolidationExecutionV1:
        effective_nodes = max_region_nodes if max_region_nodes is not None else self._max_region_nodes
        effective_edges = max_region_edges if max_region_edges is not None else self._max_region_edges
        semantic_region, semantic_meta = self._select_region(
            request=request,
            max_region_nodes=effective_nodes,
            max_region_edges=effective_edges,
            query_cache_enabled=query_cache_enabled,
        )
        nodes = [
            node
            for node in semantic_region.slice.nodes
            if node.anchor_scope == request.anchor_scope and (request.subject_ref is None or node.subject_ref in (None, request.subject_ref))
        ]
        node_ids = [node.node_id for node in nodes[: effective_nodes]]
        node_set = set(node_ids)
        edges = [
            edge
            for edge in semantic_region.slice.edges
            if edge.source.node_id in node_set and edge.target.node_id in node_set
        ][: effective_edges]
        edge_ids = [edge.edge_id for edge in edges]
        if not node_ids:
            state = self._store.snapshot()
            node_ids, edge_ids = self._select_region_from_state(
                request=request,
                state=state,
                max_region_nodes=effective_nodes,
                max_region_edges=effective_edges,
            )
            nodes = [state.nodes[nid] for nid in node_ids if nid in state.nodes]
            edges = [state.edges[eid] for eid in edge_ids if eid in state.edges]
            semantic_source = "local_fallback"
            semantic_degraded = True
        else:
            semantic_source = semantic_region.source_kind
            semantic_degraded = semantic_region.degraded

        contradiction_count = sum(1 for n in nodes if n.node_kind == "contradiction" and not bool(n.metadata.get("resolved", False)))
        evidence_gap_count = sum(1 for n in nodes if bool(n.metadata.get("frontier_hypothesis_marker", False)))
        mean_activation = sum(n.signals.activation.activation for n in nodes) / float(max(1, len(nodes)))
        mean_pressure = sum(float(n.metadata.get("dynamic_pressure") or 0.0) for n in nodes) / float(max(1, len(nodes)))

        degree: dict[str, int] = defaultdict(int)
        for edge in edges:
            degree[edge.source.node_id] += 1
            degree[edge.target.node_id] += 1
        isolated_frontier_count = sum(
            1
            for n in nodes
            if bool(n.metadata.get("frontier_source_authority")) and degree.get(n.node_id, 0) == 0
        )

        digest = self._compare_with_prior(
            node_ids=node_ids,
            edge_ids=edge_ids,
            mean_activation=mean_activation,
            mean_pressure=mean_pressure,
            contradiction_count=contradiction_count,
            evidence_gap_count=evidence_gap_count,
            isolated_frontier_count=isolated_frontier_count,
            prior_cycle=prior_cycle,
        )

        decisions = self._decide_outcomes(
            request=request,
            node_ids=node_ids,
            contradiction_count=contradiction_count,
            evidence_gap_count=evidence_gap_count,
            mean_activation=mean_activation,
            mean_pressure=mean_pressure,
            isolated_frontier_count=isolated_frontier_count,
            digest=digest,
            cognition_report=cognition_report,
            landing_result=landing_result,
        )

        counts = Counter(d.outcome for d in decisions)
        unresolved = ["contradiction_region" for _ in decisions if _.outcome in {"requeue_review", "maintain_priority"}]
        confidence = sum(d.confidence for d in decisions) / float(max(1, len(decisions)))

        result = GraphConsolidationResultV1(
            request_id=request.request_id,
            decisions=decisions,
            outcome_counts=dict(counts),
            regions_reviewed=[f"zone:{request.target_zone}", f"nodes:{len(node_ids)}", f"edges:{len(edge_ids)}"],
            unresolved_regions=unresolved,
            confidence=confidence,
            degraded=False,
            notes=[
                "deterministic_consolidation_cycle_v1",
                f"semantic_source:{semantic_source}",
                f"semantic_degraded:{semantic_degraded}",
                f"semantic_plan:{semantic_meta.plan_kind}",
                f"semantic_reused_cache:{semantic_meta.reused_cache}",
                f"semantic_duration_ms:{semantic_meta.duration_ms:.3f}",
                f"semantic_cache_enabled:{query_cache_enabled}",
            ],
            state_delta_digest=digest,
        )

        cycle_record = GraphReviewCycleRecordV1(
            request_id=request.request_id,
            focal_node_refs=node_ids,
            focal_edge_refs=edge_ids,
            mean_activation=mean_activation,
            mean_pressure=mean_pressure,
            contradiction_count=contradiction_count,
            evidence_gap_count=evidence_gap_count,
            isolated_frontier_count=isolated_frontier_count,
            outcome_counts=dict(counts),
            notes=[
                f"zone:{request.target_zone}",
                f"semantic_source:{semantic_source}",
                f"semantic_degraded:{semantic_degraded}",
                f"semantic_plan:{semantic_meta.plan_kind}",
                f"semantic_reused_cache:{semantic_meta.reused_cache}",
                f"semantic_cache_enabled:{query_cache_enabled}",
            ],
        )
        return GraphConsolidationExecutionV1(
            result=result,
            cycle_record=cycle_record,
            semantic_source=semantic_source,
            semantic_degraded=semantic_degraded,
            semantic_plan_kind=semantic_meta.plan_kind,
            semantic_reused_cache=semantic_meta.reused_cache,
        )

    def _select_region(
        self,
        *,
        request: GraphConsolidationRequestV1,
        max_region_nodes: int,
        max_region_edges: int,
        query_cache_enabled: bool,
    ) -> tuple[SubstrateQueryResultV1, SubstrateQueryExecutionMetaV1]:
        coordinator = SubstrateSemanticReadCoordinator(store=self._store, cache_enabled=query_cache_enabled)
        plan = SubstrateQueryPlanner.consolidation_region(
            target_zone=request.target_zone,
            focal_node_refs=list(request.focal_node_refs),
            max_nodes=max_region_nodes,
            max_edges=max_region_edges,
        )
        execution = coordinator.execute(plan)
        return execution.results[0], execution.meta

    def _select_region_from_state(
        self,
        *,
        request: GraphConsolidationRequestV1,
        state,
        max_region_nodes: int,
        max_region_edges: int,
    ) -> tuple[list[str], list[str]]:
        node_ids = [nid for nid in request.focal_node_refs if nid in state.nodes]
        if not node_ids:
            node_ids = [
                node.node_id
                for node in state.nodes.values()
                if node.anchor_scope == request.anchor_scope and (request.subject_ref is None or node.subject_ref in (None, request.subject_ref))
            ]
        node_ids = node_ids[: max_region_nodes]
        node_set = set(node_ids)
        edge_ids = [
            eid
            for eid, edge in state.edges.items()
            if edge.source.node_id in node_set and edge.target.node_id in node_set
        ][: max_region_edges]
        return node_ids, edge_ids

    @staticmethod
    def _compare_with_prior(
        *,
        node_ids: list[str],
        edge_ids: list[str],
        mean_activation: float,
        mean_pressure: float,
        contradiction_count: int,
        evidence_gap_count: int,
        isolated_frontier_count: int,
        prior_cycle: GraphReviewCycleRecordV1 | None,
    ) -> GraphStateDeltaDigestV1:
        if prior_cycle is None:
            return GraphStateDeltaDigestV1(
                node_persistence_ratio=0.0,
                edge_persistence_ratio=0.0,
                activation_delta=mean_activation,
                pressure_delta=mean_pressure,
                contradiction_delta=contradiction_count,
                evidence_gap_delta=evidence_gap_count,
                isolated_frontier_delta=isolated_frontier_count,
            )

        node_overlap = len(set(node_ids).intersection(prior_cycle.focal_node_refs))
        edge_overlap = len(set(edge_ids).intersection(prior_cycle.focal_edge_refs))
        node_persistence = node_overlap / float(max(1, len(set(node_ids).union(prior_cycle.focal_node_refs))))
        edge_persistence = edge_overlap / float(max(1, len(set(edge_ids).union(prior_cycle.focal_edge_refs))))

        return GraphStateDeltaDigestV1(
            node_persistence_ratio=node_persistence,
            edge_persistence_ratio=edge_persistence,
            activation_delta=mean_activation - prior_cycle.mean_activation,
            pressure_delta=mean_pressure - prior_cycle.mean_pressure,
            contradiction_delta=contradiction_count - prior_cycle.contradiction_count,
            evidence_gap_delta=evidence_gap_count - prior_cycle.evidence_gap_count,
            isolated_frontier_delta=isolated_frontier_count - prior_cycle.isolated_frontier_count,
        )

    @staticmethod
    def _decide_outcomes(
        *,
        request: GraphConsolidationRequestV1,
        node_ids: list[str],
        contradiction_count: int,
        evidence_gap_count: int,
        mean_activation: float,
        mean_pressure: float,
        isolated_frontier_count: int,
        digest: GraphStateDeltaDigestV1,
        cognition_report: GraphCognitionReportV1 | None,
        landing_result: FrontierLandingExecutionResultV1 | None,
    ) -> list[GraphConsolidationDecisionV1]:
        decisions: list[GraphConsolidationDecisionV1] = []

        if request.target_zone == "self_relationship_graph":
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="operator_only",
                    reason="strict-zone consolidation remains operator mediated",
                    confidence=0.9,
                    zone=request.target_zone,
                    priority=80,
                    notes=["strict_zone_guardrail"],
                    evidence_summary="self/relationship zone cannot self-escalate",
                )
            )
            return decisions

        if contradiction_count > 0 and mean_pressure >= 0.6:
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="maintain_priority",
                    reason="persistent contradiction under high pressure",
                    confidence=0.84,
                    zone=request.target_zone,
                    priority=90,
                    notes=["contradiction_persistent", f"pressure:{mean_pressure:.3f}"],
                    evidence_summary="contradiction unresolved and still salient",
                )
            )

        elif contradiction_count > 0:
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="requeue_review",
                    reason="contradiction remains unresolved",
                    confidence=0.78,
                    zone=request.target_zone,
                    priority=70,
                    notes=["needs_followup_review"],
                    evidence_summary="contradiction present without priority-level pressure",
                )
            )

        if evidence_gap_count > 0:
            if mean_activation < 0.18 and digest.evidence_gap_delta <= 0:
                outcome = "retire"
                reason = "evidence-gap markers stale and unsalient"
                priority = 20
            else:
                outcome = "requeue_review"
                reason = "evidence gaps still active"
                priority = 65
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome=outcome,
                    reason=reason,
                    confidence=0.72,
                    zone=request.target_zone,
                    priority=priority,
                    notes=[f"evidence_gap_count:{evidence_gap_count}"],
                    evidence_summary="frontier hypothesis marker persistence",
                )
            )

        if isolated_frontier_count > 0 and mean_activation < 0.25:
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="damp",
                    reason="frontier-induced structures are isolated with low activation",
                    confidence=0.7,
                    zone=request.target_zone,
                    priority=30,
                    notes=[f"isolated_frontier:{isolated_frontier_count}"],
                    evidence_summary="low-integration frontier region",
                )
            )

        if not decisions:
            stable = digest.node_persistence_ratio >= 0.65 and contradiction_count == 0 and isolated_frontier_count == 0
            if stable and mean_activation >= 0.2:
                outcome = "reinforce"
                reason = "region stability and integration improving"
                priority = 60
            elif mean_activation < 0.15 and mean_pressure < 0.2:
                outcome = "damp"
                reason = "region weak and cooling down"
                priority = 25
            else:
                outcome = "keep_provisional"
                reason = "region plausible but not yet consolidated"
                priority = 50

            if request.target_zone == "autonomy_graph" and outcome == "reinforce":
                outcome = "keep_provisional"
                reason = "autonomy zone remains conservative"

            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome=outcome,
                    reason=reason,
                    confidence=0.76,
                    zone=request.target_zone,
                    priority=priority,
                    notes=[f"node_persistence:{digest.node_persistence_ratio:.3f}"],
                    evidence_summary="prior-vs-current region comparison",
                )
            )

        if cognition_report and cognition_report.goal_pressure.pressure_score > 0.75 and request.target_zone == "autonomy_graph":
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="maintain_priority",
                    reason="goal pressure remains elevated in autonomy zone",
                    confidence=cognition_report.goal_pressure.confidence,
                    zone=request.target_zone,
                    priority=88,
                    notes=["autonomy_pressure_followup"],
                    evidence_summary="goal pressure state from cognition report",
                )
            )

        if landing_result and landing_result.landing_result.materialization_summary.get("materialized_items", 0) == 0:
            decisions.append(
                GraphConsolidationDecisionV1(
                    target_refs=node_ids[:8],
                    outcome="keep_provisional",
                    reason="recent landing had no materialized outcomes",
                    confidence=0.69,
                    zone=request.target_zone,
                    priority=45,
                    notes=["landing_without_materialization"],
                    evidence_summary="phase7 landing summary indicates no landed items",
                )
            )

        return decisions
