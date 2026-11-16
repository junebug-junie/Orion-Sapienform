from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.frontier_curiosity import (
    FrontierInvocationDecisionV1,
    FrontierInvocationPlanV1,
    FrontierInvocationRunResultV1,
    FrontierInvocationSignalV1,
)
from orion.core.schemas.frontier_expansion import FrontierExpansionRequestV1
from orion.core.schemas.frontier_landing import FrontierLandingRequestV1
from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1
from orion.substrate.frontier_expansion import FrontierExpansionResultV1, FrontierExpansionService
from orion.substrate.frontier_landing import FrontierLandingEvaluator, FrontierLandingExecutionResultV1
from orion.substrate.query_planning import (
    SubstrateQueryPlanStepV1,
    SubstrateQueryPlanV1,
    SubstrateQueryPlanner,
    SubstrateSemanticReadCoordinator,
)
from orion.substrate.store import SubstrateGraphStore


@dataclass(frozen=True)
class FrontierCuriosityExecutionV1:
    invocation: FrontierInvocationRunResultV1
    expansion: FrontierExpansionResultV1 | None
    landing: FrontierLandingExecutionResultV1 | None


class FrontierCuriosityEvaluator:
    def __init__(self, *, store: SubstrateGraphStore, max_focal_nodes: int = 8, max_focal_edges: int = 12) -> None:
        self._store = store
        self._max_focal_nodes = max_focal_nodes
        self._max_focal_edges = max_focal_edges

    def evaluate(
        self,
        *,
        anchor_scope: str,
        subject_ref: str | None,
        operator_requested: bool = False,
        query_cache_enabled: bool = True,
        endogenous_signals: list[FrontierInvocationSignalV1] | None = None,
    ) -> FrontierInvocationRunResultV1:
        coordinator = SubstrateSemanticReadCoordinator(store=self._store, cache_enabled=query_cache_enabled)
        seed_execution = coordinator.execute(SubstrateQueryPlanner.curiosity_seed(max_nodes=64, max_edges=128))
        hotspot, concept, contradiction = seed_execution.results
        state = self._store.snapshot() if (hotspot.degraded and concept.degraded and contradiction.degraded) else None
        all_nodes: list[BaseSubstrateNodeV1] = []
        if state is not None:
            all_nodes = list(state.nodes.values())
            semantic_source = "local_fallback"
            semantic_degraded = True
        else:
            node_map: dict[str, BaseSubstrateNodeV1] = {}
            for result in (hotspot, concept, contradiction):
                for node in result.slice.nodes:
                    node_map[node.node_id] = node
            all_nodes = list(node_map.values())
            semantic_source = "graphdb" if all(r.source_kind == "graphdb" for r in (hotspot, concept, contradiction)) else "mixed"
            semantic_degraded = bool(hotspot.degraded or concept.degraded or contradiction.degraded)

        signals = self._derive_signals(
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            all_nodes=all_nodes,
            query_coordinator=coordinator,
            operator_requested=operator_requested,
        )
        # Rung 5: intrinsic curiosity_candidate seeds (endogenous_curiosity.py)
        # ride the same decision/plan path as derived signals — no separate
        # invocation authority, same zone guardrails.
        if endogenous_signals:
            signals.extend(endogenous_signals)
        decision = self._decide(signals=signals)
        plan = self._build_plan(decision=decision, anchor_scope=anchor_scope, subject_ref=subject_ref)
        return FrontierInvocationRunResultV1(
            signals=signals,
            decision=decision,
            plan=plan,
            notes=[
                "deterministic_curiosity_policy_v1",
                f"semantic_source:{semantic_source}",
                f"semantic_degraded:{semantic_degraded}",
                f"semantic_plan:{seed_execution.meta.plan_kind}",
                f"semantic_reused_cache:{seed_execution.meta.reused_cache}",
                f"semantic_duration_ms:{seed_execution.meta.duration_ms:.3f}",
                f"semantic_cache_enabled:{query_cache_enabled}",
            ],
        )

    def _derive_signals(self, *, anchor_scope: str, subject_ref: str | None, all_nodes: list[BaseSubstrateNodeV1], query_coordinator: SubstrateSemanticReadCoordinator, operator_requested: bool) -> list[FrontierInvocationSignalV1]:
        signals: list[FrontierInvocationSignalV1] = []

        # `ontology_sparse_region` (-> `ontology_expand`) was retired 2026-10-10.
        # It fired whenever the read slice held concepts but no ontology_branch
        # nodes; nothing in Orion creates ontology_branch nodes, so it read a
        # constant 1.0 and won every evaluator decision with a task nothing
        # executes. Kill means kill: no flag, no fallback. See
        # docs/superpowers/pr-reports/2026-10-10-retire-ontology-sparse-region-pr.md.
        #
        # If you are waiting for a reading link to show up in focal_edge_refs on
        # these rows: it will not, they are gone. A reading link now shows up on
        # stored ENDOGENOUS seeds, filled after this decision by
        # orion/substrate/curiosity_seed_neighborhood.py (accepted claims only):
        # - an accepted reading claim mints a seed on its two ends (note
        #   source:reading_link_accepted), so the link is an internal edge ->
        #   focal_edge_refs (orion/substrate/link_accepted_seeds.py);
        # - any other seed touching one end -> boundary_edge_refs, with
        #   projection_endpoint_node_refs naming the linked reading concept.
        # Proving SQL (substrate_endogenous_curiosity_candidates):
        #   select c.generated_at, s->'focal_node_refs', s->'focal_edge_refs'
        #   from substrate_endogenous_curiosity_candidates c,
        #        jsonb_array_elements(c.candidates_json) s
        #   where s->'notes' ? 'source:reading_link_accepted'
        #     and jsonb_array_length(coalesce(s->'focal_edge_refs','[]')) > 0
        #   order by c.generated_at desc limit 10;

        # Four report-driven branches were retired 2026-11-16 (#3393, Patch F):
        # contradiction_hotspot, concept_instability (drift),
        # unresolved_pressure_region (goal pressure) and the identity-conflict
        # self_relationship candidate. Their only live caller (the endogenous
        # curiosity tick) fed them a hardcoded "all calm" report, so none fired
        # in 30 days of stored candidates; the real interpreter
        # (orion/graph_cognition, deleted with them) read drift as a function of
        # node count, not change. Kill means kill: no flag, no fallback.
        # See docs/superpowers/specs/2026-11-15-patch-f-curiosity-tick-stub-proposal.md.
        #
        # If you add a contradiction / goal / tension producer: the durable
        # substrate store refuses every node kind outside
        # DURABLE_NODE_KINDS (orion/substrate/falkor_codec.py) at write time,
        # before its cache, so such nodes never reach this evaluator. Widen that
        # whitelist (with a decoder) in the same patch as the producer, then
        # build the consumer here with its own metric-gate record and test.

        gap_markers = [n.node_id for n in all_nodes if bool(n.metadata.get("frontier_hypothesis_marker"))]
        if len(gap_markers) >= 2:
            node_refs, edge_refs = self._select_region(zone="concept_graph", preferred_node_ids=gap_markers, query_coordinator=query_coordinator)
            signals.append(
                FrontierInvocationSignalV1(
                    signal_type="evidence_gap_cluster",
                    anchor_scope=anchor_scope,
                    subject_ref=subject_ref,
                    target_zone="concept_graph",
                    task_type_candidate="evidence_gap_scan",
                    focal_node_refs=node_refs,
                    focal_edge_refs=edge_refs,
                    signal_strength=min(1.0, 0.45 + (0.08 * len(gap_markers))),
                    evidence_summary="repeated frontier evidence-gap hypothesis markers",
                    confidence=0.68,
                )
            )

        if operator_requested:
            top_nodes = [node.node_id for node in all_nodes[: self._max_focal_nodes]]
            node_refs, edge_refs = self._select_region(zone="concept_graph", preferred_node_ids=top_nodes, query_coordinator=query_coordinator)
            signals.append(
                FrontierInvocationSignalV1(
                    signal_type="explicit_operator_request",
                    anchor_scope=anchor_scope,
                    subject_ref=subject_ref,
                    target_zone="concept_graph",
                    task_type_candidate="concept_expand",
                    focal_node_refs=node_refs,
                    focal_edge_refs=edge_refs,
                    signal_strength=0.95,
                    evidence_summary="explicit operator-approved curiosity invocation",
                    confidence=0.95,
                    notes=["operator_override"],
                )
            )

        return signals

    def _decide(self, *, signals: list[FrontierInvocationSignalV1]) -> FrontierInvocationDecisionV1:
        if not signals:
            return FrontierInvocationDecisionV1(outcome="noop", confidence=0.0, bounded_context_reason="no qualifying curiosity signals")

        top = sorted(signals, key=lambda s: (s.signal_strength, s.confidence), reverse=True)[0]

        if top.target_zone == "self_relationship_graph":
            return FrontierInvocationDecisionV1(
                outcome="operator_only",
                chosen_task_type=top.task_type_candidate,
                target_zone=top.target_zone,
                chosen_focal_node_refs=top.focal_node_refs,
                chosen_focal_edge_refs=top.focal_edge_refs,
                bounded_context_reason="strict-zone invocation requires explicit operator path",
                confidence=top.confidence,
                block_reason="strict_zone",
                notes=["self_relationship_guardrail"],
            )

        if top.target_zone == "autonomy_graph" and top.signal_strength < 0.78:
            return FrontierInvocationDecisionV1(
                outcome="defer",
                chosen_task_type=top.task_type_candidate,
                target_zone=top.target_zone,
                chosen_focal_node_refs=top.focal_node_refs,
                chosen_focal_edge_refs=top.focal_edge_refs,
                bounded_context_reason="autonomy zone requires stronger signal threshold",
                confidence=top.confidence,
                notes=["autonomy_conservative_threshold"],
            )

        if top.signal_strength < 0.5:
            return FrontierInvocationDecisionV1(
                outcome="noop",
                chosen_task_type=top.task_type_candidate,
                target_zone=top.target_zone,
                chosen_focal_node_refs=top.focal_node_refs,
                chosen_focal_edge_refs=top.focal_edge_refs,
                bounded_context_reason="signal below invocation threshold",
                confidence=top.confidence,
            )

        return FrontierInvocationDecisionV1(
            outcome="invoke",
            chosen_task_type=top.task_type_candidate,
            target_zone=top.target_zone,
            chosen_focal_node_refs=top.focal_node_refs,
            chosen_focal_edge_refs=top.focal_edge_refs,
            bounded_context_reason=f"invoke based on {top.signal_type}",
            confidence=top.confidence,
            notes=[f"signal_strength:{top.signal_strength:.3f}"],
        )

    def _build_plan(self, *, decision: FrontierInvocationDecisionV1, anchor_scope: str, subject_ref: str | None) -> FrontierInvocationPlanV1 | None:
        if decision.outcome != "invoke" or decision.chosen_task_type is None or decision.target_zone is None:
            return None

        request = FrontierExpansionRequestV1(
            task_type=decision.chosen_task_type,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            target_zone=decision.target_zone,
            topic=decision.bounded_context_reason or "frontier curiosity invocation",
            expansion_goal=f"address {decision.chosen_task_type} in bounded graph region",
            graph_region={"focal_node_ids": decision.chosen_focal_node_refs, "focal_edge_ids": decision.chosen_focal_edge_refs, "max_hops": 2},
            constraints={"bounded_invocation": True, "max_nodes": self._max_focal_nodes, "max_edges": self._max_focal_edges},
        )
        expected_posture = {
            "world_ontology": "fast_track_proposal",
            "concept_graph": "moderate_proposal",
            "autonomy_graph": "conservative_proposal",
            "self_relationship_graph": "strict_proposal_only",
        }[decision.target_zone]

        return FrontierInvocationPlanV1(
            request_payload_summary=f"{decision.chosen_task_type} over bounded region ({len(decision.chosen_focal_node_refs)} nodes)",
            selected_node_refs=decision.chosen_focal_node_refs,
            selected_edge_refs=decision.chosen_focal_edge_refs,
            selected_graph_cognition_refs=[],
            task_type=decision.chosen_task_type,
            target_zone=decision.target_zone,
            expected_safety_posture=expected_posture,
            request=request,
        )

    def _select_region(self, *, zone: str, preferred_node_ids: list[str], query_coordinator: SubstrateSemanticReadCoordinator) -> tuple[list[str], list[str]]:
        focal_execution = query_coordinator.execute(
            SubstrateQueryPlanV1(
                plan_kind="curiosity_focal_region",
                steps=(SubstrateQueryPlanStepV1("focal_slice", {"node_ids": list(preferred_node_ids), "max_edges": self._max_focal_edges}),),
            )
        )
        query = focal_execution.results[0]
        if query.slice.nodes:
            allowed_nodes = list(query.slice.nodes)
            allowed_edges = list(query.slice.edges)
        else:
            fallback_execution = query_coordinator.execute(
                SubstrateQueryPlanV1(
                    plan_kind="curiosity_hotspot_fallback",
                    steps=(
                        SubstrateQueryPlanStepV1(
                            "hotspot_region",
                            {
                                "limit_nodes": max(self._max_focal_nodes, 16),
                                "limit_edges": max(self._max_focal_edges, 24),
                            },
                        ),
                    ),
                )
            )
            fallback = fallback_execution.results[0]
            allowed_nodes = list(fallback.slice.nodes)
            allowed_edges = list(fallback.slice.edges)
        if zone == "self_relationship_graph":
            allowed_nodes = [node for node in allowed_nodes if node.node_kind in {"hypothesis", "tension", "goal", "contradiction"}]

        preferred = [node for node in allowed_nodes if node.node_id in preferred_node_ids]
        if not preferred:
            preferred = allowed_nodes

        preferred.sort(
            key=lambda n: (
                float(n.metadata.get("dynamic_pressure") or 0.0),
                n.signals.activation.activation,
                n.signals.salience,
            ),
            reverse=True,
        )
        nodes = [node.node_id for node in preferred[: self._max_focal_nodes]]
        node_set = set(nodes)

        edges: list[str] = [
            edge.edge_id
            for edge in allowed_edges
            if edge.source.node_id in node_set and edge.target.node_id in node_set
        ]
        return nodes[: self._max_focal_nodes], edges[: self._max_focal_edges]


class FrontierCuriosityOrchestrator:
    def __init__(
        self,
        *,
        curiosity_evaluator: FrontierCuriosityEvaluator,
        expansion_service: FrontierExpansionService,
        landing_evaluator: FrontierLandingEvaluator,
    ) -> None:
        self._curiosity_evaluator = curiosity_evaluator
        self._expansion_service = expansion_service
        self._landing_evaluator = landing_evaluator

    def run(
        self,
        *,
        anchor_scope: str,
        subject_ref: str | None,
        operator_requested: bool = False,
    ) -> FrontierCuriosityExecutionV1:
        invocation = self._curiosity_evaluator.evaluate(
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            operator_requested=operator_requested,
        )
        if invocation.decision.outcome != "invoke" or invocation.plan is None:
            return FrontierCuriosityExecutionV1(invocation=invocation, expansion=None, landing=None)

        expansion = self._expansion_service.expand(request=invocation.plan.request)
        landing_request = FrontierLandingRequestV1(
            bundle_id=expansion.delta_bundle.bundle_id,
            request_id=expansion.delta_bundle.request_id,
            target_zone=expansion.delta_bundle.target_zone,
            landing_context={"subject_ref": subject_ref, "invocation_plan_id": invocation.plan.plan_id},
            graph_cognition_brief_refs=invocation.plan.selected_graph_cognition_refs,
        )
        landing = self._landing_evaluator.evaluate_and_land(request=landing_request, bundle=expansion.delta_bundle)
        return FrontierCuriosityExecutionV1(invocation=invocation, expansion=expansion, landing=landing)
