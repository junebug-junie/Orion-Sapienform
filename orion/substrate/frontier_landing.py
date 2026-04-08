from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    HypothesisNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_expansion import FrontierGraphDeltaBundleV1
from orion.core.schemas.frontier_landing import (
    FrontierDeltaLandingDecisionV1,
    FrontierLandingRequestV1,
    FrontierLandingResultV1,
)
from orion.substrate.materializer import MaterializationResultV1, SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore


@dataclass(frozen=True)
class FrontierLandingExecutionResultV1:
    landing_result: FrontierLandingResultV1
    materialization_result: MaterializationResultV1 | None


class FrontierLandingEvaluator:
    """Deterministic landing evaluator for frontier graph-delta bundles."""

    def __init__(self, *, store: InMemorySubstrateGraphStore) -> None:
        self._materializer = SubstrateGraphMaterializer(store=store)

    def evaluate_and_land(
        self,
        *,
        request: FrontierLandingRequestV1,
        bundle: FrontierGraphDeltaBundleV1,
    ) -> FrontierLandingExecutionResultV1:
        decisions: list[FrontierDeltaLandingDecisionV1] = []
        materialize_nodes = []
        materialize_edges = []

        for node in bundle.candidate_nodes:
            decision = self._decide_candidate(
                delta_item_id=f"node:{node.node_id}",
                zone=bundle.target_zone,
                confidence=bundle.confidence,
                risk_tier=node.risk_tier,
                node_kind=node.node_kind,
                has_conflict=bool(node.metadata.get("frontier_identity_protected", False)),
            )
            decisions.append(decision)
            if decision.materialize_now:
                enriched = node.model_copy(
                    update={
                        "promotion_state": decision.suggested_promotion_state,
                        "metadata": {
                            **node.metadata,
                            "frontier_source_authority": bundle.source_provenance.source_authority,
                            "frontier_provider": bundle.source_provenance.provider,
                            "frontier_model": bundle.source_provenance.model,
                            "frontier_landing_decision": decision.decision,
                        },
                    }
                )
                materialize_nodes.append(enriched)

        for edge in bundle.candidate_edges:
            decision = self._decide_edge(
                delta_item_id=f"edge:{edge.edge_id}",
                zone=bundle.target_zone,
                confidence=bundle.confidence,
                risk=edge.metadata.get("risk_tier", "low"),
                protected=bool(edge.metadata.get("frontier_identity_protected", False)),
            )
            decisions.append(decision)
            if decision.materialize_now:
                materialize_edges.append(
                    edge.model_copy(
                        update={
                            "metadata": {
                                **edge.metadata,
                                "frontier_source_authority": bundle.source_provenance.source_authority,
                                "frontier_provider": bundle.source_provenance.provider,
                                "frontier_model": bundle.source_provenance.model,
                                "frontier_landing_decision": decision.decision,
                            }
                        }
                    )
                )

        for idx, summary in enumerate(bundle.contradiction_candidates):
            decision = self._decide_contradiction_or_gap(
                delta_item_id=f"contradiction:{idx}", zone=bundle.target_zone, confidence=bundle.confidence
            )
            decisions.append(decision)
            if decision.materialize_now:
                materialize_nodes.append(self._make_gap_hypothesis(bundle=bundle, text=f"Contradiction candidate: {summary}"))

        for idx, question in enumerate(bundle.evidence_gap_candidates):
            decision = self._decide_contradiction_or_gap(delta_item_id=f"evidence_gap:{idx}", zone=bundle.target_zone, confidence=bundle.confidence)
            decisions.append(decision)
            if decision.materialize_now:
                materialize_nodes.append(self._make_gap_hypothesis(bundle=bundle, text=f"Evidence gap: {question}"))

        mat_result = None
        if materialize_nodes or materialize_edges:
            record = SubstrateGraphRecordV1(
                anchor_scope="orion",
                subject_ref=request.landing_context.get("subject_ref"),
                nodes=materialize_nodes,
                edges=materialize_edges,
            )
            mat_result = self._materializer.apply_record(record)

        counts = Counter(item.decision for item in decisions)
        blocked_counts = Counter(item.blocked_reason for item in decisions if item.blocked_reason)
        hitl_count = sum(1 for item in decisions if item.hitl_required)
        materialized_count = sum(1 for item in decisions if item.materialize_now)
        confidence = (sum(item.confidence for item in decisions) / len(decisions)) if decisions else 0.0

        landing_result = FrontierLandingResultV1(
            bundle_id=bundle.bundle_id,
            request_id=request.request_id,
            target_zone=bundle.target_zone,
            decisions=decisions,
            outcome_counts=dict(counts),
            hitl_summary={"required": hitl_count},
            materialization_summary={"materialized_items": materialized_count, "materialized_nodes": len(materialize_nodes), "materialized_edges": len(materialize_edges)},
            blocked_summary={str(k): int(v) for k, v in blocked_counts.items()},
            confidence=confidence,
            degraded=False,
            notes=[f"landing_posture:{bundle.suggested_landing_posture}"] + list(bundle.notes),
        )
        return FrontierLandingExecutionResultV1(landing_result=landing_result, materialization_result=mat_result)

    @staticmethod
    def _decide_candidate(*, delta_item_id: str, zone: str, confidence: float, risk_tier: str, node_kind: str, has_conflict: bool) -> FrontierDeltaLandingDecisionV1:
        if zone == "self_relationship_graph":
            if node_kind != "hypothesis":
                return FrontierDeltaLandingDecisionV1(
                    delta_item_id=delta_item_id,
                    decision="hitl_required",
                    target_zone=zone,
                    suggested_promotion_state="proposed",
                    hitl_required=True,
                    confidence=confidence,
                    risk_tier="high",
                    notes=["strict_zone_non_hypothesis"],
                )
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="proposed_only",
                target_zone=zone,
                suggested_promotion_state="proposed",
                hitl_required=False,
                confidence=confidence,
                risk_tier="high",
                notes=["strict_zone_hypothesis_only"],
            )

        if zone == "autonomy_graph" and (risk_tier in {"medium", "high"} or has_conflict):
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="hitl_required",
                target_zone=zone,
                suggested_promotion_state="proposed",
                hitl_required=True,
                confidence=confidence,
                risk_tier="high" if risk_tier == "high" else "medium",
                notes=["autonomy_requires_review"],
            )

        if risk_tier == "high" and zone in {"world_ontology", "concept_graph"}:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="blocked_due_to_risk",
                target_zone=zone,
                suggested_promotion_state="rejected",
                hitl_required=False,
                blocked_reason="risk",
                confidence=confidence,
                risk_tier="high",
                notes=["risk_threshold_exceeded"],
            )

        if zone == "world_ontology" and confidence >= 0.75:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="materialize_now",
                target_zone=zone,
                suggested_promotion_state="provisional",
                materialize_now=True,
                confidence=confidence,
                risk_tier=risk_tier,
                notes=["world_zone_threshold_met"],
            )

        if zone == "concept_graph" and confidence >= 0.82:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="materialize_now",
                target_zone=zone,
                suggested_promotion_state="provisional",
                materialize_now=True,
                confidence=confidence,
                risk_tier=risk_tier,
                notes=["concept_zone_threshold_met"],
            )

        if confidence >= 0.6:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="provisional",
                target_zone=zone,
                suggested_promotion_state="provisional",
                confidence=confidence,
                risk_tier=risk_tier,
                notes=["confidence_mid"],
            )

        return FrontierDeltaLandingDecisionV1(
            delta_item_id=delta_item_id,
            decision="proposed_only",
            target_zone=zone,
            suggested_promotion_state="proposed",
            confidence=confidence,
            risk_tier=risk_tier,
            notes=["default_proposal"],
        )

    @staticmethod
    def _decide_edge(*, delta_item_id: str, zone: str, confidence: float, risk: str, protected: bool) -> FrontierDeltaLandingDecisionV1:
        if zone in {"autonomy_graph", "self_relationship_graph"} and protected:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="hitl_required" if zone == "autonomy_graph" else "blocked_due_to_zone",
                target_zone=zone,
                suggested_promotion_state="proposed",
                hitl_required=zone == "autonomy_graph",
                blocked_reason="zone" if zone == "self_relationship_graph" else None,
                confidence=confidence,
                risk_tier="high" if zone == "self_relationship_graph" else "medium",
                notes=["protected_edge"],
            )
        if risk == "high":
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="blocked_due_to_risk",
                target_zone=zone,
                suggested_promotion_state="rejected",
                blocked_reason="risk",
                confidence=confidence,
                risk_tier="high",
                notes=["edge_high_risk"],
            )
        if zone == "world_ontology" and confidence >= 0.78:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="materialize_now",
                target_zone=zone,
                suggested_promotion_state="provisional",
                materialize_now=True,
                confidence=confidence,
                risk_tier="low",
                notes=["edge_world_materialize"],
            )
        if zone == "concept_graph" and confidence >= 0.86:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="materialize_now",
                target_zone=zone,
                suggested_promotion_state="provisional",
                materialize_now=True,
                confidence=confidence,
                risk_tier="low",
                notes=["edge_concept_materialize"],
            )
        return FrontierDeltaLandingDecisionV1(
            delta_item_id=delta_item_id,
            decision="proposed_only",
            target_zone=zone,
            suggested_promotion_state="proposed",
            confidence=confidence,
            risk_tier="medium" if zone == "autonomy_graph" else "low",
            notes=["edge_proposal"],
        )

    @staticmethod
    def _decide_contradiction_or_gap(*, delta_item_id: str, zone: str, confidence: float) -> FrontierDeltaLandingDecisionV1:
        if zone in {"autonomy_graph", "self_relationship_graph"}:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="proposed_only",
                target_zone=zone,
                suggested_promotion_state="proposed",
                confidence=confidence,
                risk_tier="medium" if zone == "autonomy_graph" else "high",
                notes=["bounded_gap_or_contradiction"],
            )
        if confidence >= 0.7:
            return FrontierDeltaLandingDecisionV1(
                delta_item_id=delta_item_id,
                decision="materialize_now",
                target_zone=zone,
                suggested_promotion_state="provisional",
                materialize_now=True,
                confidence=confidence,
                risk_tier="low",
                notes=["hypothesis_marker_materialized"],
            )
        return FrontierDeltaLandingDecisionV1(
            delta_item_id=delta_item_id,
            decision="provisional",
            target_zone=zone,
            suggested_promotion_state="provisional",
            confidence=confidence,
            risk_tier="low",
            notes=["hypothesis_marker_provisional"],
        )

    @staticmethod
    def _make_gap_hypothesis(*, bundle: FrontierGraphDeltaBundleV1, text: str) -> HypothesisNodeV1:
        now = datetime.now(timezone.utc)
        return HypothesisNodeV1(
            anchor_scope="orion",
            subject_ref=None,
            hypothesis_text=text,
            temporal=SubstrateTemporalWindowV1(observed_at=now),
            provenance=SubstrateProvenanceV1(
                authority="local_inferred",
                source_kind="frontier.landing",
                source_channel="orion:frontier",
                producer="frontier_landing",
                model_name=bundle.source_provenance.model,
            ),
            promotion_state="proposed",
            risk_tier="low",
            metadata={
                "frontier_source_authority": bundle.source_provenance.source_authority,
                "frontier_provider": bundle.source_provenance.provider,
                "frontier_model": bundle.source_provenance.model,
                "frontier_hypothesis_marker": True,
            },
        )
