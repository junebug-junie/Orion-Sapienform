from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orion.core.schemas.substrate_mutation import MutationClassV1, MutationPatchV1, MutationRiskTierV1


@dataclass(frozen=True)
class MutationClassContract:
    mutation_class: MutationClassV1
    allowed_targets: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    risk_tier: MutationRiskTierV1
    required_evidence_types: tuple[str, ...]
    trial_method: str
    evaluation_metrics: tuple[str, ...]
    auto_promote_default: bool = False


CONTRACTS: dict[MutationClassV1, MutationClassContract] = {
    "routing_threshold_patch": MutationClassContract(
        mutation_class="routing_threshold_patch",
        allowed_targets=("routing",),
        allowed_fields=("chat_reflective_lane_threshold", "autonomy_route_threshold"),
        bounds={
            "chat_reflective_lane_threshold": (0.0, 1.0),
            "autonomy_route_threshold": (0.0, 1.0),
        },
        risk_tier="low",
        required_evidence_types=("telemetry:runtime_outcome", "telemetry:route_selection"),
        trial_method="replay_route_comparison",
        evaluation_metrics=("success_rate_delta", "latency_ms_delta"),
        auto_promote_default=True,
    ),
    "recall_weighting_patch": MutationClassContract(
        mutation_class="recall_weighting_patch",
        allowed_targets=("recall",),
        allowed_fields=("semantic_weight", "episodic_weight", "recency_weight"),
        bounds={"semantic_weight": (0.0, 1.0), "episodic_weight": (0.0, 1.0), "recency_weight": (0.0, 1.0)},
        risk_tier="low",
        required_evidence_types=("telemetry:recall_miss", "telemetry:retrieval_quality"),
        trial_method="replay_recall_comparison",
        evaluation_metrics=("retrieval_quality_delta", "hallucination_rate_delta"),
        auto_promote_default=True,
    ),
    "recall_strategy_profile_candidate": MutationClassContract(
        mutation_class="recall_strategy_profile_candidate",
        allowed_targets=("recall_strategy",),
        allowed_fields=(
            "failure_category",
            "v1_v2_comparison_evidence",
            "anchor_plan_summary",
            "selected_evidence_cards",
            "contributing_recall_evidence_history",
            "recall_strategy_readiness",
            "suggested_operator_action",
            "why_v2_may_improve",
            "not_applied_status",
            "shadow_only_status",
        ),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:recall_miss", "telemetry:recall_compare"),
        trial_method="operator_review_only",
        evaluation_metrics=(),
        auto_promote_default=False,
    ),
    "recall_anchor_policy_candidate": MutationClassContract(
        mutation_class="recall_anchor_policy_candidate",
        allowed_targets=("recall_anchor_policy",),
        allowed_fields=(
            "failure_category",
            "v1_v2_comparison_evidence",
            "anchor_plan_summary",
            "selected_evidence_cards",
            "contributing_recall_evidence_history",
            "recall_strategy_readiness",
            "suggested_operator_action",
            "why_v2_may_improve",
            "not_applied_status",
            "shadow_only_status",
        ),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:missing_exact_anchor", "telemetry:recall_compare"),
        trial_method="operator_review_only",
        evaluation_metrics=(),
        auto_promote_default=False,
    ),
    "recall_page_index_profile_candidate": MutationClassContract(
        mutation_class="recall_page_index_profile_candidate",
        allowed_targets=("recall_page_index_profile",),
        allowed_fields=(
            "failure_category",
            "v1_v2_comparison_evidence",
            "anchor_plan_summary",
            "selected_evidence_cards",
            "contributing_recall_evidence_history",
            "recall_strategy_readiness",
            "suggested_operator_action",
            "why_v2_may_improve",
            "not_applied_status",
            "shadow_only_status",
        ),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:page_index_gap", "telemetry:recall_compare"),
        trial_method="operator_review_only",
        evaluation_metrics=(),
        auto_promote_default=False,
    ),
    "recall_graph_expansion_policy_candidate": MutationClassContract(
        mutation_class="recall_graph_expansion_policy_candidate",
        allowed_targets=("recall_graph_expansion_policy",),
        allowed_fields=(
            "failure_category",
            "v1_v2_comparison_evidence",
            "anchor_plan_summary",
            "selected_evidence_cards",
            "contributing_recall_evidence_history",
            "recall_strategy_readiness",
            "suggested_operator_action",
            "why_v2_may_improve",
            "not_applied_status",
            "shadow_only_status",
        ),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:semantic_neighbor", "telemetry:recall_compare"),
        trial_method="operator_review_only",
        evaluation_metrics=(),
        auto_promote_default=False,
    ),
    "graph_consolidation_param_patch": MutationClassContract(
        mutation_class="graph_consolidation_param_patch",
        allowed_targets=("graph_consolidation",),
        allowed_fields=("query_limit_nodes", "query_limit_edges", "normal_revisit_seconds"),
        bounds={"query_limit_nodes": (8, 256), "query_limit_edges": (16, 512), "normal_revisit_seconds": (300, 172800)},
        risk_tier="medium",
        required_evidence_types=("telemetry:review_churn", "telemetry:queue_stall"),
        trial_method="replay_consolidation_comparison",
        evaluation_metrics=("queue_resolution_delta", "requeue_rate_delta"),
        auto_promote_default=True,
    ),
    "approved_prompt_profile_variant_promotion": MutationClassContract(
        mutation_class="approved_prompt_profile_variant_promotion",
        allowed_targets=("prompt_profile",),
        allowed_fields=("profile_variant_id",),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:quality_score", "telemetry:operator_approval"),
        trial_method="replay_prompt_profile_comparison",
        evaluation_metrics=("quality_score_delta", "safety_incident_delta"),
        auto_promote_default=False,
    ),
    "cognitive_contradiction_reconciliation": MutationClassContract(
        mutation_class="cognitive_contradiction_reconciliation",
        allowed_targets=("cognitive_contradiction_reconciliation",),
        allowed_fields=("pressure_kind", "affected_surface", "suggested_operator_action", "blast_radius", "reversible_recommendation", "not_applied_status"),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:contradiction", "telemetry:lineage"),
        trial_method="operator_review_only",
        evaluation_metrics=("operator_acceptance_rate",),
        auto_promote_default=False,
    ),
    "cognitive_identity_continuity_adjustment": MutationClassContract(
        mutation_class="cognitive_identity_continuity_adjustment",
        allowed_targets=("cognitive_identity_continuity_adjustment",),
        allowed_fields=("pressure_kind", "affected_surface", "suggested_operator_action", "blast_radius", "reversible_recommendation", "not_applied_status"),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:identity_continuity", "telemetry:memory"),
        trial_method="operator_review_only",
        evaluation_metrics=("operator_acceptance_rate",),
        auto_promote_default=False,
    ),
    "cognitive_stance_continuity_adjustment": MutationClassContract(
        mutation_class="cognitive_stance_continuity_adjustment",
        allowed_targets=("cognitive_stance_continuity_adjustment",),
        allowed_fields=("pressure_kind", "affected_surface", "suggested_operator_action", "blast_radius", "reversible_recommendation", "not_applied_status"),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:stance_drift",),
        trial_method="operator_review_only",
        evaluation_metrics=("operator_acceptance_rate",),
        auto_promote_default=False,
    ),
    "cognitive_social_continuity_repair": MutationClassContract(
        mutation_class="cognitive_social_continuity_repair",
        allowed_targets=("cognitive_social_continuity_repair",),
        allowed_fields=("pressure_kind", "affected_surface", "suggested_operator_action", "blast_radius", "reversible_recommendation", "not_applied_status"),
        bounds={},
        risk_tier="medium",
        required_evidence_types=("telemetry:social_continuity",),
        trial_method="operator_review_only",
        evaluation_metrics=("operator_acceptance_rate",),
        auto_promote_default=False,
    ),
}


def contract_for(mutation_class: MutationClassV1) -> MutationClassContract:
    return CONTRACTS[mutation_class]


def validate_patch(patch: MutationPatchV1) -> list[str]:
    contract = contract_for(patch.mutation_class)
    warnings: list[str] = []
    if patch.target_ref not in contract.allowed_targets:
        warnings.append("target_ref_not_allowed")
    for field, value in patch.patch.items():
        if field not in contract.allowed_fields:
            warnings.append(f"field_not_allowed:{field}")
            continue
        bound = contract.bounds.get(field)
        if bound is not None and isinstance(value, (int, float)):
            lo, hi = bound
            if float(value) < lo or float(value) > hi:
                warnings.append(f"field_out_of_bounds:{field}")
    if not patch.rollback_payload:
        warnings.append("rollback_payload_required")
    return warnings


def metric_passed(metrics: dict[str, float], key: str, minimum: float = 0.0) -> bool:
    value: Any = metrics.get(key)
    if not isinstance(value, (int, float)):
        return False
    return float(value) >= minimum
