from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationPressureV1, MutationProposalV1
from orion.substrate.mutation_contracts import CONTRACTS
from orion.substrate.recall_strategy_readiness import readiness_for_pressure


SURFACE_TO_CLASS = {
    "routing": "routing_threshold_patch",
    "recall": "recall_strategy_profile_candidate",
    "recall_strategy_profile": "recall_strategy_profile_candidate",
    "recall_anchor_policy": "recall_anchor_policy_candidate",
    "recall_page_index_profile": "recall_page_index_profile_candidate",
    "recall_graph_expansion_policy": "recall_graph_expansion_policy_candidate",
    "graph_consolidation": "graph_consolidation_param_patch",
    "prompt_profile": "approved_prompt_profile_variant_promotion",
    "cognitive_contradiction_reconciliation": "cognitive_contradiction_reconciliation",
    "cognitive_identity_continuity_adjustment": "cognitive_identity_continuity_adjustment",
    "cognitive_stance_continuity_adjustment": "cognitive_stance_continuity_adjustment",
    "cognitive_social_continuity_repair": "cognitive_social_continuity_repair",
}

SURFACE_TO_LANE = {
    "routing": "operational",
    "recall": "operational",
    "recall_strategy_profile": "operational",
    "recall_anchor_policy": "operational",
    "recall_page_index_profile": "operational",
    "recall_graph_expansion_policy": "operational",
    "graph_consolidation": "operational",
    "prompt_profile": "operational",
    "cognitive_contradiction_reconciliation": "cognitive",
    "cognitive_identity_continuity_adjustment": "cognitive",
    "cognitive_stance_continuity_adjustment": "cognitive",
    "cognitive_social_continuity_repair": "cognitive",
}


@dataclass(frozen=True)
class ProposalFactory:
    def from_pressure(self, pressure: MutationPressureV1) -> MutationProposalV1 | None:
        mutation_class = SURFACE_TO_CLASS.get(pressure.target_surface)
        if mutation_class is None:
            return None
        contract = CONTRACTS[mutation_class]
        patch_payload = _default_patch_for_class(mutation_class)
        rollback_payload = _default_rollback_for_class(mutation_class)
        lane = SURFACE_TO_LANE.get(pressure.target_surface, "operational")
        target_ref = contract.allowed_targets[0]
        notes: list[str] = []
        if lane == "cognitive":
            template = _cognitive_template_for_surface(pressure.target_surface)
            patch_payload = {
                "pressure_kind": pressure.pressure_kind,
                "affected_surface": pressure.target_surface,
                "suggested_operator_action": template["suggested_operator_action"],
                "blast_radius": template["blast_radius"],
                "reversible_recommendation": template["rollback_notes"],
                "not_applied_status": "draft_only_not_applied",
            }
            rollback_payload = {"draft_recommendation": "discard_and_revert_to_previous_draft"}
            target_ref = pressure.target_surface
            notes = [
                f"blast_radius:{template['blast_radius']}",
                f"suggested_operator_action:{template['suggested_operator_action']}",
                f"rollback_notes:{template['rollback_notes']}",
                "reversible_recommendation:draft_only_not_applied",
                "explicit_status:not_applied",
            ]
        elif pressure.target_surface.startswith("recall_") or pressure.target_surface == "recall":
            recall_payload, recall_notes = _recall_candidate_payload(pressure=pressure)
            patch_payload = recall_payload
            rollback_payload = {"draft_recommendation": "discard_shadow_recall_candidate"}
            target_ref = contract.allowed_targets[0]
            notes = recall_notes
        return MutationProposalV1(
            lane=lane,
            mutation_class=mutation_class,
            risk_tier=contract.risk_tier,
            target_surface=pressure.target_surface,
            anchor_scope=pressure.anchor_scope,
            subject_ref=pressure.subject_ref,
            rationale=_proposal_rationale(pressure, lane=lane, target_surface=pressure.target_surface),
            expected_effect=_proposal_expected_effect(pressure, lane=lane),
            evidence_refs=pressure.evidence_refs[:32] or [f"pressure:{pressure.pressure_id}"],
            source_signal_ids=pressure.source_signal_ids[:32] or [f"pressure:{pressure.pressure_id}"],
            source_pressure_id=pressure.pressure_id,
            patch=MutationPatchV1(
                mutation_class=mutation_class,
                target_surface=pressure.target_surface,
                target_ref=target_ref,
                patch=patch_payload,
                rollback_payload=rollback_payload,
            ),
            notes=notes,
        )


def _default_patch_for_class(mutation_class: str) -> dict[str, float | str]:
    if mutation_class == "routing_threshold_patch":
        return {"chat_reflective_lane_threshold": 0.58}
    if mutation_class == "recall_weighting_patch":
        return {"semantic_weight": 0.55, "episodic_weight": 0.30, "recency_weight": 0.15}
    if mutation_class.startswith("recall_") and mutation_class.endswith("_candidate"):
        return {
            "failure_category": "recall_miss_or_dissatisfaction",
            "v1_v2_comparison_evidence": [],
            "anchor_plan_summary": [],
            "selected_evidence_cards": [],
            "contributing_recall_evidence_history": [],
            "recall_strategy_readiness": {},
            "suggested_operator_action": "review_recall_v2_shadow_comparison_and_update_policy_manually",
            "why_v2_may_improve": "v2_applies_deterministic_anchors_before_fuzzy_similarity",
            "not_applied_status": "proposal_only_not_applied",
            "shadow_only_status": "recall_v2_shadow_only",
        }
    if mutation_class == "graph_consolidation_param_patch":
        return {"query_limit_nodes": 96, "query_limit_edges": 192, "normal_revisit_seconds": 3600}
    if mutation_class == "cognitive_contradiction_reconciliation":
        return {
            "pressure_kind": "contradiction_pressure",
            "affected_surface": "cognitive_contradiction_reconciliation",
            "suggested_operator_action": "review_contradiction_evidence_and_stage_reconciliation",
            "blast_radius": "contradiction_reconciliation_notes_only",
            "reversible_recommendation": "draft_only_recommendation_not_applied",
            "not_applied_status": "draft_only_not_applied",
        }
    if mutation_class == "cognitive_identity_continuity_adjustment":
        return {
            "pressure_kind": "identity_continuity_pressure",
            "affected_surface": "cognitive_identity_continuity_adjustment",
            "suggested_operator_action": "review_identity_continuity_evidence_before_any_commit",
            "blast_radius": "identity_continuity_summary_only",
            "reversible_recommendation": "draft_only_recommendation_not_applied",
            "not_applied_status": "draft_only_not_applied",
        }
    if mutation_class == "cognitive_stance_continuity_adjustment":
        return {
            "pressure_kind": "stance_drift_pressure",
            "affected_surface": "cognitive_stance_continuity_adjustment",
            "suggested_operator_action": "review_stance_drift_and_adjust_stance_contract_manually",
            "blast_radius": "stance_continuity_summary_only",
            "reversible_recommendation": "draft_only_recommendation_not_applied",
            "not_applied_status": "draft_only_not_applied",
        }
    if mutation_class == "cognitive_social_continuity_repair":
        return {
            "pressure_kind": "social_continuity_pressure",
            "affected_surface": "cognitive_social_continuity_repair",
            "suggested_operator_action": "review_social_addressedness_evidence_and_stage_repair",
            "blast_radius": "social_continuity_bridge_only",
            "reversible_recommendation": "draft_only_recommendation_not_applied",
            "not_applied_status": "draft_only_not_applied",
        }
    return {"profile_variant_id": "approved-baseline-v2"}


def _default_rollback_for_class(mutation_class: str) -> dict[str, float | str]:
    if mutation_class == "routing_threshold_patch":
        return {"chat_reflective_lane_threshold": 0.50}
    if mutation_class == "recall_weighting_patch":
        return {"semantic_weight": 0.50, "episodic_weight": 0.35, "recency_weight": 0.15}
    if mutation_class == "graph_consolidation_param_patch":
        return {"query_limit_nodes": 64, "query_limit_edges": 128, "normal_revisit_seconds": 7200}
    if mutation_class.startswith("cognitive_") or mutation_class.startswith("recall_"):
        return {"draft_recommendation": "discard_and_revert_to_previous_draft"}
    return {"profile_variant_id": "approved-baseline-v1"}


def _proposal_rationale(pressure: MutationPressureV1, *, lane: str, target_surface: str) -> str:
    if lane != "cognitive":
        return f"pressure:{pressure.pressure_kind} score:{pressure.pressure_score:.2f}"
    template = _cognitive_template_for_surface(target_surface)
    return (
        f"cognitive_pressure:{pressure.pressure_kind} score:{pressure.pressure_score:.2f}; "
        f"blast_radius:{template['blast_radius']}; suggested_action:{template['suggested_operator_action']}"
    )


def _proposal_expected_effect(pressure: MutationPressureV1, *, lane: str) -> str:
    if lane != "cognitive":
        return f"reduce_{pressure.pressure_kind}"
    return f"operator_review_{pressure.pressure_kind}_with_bounded_cognitive_adjustment"


def _cognitive_template_for_surface(surface: str) -> dict[str, str]:
    defaults = {
        "blast_radius": "stance_brief_and_related_reflective_context_only",
        "suggested_operator_action": "review_cognitive_lineage_and_choose_manual_adoption_or_reject",
        "rollback_notes": "revert_to_prior_brief_variant_and_revalidate_lineage",
        "profile_variant_id": "cognitive.operator_review.required",
    }
    if surface == "cognitive_contradiction_reconciliation":
        return {
            **defaults,
            "blast_radius": "contradiction_reconciliation_notes_only",
            "suggested_operator_action": "review_contradiction_evidence_and_stage_reconciliation",
            "profile_variant_id": "cognitive.contradiction.reconciliation.review",
        }
    if surface == "cognitive_identity_continuity_adjustment":
        return {
            **defaults,
            "blast_radius": "identity_continuity_summary_only",
            "suggested_operator_action": "review_identity_continuity_evidence_before_any_commit",
            "profile_variant_id": "cognitive.identity.continuity.review",
        }
    if surface == "cognitive_stance_continuity_adjustment":
        return {
            **defaults,
            "blast_radius": "stance_continuity_summary_only",
            "suggested_operator_action": "review_stance_drift_and_adjust_stance_contract_manually",
            "profile_variant_id": "cognitive.stance.continuity.review",
        }
    if surface == "cognitive_social_continuity_repair":
        return {
            **defaults,
            "blast_radius": "social_continuity_bridge_only",
            "suggested_operator_action": "review_social_addressedness_evidence_and_stage_repair",
            "profile_variant_id": "cognitive.social.continuity.repair.review",
        }
    return defaults


def _why_v2_rationale(*, failure_category: str, compare: dict[str, object] | None) -> str:
    if isinstance(compare, dict):
        delta = compare.get("selected_count_delta")
        v2c = compare.get("v2_selected_count")
        v1c = compare.get("v1_selected_count")
        if isinstance(delta, (int, float)) and float(delta) > 0:
            return (
                f"v2_shadow_selected_more_items_than_v1_delta={delta} "
                f"(v1_count={v1c} v2_count={v2c}); anchor-first ranking may reduce_{failure_category}"
            )
        if isinstance(delta, (int, float)) and float(delta) < 0:
            return (
                f"v2_shadow_selected_fewer_items_delta={delta}; still_review_anchor_policy_vs_v1_fusion "
                f"for_category={failure_category}"
            )
    return (
        "v2_shadow_runs_anchor_first_hybrid_retrieval_pageindex_vector_rdf_sql_with_explicit_card_lineage; "
        f"v1_production_path_unchanged_review_for_category={failure_category}"
    )


def _recall_candidate_payload(*, pressure: MutationPressureV1) -> tuple[dict[str, object], list[str]]:
    snap = pressure.recall_evidence_snapshot or {}
    compare_obj = snap.get("recall_compare") if isinstance(snap.get("recall_compare"), dict) else None
    anchor_obj = snap.get("anchor_plan") if isinstance(snap.get("anchor_plan"), dict) else None
    cards_obj = snap.get("selected_evidence_cards") if isinstance(snap.get("selected_evidence_cards"), list) else None

    compare_refs = [ref for ref in pressure.evidence_refs if str(ref).startswith("recall_compare:")][:8]
    anchor_refs = [ref for ref in pressure.evidence_refs if str(ref).startswith("anchor_plan:")][:8]
    card_refs = [ref for ref in pressure.evidence_refs if str(ref).startswith("selected_card:")][:12]

    v1_v2_evidence: object = compare_obj if compare_obj is not None else compare_refs
    anchor_summary: object = anchor_obj if anchor_obj is not None else anchor_refs
    selected_cards: object = cards_obj if cards_obj is not None else card_refs

    failure_category = str(snap.get("failure_category") or "").strip() or pressure.pressure_kind.replace("pressure_event:", "")
    suggested_action = {
        "recall_miss_or_dissatisfaction": "review_recall_v2_shadow_eval_and_expand_retrieval_breadth_manually",
        "missing_exact_anchor": "tighten_exact_anchor_gate_before_vector_retrieval",
        "irrelevant_semantic_neighbor": "reduce_vector_weight_and_increase_anchor_lock_requirement",
        "stale_memory_selected": "tighten_temporal_filters_for_recall_selection",
        "unsupported_memory_claim": "require_supporting_evidence_card_before_final_answer",
    }.get(failure_category, "review_recall_strategy_profile_and_shadow_eval")
    why = _why_v2_rationale(failure_category=failure_category, compare=compare_obj)
    history = list(pressure.recall_evidence_history or [])[-8:]
    readiness = readiness_for_pressure(pressure)
    readiness_dump = readiness.model_dump(mode="json")
    payload: dict[str, object] = {
        "failure_category": failure_category,
        "v1_v2_comparison_evidence": v1_v2_evidence,
        "anchor_plan_summary": anchor_summary,
        "selected_evidence_cards": selected_cards,
        "contributing_recall_evidence_history": history,
        "recall_strategy_readiness": readiness_dump,
        "suggested_operator_action": suggested_action,
        "why_v2_may_improve": why,
        "not_applied_status": "proposal_only_not_applied",
        "shadow_only_status": "recall_v2_shadow_only",
    }
    notes = [
        f"failure_category:{failure_category}",
        f"v1_v2_evidence_mode:{'structured' if compare_obj is not None else 'refs'}",
        f"anchor_mode:{'structured' if anchor_obj is not None else 'refs'}",
        f"selected_cards_mode:{'structured' if cards_obj is not None else 'refs'}",
        f"evidence_history_entries:{len(history)}",
        f"readiness_recommendation:{readiness.recommendation}",
        "explicit_status:not_applied",
        "shadow_only:recall_v2",
    ]
    return payload, notes
