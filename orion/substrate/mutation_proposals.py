from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationPressureV1, MutationProposalV1
from orion.substrate.mutation_contracts import CONTRACTS


SURFACE_TO_CLASS = {
    "routing": "routing_threshold_patch",
    "recall": "recall_weighting_patch",
    "graph_consolidation": "graph_consolidation_param_patch",
    "prompt_profile": "approved_prompt_profile_variant_promotion",
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
        return MutationProposalV1(
            lane="operational",
            mutation_class=mutation_class,
            risk_tier=contract.risk_tier,
            target_surface=pressure.target_surface,
            anchor_scope=pressure.anchor_scope,
            subject_ref=pressure.subject_ref,
            rationale=f"pressure:{pressure.pressure_kind} score:{pressure.pressure_score:.2f}",
            expected_effect=f"reduce_{pressure.pressure_kind}",
            evidence_refs=pressure.evidence_refs[:32] or [f"pressure:{pressure.pressure_id}"],
            source_signal_ids=pressure.source_signal_ids[:32] or [f"pressure:{pressure.pressure_id}"],
            source_pressure_id=pressure.pressure_id,
            patch=MutationPatchV1(
                mutation_class=mutation_class,
                target_surface=pressure.target_surface,
                target_ref=contract.allowed_targets[0],
                patch=patch_payload,
                rollback_payload=rollback_payload,
            ),
        )


def _default_patch_for_class(mutation_class: str) -> dict[str, float | str]:
    if mutation_class == "routing_threshold_patch":
        return {"chat_reflective_lane_threshold": 0.58}
    if mutation_class == "recall_weighting_patch":
        return {"semantic_weight": 0.55, "episodic_weight": 0.30, "recency_weight": 0.15}
    if mutation_class == "graph_consolidation_param_patch":
        return {"query_limit_nodes": 96, "query_limit_edges": 192, "normal_revisit_seconds": 3600}
    return {"profile_variant_id": "approved-baseline-v2"}


def _default_rollback_for_class(mutation_class: str) -> dict[str, float | str]:
    if mutation_class == "routing_threshold_patch":
        return {"chat_reflective_lane_threshold": 0.50}
    if mutation_class == "recall_weighting_patch":
        return {"semantic_weight": 0.50, "episodic_weight": 0.35, "recency_weight": 0.15}
    if mutation_class == "graph_consolidation_param_patch":
        return {"query_limit_nodes": 64, "query_limit_edges": 128, "normal_revisit_seconds": 7200}
    return {"profile_variant_id": "approved-baseline-v1"}
