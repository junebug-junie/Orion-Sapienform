from __future__ import annotations

from dataclasses import dataclass, field

from orion.core.schemas.substrate_mutation import MutationDecisionV1, MutationProposalV1, MutationTrialV1
from orion.substrate.mutation_contracts import CONTRACTS, validate_patch

_PROMOTED_GOAL_STATUSES = frozenset({"planned", "executing"})


def unpromoted_goal_blocks_execution(*, goal_proposal_status: str | None) -> bool:
    status = str(goal_proposal_status or "proposed").strip().lower()
    return status not in _PROMOTED_GOAL_STATUSES


def _goal_proposal_status_from_proposal(proposal: MutationProposalV1) -> str | None:
    for note in proposal.notes:
        prefix = "autonomy_goal_proposal_status="
        if str(note).startswith(prefix):
            return str(note)[len(prefix) :].strip()
    patch_payload = proposal.patch.patch if isinstance(proposal.patch.patch, dict) else {}
    if isinstance(patch_payload, dict):
        for key in ("goal_proposal_status", "proposal_status", "autonomy_goal_proposal_status"):
            value = patch_payload.get(key)
            if value:
                return str(value)
    return None


def _proposal_targets_autonomy_goal_execution(proposal: MutationProposalV1) -> bool:
    if any(str(note).startswith("autonomy_goal_execute:") for note in proposal.notes):
        return True
    patch_payload = proposal.patch.patch if isinstance(proposal.patch.patch, dict) else {}
    return bool(isinstance(patch_payload, dict) and patch_payload.get("autonomy_goal_execute"))


@dataclass
class DecisionPolicy:
    auto_promote_allowlist: set[str] = field(default_factory=lambda: {"routing_threshold_patch", "graph_consolidation_param_patch"})
    operator_gated_classes: set[str] = field(
        default_factory=lambda: {
            "approved_prompt_profile_variant_promotion",
            "recall_strategy_profile_candidate",
            "recall_anchor_policy_candidate",
            "recall_page_index_profile_candidate",
            "recall_graph_expansion_policy_candidate",
        }
    )
    operator_gated_surfaces: set[str] = field(
        default_factory=lambda: {
            "policy_profile",
            "cognitive_contradiction_reconciliation",
            "cognitive_identity_continuity_adjustment",
            "cognitive_stance_continuity_adjustment",
            "cognitive_social_continuity_repair",
        }
    )


@dataclass
class DecisionEngine:
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)

    def decide(
        self,
        *,
        proposal: MutationProposalV1,
        trial: MutationTrialV1 | None,
        has_replay_and_baseline: bool,
        active_surface_exists: bool,
    ) -> MutationDecisionV1:
        if _proposal_targets_autonomy_goal_execution(proposal):
            goal_status = _goal_proposal_status_from_proposal(proposal)
            if unpromoted_goal_blocks_execution(goal_proposal_status=goal_status):
                return MutationDecisionV1(
                    proposal_id=proposal.proposal_id,
                    action="reject",
                    reason="unpromoted_goal_execution_blocked",
                    notes=["autonomy_goals_require_operator_promote"],
                )
        warnings = validate_patch(proposal.patch)
        if warnings:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="reject", reason="invalid_patch_contract", notes=warnings)
        if active_surface_exists:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="hold", reason="active_surface_mutation_exists")
        if not proposal.evidence_refs or not proposal.source_signal_ids:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="reject", reason="missing_signal_evidence_refs")
        if not proposal.patch.rollback_payload:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="reject", reason="rollback_payload_missing")
        if trial is None:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="hold", reason="trial_missing")
        if trial.status != "passed":
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="reject", reason=f"trial_{trial.status}", notes=list(trial.notes))
        if not has_replay_and_baseline:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="hold", reason="missing_replay_or_baseline")
        if proposal.lane == "cognitive":
            return MutationDecisionV1(
                proposal_id=proposal.proposal_id,
                action="require_review",
                reason="cognitive_lane_operator_gated",
                requires_operator_review=True,
                notes=["cognitive_lane_proposal_only"],
            )

        if proposal.mutation_class in self.policy.operator_gated_classes or proposal.target_surface in self.policy.operator_gated_surfaces:
            return MutationDecisionV1(
                proposal_id=proposal.proposal_id,
                action="require_review",
                reason="operator_gated_class",
                requires_operator_review=True,
            )
        contract = CONTRACTS[proposal.mutation_class]
        if proposal.mutation_class in self.policy.auto_promote_allowlist and contract.auto_promote_default:
            return MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote", reason="low_risk_trial_passed")
        return MutationDecisionV1(
            proposal_id=proposal.proposal_id,
            action="require_review",
            reason="class_not_auto_promotable",
            requires_operator_review=True,
        )
