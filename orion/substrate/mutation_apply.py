from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orion.core.schemas.substrate_mutation import MutationAdoptionV1, MutationDecisionV1, MutationProposalV1
from orion.substrate.mutation_control_surface import (
    get_chat_reflective_lane_threshold,
    set_chat_reflective_lane_threshold,
)


@dataclass
class PatchApplier:
    """Applies typed patches to bounded in-memory surfaces."""

    surfaces: dict[str, dict[str, Any]]

    def apply(self, *, proposal: MutationProposalV1, decision: MutationDecisionV1) -> MutationAdoptionV1 | None:
        if decision.action != "auto_promote":
            return None
        if not proposal.patch.rollback_payload:
            return None
        if proposal.mutation_class == "routing_threshold_patch":
            live_threshold = get_chat_reflective_lane_threshold()
            patch_threshold = proposal.patch.patch.get("chat_reflective_lane_threshold")
            rollback_payload = dict(proposal.patch.rollback_payload)
            rollback_payload.setdefault("chat_reflective_lane_threshold", live_threshold)
            if patch_threshold is not None:
                set_chat_reflective_lane_threshold(
                    value=float(patch_threshold),
                    actor="mutation_apply",
                    proposal_id=proposal.proposal_id,
                    decision_id=decision.decision_id,
                )
            proposal = proposal.model_copy(
                update={"patch": proposal.patch.model_copy(update={"rollback_payload": rollback_payload})}
            )
        current = self.surfaces.setdefault(proposal.target_surface, {})
        current.update(proposal.patch.patch)
        return MutationAdoptionV1(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            target_surface=proposal.target_surface,
            applied_patch=dict(proposal.patch.patch),
            rollback_payload=dict(proposal.patch.rollback_payload),
            rollback_window_sec=900,
        )

    def rollback(self, *, adoption: MutationAdoptionV1) -> None:
        threshold = adoption.rollback_payload.get("chat_reflective_lane_threshold")
        if threshold is not None:
            set_chat_reflective_lane_threshold(
                value=float(threshold),
                actor="mutation_rollback",
                proposal_id=adoption.proposal_id,
                decision_id=adoption.decision_id,
            )
        current = self.surfaces.setdefault(adoption.target_surface, {})
        current.update(adoption.rollback_payload)
