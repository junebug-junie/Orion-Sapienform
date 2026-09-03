from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orion.core.schemas.substrate_mutation import MutationAdoptionV1, MutationDecisionV1, MutationProposalV1
from orion.substrate.mutation_control_surface import (
    ControlSurfaceWriteError,
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
        if str(proposal.mutation_class).startswith("recall_") and str(proposal.mutation_class).endswith("_candidate"):
            return None
        if proposal.mutation_class == "recall_weighting_patch":
            return None
        if proposal.mutation_class == "routing_threshold_patch":
            live_threshold = get_chat_reflective_lane_threshold()
            patch_threshold = proposal.patch.patch.get("chat_reflective_lane_threshold")
            rollback_payload = dict(proposal.patch.rollback_payload)
            # Overwrite, do not setdefault. The proposal already carries a
            # hardcoded fallback from _default_rollback_for_class, so setdefault
            # was always a no-op and this observed reading was read and thrown
            # away. That made every recorded rollback value a constant rather
            # than a measurement: undo would restore whatever someone typed into
            # mutation_proposals.py, not what was actually live. It happened to
            # match once (2026-09-02, both 0.5) purely by coincidence.
            rollback_payload["chat_reflective_lane_threshold"] = live_threshold
            if patch_threshold is not None:
                try:
                    set_chat_reflective_lane_threshold(
                        value=float(patch_threshold),
                        actor="mutation_apply",
                        proposal_id=proposal.proposal_id,
                        decision_id=decision.decision_id,
                    )
                except ControlSurfaceWriteError:
                    # The live value did not move, so there is nothing to adopt.
                    # Returning an adoption here would take the surface lock and
                    # write a record claiming a change that never happened.
                    return None
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
