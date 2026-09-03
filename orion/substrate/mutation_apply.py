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

    @staticmethod
    def _is_noop(*, patch: dict[str, Any], live_threshold: float) -> bool:
        """Would writing this patch leave the surface exactly as it is?

        Every key must be comparable AND already match. The contract allows
        ``autonomy_route_threshold`` alongside the lane threshold, so judging a
        multi-key patch on one key would skip the whole apply and silently drop
        a real change to the other -- worse than the no-op it prevents.

        Compares the value that would actually be written: the setter clamps to
        ``[0, 1]``, so an out-of-range patch over a saturated surface writes
        nothing while looking like a change.
        """
        if set(patch) - {"chat_reflective_lane_threshold"}:
            return False
        patch_threshold = patch.get("chat_reflective_lane_threshold")
        if patch_threshold is None:
            return False
        return max(0.0, min(1.0, float(patch_threshold))) == float(live_threshold)

    def noop_reason(self, *, proposal: MutationProposalV1) -> str | None:
        """Why applying this proposal would change nothing, or None if it would.

        Called by the worker only *after* ``apply`` has already declined, to
        explain the refusal in the record. It re-reads the surface, which is
        acceptable on that rare path and deliberately avoided on the common one.

        The routing patch value is a hardcoded constant
        (``_default_patch_for_class`` returns 0.58 for every
        ``routing_threshold_patch``), so once the surface reaches that value
        every subsequent proposal re-applies the number already live. Confirmed
        in production 2026-09-03: the first cycle after the surface lock was
        released adopted 0.58 over a live 0.58 and wrote a history row reading
        ``0.58 -> 0.58``. Left alone that repeats every rollback window forever,
        and each adoption holds the surface lock for the whole window, blocking
        real proposals behind a change that is not a change.

        Surfaces it cannot compare return None, because "cannot tell" must not
        read as "no change".
        """
        if proposal.mutation_class != "routing_threshold_patch":
            return None
        try:
            live_threshold = get_chat_reflective_lane_threshold()
        except Exception:
            return None
        if self._is_noop(patch=dict(proposal.patch.patch), live_threshold=live_threshold):
            return f"patch_is_noop:chat_reflective_lane_threshold={live_threshold}"
        return None

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
            if self._is_noop(patch=proposal.patch.patch, live_threshold=live_threshold):
                # Nothing to adopt. Decided here rather than in the worker so the
                # happy path reads the control surface exactly as often as it did
                # before -- an extra read is not free: it deterministically broke
                # a hub test three modules away, because this suite's fixtures
                # assign the store global by raw assignment and never restore it,
                # so behaviour depends on when the surface is first touched.
                return None
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
