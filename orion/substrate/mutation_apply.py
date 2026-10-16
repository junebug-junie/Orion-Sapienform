from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from orion.core.schemas.substrate_mutation import MutationAdoptionV1, MutationDecisionV1, MutationProposalV1
from orion.substrate.mutation_contracts import RETIRED_MUTATION_CLASSES
from orion.substrate.mutation_control_surface import (
    ControlSurfaceWriteError,
    get_chat_reflective_lane_threshold,
    get_prediction_expectation,
    set_chat_reflective_lane_threshold,
    set_prediction_expectation,
)
from orion.substrate.prediction_expectation import (
    MUTATION_CLASS as PREDICTION_EXPECTATION_CLASS,
    domain_for_surface,
    expectation_from_payload,
)

# Rung 6: how long an adopted expectation must survive the post-adoption
# replay check before its surface lock is released. Long enough for the
# execution domain (~70 readings/hour live) to clear the monitor's
# MIN_POST_ADOPTION_READINGS several times over.
PREDICTION_EXPECTATION_ROLLBACK_WINDOW_SEC = 6 * 3600


@dataclass
class PatchApplier:
    """Applies typed patches to bounded in-memory surfaces.

    Live surfaces: ``prediction_expectation_patch`` writes the substrate
    control surface (rung 6). The graph_consolidation policy-store staging
    branch was removed 2026-10-16 with that class's retirement.
    """

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
        if proposal.mutation_class == PREDICTION_EXPECTATION_CLASS:
            spec = domain_for_surface(proposal.target_surface)
            if spec is None:
                return None
            try:
                live = get_prediction_expectation(spec.domain)
            except Exception:
                return None
            if expectation_from_payload(proposal.patch.patch, default=live) == live:
                return f"patch_is_noop:{spec.control_surface_key}=alpha:{live.alpha},min_variance:{live.min_variance}"
            return None
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
        if proposal.mutation_class in RETIRED_MUTATION_CLASSES:
            # Defense in depth alongside DecisionEngine.decide()'s early
            # reject: refuses even a hand-built MutationDecisionV1 with
            # action="auto_promote" passed straight to apply() (a debug
            # tool, a test, a replayed historical decision), not just one
            # that went through decide() itself. See
            # mutation_contracts.py's RETIRED_MUTATION_CLASSES docstring --
            # without this, this method's own live write below (routing's
            # branch) would still execute for real.
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
        if proposal.mutation_class == PREDICTION_EXPECTATION_CLASS:
            return self._apply_prediction_expectation(proposal=proposal, decision=decision)
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

    def _apply_prediction_expectation(
        self, *, proposal: MutationProposalV1, decision: MutationDecisionV1
    ) -> MutationAdoptionV1 | None:
        """Write Orion's revised expectation to the live control surface.

        The rollback payload is the value read live at apply time, never the
        proposal's own copy: undo restores what was actually running.
        """
        spec = domain_for_surface(proposal.target_surface)
        if spec is None or proposal.patch.target_ref != spec.domain:
            return None
        live = get_prediction_expectation(spec.domain)
        candidate = expectation_from_payload(proposal.patch.patch, default=live)
        if candidate == live or not spec.in_bounds(candidate):
            return None
        try:
            set_prediction_expectation(
                spec.domain,
                alpha=candidate.alpha,
                min_variance=candidate.min_variance,
                actor="mutation_apply",
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
            )
        except ControlSurfaceWriteError:
            return None
        return MutationAdoptionV1(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            target_surface=proposal.target_surface,
            applied_patch=candidate.as_patch(),
            rollback_payload=live.as_patch(),
            rollback_window_sec=PREDICTION_EXPECTATION_ROLLBACK_WINDOW_SEC,
        )

    def rollback(self, *, adoption: MutationAdoptionV1) -> None:
        spec = domain_for_surface(adoption.target_surface)
        if spec is not None:
            previous = expectation_from_payload(adoption.rollback_payload, default=spec.default)
            set_prediction_expectation(
                spec.domain,
                alpha=previous.alpha,
                min_variance=previous.min_variance,
                actor="mutation_rollback",
                proposal_id=adoption.proposal_id,
                decision_id=adoption.decision_id,
            )
            return
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
