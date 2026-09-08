from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from orion.core.schemas.substrate_mutation import MutationAdoptionV1, MutationDecisionV1, MutationProposalV1
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.substrate.mutation_contracts import CONTRACTS, RETIRED_MUTATION_CLASSES
from orion.substrate.mutation_control_surface import (
    ControlSurfaceWriteError,
    get_chat_reflective_lane_threshold,
    set_chat_reflective_lane_threshold,
)
from orion.substrate.policy_profiles import SubstratePolicyProfileStore

# graph_consolidation_param_patch's allowed fields (mutation_contracts.py's
# CONTRACTS entry) are a direct 1:1 match to SubstratePolicyOverridesV1 fields
# -- same names, same bounds. Read from the contract itself rather than
# hand-duplicated here, so the two can't silently drift apart.
_GRAPH_CONSOLIDATION_OVERRIDE_FIELDS = CONTRACTS["graph_consolidation_param_patch"].allowed_fields

# The evidence this proposal class is built from is always operator_review-
# surface, world_ontology-zone telemetry (mutation_detectors.TARGET_SURFACE_BY_ZONE).
# A staged profile must carry that same scope, not the default empty/global
# one -- an empty rollout_scope matches every review, everywhere, once an
# operator ever promotes it (SubstratePolicyProfileStore._matches_scope()).
_GRAPH_CONSOLIDATION_ROLLOUT_SCOPE = SubstratePolicyRolloutScopeV1(
    invocation_surfaces=["operator_review"],
    target_zones=["world_ontology"],
)


@dataclass
class PatchApplier:
    """Applies typed patches to bounded in-memory surfaces.

    ``policy_store``, when given, is where ``graph_consolidation_param_patch``
    actually lands: a real, live-read ``SubstratePolicyProfileStore`` (the
    same one ``GraphReviewRuntimeExecutor._resolve_policy()`` consults),
    instead of the ``surfaces`` dict below -- which nothing in production
    reads back. ``None`` (the default, used by every existing test and the
    standalone smoke worker) keeps the old surfaces-dict-only behavior
    unchanged.
    """

    surfaces: dict[str, dict[str, Any]]
    policy_store: SubstratePolicyProfileStore | None = None

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
        if proposal.mutation_class == "graph_consolidation_param_patch" and self.policy_store is not None:
            try:
                overrides = SubstratePolicyOverridesV1(
                    **{
                        field: proposal.patch.patch[field]
                        for field in _GRAPH_CONSOLIDATION_OVERRIDE_FIELDS
                        if field in proposal.patch.patch
                    }
                )
                result = self.policy_store.adopt(
                    SubstratePolicyAdoptionRequestV1(
                        source_recommendation_id=proposal.proposal_id,
                        rollout_scope=_GRAPH_CONSOLIDATION_ROLLOUT_SCOPE,
                        policy_overrides=overrides,
                        # Staged, not activated: the store is documented as
                        # manual/operator-controlled, and GraphReviewRuntimeExecutor
                        # only resolves *active* profiles. This gives the loop a
                        # real, durable, audited target -- an operator still has to
                        # promote it before it changes live review behavior.
                        activate_now=False,
                        operator_id="mutation_apply",
                        rationale=f"autonomous proposal {proposal.proposal_id}",
                        notes=[f"decision:{decision.decision_id}"],
                    )
                )
            except Exception:
                # Bad bounds (shouldn't happen -- the contract enforces the
                # same bounds upstream, but this must not crash a scheduled
                # cycle tick over one bad proposal) or a degraded store.
                # Nothing durable happened -- same "cannot tell, don't
                # pretend" stance as routing's ControlSurfaceWriteError
                # branch above. Logged (not silent): an unattributable
                # failure here is exactly the kind of thing that looks like
                # a routine no-op refusal from the outside.
                logging.getLogger(__name__).warning(
                    "graph_consolidation_param_patch policy-store staging failed for proposal %s",
                    proposal.proposal_id,
                    exc_info=True,
                )
                return None
            rollback_payload = dict(proposal.patch.rollback_payload)
            # Undoing a staged-but-never-activated profile means discarding
            # it, not restoring some prior live value -- nothing went live.
            # rollback() below reads this key to know which profile that is.
            rollback_payload["policy_profile_id"] = result.profile_id
            proposal = proposal.model_copy(
                update={"patch": proposal.patch.model_copy(update={"rollback_payload": rollback_payload})}
            )
            # The real write already landed in the policy store above --
            # skip the decorative surfaces-dict write below entirely (unlike
            # routing's branch, which still needs it: the control-surface
            # store is separate from `self.surfaces`, but policy_profiles
            # IS graph_consolidation's real surface now, not a second one).
            return MutationAdoptionV1(
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                target_surface=proposal.target_surface,
                applied_patch=dict(proposal.patch.patch),
                rollback_payload=dict(proposal.patch.rollback_payload),
                rollback_window_sec=900,
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
        if "policy_profile_id" in adoption.rollback_payload:
            # A graph_consolidation_param_patch adoption only ever staged a
            # profile (see apply() above) -- it never went live, so there is
            # nothing to restore. The profile itself is left in place (state
            # stays "staged"); it is simply never activated, and the store's
            # own max_profiles trim will eventually drop it like any other
            # unused staged profile. Returning here (not falling through to
            # the generic surfaces-dict write below) keeps this true: writing
            # policy_profile_id into self.surfaces would recreate exactly
            # the decorative second surface apply() was fixed to stop using.
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
