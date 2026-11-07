from __future__ import annotations

from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1


def build_cortex_request_envelope(
    *,
    candidate: ProposalCandidateV1,
    decision: PolicyDecisionV1,
    route: CortexRouteTemplateV1,
    field_tick_id: str,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "verb": route.cortex_verb,
        "mode": route.cortex_mode,
        "source": "orion-execution-dispatch-runtime",
        "origin": "endogenous.dispatch",
        "dry_run": dry_run,
        "context": {
            "proposal_id": candidate.proposal_id,
            "decision_id": decision.decision_id,
            "field_tick_id": field_tick_id,
            "target_id": candidate.target_id,
            "target_kind": candidate.target_kind,
            "allowed_scope": route.allowed_scope,
            "origin": "endogenous.dispatch",
            # Real, already-computed grounding data for the substrate.inspect/
            # summarize/observe prompts (see orion/cognition/prompts/
            # substrate_*.j2's "REAL TELEMETRY" section). motivating_dimensions
            # is the actual field_pressures()/template_match_score() dimension
            # scores (orion/proposals/builder.py::_build_candidate()) that
            # caused this proposal to exist -- the only real numbers the model
            # gets instead of a bare target_id. priority_score/risk_score are
            # likewise already computed on the candidate; included so the
            # model can ground "why does this matter" without inventing a
            # justification.
            "motivating_dimensions": dict(candidate.motivating_dimensions),
            "priority_score": candidate.priority_score,
            "risk_score": candidate.risk_score,
            # Route-declared static args. 2026-11-06 (ladder retirement PR 4):
            # the derived maintenance `mode` (execute unless dry_run) is gone
            # with the maintenance scope, so dispatch never asks a skill to
            # execute. Config cannot either (CortexRouteTemplateV1 refuses
            # `mode` in skill_args).
            **({"skill_args": dict(route.skill_args)} if route.skill_args else {}),
        },
        # NOTE on what this dict is and is not. Verified 2026-08-12: the
        # dispatch worker forwards only `envelope["context"]` to cortex-exec,
        # so nothing here crosses the wire, and the only other reader
        # stringifies it onto the stored candidate. These are therefore
        # HONEST METADATA on the persisted record, not gates. The real gate is
        # builder.py's scope check. An earlier comment here claimed the
        # executor "might act on" a false value; it cannot, because it never
        # sees one. Getting them right still matters -- the stored candidate is
        # the audit record for a destructive action.
        "constraints": {
            # 2026-08-12 .. 2026-11-06 these three were False for the
            # maintenance_bounded scope (the docker prunes). That scope is gone
            # (ladder retirement PR 4), so no admitted route writes host files.
            "read_only": True,
            "dry_run": dry_run,
            "no_external_side_effects": True,
            "no_file_writes": True,
            "no_service_restarts": True,
            "no_operator_notifications": True,
            "no_stream_replay": True,
            "no_stream_purge": True,
            "no_catalog_write": True,
        },
    }
