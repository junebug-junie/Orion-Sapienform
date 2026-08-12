from __future__ import annotations

from orion.execution_dispatch.policy import MAINTENANCE_SCOPE, CortexRouteTemplateV1
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
            # Mutating routes carry an explicit run mode for the target skill.
            # `dry_run` already flows from the dispatch runtime's own mode, so
            # a real prune needs BOTH that and this -- and then still has to
            # pass the skill's own measured gate. Three independent switches,
            # none of them defaulted open.
            **(
                {"skill_args": {"mode": "execute" if not dry_run else "preview"}}
                if route.allowed_scope == MAINTENANCE_SCOPE
                else {}
            ),
        },
        "constraints": {
            # 2026-08-12: was hardcoded True. Now derived from the route's own
            # scope, because a route that has passed builder.py's
            # `maintenance_bounded` gate is by definition not read-only, and
            # telling the executor otherwise would be a lie that the executor
            # might act on.
            "read_only": route.allowed_scope != MAINTENANCE_SCOPE,
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
