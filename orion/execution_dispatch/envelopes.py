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
        },
        "constraints": {
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
