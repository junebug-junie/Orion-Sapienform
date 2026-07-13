from __future__ import annotations

from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1
from orion.schemas.self_state import SelfStateV1


def build_cortex_request_envelope(
    *,
    candidate: ProposalCandidateV1,
    decision: PolicyDecisionV1,
    route: CortexRouteTemplateV1,
    self_state: SelfStateV1,
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
            "self_state_id": self_state.self_state_id,
            "target_id": candidate.target_id,
            "target_kind": candidate.target_kind,
            "allowed_scope": route.allowed_scope,
            "origin": "endogenous.dispatch",
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
