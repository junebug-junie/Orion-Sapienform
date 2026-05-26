from __future__ import annotations

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.proposals.templates import cast_policy_gate, cast_proposal_kind, cast_proposed_effect, cast_target_kind
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from datetime import datetime, timezone

NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def test_transport_envelope_has_dry_run_constraints() -> None:
    candidate = ProposalCandidateV1(
        proposal_id="proposal:inspect_transport_status:s",
        title="t",
        description="d",
        target_id="capability:transport",
        target_kind=cast_target_kind("capability"),
        priority_score=0.5,
        urgency_score=0.3,
        confidence_score=0.8,
        risk_score=0.05,
        reversibility_score=1.0,
        proposal_kind=cast_proposal_kind("inspect"),
        proposed_effect=cast_proposed_effect("increase_observability"),
        required_policy_gate=cast_policy_gate("read_only"),
        execution_intent={"template": "inspect_transport_status"},
    )
    decision = PolicyDecisionV1(
        decision_id="policy.decision:1",
        proposal_id=candidate.proposal_id,
        decision="approved_read_only",
        policy_gate="read_only",
        autonomy_tier="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.8,
        allowed_scope="inspect_only",
    )
    self_state = SelfStateV1(
        self_state_id="self.state:1",
        generated_at=NOW,
        source_field_tick_id="t",
        source_field_generated_at=NOW,
        source_attention_frame_id="a",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.5,
        overall_confidence=0.7,
        dimensions={
            "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.8, confidence=0.7)
        },
    )
    route = CortexRouteTemplateV1(
        dispatch_kind="inspect",
        cortex_verb="substrate.inspect.transport",
        cortex_mode="brain",
        allowed_scope="inspect_only",
    )
    envelope = build_cortex_request_envelope(
        candidate=candidate,
        decision=decision,
        route=route,
        self_state=self_state,
        dry_run=True,
    )
    constraints = envelope["constraints"]
    assert constraints["dry_run"] is True
    assert constraints["no_stream_replay"] is True
    assert constraints["no_stream_purge"] is True
    assert constraints["no_catalog_write"] is True
    assert constraints["no_service_restarts"] is True
