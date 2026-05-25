from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1
from orion.schemas.self_state import SelfStateV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

ROUTE = CortexRouteTemplateV1(
    dispatch_kind="inspect",
    cortex_verb="substrate.inspect",
    cortex_mode="brain",
    allowed_scope="inspect_only",
)


def _candidate() -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id="proposal:inspect:state",
        proposal_kind="inspect",
        title="inspect",
        description="test",
        target_id="capability:orchestration",
        target_kind="capability",
        priority_score=0.5,
        urgency_score=0.4,
        confidence_score=0.9,
        risk_score=0.05,
        reversibility_score=1.0,
        proposed_effect="increase_observability",
        required_policy_gate="read_only",
        execution_intent={"mode": "descriptive_only"},
    )


def _decision() -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
    )


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:pf1",
        generated_at=NOW,
        source_field_tick_id="tick_live",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_live",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.6,
        overall_confidence=0.9,
    )


def test_envelope_includes_verb_mode_source_dry_run() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    assert env["verb"] == "substrate.inspect"
    assert env["mode"] == "brain"
    assert env["source"] == "orion-execution-dispatch-runtime"
    assert env["dry_run"] is True


def test_envelope_includes_refs() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    ctx = env["context"]
    assert ctx["proposal_id"] == "proposal:inspect:state"
    assert ctx["decision_id"] == "policy.decision:proposal:inspect:substrate_policy.v1"
    assert ctx["self_state_id"] == "self.state:pf1"
    assert ctx["allowed_scope"] == "inspect_only"


def test_envelope_read_only_constraints() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    c = env["constraints"]
    assert c["read_only"] is True
    assert c["no_file_writes"] is True
    assert c["no_service_restarts"] is True


def test_envelope_no_field_state_or_prompts() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    blob = str(env).lower()
    assert "prompt" not in blob
    assert "llm" not in blob
    assert "field_state" not in blob
    assert "dimensions" not in blob
