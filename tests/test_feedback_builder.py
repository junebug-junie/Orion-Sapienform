from datetime import datetime, timezone
from pathlib import Path

from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.feedback.builder import build_feedback_frame, stable_feedback_frame_id
from orion.feedback.policy import load_feedback_policy
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
DISPATCH_POLICY = load_execution_dispatch_policy(
    REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
)
FEEDBACK_POLICY = load_feedback_policy(REPO / "config" / "feedback" / "feedback_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def _dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)


def _self_state(self_state_id: str, scores: dict[str, float]) -> SelfStateV1:
    return SelfStateV1(
        self_state_id=self_state_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.9,
        dimensions={k: _dim(k, v) for k, v in scores.items()},
    )


def _proposal() -> ProposalFrameV1:
    def cand(pid: str, kind: str) -> ProposalCandidateV1:
        return ProposalCandidateV1(
            proposal_id=pid,
            proposal_kind=kind,
            title=pid,
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

    state = _self_state("self.state:before", {"execution_pressure": 1.0})
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.6,
        overall_risk=0.3,
        candidates=[cand("proposal:inspect:state", "inspect")],
    )


def _policy_frame(proposal: ProposalFrameV1) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
    )
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=proposal.source_self_state_id,
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )


def _dispatch_dry_run() -> ExecutionDispatchFrameV1:
    proposal = _proposal()
    policy_frame = _policy_frame(proposal)
    before = _self_state("self.state:before", {"execution_pressure": 1.0})
    return build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=before,
        policy=DISPATCH_POLICY,
        now=NOW,
    )


def test_dry_run_produces_dry_run_only() -> None:
    dispatch = _dispatch_dry_run()
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=_policy_frame(_proposal()),
        proposal_frame=_proposal(),
        self_state_before=_self_state("self.state:before", {"execution_pressure": 1.0}),
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "dry_run_only"
    assert any(o.outcome_kind == "dry_run" for o in frame.observations)


def test_prepared_only_dispatch() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:prep:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:prep",
        source_proposal_frame_id="proposal.frame:prep",
        source_self_state_id="self.state:prep",
        dispatch_mode="prepare_only",
        candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="prepared",
                dispatch_mode="prepare_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
        dispatch_attempted=False,
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "prepared_only"


def test_prepared_for_dispatch_candidate_observation() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:pfd:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pfd",
        source_proposal_frame_id="proposal.frame:pfd",
        source_self_state_id="self.state:pfd",
        dispatch_mode="dispatch_read_only",
        candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="prepared_for_dispatch",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert any(o.outcome_kind == "prepared_for_dispatch" for o in frame.observations)


def test_blocked_candidate_observation() -> None:
    dispatch = _dispatch_dry_run()
    dispatch = dispatch.model_copy(
        update={
            "blocked_candidates": [
                ExecutionDispatchCandidateV1(
                    dispatch_id="dispatch:proposal:blocked:execution_dispatch_policy.v1",
                    source_decision_id="pd2",
                    source_proposal_id="proposal:blocked:state",
                    dispatch_status="blocked",
                    dispatch_mode="dry_run",
                    dispatch_kind="inspect",
                    target_id="t1",
                    target_kind="capability",
                    risk_score=0.3,
                    confidence_score=0.9,
                    blocked_by=["rejected"],
                )
            ],
            "blocked_count": 1,
        }
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert any(o.outcome_kind == "blocked" for o in frame.observations)


def test_missing_cortex_result_absence() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:ro:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:ro",
        source_proposal_frame_id="proposal.frame:ro",
        source_self_state_id="self.state:ro",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:inspect",
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status in ("absent", "mixed")
    assert len(frame.absence_evidence) >= 1


def test_successful_cortex_result_completed() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:ok:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:ok",
        source_proposal_frame_id="proposal.frame:ok",
        source_self_state_id="self.state:ok",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:inspect",
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[
            {"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "success"}
        ],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "completed"
    assert any(o.outcome_kind == "completed" for o in frame.observations)


def test_failed_cortex_result() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:fail:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:fail",
        source_proposal_frame_id="proposal.frame:fail",
        source_self_state_id="self.state:fail",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:inspect",
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[
            {"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "failed"}
        ],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "failed"


def test_self_state_improvement_positive_evidence() -> None:
    dispatch = _dispatch_dry_run()
    before = _self_state("self.state:before", {"execution_pressure": 1.0, "agency_readiness": 0.2})
    after = _self_state("self.state:after", {"execution_pressure": 0.5, "agency_readiness": 0.6})
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=before,
        self_state_after=after,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert len(frame.positive_evidence) >= 1
    assert any(o.outcome_kind == "improved" for o in frame.observations)


def test_self_state_worsening_negative_evidence() -> None:
    dispatch = _dispatch_dry_run()
    before = _self_state("self.state:before", {"agency_readiness": 0.8})
    after = _self_state("self.state:after", {"agency_readiness": 0.2})
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=before,
        self_state_after=after,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert len(frame.negative_evidence) >= 1
    assert any(o.outcome_kind == "worsened" for o in frame.observations)


def test_stable_frame_id() -> None:
    dispatch = _dispatch_dry_run()
    expected = stable_feedback_frame_id(
        dispatch_frame_id=dispatch.frame_id,
        policy_id=FEEDBACK_POLICY.policy_id,
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.frame_id == expected


def test_partial_dispatch_completed_and_absent_is_mixed() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:partial:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:partial",
        source_proposal_frame_id="proposal.frame:partial",
        source_self_state_id="self.state:partial",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=2,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:inspect",
            ),
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:summarize:execution_dispatch_policy.v1",
                source_decision_id="pd2",
                source_proposal_id="proposal:summarize:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="summarize",
                target_id="t2",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:summarize",
            ),
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[
            {"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "success"}
        ],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "mixed"
    assert len(frame.absence_evidence) >= 1


def test_completed_and_failed_is_mixed() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:mix:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:mix",
        source_proposal_frame_id="proposal.frame:mix",
        source_self_state_id="self.state:mix",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=2,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
                dispatched_at=NOW,
                result_ref="stub:result:inspect",
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[
            {"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "success"},
            {"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "failed"},
        ],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "mixed"


def test_no_mutation_side_effects() -> None:
    dispatch = _dispatch_dry_run()
    policy_frame = _policy_frame(_proposal())
    proposal = _proposal()
    dispatch_dump = dispatch.model_dump()
    policy_dump = policy_frame.model_dump()
    proposal_dump = proposal.model_dump()
    build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert dispatch.model_dump() == dispatch_dump
    assert policy_frame.model_dump() == policy_dump
    assert proposal.model_dump() == proposal_dump
