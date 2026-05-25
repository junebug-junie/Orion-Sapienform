from datetime import datetime, timezone
from pathlib import Path

from orion.consolidation.expectation import build_expectations_from_motifs
from orion.consolidation.motif import detect_motifs
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import MotifObservationV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_consolidation_policy(REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
POLICY_ID = POLICY.policy_id


def _dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)


def _self_state(self_state_id: str) -> SelfStateV1:
    return SelfStateV1(
        self_state_id=self_state_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.7,
        overall_confidence=0.9,
        dimensions={
            "execution_pressure": _dim("execution_pressure", 0.8),
            "reliability_pressure": _dim("reliability_pressure", 0.2),
        },
    )


def _attention_frame(frame_id: str) -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="athena",
        target_kind="node",
        salience_score=0.8,
        pressure_score=0.7,
        novelty_score=0.1,
        urgency_score=0.5,
        confidence_score=0.9,
    )
    return FieldAttentionFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        overall_salience=0.8,
        node_targets=[target],
        dominant_targets=[target],
    )


def _policy_frame(frame_id: str) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id=f"policy.decision:{frame_id}",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
    )
    return PolicyDecisionFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:test",
        source_self_state_id="self.state:test",
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
        execution_allowed=False,
    )


def _review_policy_frame(frame_id: str) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id=f"policy.decision:{frame_id}",
        proposal_id="proposal:exec:state",
        decision="requires_operator_review",
        policy_gate="operator_review",
        risk_score=0.6,
        reversibility_score=0.4,
        confidence_score=0.8,
        allowed_scope="operator_review_required",
    )
    return PolicyDecisionFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:review",
        source_self_state_id="self.state:review",
        decisions=[decision],
        review_required_decisions=[decision],
        overall_risk=0.6,
        operator_review_required=True,
        execution_allowed=False,
    )


def _feedback_frame(frame_id: str) -> FeedbackFrameV1:
    return FeedbackFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_execution_dispatch_frame_id=f"execution.dispatch.frame:{frame_id}",
        outcome_status="dry_run_only",
        outcome_score=0.5,
        confidence_score=0.9,
    )


def _dispatch_blocked(frame_id: str) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_policy_frame_id="policy.frame:blocked",
        source_proposal_frame_id="proposal.frame:blocked",
        source_self_state_id="self.state:blocked",
        blocked_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id=f"dispatch:blocked:{frame_id}",
                source_decision_id="pd1",
                source_proposal_id="proposal:blocked",
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
        blocked_count=1,
    )


def _empty_window(**overrides: object) -> ConsolidationWindowData:
    defaults: dict[str, object] = {
        "window_start": START,
        "window_end": NOW,
        "self_states": [],
        "attention_frames": [],
        "proposal_frames": [],
        "policy_frames": [],
        "dispatch_frames": [],
        "feedback_frames": [],
    }
    defaults.update(overrides)
    return ConsolidationWindowData(**defaults)  # type: ignore[arg-type]


def _motif(label: str, *, confidence: float = 1.0, recurrence: int = 3) -> MotifObservationV1:
    return MotifObservationV1(
        motif_id=f"motif:{label}:{POLICY_ID}",
        motif_kind="self_state_pattern",
        label=label,
        recurrence_count=recurrence,
        support_score=0.75,
        confidence_score=confidence,
        evidence_frame_ids=[f"evidence:{label}:1", f"evidence:{label}:2"],
    )


def _expectations_for_window(window: ConsolidationWindowData):
    motifs = detect_motifs(window=window, policy=POLICY)
    return build_expectations_from_motifs(
        motifs=motifs,
        feedback_frames=window.feedback_frames,
        policy=POLICY,
    )


def test_loaded_but_reliable_creates_reliability_clear_expectation() -> None:
    window = _empty_window(
        self_states=[_self_state("self.state:1"), _self_state("self.state:2"), _self_state("self.state:3")]
    )
    expectations = _expectations_for_window(window)
    match = [e for e in expectations if e.expected_outcome_kind == "reliability_clear"]
    assert len(match) == 1
    assert match[0].trigger_motif_id == f"motif:loaded_but_reliable:{POLICY_ID}"


def test_read_only_policy_loop_creates_read_only_approved_expectation() -> None:
    window = _empty_window(
        policy_frames=[_policy_frame("policy.frame:1"), _policy_frame("policy.frame:2"), _policy_frame("policy.frame:3")]
    )
    expectations = _expectations_for_window(window)
    match = [e for e in expectations if e.expected_outcome_kind == "read_only_approved"]
    assert len(match) == 1


def test_dry_run_feedback_loop_creates_dry_run_feedback_expectation() -> None:
    window = _empty_window(
        feedback_frames=[
            _feedback_frame("feedback.frame:1"),
            _feedback_frame("feedback.frame:2"),
            _feedback_frame("feedback.frame:3"),
        ]
    )
    expectations = _expectations_for_window(window)
    match = [e for e in expectations if e.expected_outcome_kind == "dry_run_feedback"]
    assert len(match) == 1


def test_blocked_review_loop_creates_policy_review_required_expectation() -> None:
    window = _empty_window(
        policy_frames=[_review_policy_frame("policy.frame:review:1"), _review_policy_frame("policy.frame:review:2")],
        dispatch_frames=[_dispatch_blocked("execution.dispatch.frame:blocked:1")],
    )
    expectations = _expectations_for_window(window)
    match = [e for e in expectations if e.expected_outcome_kind == "policy_review_required"]
    assert len(match) == 1


def test_expectation_confidence_follows_motif_confidence() -> None:
    motifs = [_motif("loaded_but_reliable", confidence=0.42)]
    expectations = build_expectations_from_motifs(motifs=motifs, feedback_frames=[], policy=POLICY)
    assert expectations[0].confidence_score == 0.42
    assert expectations[0].support_count == 3


def test_evidence_refs_are_preserved() -> None:
    motifs = [_motif("loaded_but_reliable")]
    expectations = build_expectations_from_motifs(motifs=motifs, feedback_frames=[], policy=POLICY)
    assert expectations[0].evidence_refs == ["evidence:loaded_but_reliable:1", "evidence:loaded_but_reliable:2"]


def test_attention_saturated_execution_creates_execution_pressure_high() -> None:
    window = _empty_window(
        attention_frames=[
            _attention_frame("attention.frame:1"),
            _attention_frame("attention.frame:2"),
            _attention_frame("attention.frame:3"),
        ]
    )
    expectations = _expectations_for_window(window)
    match = [e for e in expectations if e.expected_outcome_kind == "execution_pressure_high"]
    assert len(match) == 1
