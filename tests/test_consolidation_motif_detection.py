from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.consolidation.motif import detect_motifs
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_consolidation_policy(REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)


def _dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)


def _self_state(
    self_state_id: str,
    *,
    overall_condition: str = "loaded",
    execution_pressure: float = 0.8,
    reliability_pressure: float = 0.2,
) -> SelfStateV1:
    return SelfStateV1(
        self_state_id=self_state_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        overall_condition=overall_condition,
        overall_intensity=0.7,
        overall_confidence=0.9,
        dimensions={
            "execution_pressure": _dim("execution_pressure", execution_pressure),
            "reliability_pressure": _dim("reliability_pressure", reliability_pressure),
        },
    )


def _attention_frame(frame_id: str, *, salience: float = 0.8) -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="athena",
        target_kind="node",
        salience_score=salience,
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
        overall_salience=salience,
        node_targets=[target],
        dominant_targets=[target],
    )


def _policy_frame(frame_id: str, *, execution_allowed: bool = False, read_only: bool = True) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id=f"policy.decision:{frame_id}",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only" if read_only else "rejected",
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
        approved_decisions=[decision] if read_only else [],
        overall_risk=0.05,
        execution_allowed=execution_allowed,
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


def _feedback_frame(
    frame_id: str,
    *,
    outcome_status: str = "dry_run_only",
    with_unchanged_delta: bool = False,
) -> FeedbackFrameV1:
    observations: list[OutcomeObservationV1] = []
    if with_unchanged_delta:
        observations.append(
            OutcomeObservationV1(
                observation_id=f"obs:{frame_id}:delta",
                source_kind="self_state_delta",
                source_id="self.state:before",
                outcome_kind="unchanged",
                score=0.5,
                confidence=0.9,
                observed_at=NOW,
            )
        )
    return FeedbackFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_execution_dispatch_frame_id=f"execution.dispatch.frame:{frame_id}",
        outcome_status=outcome_status,
        outcome_score=0.5,
        confidence_score=0.9,
        observations=observations,
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


def _motif_by_label(motifs: list, label: str):
    matches = [m for m in motifs if m.label == label]
    assert len(matches) == 1, f"expected one motif {label}, got {matches}"
    return matches[0]


def test_loaded_but_reliable_detected() -> None:
    window = _empty_window(
        self_states=[
            _self_state("self.state:1"),
            _self_state("self.state:2"),
            _self_state("self.state:3"),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "loaded_but_reliable")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "self_state_pattern"


def test_attention_saturated_execution_detected() -> None:
    window = _empty_window(
        attention_frames=[
            _attention_frame("attention.frame:1"),
            _attention_frame("attention.frame:2"),
            _attention_frame("attention.frame:3"),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "attention_saturated_execution")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "attention_pattern"


def test_read_only_policy_loop_detected() -> None:
    window = _empty_window(
        policy_frames=[
            _policy_frame("policy.frame:1"),
            _policy_frame("policy.frame:2"),
            _policy_frame("policy.frame:3"),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "read_only_policy_loop")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "proposal_policy_pattern"


def test_dry_run_feedback_loop_detected() -> None:
    window = _empty_window(
        feedback_frames=[
            _feedback_frame("feedback.frame:1"),
            _feedback_frame("feedback.frame:2"),
            _feedback_frame("feedback.frame:3"),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "dry_run_feedback_loop")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "dispatch_feedback_pattern"


def test_blocked_review_loop_detected() -> None:
    window = _empty_window(
        policy_frames=[
            _review_policy_frame("policy.frame:review:1"),
            _review_policy_frame("policy.frame:review:2"),
        ],
        dispatch_frames=[_dispatch_blocked("execution.dispatch.frame:blocked:1")],
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "blocked_review_loop")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "dispatch_feedback_pattern"


def test_stable_after_dry_run_detected() -> None:
    window = _empty_window(
        feedback_frames=[
            _feedback_frame("feedback.frame:stable:1", with_unchanged_delta=True),
            _feedback_frame("feedback.frame:stable:2", with_unchanged_delta=True),
            _feedback_frame("feedback.frame:stable:3", with_unchanged_delta=True),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "stable_after_dry_run")
    assert motif.recurrence_count == 3
    assert motif.motif_kind == "stability_pattern"


def test_motif_scores_and_evidence() -> None:
    window = _empty_window(
        self_states=[
            _self_state("self.state:1"),
            _self_state("self.state:2"),
            _self_state("self.state:3"),
            _self_state("self.state:4", overall_condition="steady"),
        ]
    )
    motifs = detect_motifs(window=window, policy=POLICY)
    motif = _motif_by_label(motifs, "loaded_but_reliable")
    assert motif.recurrence_count == 3
    assert motif.support_score == 0.75
    assert motif.confidence_score == 1.0
    assert len(motif.evidence_frame_ids) == 3
    assert "self.state:1" in motif.evidence_frame_ids
    assert motif.first_seen_at == NOW
    assert motif.last_seen_at == NOW
    assert motif.dominant_dimensions["execution_pressure"] == pytest.approx(0.8)
    assert motif.dominant_dimensions["reliability_pressure"] == pytest.approx(0.2)
