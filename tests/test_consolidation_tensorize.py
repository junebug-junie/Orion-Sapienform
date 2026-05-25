from datetime import datetime, timezone
from pathlib import Path

import orion.consolidation.tensorize as tensorize_module
from orion.consolidation.policy import ConsolidationPolicyV1, load_consolidation_policy
from orion.consolidation.tensorize import build_sparse_tensor_slices
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import MotifObservationV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_consolidation_policy(REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)


def _dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)


def _self_state(self_state_id: str, *, attention_frame_id: str = "attention.frame:1") -> SelfStateV1:
    return SelfStateV1(
        self_state_id=self_state_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id=attention_frame_id,
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


def _proposal_frame(frame_id: str) -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_self_state_id="self.state:1",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="attention.frame:1",
        source_field_tick_id="tick",
        overall_action_pressure=0.5,
        overall_risk=0.1,
        candidates=[
            ProposalCandidateV1(
                proposal_id="proposal:inspect:1",
                proposal_kind="inspect",
                title="Inspect",
                description="Inspect state",
                target_id="athena",
                target_kind="node",
                priority_score=0.5,
                urgency_score=0.4,
                confidence_score=0.8,
                risk_score=0.1,
                reversibility_score=1.0,
                proposed_effect="increase_observability",
            )
        ],
    )


def _policy_frame(frame_id: str) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id=f"policy.decision:{frame_id}",
        proposal_id="proposal:inspect:1",
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
        source_proposal_frame_id="proposal.frame:1",
        source_self_state_id="self.state:1",
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
        execution_allowed=False,
    )


def _dispatch_frame(frame_id: str) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_policy_frame_id="policy.frame:1",
        source_proposal_frame_id="proposal.frame:1",
        source_self_state_id="self.state:1",
        candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:1",
                source_decision_id="policy.decision:1",
                source_proposal_id="proposal:inspect:1",
                dispatch_status="dry_run",
                dispatch_mode="dry_run",
                dispatch_kind="inspect",
                target_id="athena",
                target_kind="node",
                risk_score=0.1,
                confidence_score=0.9,
            )
        ],
    )


def _feedback_frame(frame_id: str, *, dispatch_frame_id: str) -> FeedbackFrameV1:
    return FeedbackFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_execution_dispatch_frame_id=dispatch_frame_id,
        outcome_status="dry_run_only",
        outcome_score=0.5,
        confidence_score=0.9,
    )


def _motif(label: str) -> MotifObservationV1:
    return MotifObservationV1(
        motif_id=f"motif:{label}:consolidation_policy.v1",
        motif_kind="self_state_pattern",
        label=label,
        recurrence_count=3,
        support_score=0.75,
        confidence_score=1.0,
        evidence_frame_ids=["self.state:1", "feedback.frame:1"],
    )


def _window(**overrides: object) -> ConsolidationWindowData:
    defaults: dict[str, object] = {
        "window_start": START,
        "window_end": NOW,
        "self_states": [_self_state("self.state:1")],
        "attention_frames": [_attention_frame("attention.frame:1")],
        "proposal_frames": [_proposal_frame("proposal.frame:1")],
        "policy_frames": [_policy_frame("policy.frame:1")],
        "dispatch_frames": [_dispatch_frame("execution.dispatch.frame:1")],
        "feedback_frames": [_feedback_frame("feedback.frame:1", dispatch_frame_id="execution.dispatch.frame:1")],
    }
    defaults.update(overrides)
    return ConsolidationWindowData(**defaults)  # type: ignore[arg-type]


def _slices(**overrides: object):
    window = _window(**overrides)
    motifs = [_motif("loaded_but_reliable")]
    return build_sparse_tensor_slices(
        window=window,
        motifs=motifs,
        expectations=[],
        policy=POLICY,
    )


def _slice_by_kind(slices, kind: str):
    matches = [item for item in slices if item.tensor_kind == kind]
    assert len(matches) == 1, f"expected one slice for {kind}, got {matches}"
    return matches[0]


def test_field_attention_self_tensor_has_expected_axes() -> None:
    tensor = _slice_by_kind(_slices(), "field_attention_self")
    assert tensor.axes == ["time_bucket", "self_condition", "attention_target", "dimension"]


def test_policy_dispatch_feedback_tensor_has_expected_axes() -> None:
    tensor = _slice_by_kind(_slices(), "policy_dispatch_feedback")
    assert tensor.axes == ["proposal_kind", "policy_decision", "dispatch_status", "feedback_outcome"]


def test_motif_condition_outcome_tensor_has_expected_axes() -> None:
    tensor = _slice_by_kind(_slices(), "motif_condition_outcome")
    assert tensor.axes == ["motif", "self_condition", "outcome_status"]


def test_coordinates_are_sparse_dicts() -> None:
    tensor = _slice_by_kind(_slices(), "field_attention_self")
    assert tensor.coordinates
    assert all(isinstance(coord, dict) for coord in tensor.coordinates)
    assert all(isinstance(key, str) and isinstance(value, str) for coord in tensor.coordinates for key, value in coord.items())


def test_values_length_equals_coordinates_length() -> None:
    for tensor in _slices():
        assert len(tensor.values) == len(tensor.coordinates)


def test_coordinate_count_is_capped() -> None:
    capped_policy = POLICY.model_copy(update={"tensor": POLICY.tensor.model_copy(update={"max_coordinates": 2})})
    window = _window(
        self_states=[
            _self_state("self.state:1"),
            _self_state("self.state:2", attention_frame_id="attention.frame:2"),
            _self_state("self.state:3", attention_frame_id="attention.frame:3"),
        ],
        attention_frames=[
            _attention_frame("attention.frame:1"),
            _attention_frame("attention.frame:2"),
            _attention_frame("attention.frame:3"),
        ],
    )
    tensor = build_sparse_tensor_slices(
        window=window,
        motifs=[_motif("loaded_but_reliable")],
        expectations=[],
        policy=capped_policy,
    )[0]
    assert len(tensor.coordinates) <= 2
    assert len(tensor.values) <= 2


def test_tensor_evidence_refs_are_present() -> None:
    for tensor in _slices():
        assert tensor.evidence_refs


def test_no_model_training_or_embedding_code() -> None:
    source = Path(tensorize_module.__file__).read_text(encoding="utf-8")
    forbidden = ("sklearn", "torch", "tensorflow", "embedding", "train(")
    assert not any(token in source for token in forbidden)
