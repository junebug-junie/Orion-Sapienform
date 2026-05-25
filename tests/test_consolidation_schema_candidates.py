from datetime import datetime, timezone
from pathlib import Path

import orion.consolidation.schema_candidates as schema_candidates_module
from orion.consolidation.builder import build_consolidation_frame
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.schema_candidates import build_schema_candidates
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import ExpectationV1, MotifObservationV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
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


def _motif(label: str) -> MotifObservationV1:
    return MotifObservationV1(
        motif_id=f"motif:{label}:{POLICY_ID}",
        motif_kind="self_state_pattern",
        label=label,
        recurrence_count=3,
        support_score=0.75,
        confidence_score=1.0,
        evidence_frame_ids=[f"evidence:{label}:1"],
    )


def _expectation(motif: MotifObservationV1, kind: str) -> ExpectationV1:
    return ExpectationV1(
        expectation_id=f"expectation:{motif.motif_id}:{kind}",
        trigger_motif_id=motif.motif_id,
        expected_outcome_kind=kind,  # type: ignore[arg-type]
        confidence_score=motif.confidence_score,
        support_count=motif.recurrence_count,
    )


def test_loaded_but_reliable_produces_prior_candidate() -> None:
    motif = _motif("loaded_but_reliable")
    candidates = build_schema_candidates(
        motifs=[motif],
        expectations=[_expectation(motif, "reliability_clear")],
        policy=POLICY,
    )
    match = [item for item in candidates if item.candidate_kind == "prior_candidate"]
    assert len(match) == 1
    assert match[0].label == "loaded_but_reliable_operating_mode"


def test_dry_run_feedback_loop_produces_habit_candidate() -> None:
    motif = _motif("dry_run_feedback_loop")
    candidates = build_schema_candidates(
        motifs=[motif],
        expectations=[_expectation(motif, "dry_run_feedback")],
        policy=POLICY,
    )
    match = [item for item in candidates if item.candidate_kind == "habit_candidate"]
    assert len(match) == 1
    assert match[0].proposed_schema["expected_feedback"] == "dry_run_only"


def test_read_only_policy_loop_produces_policy_candidate() -> None:
    motif = _motif("read_only_policy_loop")
    candidates = build_schema_candidates(
        motifs=[motif],
        expectations=[_expectation(motif, "read_only_approved")],
        policy=POLICY,
    )
    match = [item for item in candidates if item.candidate_kind == "policy_candidate"]
    assert len(match) == 1
    assert match[0].proposed_schema["gate"] == "read_only"


def test_promotion_status_is_candidate_only() -> None:
    motif = _motif("loaded_but_reliable")
    candidates = build_schema_candidates(motifs=[motif], expectations=[], policy=POLICY)
    assert candidates
    assert all(candidate.promotion_status == "candidate_only" for candidate in candidates)


def test_candidate_includes_evidence_refs() -> None:
    motif = _motif("loaded_but_reliable")
    candidates = build_schema_candidates(motifs=[motif], expectations=[], policy=POLICY)
    assert candidates[0].evidence_refs == ["evidence:loaded_but_reliable:1"]


def test_no_rdf_writes() -> None:
    source = Path(schema_candidates_module.__file__).read_text(encoding="utf-8")
    assert "rdf" not in source.lower()
    assert "graphdb" not in source.lower()


def test_builder_emits_schema_candidates_without_policy_mutation() -> None:
    window = ConsolidationWindowData(
        window_start=START,
        window_end=NOW,
        self_states=[_self_state("self.state:1"), _self_state("self.state:2"), _self_state("self.state:3")],
        attention_frames=[],
        proposal_frames=[],
        policy_frames=[_policy_frame("policy.frame:1"), _policy_frame("policy.frame:2"), _policy_frame("policy.frame:3")],
        dispatch_frames=[_dispatch_blocked("execution.dispatch.frame:1")],
        feedback_frames=[
            _feedback_frame("feedback.frame:1"),
            _feedback_frame("feedback.frame:2"),
            _feedback_frame("feedback.frame:3"),
        ],
    )
    frame = build_consolidation_frame(window=window, policy=POLICY, generated_at=NOW)
    assert frame.schema_candidates
    assert all(candidate.promotion_status == "candidate_only" for candidate in frame.schema_candidates)
