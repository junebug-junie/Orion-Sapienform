from datetime import datetime, timezone
from pathlib import Path

from orion.proposals.builder import build_proposal_frame, stable_proposal_frame_id
from orion.proposals.policy import load_proposal_policy
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _loaded_self_state() -> SelfStateV1:
    def dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
        return SelfStateDimensionV1(
            dimension_id=dimension_id,
            score=score,
            confidence=0.9,
        )

    return SelfStateV1(
        self_state_id="self.state:tick_live:frame_live:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_live",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_live:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.655,
        overall_confidence=0.9,
        dimensions={
            "execution_pressure": dim("execution_pressure", 1.0),
            "reasoning_pressure": dim("reasoning_pressure", 0.9),
            "resource_pressure": dim("resource_pressure", 1.0),
            "agency_readiness": dim("agency_readiness", 0.6),
            "reliability_pressure": dim("reliability_pressure", 0.0),
            "field_intensity": dim("field_intensity", 0.7),
            "uncertainty": dim("uncertainty", 0.2),
        },
        dominant_attention_targets=[
            "field:recent_perturbations",
            "node:athena",
            "capability:orchestration",
            "capability:graph",
        ],
        summary_labels=[
            "attention_saturated",
            "execution_loaded",
            "field_active",
            "orchestration_pressurized",
            "reliability_clear",
            "resource_pressurized",
        ],
    )


def test_frame_references_source_self_state() -> None:
    state = _loaded_self_state()
    frame = build_proposal_frame(self_state=state, attention=None, field=None, policy=POLICY, now=NOW)
    assert frame.source_self_state_id == state.self_state_id


def test_at_least_one_candidate() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    assert len(frame.candidates) >= 1


def test_inspect_or_summarize_candidate_present() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    kinds = {c.proposal_kind for c in frame.candidates}
    assert "inspect" in kinds or "summarize" in kinds


def test_read_only_candidates_not_execution_policy() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    for candidate in frame.candidates:
        if candidate.proposal_kind in ("observe", "inspect", "summarize", "defer"):
            assert candidate.required_policy_gate in ("none", "read_only")
        assert candidate.execution_intent.get("mode") == "descriptive_only"


def test_no_execution_in_candidates() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    for candidate in frame.candidates + frame.suppressed_candidates:
        assert "execute" not in candidate.proposal_kind
        assert candidate.execution_intent.get("note") != "approved_for_execution"


def test_policy_required_when_operator_review_present() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    has_operator = any(
        c.required_policy_gate == "operator_review" for c in frame.candidates
    )
    if has_operator or frame.overall_risk >= POLICY.thresholds.policy_required_above_risk:
        assert frame.policy_required is True


def test_frame_id_stable() -> None:
    state = _loaded_self_state()
    a = build_proposal_frame(self_state=state, attention=None, field=None, policy=POLICY, now=NOW)
    b = build_proposal_frame(self_state=state, attention=None, field=None, policy=POLICY, now=NOW)
    assert a.frame_id == b.frame_id
    assert a.frame_id == stable_proposal_frame_id(
        self_state_id=state.self_state_id,
        policy_id=POLICY.policy_id,
    )


def test_evidence_refs_include_self_state() -> None:
    state = _loaded_self_state()
    frame = build_proposal_frame(self_state=state, attention=None, field=None, policy=POLICY, now=NOW)
    assert any(
        ref.startswith("self_state:") for c in frame.candidates for ref in c.evidence_refs
    )


def test_suppressed_candidates_separate() -> None:
    frame = build_proposal_frame(
        self_state=_loaded_self_state(),
        attention=None,
        field=None,
        policy=POLICY,
        now=NOW,
    )
    active_ids = {c.proposal_id for c in frame.candidates}
    suppressed_ids = {c.proposal_id for c in frame.suppressed_candidates}
    assert active_ids.isdisjoint(suppressed_ids)
