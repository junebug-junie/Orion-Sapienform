from datetime import datetime, timezone
from pathlib import Path

from orion.proposals.builder import (
    ATTENTION_FIRST_TARGET_BINDING,
    _build_candidate,
    build_proposal_frame,
    stable_proposal_frame_id,
)
from orion.proposals.policy import ProposalTemplateV1, load_proposal_policy
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _loaded_field() -> FieldStateV1:
    """2026-07-22 (SelfStateV1 burn): replaces the old SelfStateV1 fixture.
    node_vectors carry the same real pressure values the old dimension fixture
    hand-set directly, so PRESSURE_DIMENSIONS scoring behaves the same for the
    dimensions that survive the burn (execution/resource/reasoning_pressure)."""
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_live",
        node_vectors={
            "node:athena": {
                "execution_pressure": 1.0,
                "reasoning_pressure": 0.9,
                "pressure": 1.0,
                "reliability_pressure": 0.0,
            },
        },
    )


def _loaded_attention(field_tick_id: str = "tick_live") -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:tick_live:field_attention_policy.v1",
        generated_at=NOW,
        source_field_tick_id=field_tick_id,
        source_field_generated_at=NOW,
        overall_salience=0.6,
        dominant_targets=[
            FieldAttentionTargetV1(
                target_id="field:recent_perturbations",
                target_kind="field",
                salience_score=0.9,
                pressure_score=0.9,
                novelty_score=0.5,
                urgency_score=0.5,
                confidence_score=0.9,
            ),
            FieldAttentionTargetV1(
                target_id="node:athena",
                target_kind="node",
                salience_score=0.7,
                pressure_score=0.7,
                novelty_score=0.3,
                urgency_score=0.4,
                confidence_score=0.9,
            ),
        ],
    )


def test_frame_references_source_field() -> None:
    field = _loaded_field()
    frame = build_proposal_frame(field=field, attention=None, policy=POLICY, now=NOW)
    assert frame.source_field_tick_id == field.tick_id


def test_at_least_one_candidate() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    assert len(frame.candidates) >= 1


def test_inspect_or_summarize_candidate_present() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    kinds = {c.proposal_kind for c in frame.candidates}
    assert "inspect" in kinds or "summarize" in kinds


def test_read_only_candidates_not_execution_policy() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    for candidate in frame.candidates:
        if candidate.proposal_kind in ("observe", "inspect", "summarize", "defer"):
            assert candidate.required_policy_gate in ("none", "read_only")
        assert candidate.execution_intent.get("mode") == "descriptive_only"


def test_no_execution_in_candidates() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    for candidate in frame.candidates + frame.suppressed_candidates:
        assert "execute" not in candidate.proposal_kind
        assert candidate.execution_intent.get("note") != "approved_for_execution"


def test_policy_required_when_operator_review_present() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    has_operator = any(
        c.required_policy_gate == "operator_review" for c in frame.candidates
    )
    if has_operator or frame.overall_risk >= POLICY.thresholds.policy_required_above_risk:
        assert frame.policy_required is True


def test_frame_id_stable() -> None:
    field = _loaded_field()
    attention = _loaded_attention()
    a = build_proposal_frame(field=field, attention=attention, policy=POLICY, now=NOW)
    b = build_proposal_frame(field=field, attention=attention, policy=POLICY, now=NOW)
    assert a.frame_id == b.frame_id
    assert a.frame_id == stable_proposal_frame_id(
        field_tick_id=field.tick_id,
        attention_frame_id=attention.frame_id,
        policy_id=POLICY.policy_id,
    )


def test_evidence_refs_include_field() -> None:
    field = _loaded_field()
    frame = build_proposal_frame(field=field, attention=None, policy=POLICY, now=NOW)
    assert any(
        ref.startswith("field:") for c in frame.candidates for ref in c.evidence_refs
    )


def test_suppressed_candidates_separate() -> None:
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=None,
        policy=POLICY,
        now=NOW,
    )
    active_ids = {c.proposal_id for c in frame.candidates}
    suppressed_ids = {c.proposal_id for c in frame.suppressed_candidates}
    assert active_ids.isdisjoint(suppressed_ids)


# --- P5: attention-bound proposal target binding -----------------------------

_BINDING_TEMPLATE = ProposalTemplateV1(
    kind="inspect",
    target_kind="capability",
    target_id="capability:orchestration",
    target_binding=ATTENTION_FIRST_TARGET_BINDING,
    proposed_effect="increase_observability",
    required_policy_gate="read_only",
    base_priority=0.34,
    base_risk=0.05,
    reversibility=1.0,
    dimensions={"execution_pressure": 0.30},
)


def _attention_with_targets(targets: list[FieldAttentionTargetV1]) -> FieldAttentionFrameV1:
    attention = _loaded_attention()
    return attention.model_copy(update={"dominant_targets": targets})


def test_binding_resolves_target_from_attention() -> None:
    targets = [
        FieldAttentionTargetV1(
            target_id="node:mycelium",
            target_kind="node",
            salience_score=0.8,
            pressure_score=0.8,
            novelty_score=0.5,
            urgency_score=0.5,
            confidence_score=0.8,
        ),
        FieldAttentionTargetV1(
            target_id="capability:orchestration",
            target_kind="capability",
            salience_score=0.5,
            pressure_score=0.5,
            novelty_score=0.3,
            urgency_score=0.3,
            confidence_score=0.8,
        ),
    ]
    field = _loaded_field()
    candidate = _build_candidate(
        template_key="inspect_attended_target",
        template=_BINDING_TEMPLATE,
        field_tick_id=field.tick_id,
        attention=_attention_with_targets(targets),
        pressures={},
        policy=POLICY,
    )
    assert candidate.target_id == "node:mycelium"
    assert candidate.target_kind == "node"
    assert candidate.binding_resolved_from == ATTENTION_FIRST_TARGET_BINDING


def test_binding_falls_back_to_literal_when_attention_absent() -> None:
    field = _loaded_field()
    candidate = _build_candidate(
        template_key="inspect_attended_target",
        template=_BINDING_TEMPLATE,
        field_tick_id=field.tick_id,
        attention=None,
        pressures={},
        policy=POLICY,
    )
    assert candidate.target_id == _BINDING_TEMPLATE.target_id
    assert candidate.target_kind == _BINDING_TEMPLATE.target_kind
    assert candidate.binding_resolved_from is None


def test_binding_falls_back_to_literal_when_kind_unaccepted() -> None:
    # "channel" is a valid FieldAttentionTargetV1.target_kind but is not in
    # ProposalCandidateV1's target_kind Literal -- must fail closed, not raise.
    targets = [
        FieldAttentionTargetV1(
            target_id="channel:orion.bus.thoughts",
            target_kind="channel",
            salience_score=0.7,
            pressure_score=0.7,
            novelty_score=0.3,
            urgency_score=0.3,
            confidence_score=0.7,
        ),
    ]
    field = _loaded_field()
    candidate = _build_candidate(
        template_key="inspect_attended_target",
        template=_BINDING_TEMPLATE,
        field_tick_id=field.tick_id,
        attention=_attention_with_targets(targets),
        pressures={},
        policy=POLICY,
    )
    assert candidate.target_id == _BINDING_TEMPLATE.target_id
    assert candidate.target_kind == _BINDING_TEMPLATE.target_kind
    assert candidate.binding_resolved_from is None


def test_binding_resolved_from_none_for_non_binding_template() -> None:
    template = POLICY.proposal_templates["inspect_execution_pressure"]
    assert template.target_binding is None
    field = _loaded_field()
    candidate = _build_candidate(
        template_key="inspect_execution_pressure",
        template=template,
        field_tick_id=field.tick_id,
        attention=None,
        pressures={},
        policy=POLICY,
    )
    assert candidate.binding_resolved_from is None
    assert candidate.target_id == template.target_id


def test_build_proposal_frame_includes_binding_resolved_candidate() -> None:
    targets = [
        FieldAttentionTargetV1(
            target_id="field:recent_perturbations",
            target_kind="field",
            salience_score=0.9,
            pressure_score=0.9,
            novelty_score=0.5,
            urgency_score=0.5,
            confidence_score=0.9,
        ),
    ]
    frame = build_proposal_frame(
        field=_loaded_field(),
        attention=_attention_with_targets(targets),
        policy=POLICY,
        now=NOW,
    )
    attended = [
        c for c in frame.candidates + frame.suppressed_candidates
        if c.proposal_id.startswith("proposal:inspect_attended_target:")
    ]
    assert len(attended) == 1
    assert attended[0].binding_resolved_from == ATTENTION_FIRST_TARGET_BINDING
    assert attended[0].target_id == "field:recent_perturbations"
    assert attended[0].target_kind == "field"
