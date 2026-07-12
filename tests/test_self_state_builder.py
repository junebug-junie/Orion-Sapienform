from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
SELF_POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_exec_attention",
        node_vectors={
            "node:athena": {
                "execution_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1", "state_delta:exec_2"],
    )


def test_builder_references_sources() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert state.source_field_tick_id == field.tick_id
    assert state.source_attention_frame_id == attention.frame_id


def test_execution_pressure_nonzero() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_pressure" in state.dimensions
    assert state.dimensions["execution_pressure"].score > 0.0


def test_agency_readiness_present() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "agency_readiness" in state.dimensions


def test_dominant_attention_targets() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    ids = set(state.dominant_attention_targets)
    assert "node:athena" in ids or "capability:orchestration" in ids


def test_summary_labels_execution_loaded() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_loaded" in state.summary_labels


def test_self_state_id_stable() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    a = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    b = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert a.self_state_id == b.self_state_id


def test_dimension_reasons_reflect_real_evidence() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    execution_dim = state.dimensions["execution_pressure"]
    reasoning_dim = state.dimensions["reasoning_pressure"]

    # Reasons are not a fixed template repeated across every dimension.
    assert execution_dim.reasons != reasoning_dim.reasons
    assert "from field+attention channel synthesis" not in " ".join(
        execution_dim.reasons + reasoning_dim.reasons
    )

    # Reasons are derived from the same evidence already computed for
    # dominant_evidence, not an independent/duplicated computation.
    assert execution_dim.dominant_evidence
    for ev in execution_dim.dominant_evidence:
        assert any(ev in reason for reason in execution_dim.reasons)

    # A dimension with no contributing channel evidence this tick gets an
    # honest fallback, not a reused fixed string.
    social_dim = state.dimensions["social_pressure"]
    assert social_dim.dominant_evidence == []
    assert social_dim.reasons == ["no contributing channel evidence this tick"]


def _confidence_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_confidence",
        node_vectors={
            "node:athena": {
                "availability": 0.90,
                "confidence": 0.85,
                "thermal_pressure": 0.30,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "pressure": 0.70,
                "reliability_pressure": 0.50,
            },
            "capability:transport": {
                "available_capacity": 0.80,
            },
        },
    )


def test_dimension_confidence_varies_across_dimensions() -> None:
    # Phase 1 (2026-07-12): per-dimension confidence used to be a single
    # global proxy (0.5 + 0.5*len(dominant_targets)/5) copy-pasted onto all
    # 13 dimensions, carrying zero per-dimension information. It must now be
    # a real function of how many channels contributed to each dimension
    # this tick and how much they agree.
    field = _confidence_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)

    confidences = {dim_id: dim.confidence for dim_id, dim in state.dimensions.items()}
    assert len(set(confidences.values())) > 1, confidences

    # coherence has 3 agreeing contributing channels (availability=0.90,
    # confidence=0.85, available_capacity=0.80); reliability_pressure has
    # only 1 (reliability_pressure=0.50). More agreeing evidence must mean
    # higher confidence.
    assert state.dimensions["coherence"].confidence > state.dimensions["reliability_pressure"].confidence

    # A dimension with zero contributing channels this tick (no channel in
    # merged_channels maps to continuity_pressure in this fixture) reports
    # honest zero confidence, not a borrowed global proxy.
    assert state.dimensions["continuity_pressure"].confidence == 0.0

    # Confidence values stay within the schema's valid range.
    for dim in state.dimensions.values():
        assert 0.0 <= dim.confidence <= 1.0


def test_no_action_outputs() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    payload = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW).model_dump()
    forbidden = ("proposal", "action", "policy_gate", "cortex", "selected_action")
    for key in payload:
        assert not any(f in key.lower() for f in forbidden)
