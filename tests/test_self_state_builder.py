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


def test_no_action_outputs() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    payload = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW).model_dump()
    forbidden = ("proposal", "action", "policy_gate", "cortex", "selected_action")
    for key in payload:
        assert not any(f in key.lower() for f in forbidden)
