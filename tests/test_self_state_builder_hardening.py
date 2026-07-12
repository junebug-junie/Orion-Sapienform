from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(
    REPO / "config" / "attention" / "field_attention_policy.v1.yaml"
)
SELF_POLICY = load_self_state_policy(
    REPO / "config" / "self_state" / "self_state_policy.v1.yaml"
)
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _live_shaped_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_semantic_hardening",
        node_vectors={
            "node:athena": {
                "availability": 1.0,
                "confidence": 1.0,
                "available_capacity": 1.0,
                "execution_load": 1.0,
                "cpu_pressure": 0.92,
                "pressure": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=[f"state_delta:{i}" for i in range(20)],
    )


def _built_state():
    field = _live_shaped_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    return build_self_state(
        field=field, attention=attention, policy=SELF_POLICY, now=NOW
    )


def test_stabilizers_not_in_unresolved_pressures() -> None:
    state = _built_state()
    assert "availability→coherence" not in state.unresolved_pressures
    assert "confidence→coherence" not in state.unresolved_pressures
    assert "available_capacity→coherence" not in state.unresolved_pressures


def test_stabilizers_in_stabilizing_factors() -> None:
    state = _built_state()
    assert "availability=1.00" in state.stabilizing_factors
    assert "confidence=1.00" in state.stabilizing_factors
    assert "available_capacity=1.00" in state.stabilizing_factors


def test_pressure_channels_in_unresolved_pressures() -> None:
    state = _built_state()
    assert "execution_pressure→execution_pressure" in state.unresolved_pressures
    assert "pressure→resource_pressure" in state.unresolved_pressures
    # Phase 1 double-counting fix regression guard: raw node-level
    # execution_load/cpu_pressure no longer have a channel_dimension_map
    # entry (config/self_state/self_state_policy.v1.yaml), so they can never
    # be flagged as unresolved anymore — their signal reaches dimensions only
    # via the diffused capability channel names above.
    assert "execution_load→execution_pressure" not in state.unresolved_pressures
    assert "cpu_pressure→resource_pressure" not in state.unresolved_pressures


def test_dimension_evidence_is_dimension_specific() -> None:
    state = _built_state()
    execution_evidence = state.dimensions["execution_pressure"].dominant_evidence
    coherence_evidence = state.dimensions["coherence"].dominant_evidence
    assert any(
        "execution_load" in x or "execution_pressure" in x for x in execution_evidence
    )
    assert any(
        "availability" in x or "confidence" in x for x in coherence_evidence
    )
    assert execution_evidence != coherence_evidence


def test_context_channels_not_unresolved() -> None:
    state = _built_state()
    assert not any(
        "recent_perturbation_count" in x for x in state.unresolved_pressures
    )
    assert not any("overall_salience" in x for x in state.unresolved_pressures)


def test_stabilized_but_loaded_label() -> None:
    state = _built_state()
    assert state.dimensions["coherence"].score >= 0.8
    assert state.dimensions["execution_pressure"].score >= 0.7
    assert "stabilized_but_loaded" in state.summary_labels
