from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _fixtures():
    from orion.schemas.field_state import FieldStateV1

    field = FieldStateV1(
        tick_id="tick_self_transport",
        generated_at=NOW,
        topology_id="orion_field_topology",
        topology_version="v1",
        node_vectors={
            "node:athena": {
                "bus_health": 1.0,
                "delivery_confidence": 1.0,
                "contract_pressure": 1.0,
                "transport_pressure": 0.0,
            }
        },
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 1.0,
                "confidence": 1.0,
                "available_capacity": 1.0,
            }
        },
        edges=[],
        recent_perturbations=[],
    )
    attn_policy = load_attention_policy(REPO_ROOT / "config" / "attention" / "field_attention_policy.v1.yaml")
    attention = build_attention_frame(field=field, policy=attn_policy, now=NOW)
    self_policy = load_self_state_policy(REPO_ROOT / "config" / "self_state" / "self_state_policy.v1.yaml")
    return field, attention, self_policy


def test_flag_false_no_transport_dimension() -> None:
    field, attention, policy = _fixtures()
    state = build_self_state(field=field, attention=attention, policy=policy, enable_transport_influence=False)
    assert "transport_integrity" not in state.dimensions


def test_flag_true_catalog_drift_transport_integrity() -> None:
    field, attention, policy = _fixtures()
    state = build_self_state(field=field, attention=attention, policy=policy, enable_transport_influence=True)
    assert "transport_integrity" in state.dimensions
    assert state.dimensions["transport_integrity"].score == 0.5
    assert "transport_contract_drift" in state.summary_labels
