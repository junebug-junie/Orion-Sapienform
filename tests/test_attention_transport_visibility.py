from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "config" / "attention" / "field_attention_policy.v1.yaml"
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _field_with_transport_drift() -> FieldStateV1:
    return FieldStateV1(
        tick_id="tick_transport_attn",
        generated_at=NOW,
        topology_id="orion_field_topology",
        topology_version="v1",
        node_vectors={
            "node:athena": {
                "contract_pressure": 1.0,
                "transport_pressure": 0.0,
                "bus_health": 1.0,
            }
        },
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 1.0,
                "pressure": 0.0,
                "confidence": 1.0,
            }
        },
        edges=[],
        recent_perturbations=["delta_transport_1"],
    )


def test_transport_capability_becomes_attention_item() -> None:
    policy = load_attention_policy(POLICY_PATH)
    frame = build_attention_frame(field=_field_with_transport_drift(), policy=policy, now=NOW)
    transport_items = [t for t in frame.capability_targets if t.target_id == "capability:transport"]
    assert transport_items
    assert any("contract_pressure" in r for r in transport_items[0].reasons)


def test_contract_pressure_contributes_to_salience() -> None:
    policy = load_attention_policy(POLICY_PATH)
    frame = build_attention_frame(field=_field_with_transport_drift(), policy=policy, now=NOW)
    item = next(t for t in frame.capability_targets if t.target_id == "capability:transport")
    assert item.salience_score >= policy.thresholds.min_salience
    assert "contract_pressure" in item.dominant_channels
