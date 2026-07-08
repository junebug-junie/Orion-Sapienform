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
NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def test_contract_pressure_does_not_floor_reliability() -> None:
    # contract_pressure maxed (catalog drift), but NO real reliability failure.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_contract_only",
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 1.0,
                "catalog_drift_pressure": 1.0,
                "failure_pressure": 0.0,
                "execution_friction": 0.0,
            }
        },
    )
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    rel = state.dimensions["reliability_pressure"].score
    assert rel < 0.5, f"catalog/contract drift must not floor felt reliability (got {rel})"


def test_real_failure_still_raises_reliability() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_real_failure",
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 0.0,
                "failure_pressure": 1.0,
            }
        },
    )
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert state.dimensions["reliability_pressure"].score >= 0.9
