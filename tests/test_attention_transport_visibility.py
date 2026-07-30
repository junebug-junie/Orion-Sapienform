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
                "stream_backlog_pressure": 0.0,
                "stream_backlog_health": 1.0,
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


def test_capability_transport_never_becomes_an_attention_item() -> None:
    # 2026-07-30: capability attention was killed outright (no hand-weighted
    # fallback for capability_channel_weights) -- select_capability_targets
    # always returns []. This test now pins that reality instead of the old
    # (now-impossible) "transport capability shows up" behavior. Real
    # capability attention needs its own theory-grounded instrument built
    # first, per orion/attention/field_attention/selectors.py's own
    # docstring.
    policy = load_attention_policy(POLICY_PATH)
    frame = build_attention_frame(field=_field_with_transport_drift(), policy=policy, now=NOW)
    assert frame.capability_targets == []
    assert not any(t.target_id == "capability:transport" for t in frame.dominant_targets)
