"""Tests for hub presence passthrough in the self-state build path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateV1
from orion.self_state.builder import build_self_state, normalize_hub_presence
from orion.self_state.policy import load_self_state_policy

REPO = Path(__file__).resolve().parents[4]
ATTENTION_POLICY = load_attention_policy(
    REPO / "config" / "attention" / "field_attention_policy.v1.yaml"
)
SELF_POLICY = load_self_state_policy(
    REPO / "config" / "self_state" / "self_state_policy.v1.yaml"
)
NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_hub_presence",
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
        recent_perturbations=["state_delta:exec_1"],
    )


def _build_state(**kwargs) -> SelfStateV1:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    return build_self_state(
        field=field,
        attention=attention,
        policy=SELF_POLICY,
        now=NOW,
        **kwargs,
    )


# --- normalize_hub_presence ---


def test_normalize_passes_non_empty_dict_through():
    presence = {"connection_health": "fresh", "last_turn_age_sec": 5}
    assert normalize_hub_presence(presence) == presence


def test_normalize_empty_dict_becomes_none():
    assert normalize_hub_presence({}) is None


def test_normalize_none_stays_none():
    assert normalize_hub_presence(None) is None


def test_normalize_non_dict_becomes_none():
    assert normalize_hub_presence(["not", "a", "dict"]) is None  # type: ignore[arg-type]


# --- build_self_state passthrough ---


def test_build_self_state_passes_hub_presence_through():
    presence = {
        "last_turn_age_sec": 5,
        "turns_per_minute": 2.0,
        "connection_health": "fresh",
        "as_of": NOW.isoformat(),
    }
    state = _build_state(hub_presence=presence)
    assert state.hub_presence == presence
    assert state.hub_presence["connection_health"] == "fresh"


def test_build_self_state_absent_hub_presence_is_none():
    state = _build_state()
    assert state.hub_presence is None


def test_build_self_state_empty_hub_presence_is_none():
    state = _build_state(hub_presence={})
    assert state.hub_presence is None
