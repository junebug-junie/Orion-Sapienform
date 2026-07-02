"""Tests for hub presence self-state dimension."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.self_state import SelfStateV1


def _base_state(**kwargs):
    """Factory for SelfStateV1 with required fields."""
    now = datetime.now(timezone.utc)
    defaults = {
        "self_state_id": "s1",
        "generated_at": now,
        "source_field_tick_id": "t1",
        "source_field_generated_at": now,
        "source_attention_frame_id": "f1",
        "source_attention_generated_at": now,
        "overall_intensity": 0.5,
        "overall_confidence": 0.5,
    }
    defaults.update(kwargs)
    return SelfStateV1(**defaults)


def test_hub_presence_field_exists():
    """SelfStateV1 can accept hub_presence dict with heartbeat metadata."""
    metadata = {
        "last_turn_age_sec": 5,
        "turns_per_minute": 2.0,
        "connection_health": "fresh",
        "queue_depth": 0,
    }
    state = _base_state(hub_presence=metadata)
    assert state.hub_presence == metadata
    assert state.hub_presence["last_turn_age_sec"] == 5
    assert state.hub_presence["connection_health"] == "fresh"


def test_hub_presence_defaults_none():
    """hub_presence defaults to None when not provided."""
    state = _base_state()
    assert state.hub_presence is None


def test_hub_presence_accepts_dict():
    """hub_presence stores complex metadata dicts."""
    metadata = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    state = _base_state(hub_presence=metadata)
    assert state.hub_presence == metadata
    assert isinstance(state.hub_presence, dict)
