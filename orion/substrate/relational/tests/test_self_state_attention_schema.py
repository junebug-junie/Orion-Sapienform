"""Tests for attention schema self-state dimension."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.self_state import SelfStateV1

_NOW = datetime.now(timezone.utc)

def _base_state(**kwargs):
    """Factory for SelfStateV1 with required fields."""
    defaults = {
        "self_state_id": "s1",
        "generated_at": _NOW,
        "source_field_tick_id": "t1",
        "source_field_generated_at": _NOW,
        "source_attention_frame_id": "f1",
        "source_attention_generated_at": _NOW,
        "overall_intensity": 0.5,
        "overall_confidence": 0.5,
    }
    defaults.update(kwargs)
    return SelfStateV1(**defaults)


def test_attention_schema_type_field():
    """SelfStateV1 has attention_schema_type field."""
    state = _base_state(attention_schema_type="focused_single")
    assert state.attention_schema_type == "focused_single"


def test_attention_schema_dwell_ticks_field():
    """SelfStateV1 has attention_dwell_ticks field."""
    state = _base_state(attention_dwell_ticks=5)
    assert state.attention_dwell_ticks == 5


def test_attention_schema_node_count_field():
    """SelfStateV1 has attention_node_count field."""
    state = _base_state(attention_node_count=3)
    assert state.attention_node_count == 3


def test_attention_schema_defaults():
    """Attention schema fields default to None/0."""
    state = _base_state()
    assert state.attention_schema_type is None
    assert state.attention_dwell_ticks == 0
    assert state.attention_node_count == 0


def test_attention_schema_all_types_valid():
    """All attention schema type values are valid."""
    types = ["focused_single", "distributed", "open_loop", "none", "unknown"]
    for schema_type in types:
        state = _base_state(attention_schema_type=schema_type)
        assert state.attention_schema_type == schema_type
