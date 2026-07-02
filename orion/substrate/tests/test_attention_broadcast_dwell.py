"""Tests for coalition dwell and hysteresis (rung-3 focus stability)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1


def test_dwell_ticks_field_exists():
    """AttentionBroadcastProjectionV1 has dwell_ticks field."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(frame=frame, dwell_ticks=5)
    assert proj.dwell_ticks == 5


def test_coalition_stability_score_field_exists():
    """AttentionBroadcastProjectionV1 has coalition_stability_score field."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(frame=frame, coalition_stability_score=0.8)
    assert proj.coalition_stability_score == 0.8


def test_coalition_history_field_exists():
    """AttentionBroadcastProjectionV1 has coalition_history field."""
    frame = AttentionFrameV1()
    history = [{"transition": "from_a", "at_tick": 0}, {"transition": "to_b", "at_tick": 1}]
    proj = AttentionBroadcastProjectionV1(frame=frame, coalition_history=history)
    assert proj.coalition_history == history


def test_dwell_ticks_default_zero():
    """dwell_ticks defaults to 0."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(frame=frame)
    assert proj.dwell_ticks == 0


def test_coalition_stability_score_default_one():
    """coalition_stability_score defaults to 1.0."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(frame=frame)
    assert proj.coalition_stability_score == 1.0


def test_coalition_history_default_empty():
    """coalition_history defaults to empty list."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(frame=frame)
    assert proj.coalition_history == []


def test_dwell_ticks_clamped_non_negative():
    """dwell_ticks can be any non-negative int."""
    frame = AttentionFrameV1()
    for ticks in [0, 1, 10, 100]:
        proj = AttentionBroadcastProjectionV1(frame=frame, dwell_ticks=ticks)
        assert proj.dwell_ticks == ticks


def test_coalition_stability_score_bounded_0_1():
    """coalition_stability_score is clamped to [0, 1]."""
    frame = AttentionFrameV1()
    for score in [0.0, 0.5, 1.0]:
        proj = AttentionBroadcastProjectionV1(frame=frame, coalition_stability_score=score)
        assert proj.coalition_stability_score == score

    # Test validation bounds
    with pytest.raises(ValueError):
        AttentionBroadcastProjectionV1(frame=frame, coalition_stability_score=-0.1)
    with pytest.raises(ValueError):
        AttentionBroadcastProjectionV1(frame=frame, coalition_stability_score=1.1)


def test_projection_serialization_includes_dwell_fields():
    """Projection serializes with new dwell fields."""
    frame = AttentionFrameV1()
    proj = AttentionBroadcastProjectionV1(
        frame=frame,
        attended_node_ids=["node:a"],
        dwell_ticks=3,
        coalition_stability_score=0.7,
        coalition_history=[{"tick": 1}],
    )
    data = proj.model_dump()
    assert data["dwell_ticks"] == 3
    assert data["coalition_stability_score"] == 0.7
    assert data["coalition_history"] == [{"tick": 1}]
