"""Tests for coalition dwell and hysteresis (rung-3 focus stability)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import orion.substrate.attention_broadcast as attention_broadcast
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1
from orion.substrate.attention_broadcast import (
    broadcast_projection_from_frame,
    build_substrate_attention_frame,
)

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_broadcast_globals():
    attention_broadcast._coalition_history.clear()
    attention_broadcast._current_active_coalition = None
    attention_broadcast._dwell_ticks = 0
    attention_broadcast._current_dwelling_loop_id = None
    attention_broadcast._transition_history.clear()
    attention_broadcast._recent_selected_counts.clear()
    attention_broadcast._first_selected_at.clear()
    yield
    attention_broadcast._coalition_history.clear()
    attention_broadcast._current_active_coalition = None
    attention_broadcast._dwell_ticks = 0
    attention_broadcast._current_dwelling_loop_id = None
    attention_broadcast._transition_history.clear()
    attention_broadcast._recent_selected_counts.clear()
    attention_broadcast._first_selected_at.clear()


def _node(node_id: str, label: str, pressure: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata={"dynamic_pressure": pressure},
        signals=SimpleNamespace(confidence=0.8),
    )


def _tick(nodes: list) -> AttentionBroadcastProjectionV1:
    frame = build_substrate_attention_frame(nodes=nodes, now=_NOW)
    return broadcast_projection_from_frame(frame)


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


def test_coalition_history_records_activation():
    """Two ticks of the same winning coalition record one 'activated' event."""
    nodes = [_node("node:hot", "unresolved contradiction")]
    first = _tick(nodes)
    assert first.coalition_history == []
    second = _tick(nodes)
    assert second.coalition_history == [
        {"at": _NOW.isoformat(), "event": "activated", "size": 1}
    ]


def test_coalition_history_records_decay():
    """Active coalition leaving the 3-tick window records a 'decayed' event."""
    # Activate coalition {node:a} (2 ticks of the same winner).
    _tick([_node("node:a", "focus a")])
    _tick([_node("node:a", "focus a")])
    # Three distinct single-node coalitions push {node:a} out of the window
    # without any of them activating themselves.
    _tick([_node("node:b", "focus b")])
    _tick([_node("node:c", "focus c")])
    projection = _tick([_node("node:d", "focus d")])
    events = [entry["event"] for entry in projection.coalition_history]
    assert events == ["activated", "decayed"]
    decay = projection.coalition_history[-1]
    assert decay == {"at": _NOW.isoformat(), "event": "decayed", "size": 1}


def test_coalition_history_capped_at_ten():
    """Transition log never exceeds 10 entries (deque + schema cap agree)."""
    for i in range(12):
        # Each pair of identical ticks activates a fresh coalition.
        nodes = [_node(f"node:{i}", f"focus {i}")]
        _tick(nodes)
        projection = _tick(nodes)
    assert len(projection.coalition_history) == 10
    assert all(entry["event"] == "activated" for entry in projection.coalition_history)
    # Oldest activations were evicted: the surviving log covers the last 10.
    assert projection.coalition_history[-1]["size"] == 1
    # Projection validates against the schema cap (max_length=10).
    assert AttentionBroadcastProjectionV1.model_validate(
        projection.model_dump(mode="json")
    ).coalition_history == projection.coalition_history
