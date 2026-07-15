"""Regression tests for open-loop recency actually decaying over real time.

Found live 2026-07-14: a loop verdicted "resolved" on 2026-07-08 and
"dismissed" on 2026-07-10 was still winning attention-broadcast selection on
2026-07-14, with salience_features byte-identical to the 2026-07-10 snapshot
-- including recency=1.0, four days after the loop was last touched.
Root cause: SalienceHistory.first_seen_at (consumed by
orion.substrate.attention.salience._recency()'s half-life decay) was never
populated by attention_broadcast.py's _current_history(), so _recency()
always hit its "never seen before -> maximally fresh" branch and returned
1.0 unconditionally, for every loop, forever.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.substrate.attention_broadcast as attention_broadcast
from orion.substrate.attention_broadcast import (
    broadcast_projection_from_frame,
    build_substrate_attention_frame,
)

_NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_broadcast_globals(monkeypatch: pytest.MonkeyPatch):
    # Matches the live deployment (.env: ORION_ATTENTION_HABITUATION_ENABLED=true)
    # -- _current_history() (and therefore first_seen_at/_recency()) is only
    # ever consulted when this flag is on; build_substrate_attention_frame
    # passes history=None otherwise, bypassing the fix entirely. Without this,
    # these tests would pass even with the original bug reintroduced.
    monkeypatch.setenv("ORION_ATTENTION_HABITUATION_ENABLED", "true")
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


def _tick(nodes: list, *, now: datetime):
    frame = build_substrate_attention_frame(nodes=nodes, now=now)
    return broadcast_projection_from_frame(frame)


def _winning_loop_recency(projection) -> float:
    loop_id = projection.selected_open_loop_id
    assert loop_id is not None, "expected a winning loop to check recency on"
    loop = next(loop for loop in projection.frame.open_loops if loop.id == loop_id)
    assert loop.salience_features is not None
    features = loop.salience_features
    return features["recency"] if isinstance(features, dict) else features.recency


def test_first_selection_is_maximally_fresh():
    """Unchanged behavior: a loop selected for the first time is fresh (1.0)."""
    nodes = [_node("node:hot", "unresolved contradiction")]
    projection = _tick(nodes, now=_NOW)
    assert _winning_loop_recency(projection) == pytest.approx(1.0)


def test_recency_actually_decays_after_real_time_passes():
    """The bug: recency stayed 1.0 forever because first_seen_at was never
    recorded. First tick establishes first_seen_at; a much later tick on the
    SAME loop must show decayed recency, not another 1.0."""
    nodes = [_node("node:hot", "unresolved contradiction")]
    first = _tick(nodes, now=_NOW)
    assert _winning_loop_recency(first) == pytest.approx(1.0)

    # 6 hours later == one half-life (see salience._recency's docstring).
    later = _tick(nodes, now=_NOW + timedelta(hours=6))
    recency = _winning_loop_recency(later)
    assert recency < 1.0, (
        f"recency did not decay after 6 real hours -- got {recency}, "
        "first_seen_at wiring regressed"
    )
    assert recency == pytest.approx(0.5, abs=0.01)


def test_recency_matches_live_incident_scale():
    """The exact live-incident shape: a loop still winning ~4-6 real days
    after it was first seen must show recency near zero, not 1.0."""
    nodes = [_node("node:hot", "unresolved contradiction")]
    _tick(nodes, now=_NOW)
    much_later = _tick(nodes, now=_NOW + timedelta(days=5))
    recency = _winning_loop_recency(much_later)
    assert recency < 0.01, f"expected near-zero recency after 5 days, got {recency}"


def test_first_selected_at_evicted_alongside_recent_selected_counts():
    """_first_selected_at must not grow unboundedly independent of the
    existing _recent_selected_counts cap -- eviction stays paired."""
    for i in range(attention_broadcast._MAX_TRACKED_THEMES + 5):
        nodes = [_node(f"node:{i}", f"focus {i}", pressure=0.9)]
        _tick(nodes, now=_NOW + timedelta(seconds=i))
    assert len(attention_broadcast._first_selected_at) <= attention_broadcast._MAX_TRACKED_THEMES
    assert set(attention_broadcast._first_selected_at) <= set(
        attention_broadcast._recent_selected_counts
    )
