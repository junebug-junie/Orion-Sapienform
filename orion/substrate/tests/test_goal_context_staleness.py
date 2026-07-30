"""GoalContextStore staleness dead-man's-switch (Part B of
docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-design.md).

Deliberately narrow: this is a read-time hard timeout, not continuous decay -- see that
design doc for why a leaky-integrator decay was rejected here. Coverage for the
existing latest-wins / terminal-status-clears contract lives in
tests/test_voluntary_attention_wiring.py; this file only covers the new age-based path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.substrate.attention.goal_context import GoalContextStore
from orion.substrate.attention.top_down import GoalContext


def test_current_returns_fresh_goal_unmodified() -> None:
    store = GoalContextStore()
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1")
    assert store.current() is not None
    assert store.current().goal_artifact_id == "g1"


def test_current_clears_goal_past_max_age(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_GOAL_CONTEXT_MAX_AGE_SEC", "60")
    store = GoalContextStore()
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1", received_at=stale_ts)
    assert store.current() is None
    # Clearing is sticky -- a second read doesn't resurrect it.
    assert store.current() is None


def test_current_keeps_goal_just_under_max_age(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_GOAL_CONTEXT_MAX_AGE_SEC", "60")
    store = GoalContextStore()
    fresh_ts = datetime.now(timezone.utc) - timedelta(seconds=30)
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1", received_at=fresh_ts)
    assert store.current() is not None


def test_default_max_age_is_conservative_four_hours(monkeypatch) -> None:
    monkeypatch.delenv("ORION_ATTENTION_GOAL_CONTEXT_MAX_AGE_SEC", raising=False)
    store = GoalContextStore()
    almost_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=3, minutes=59)
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1", received_at=almost_four_hours_ago)
    assert store.current() is not None
    over_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=4, minutes=1)
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1", received_at=over_four_hours_ago)
    assert store.current() is None


def test_malformed_max_age_env_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_GOAL_CONTEXT_MAX_AGE_SEC", "not-a-number")
    store = GoalContextStore()
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=1)
    store._current = GoalContext(priority=0.7, goal_artifact_id="g1", received_at=fresh_ts)
    assert store.current() is not None
