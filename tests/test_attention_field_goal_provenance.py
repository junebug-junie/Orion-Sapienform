"""orion/attention/field_attention/goal_provenance.py -- pure logic, no DB/bus."""
from __future__ import annotations

from orion.attention.field_attention.goal_provenance import (
    DominanceStreak,
    top_node_substrate_target,
    update_dominance_streak,
)
from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1


def _target(target_id: str, salience: float, kind: str = "node") -> FieldAttentionTargetV1:
    return FieldAttentionTargetV1(
        target_id=target_id,
        target_kind=kind,
        salience_score=salience,
        pressure_score=salience,
        novelty_score=salience,
        urgency_score=salience,
        confidence_score=1.0,
    )


def _frame(node_targets: list[FieldAttentionTargetV1]) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="frame-1",
        generated_at="2026-07-30T00:00:00Z",
        source_field_tick_id="tick-1",
        source_field_generated_at="2026-07-30T00:00:00Z",
        overall_salience=max((t.salience_score for t in node_targets), default=0.0),
        dominant_targets=node_targets,
        node_targets=node_targets,
    )


def test_top_node_substrate_target_picks_highest_qualifying():
    real_domain = next(iter(PREDICTION_ERROR_NATIVE_TARGETS.keys()))
    frame = _frame([_target(real_domain, 0.4), _target("node:athena", 0.9)])
    winner = top_node_substrate_target(frame)
    assert winner is not None
    assert winner.target_id == real_domain


def test_top_node_substrate_target_excludes_host_targets():
    # Candidate B host targets (e.g. node:athena) are real but out of scope for this
    # producer -- confirms the filter genuinely excludes them, not just deprioritizes.
    frame = _frame([_target("node:athena", 0.95), _target("node:atlas", 0.90)])
    assert top_node_substrate_target(frame) is None


def test_top_node_substrate_target_empty_frame_returns_none():
    assert top_node_substrate_target(_frame([])) is None


def test_dominance_streak_resets_on_target_change():
    streak = DominanceStreak(target_id="a", count=5)
    new_streak, should_emit = update_dominance_streak(streak, "b", min_streak=3)
    assert new_streak.target_id == "b"
    assert new_streak.count == 1
    assert should_emit is False


def test_dominance_streak_resets_on_none_target():
    streak = DominanceStreak(target_id="a", count=5)
    new_streak, should_emit = update_dominance_streak(streak, None, min_streak=3)
    assert new_streak.target_id is None
    assert new_streak.count == 0
    assert should_emit is False


def test_dominance_streak_does_not_emit_before_min_streak():
    streak = DominanceStreak()
    for expected_count in (1, 2):
        streak, should_emit = update_dominance_streak(streak, "a", min_streak=3)
        assert streak.count == expected_count
        assert should_emit is False


def test_dominance_streak_emits_at_min_streak_and_keeps_emitting():
    streak = DominanceStreak()
    for _ in range(2):
        streak, _ = update_dominance_streak(streak, "a", min_streak=3)
    streak, should_emit = update_dominance_streak(streak, "a", min_streak=3)
    assert streak.count == 3
    assert should_emit is True
    # Real, still-dominant target keeps refreshing on every subsequent qualifying tick.
    streak, should_emit = update_dominance_streak(streak, "a", min_streak=3)
    assert streak.count == 4
    assert should_emit is True


def test_dominance_streak_flip_flop_never_emits():
    streak = DominanceStreak()
    for target_id in ("a", "b", "a", "b", "a"):
        streak, should_emit = update_dominance_streak(streak, target_id, min_streak=3)
        assert should_emit is False
