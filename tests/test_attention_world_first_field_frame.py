"""World-first field attention frame + its blast radius (goal provenance,
self-modification proposals). Spec 2026-10-07 self-calibration, section A."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.goal_provenance import (
    DominanceStreak,
    qualified_node_targets,
    top_node_substrate_target,
    update_dominance_streak,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import field_target_source_kind
from orion.attention.world_first import (
    PERCEPTION_NODE_ID,
    WORLD_CHAT_SOURCE_ID,
    chat_candidate,
    node_candidate,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _field(tick: str = "tick_wf") -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id=tick,
        node_vectors={
            "node:athena": {"cortex_exec_step_load": 0.9},
            "node:substrate.execution": {"prediction_error": 0.6},
            "node:substrate.vision_organ": {"vision_frame_staleness": 0.0},
        },
        capability_vectors={"capability:orchestration": {"execution_pressure": 0.8}},
    )


def _mag(pct: float, value: float = 0.3) -> PredictionErrorMagnitudeV1:
    band = "quiet" if pct < 0.5 else "usual" if pct < 0.9 else "high" if pct < 0.99 else "unusual"
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=20.0, percentile_now=pct, n_readings_7d=5000, band=band, trend="flat"
    )


def _node(node_id: str, pct: float, value: float = 0.3):
    return node_candidate(node_id=node_id, label=node_id, magnitude=_mag(pct, value), observed_at=NOW, now=NOW)


def _frame(cands, previous=None, tick="tick_wf") -> FieldAttentionFrameV1:
    return build_attention_frame(
        field=_field(tick), policy=POLICY, prediction_error_baselines={},
        previous_frame=previous, now=NOW, world_first_candidates=cands,
    )


def test_calm_tick_is_an_explicit_no_winner_frame() -> None:
    frame = _frame([_node("node:substrate.biometrics", 0.4, 0.02), _node("node:substrate.execution", 0.7)])
    assert frame.dominant_targets == [] and frame.node_targets == []
    assert frame.overall_salience == 0.0
    assert frame.warnings == ["world_first_no_winner"]
    # Every candidate is still traced, with its side of the seam.
    ids = {t.target_id for t in frame.suppressed_targets}
    assert {"node:substrate.biometrics", "node:substrate.execution"} <= ids
    assert all(field_target_source_kind(t) == "internal" for t in frame.suppressed_targets)


def test_old_hardcoded_winner_at_one_is_gone() -> None:
    """The bug: min-max crowned a calm node at 1.0 on every frame."""
    frame = _frame([_node("node:substrate.bus_synaptic", 0.3, 0.03)])
    assert not frame.dominant_targets
    assert all(t.salience_score < 1.0 for t in frame.suppressed_targets)


def test_body_alarm_wins_with_its_own_percentile_as_salience() -> None:
    frame = _frame([_node("node:substrate.execution", 0.995, 0.9), _node("node:substrate.biometrics", 0.6)])
    top = frame.dominant_targets[0]
    assert top.target_id == "node:substrate.execution"
    assert top.salience_score == 0.995
    assert field_target_source_kind(top) == "internal"
    assert frame.node_targets == [top]


def test_busy_chat_wins_over_a_usual_body() -> None:
    turns = [NOW.replace(day=d) for d in range(3, 9)] + [NOW.replace(minute=0) for _ in range(1)]
    turns = [t.replace(hour=11, minute=50) for t in turns]
    chat = chat_candidate(turns, now=NOW)
    frame = _frame([chat, _node("node:substrate.biometrics", 0.85)])
    top = frame.dominant_targets[0]
    assert top.target_id == WORLD_CHAT_SOURCE_ID and top.target_kind == "channel"
    assert field_target_source_kind(top) == "external"
    assert frame.node_targets == []


def test_host_and_capability_novelty_is_observed_not_attended() -> None:
    first = _frame([_node("node:substrate.execution", 0.2)], tick="t1")
    second = _frame([_node("node:substrate.execution", 0.2)], previous=first, tick="t2")
    assert second.capability_targets == [] and second.system_targets == []
    sup = {t.target_id: t for t in second.suppressed_targets}
    assert "node:athena" in sup and "capability:orchestration" in sup
    assert field_target_source_kind(sup["node:athena"]) == "internal"
    # Not double-counted: a candidate node is not ALSO scored as a host target.
    assert [t.target_id for t in second.suppressed_targets].count("node:substrate.execution") == 1


def test_goal_provenance_names_an_attended_world_winner() -> None:
    """Decision 1 (2026-11-08): a world winner may emit a goal. It outranks a
    body node here on its own-week percentile, so it names the goal."""
    frame = _frame(
        [_node(PERCEPTION_NODE_ID, 0.999, 0.9), _node("node:substrate.execution", 0.95)]
    )
    assert frame.dominant_targets[0].target_id == PERCEPTION_NODE_ID
    q = qualified_node_targets(frame)
    assert {t.target_id for t in q} == {PERCEPTION_NODE_ID, "node:substrate.execution"}
    assert top_node_substrate_target(frame).target_id == PERCEPTION_NODE_ID


def test_busy_chat_names_the_goal() -> None:
    """Chat is a ``channel`` target, never in node_targets; it must still qualify."""
    turns = [NOW.replace(day=d) for d in range(3, 9)] + [NOW.replace(minute=0) for _ in range(1)]
    turns = [t.replace(hour=11, minute=50) for t in turns]
    frame = _frame([chat_candidate(turns, now=NOW), _node("node:substrate.biometrics", 0.85)])
    assert frame.node_targets == []
    assert top_node_substrate_target(frame).target_id == WORLD_CHAT_SOURCE_ID


def _chat_and_body_frame():
    turns = [NOW.replace(day=d) for d in range(3, 9)] + [NOW.replace(minute=0) for _ in range(1)]
    turns = [t.replace(hour=11, minute=50) for t in turns]
    return _frame([chat_candidate(turns, now=NOW), _node("node:substrate.execution", 0.995, 0.9)])


def test_competition_precedence_keeps_a_world_goal_it_holds() -> None:
    """Review finding: the competition read must carry world:* refs, or a body
    node in the competition beats an attended chat on every tick."""
    frame = _chat_and_body_frame()
    assert {t.target_id for t in qualified_node_targets(frame)} == {
        WORLD_CHAT_SOURCE_ID, "node:substrate.execution"
    }
    both = {WORLD_CHAT_SOURCE_ID, "node:substrate.execution"}
    assert top_node_substrate_target(frame, competing=both).target_id in both
    assert top_node_substrate_target(
        frame, competing={WORLD_CHAT_SOURCE_ID}
    ).target_id == WORLD_CHAT_SOURCE_ID


def test_world_streak_is_held_when_competition_read_is_unknown() -> None:
    frame = _chat_and_body_frame()
    held = top_node_substrate_target(frame, competing=None, current=WORLD_CHAT_SOURCE_ID)
    assert held.target_id == WORLD_CHAT_SOURCE_ID


def test_goal_provenance_handles_a_no_winner_frame() -> None:
    frame = _frame([_node("node:substrate.execution", 0.2)])
    assert qualified_node_targets(frame) == []
    assert top_node_substrate_target(frame) is None
    streak, emit = update_dominance_streak(DominanceStreak("node:substrate.execution", 5), None, min_streak=3)
    assert streak.target_id is None and streak.count == 0 and not emit


def test_world_only_frame_yields_a_world_goal() -> None:
    frame = _frame([_node(PERCEPTION_NODE_ID, 0.97)])
    assert [t.target_id for t in qualified_node_targets(frame)] == [PERCEPTION_NODE_ID]


def test_unattended_world_source_yields_no_goal() -> None:
    """Only ATTENDED world targets qualify: a quiet camera stays observed."""
    frame = _frame([_node(PERCEPTION_NODE_ID, 0.2)])
    assert not frame.dominant_targets and qualified_node_targets(frame) == []


# The proposal-binding tests (orion.proposals.builder) retired with the proposal rung, 2026-11-07.


def test_goal_set_is_not_widened_beyond_the_native_five() -> None:
    """Cabinet is a body node that can win world-first attention, but a goal
    on it would bias the broadcast toward a node a real self-reversible
    action binds to. Widening the goal set is a separate decision."""
    frame = _frame([_node("node:substrate.cabinet", 0.99), _node("node:substrate.codebase", 0.98)])
    assert {t.target_id for t in frame.node_targets} == {"node:substrate.cabinet", "node:substrate.codebase"}
    assert qualified_node_targets(frame) == []


def test_goal_refuses_a_native_id_marked_external() -> None:
    """Defense in depth: the internal marker is checked, not just the id list."""
    t = FieldAttentionTargetV1(
        target_id="node:substrate.execution", target_kind="node", salience_score=0.99,
        pressure_score=0.9, novelty_score=0.0, urgency_score=0.99, confidence_score=1.0,
        evidence_refs=["source_kind:external"],
    )
    frame = FieldAttentionFrameV1(
        frame_id="f", generated_at=NOW, source_field_tick_id="t", source_field_generated_at=NOW,
        overall_salience=0.99, dominant_targets=[t], node_targets=[t],
    )
    assert qualified_node_targets(frame) == []
