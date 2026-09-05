"""The goal's subject must survive the trip into attention.

`GoalContextStore.update_from_goal` used to keep only `priority` and
`goal_artifact_id`, silently dropping `field_target_id`. That drop was the
upstream half of the "voluntary override can never fire" defect: by the time
`relevance(goal, loop)` ran, the goal's subject was already gone, so relevance
fell back to a per-loop constant and every candidate got identical bias.

These tests fail against that version.
"""
from __future__ import annotations

from orion.core.schemas.drives import ArtifactProvenance
from orion.schemas.attention_frame import OpenLoopV1
from orion.schemas.field_goal import FieldGoalProvenanceV1
from orion.substrate.attention.goal_context import GoalContextStore
from orion.substrate.attention.top_down import relevance


def _goal(field_target_id: str = "node:substrate.biometrics", priority: float = 0.7):
    return FieldGoalProvenanceV1(
        artifact_id="goal-1",
        subject="attention",
        model_layer="field_attention",
        entity_id=field_target_id,
        kind="memory.field_goals.proposed.v1",
        correlation_id="c-1",
        provenance=ArtifactProvenance(intake_channel="internal.attention_runtime"),
        field_target_id=field_target_id,
        target_kind="node",
        salience_score=priority,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=priority,
        proposal_status="active",
    )


def _loop(loop_id: str, source: str | None) -> OpenLoopV1:
    return OpenLoopV1(
        id=loop_id,
        description="d",
        source_refs=[source] if source else [],
    )


def test_store_carries_the_goal_target_through():
    store = GoalContextStore()
    store.update_from_goal(_goal("node:substrate.execution"))
    held = store.current()
    assert held is not None
    assert held.target_id == "node:substrate.execution"
    assert held.priority == 0.7


def test_carried_target_actually_discriminates_between_loops():
    """End-to-end: a real goal object in, a real relevance verdict out.

    This is the assertion that would have caught the original defect. Under the
    old code both loops returned the same relevance (the fabricated 0.55
    concept_value floor), so no goal could distinguish them.
    """
    store = GoalContextStore()
    store.update_from_goal(_goal("node:substrate.execution"))
    goal = store.current()
    assert goal is not None

    on_target = _loop("a", "node:substrate.execution")
    off_target = _loop("b", "node:substrate.chat")

    assert relevance(goal, on_target) == 1.0
    assert relevance(goal, off_target) == 0.0
    # The discriminating part -- not merely "both are in [0,1]".
    assert relevance(goal, on_target) != relevance(goal, off_target)


def test_two_different_goals_disagree_about_the_same_loop():
    """Relevance must be a function of the goal, not only of the loop.

    The old body took `goal` and never read it, so this could not hold no
    matter what the loops looked like.
    """
    loop = _loop("a", "node:substrate.execution")

    store_a = GoalContextStore()
    store_a.update_from_goal(_goal("node:substrate.execution"))
    store_b = GoalContextStore()
    store_b.update_from_goal(_goal("node:substrate.chat"))

    goal_a, goal_b = store_a.current(), store_b.current()
    assert goal_a is not None and goal_b is not None
    assert relevance(goal_a, loop) == 1.0
    assert relevance(goal_b, loop) == 0.0


def test_cleared_goal_pushes_on_nothing():
    """No held goal -> no bias anywhere. Deliberate: absence must not fabricate."""
    store = GoalContextStore()
    assert store.current() is None


def test_relevance_joins_a_real_substrate_node():
    """Pin the join through the REAL producer chain, not hand-built loops.

    Every other test in this area constructs `source_refs` by hand, so all of
    them would keep passing if the producer stopped putting the node id there.
    That is the exact silent-constant failure this whole patch exists to undo:
    if `evidence_refs` is ever reordered, prefixed, normalized, or reduced to
    turn ids, `relevance()` returns 0.0 for every loop, override becomes
    impossible again, and the only outward symptom is `top_down_bias_max`
    quietly going to 0.0.

    So this walks the actual path:
        substrate node -> substrate_pressure_signals -> build_open_loops
                       -> relevance(goal, loop)
    """
    from types import SimpleNamespace

    from orion.substrate.attention.scoring import build_open_loops, merge_signals
    from orion.substrate.attention_broadcast import substrate_pressure_signals

    def _node(node_id: str, label: str, pressure: float):
        return SimpleNamespace(
            node_id=node_id,
            label=label,
            metadata={"dynamic_pressure": pressure, "prediction_error": pressure},
            signals=None,
        )

    wanted = "node:substrate.execution"
    nodes = [
        _node(wanted, "Execution prediction error", 0.30),
        _node("node:substrate.chat", "Chat prediction error", 0.90),
    ]

    signals = substrate_pressure_signals(nodes, min_salience=0.05, limit=24)
    loops = build_open_loops(
        signals=merge_signals(signals, limit=15),
        ctx={}, inputs={}, belief_lineage=[],
        direct_turn=False, generic_reversal=False, stale_thread_active=False,
        max_open=5,
    )
    assert len(loops) == 2, "both nodes should clear the threshold"

    # The producer must have carried the node id through to source_refs.
    by_ref = {ref: loop for loop in loops for ref in loop.source_refs}
    assert wanted in by_ref, (
        "substrate_pressure_signals -> build_open_loops dropped the node id "
        f"from source_refs; got {[l.source_refs for l in loops]}"
    )

    store = GoalContextStore()
    store.update_from_goal(_goal(wanted))
    goal = store.current()
    assert goal is not None

    on_target = by_ref[wanted]
    off_target = next(l for l in loops if wanted not in l.source_refs)
    assert relevance(goal, on_target) == 1.0
    assert relevance(goal, off_target) == 0.0
