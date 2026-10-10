"""World-first: a reverie about a WORLD winner (Juniper talking, a camera
surprise) never becomes a self_state policy-review proposal."""
from __future__ import annotations

from orion.reverie.proposal import spontaneous_thought_to_candidate
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

GROUNDED = "The loop ol-1 keeps winning and has not discharged; conflict at its node recurs."


def _thought(attended: list[str]) -> SpontaneousThoughtV1:
    return SpontaneousThoughtV1(
        thought_id="th-w", correlation_id="c-w",
        coalition=CoalitionSnapshotV1(
            attended_node_ids=attended, selected_open_loop_id="ol-1", open_loop_ids=["ol-1"],
            generated_at="2026-10-10T00:00:00Z",
        ),
        interpretation=GROUNDED, salience=0.9, evidence_refs=["ol-1"],
    )


def test_world_only_coalition_never_proposes() -> None:
    assert spontaneous_thought_to_candidate(_thought(["world:chat"]), fallback_target_id="ss") is None
    assert spontaneous_thought_to_candidate(
        _thought(["node:substrate.perception"]), fallback_target_id="ss"
    ) is None


def test_body_coalition_still_proposes() -> None:
    c = spontaneous_thought_to_candidate(_thought(["node:substrate.execution"]), fallback_target_id="ss")
    assert c is not None and c.required_policy_gate == "operator_review"
