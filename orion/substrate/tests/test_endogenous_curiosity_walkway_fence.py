"""node:substrate.perception_walkway is recorded but not yet a curiosity seed
(2026-10-15): endogenous curiosity reads ANY node carrying prediction_error,
so the fence in orion/schemas/attention_candidate.py is what keeps it out."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from orion.schemas.attention_candidate import PERCEPTION_NODE_ID, PERCEPTION_WALKWAY_NODE_ID
from orion.substrate.endogenous_curiosity import _prediction_error_candidates

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def _node(node_id: str, pe: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": pe},
        temporal=SimpleNamespace(observed_at=NOW),
    )


def test_walkway_node_never_seeds_curiosity_but_room_node_still_does() -> None:
    candidates = _prediction_error_candidates(
        [_node(PERCEPTION_WALKWAY_NODE_ID, 1.0), _node(PERCEPTION_NODE_ID, 0.9)],
        anchor_scope="orion",
        subject_ref="entity:orion",
        min_error=0.1,
        now=NOW,
    )
    refs = [c.focal_node_refs[0] for c in candidates]
    assert refs == [PERCEPTION_NODE_ID]
