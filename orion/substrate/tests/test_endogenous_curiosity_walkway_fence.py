"""Per-consumer fence for node:substrate.perception_walkway (2026-10-15).

Curiosity reads the walkway camera's surprise node like the room camera's;
attention still does not. Endogenous curiosity reads ANY node carrying
prediction_error, so the fences in orion/schemas/attention_candidate.py are
what decide, per consumer."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from orion.schemas.attention_candidate import (
    NO_ATTENTION_CONSUMER_NODE_IDS,
    NO_CURIOSITY_CONSUMER_NODE_IDS,
    PERCEPTION_NODE_ID,
    PERCEPTION_WALKWAY_NODE_ID,
)
from orion.substrate.endogenous_curiosity import _prediction_error_candidates

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def _node(node_id: str, pe: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": pe},
        temporal=SimpleNamespace(observed_at=NOW),
    )


def test_walkway_node_seeds_curiosity_alongside_the_room_node() -> None:
    candidates = _prediction_error_candidates(
        [_node(PERCEPTION_WALKWAY_NODE_ID, 1.0), _node(PERCEPTION_NODE_ID, 0.9)],
        anchor_scope="orion",
        subject_ref="entity:orion",
        min_error=0.55,
        now=NOW,
    )
    assert [c.focal_node_refs[0] for c in candidates] == [PERCEPTION_WALKWAY_NODE_ID, PERCEPTION_NODE_ID]


def test_walkway_node_below_min_error_does_not_seed() -> None:
    # Live 2026-10-15 night sample: p99 0.508 < min_error 0.55.
    assert _prediction_error_candidates(
        [_node(PERCEPTION_WALKWAY_NODE_ID, 0.508)],
        anchor_scope="orion", subject_ref="entity:orion", min_error=0.55, now=NOW,
    ) == []


def test_fences_are_per_consumer() -> None:
    assert PERCEPTION_WALKWAY_NODE_ID not in NO_CURIOSITY_CONSUMER_NODE_IDS
    assert PERCEPTION_WALKWAY_NODE_ID in NO_ATTENTION_CONSUMER_NODE_IDS
