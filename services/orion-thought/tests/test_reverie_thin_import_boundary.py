"""Import-boundary tests: orion-thought must not load orion.substrate."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.reverie.referent_loader import TurnReferentRow
from orion.reverie.semantic_lift import resolve_concern_cards
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1


def test_stable_hash_id_deterministic() -> None:
    from orion.core.ids import stable_hash_id

    a = stable_hash_id("concern", ["harness_closure:abc"])
    b = stable_hash_id("concern", ["harness_closure:abc"])
    assert a == b
    assert a == "concern_0ebb4258d3d772159cf641a6"


def test_concern_card_from_harness_turn_without_substrate_import() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-thin",
        user_message_excerpt="Will the deploy slip if we cut testing?",
        stance_imperative="Name the testing tradeoff before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert card.card_id.startswith("concern_")


def test_resolve_concern_cards_builds_real_cards_unmocked() -> None:
    row = TurnReferentRow(
        correlation_id="corr-thin",
        coalition_ref="harness_closure:corr-thin",
        user_message_excerpt="What's my last PR title?",
        stance_imperative="Search PR metadata for the most recent pull request title.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    loader = MagicMock()
    loader.load_by_coalition_ref.return_value = row
    broadcast = AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(
            open_loops=[
                OpenLoopV1(
                    id="ol-1",
                    description="pr title",
                    source_refs=["harness_closure:corr-thin"],
                )
            ]
        ),
        attended_node_ids=["harness_closure:corr-thin"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.5,
    )
    cards = resolve_concern_cards(broadcast, referent_loader=loader)
    assert len(cards) == 1
    assert "PR title" in cards[0].human_text
