from datetime import datetime, timezone

from orion.schemas.reverie import ConcernCardV1


def test_concern_card_harness_turn_template() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:abc",
        user_message_excerpt="Will the surgery timeline slip?",
        stance_imperative="Stay with the worry before fixing it.",
        created_at=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
    )
    assert card.coalition_ref == "harness_closure:abc"
    assert "Will the surgery timeline slip?" in card.human_text
    assert "Stay with the worry" in card.human_text
    assert card.source_kind == "harness_turn"
    assert len(card.human_text) >= 40
