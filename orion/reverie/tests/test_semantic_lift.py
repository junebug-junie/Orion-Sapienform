from datetime import datetime, timezone

from orion.reverie.referent_loader import TurnReferentRow, parse_harness_closure_ref
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


def test_parse_harness_closure_ref() -> None:
    assert parse_harness_closure_ref("harness_closure:abc-123") == "abc-123"
    assert parse_harness_closure_ref("node:foo") is None


def test_turn_referent_row_to_card() -> None:
    row = TurnReferentRow(
        correlation_id="abc-123",
        coalition_ref="harness_closure:abc-123",
        user_message_excerpt="What about the deadline?",
        stance_imperative="Acknowledge the pressure first.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    card = row.to_concern_card(now=datetime(2026, 7, 7, 1, 0, tzinfo=timezone.utc))
    assert card is not None
    assert "deadline" in card.human_text.lower()
