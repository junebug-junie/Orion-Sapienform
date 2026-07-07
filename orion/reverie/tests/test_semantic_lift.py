from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.reverie.referent_loader import TurnReferentRow, parse_harness_closure_ref
from orion.reverie.semantic_lift import (
    enforce_semantic_quality,
    infra_vocabulary_hit,
    referent_overlap,
    resolve_concern_cards,
    reverie_semantic_gate,
)
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1


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


def _broadcast_with_harness_ref(corr: str = "corr-1") -> AttentionBroadcastProjectionV1:
    loop = OpenLoopV1(
        id="ol-1",
        description="deploy worry",
        source_refs=[f"harness_closure:{corr}"],
    )
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[loop]),
        attended_node_ids=[f"harness_closure:{corr}"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.5,
    )


def test_resolve_concern_cards_from_selected_loop_source_ref() -> None:
    row = TurnReferentRow(
        correlation_id="corr-1",
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="Can we still make Friday's deploy?",
        stance_imperative="Name the schedule risk before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    loader = MagicMock()
    loader.load_by_coalition_ref.return_value = row
    cards = resolve_concern_cards(_broadcast_with_harness_ref(), referent_loader=loader)
    assert len(cards) == 1
    assert "Friday" in cards[0].human_text


def test_reverie_semantic_gate_skips_when_no_cards() -> None:
    assert reverie_semantic_gate([]) == "skip"


def test_reverie_semantic_gate_proceeds_when_card_substantive() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:x",
        user_message_excerpt="What happens if the migration fails overnight?",
        stance_imperative="Stay with the operational fear before planning.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert reverie_semantic_gate([card]) == "proceed"


_COALITION = CoalitionSnapshotV1(
    attended_node_ids=["harness_closure:corr-1"],
    selected_open_loop_id="ol-1",
    open_loop_ids=["ol-1"],
    generated_at="2026-07-07T00:00:00Z",
)


def test_infra_vocabulary_detects_mechanism_narration() -> None:
    assert infra_vocabulary_hit("The coalition centers on two open loops with substrate pressure.")


def test_referent_overlap_requires_shared_tokens() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="I'm worried the deploy will slip again.",
        stance_imperative="Name the slip risk plainly.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert referent_overlap("I keep worrying the deploy will slip again.", [card])


def test_enforce_semantic_quality_stamps_infra_vocabulary_hollow() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="I'm worried the deploy will slip again.",
        stance_imperative="Name the slip risk plainly.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    thought = SpontaneousThoughtV1(
        thought_id="t",
        correlation_id="c",
        coalition=_COALITION,
        interpretation="Two open loops dominate the coalition stability score this tick.",
        evidence_refs=["ol-1"],
    )
    out = enforce_semantic_quality(thought, [card])
    assert out.hollow and out.hollow_reason == "infra_vocabulary"
