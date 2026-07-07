"""Eval: reverie semantic lift quality bar — referent, voice, grounding."""

from datetime import datetime, timezone

from orion.reverie.semantic_lift import enforce_semantic_quality, infra_vocabulary_hit
from orion.schemas.reverie import ConcernCardV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

_COALITION = CoalitionSnapshotV1(
    attended_node_ids=["harness_closure:c-1"],
    selected_open_loop_id="ol-1",
    open_loop_ids=["ol-1"],
    generated_at="2026-07-07T00:00:00Z",
)

_CARD = ConcernCardV1.from_harness_turn(
    coalition_ref="harness_closure:c-1",
    user_message_excerpt="What if the deploy slips when we cut testing?",
    stance_imperative="Name the testing tradeoff before reassuring.",
    created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
)
assert _CARD is not None

GOOD = (
    "I keep worrying the deploy might slip if we cut testing — that tradeoff is still unresolved.",
    ["ol-1"],
)
BAD_META = (
    "The coalition centers on two open loops with high substrate pressure and stability scores.",
    ["ol-1"],
)


def test_good_fixture_passes_semantic_gates():
    t = SpontaneousThoughtV1(
        thought_id="g", correlation_id="c", coalition=_COALITION,
        interpretation=GOOD[0], evidence_refs=GOOD[1],
    )
    out = enforce_semantic_quality(t, [_CARD])
    assert not out.hollow
    assert not infra_vocabulary_hit(GOOD[0])


def test_bad_meta_fixture_fails_infra_vocab():
    assert infra_vocabulary_hit(BAD_META[0])
    t = SpontaneousThoughtV1(
        thought_id="b", correlation_id="c", coalition=_COALITION,
        interpretation=BAD_META[0], evidence_refs=BAD_META[1],
    )
    out = enforce_semantic_quality(t, [_CARD])
    assert out.hollow_reason == "infra_vocabulary"
