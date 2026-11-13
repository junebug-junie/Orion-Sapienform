"""Personality rev 4 in Prime, pure parts: the choice point (orion/prime/choice.py), the tending lines
(orion/prime/tending.py) and their place in the Prime block (orion/prime/block.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.prime.block import BLOCK_CAP, check_prime_citations, render_prime_block
from orion.prime.choice import ShownConcern, body_situation, market_ranks, prime_choice
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from orion.prime.tending import TendingV1, tendencies_from_frame
from orion.prime.thought import PrimeAnswerV1

NOW = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)
A = ShownConcern("c-a", "the bee paper", "seed:1")
B = ShownConcern("c-b", "whose car is it", "ask:1")
C = ShownConcern("c-c", "a promise from chat", "turn:x")
RANKED = ["ask:1", "other", "seed:1"]


def _choice(answer: dict, shown=(A, B, C), **kw):
    return prime_choice(PrimeAnswerV1.model_validate(answer), list(shown), thought_id="t1", occurred_at=NOW,
                        ranked_event_ids=RANKED, **kw)


def test_market_rank_is_the_event_order_among_shown_concerns_only():
    assert market_ranks([A, B, C], RANKED) == {"c-b": 1, "c-a": 2, "c-c": None}
    assert market_ranks([A, B], []) == {"c-a": None, "c-b": None}


def test_explicit_pick_records_chosen_and_the_rest_as_alternatives():
    choice, why = _choice({"pick": "c-a"})
    assert why == "chosen" and choice.choice_id == "prime:t1"
    assert (choice.chosen.key, choice.chosen_market_rank) == ("c-a", 2)  # departed from the market's top
    assert [o.key for o in choice.alternatives] == ["c-b", "c-c"]


@pytest.mark.parametrize("pick", ["none", None, "no move", "NONE"])
def test_none_is_declined_all(pick):
    choice, why = _choice({"pick": pick})
    assert why == "declined_all" and choice.declined_all and len(choice.alternatives) == 3


def test_declined_all_needs_only_one_option_but_a_pick_needs_two():
    choice, _ = _choice({"pick": "none"}, shown=(A,))
    assert choice.declined_all
    assert _choice({"pick": "c-a"}, shown=(A,)) == (None, "one_concern_picked")


def test_unusable_picks_record_nothing_with_a_reason():
    assert _choice({"pick": "new"}) == (None, "pick_not_shown")
    assert _choice({"pick": "c-zzz"}) == (None, "pick_not_shown")
    assert _choice({}) == (None, "no_pick")
    assert _choice({"pick": "c-a"}, shown=()) == (None, "no_concerns")


def test_missing_pick_falls_back_to_the_move_concern():
    move = {"kind": "curiosity_focus", "question": "q?", "why": "w", "concern_id": "c-b"}
    choice, _ = _choice({"move": move})
    assert choice.chosen.key == "c-b" and choice.chosen_market_rank == 1


def test_an_explicit_pick_wins_over_the_move():
    move = {"kind": "curiosity_focus", "question": "q?", "why": "w", "concern_id": "c-b"}
    choice, _ = _choice({"pick": "c-a", "move": move})
    assert choice.chosen.key == "c-a"


def test_situation_is_flat_and_valid():
    body = {"strain": {"value": 0.42, "constraint": "heat"}}
    sit = body_situation(body, ["energy", "heat"], event_mode=True)
    assert sit == {"body_read": True, "event_mode": True, "binding": "energy,heat", "binding_count": 2,
                   "strain": 0.42, "strain_constraint": "heat"}
    choice, _ = _choice({"pick": "c-a"}, situation=sit)
    assert choice.situation == sit
    assert body_situation(None, None, event_mode=False) == {"body_read": False, "event_mode": False}


def test_long_concern_text_is_cut_to_the_option_cap():
    long = ShownConcern("c-l", "x" * 900)
    choice, _ = _choice({"pick": "c-l"}, shown=(long, A))
    assert len(choice.chosen.text) == 400


# --- tending ------------------------------------------------------------------------------------

def _disp(did, weight, *, origin="choice", status="active", label=None):
    return {"disposition_id": did, "label": label or f"{did} over other, 5 of 7", "origin": origin, "status": status,
            "weight": weight, "evidence_count": 7, "first_seen_at": NOW.isoformat(),
            "last_reinforced_at": NOW.isoformat()}


def _frame(*disps, at=NOW):
    return {"schema_version": "personality.frame.v1", "generated_at": at.isoformat(), "dispositions": list(disps)}


def test_tending_shows_only_choice_nodes_with_ten_weighted_choices_top_two():
    frame = _frame(_disp("d1", 30), _disp("d2", 10), _disp("d3", 11), _disp("d4", 9.99),
                   _disp("d5", 50, origin="recurrence"))
    tends, why = tendencies_from_frame(frame, now=NOW)
    assert why is None and [t.disposition_id for t in tends] == ["d1", "d3"]
    assert tends[0].line() == "tending: d1 over other, 5 of 7 [disposition:d1]"


def test_tending_skips_stale_invalid_and_thin_frames():
    assert tendencies_from_frame(_frame(_disp("d1", 30), at=NOW - timedelta(hours=2)), now=NOW) == ([], "stale")
    assert tendencies_from_frame({"nope": 1}, now=NOW) == ([], "invalid_frame")
    assert tendencies_from_frame(_frame(_disp("d1", 2)), now=NOW) == ([], "below_threshold")
    # Before the reducer's first commit the frame is dated at the epoch: stale, never current.
    assert tendencies_from_frame(_frame(at=datetime(1970, 1, 1, tzinfo=timezone.utc)), now=NOW)[1] == "stale"


# --- the block --------------------------------------------------------------------------------

T1 = TendingV1("disp-aaa", "people over pipeline, 11 of 14", 12.0)
T2 = TendingV1("disp-bbb", "reading over walkway, 10 of 12", 10.5)


def _state(concern_words: str = "short") -> PrimeStateV1:
    at = NOW - timedelta(hours=1)
    concerns = [PrimeConcernV1(concern_id=f"c{i}", text=concern_words, opened_at=at, last_evidence_at=at)
                for i in range(3)]
    return PrimeStateV1(concerns=concerns, last_thought_at=NOW - timedelta(minutes=5), thoughts_total=9,
                        last_reading="quiet", seen_binding=[])


def _render(state, tending=()):
    return render_prime_block(state, ticked_at=NOW - timedelta(seconds=5), now=NOW, tending=tending)


def test_block_renders_tending_lines_with_disposition_refs_and_counts_them():
    b = _render(_state(), [T1, T2])
    assert "- tending: people over pipeline, 11 of 14 [disposition:disp-aaa]" in b.text
    assert "[disposition:disp-bbb]" in b.text and b.tending == 2
    assert "disposition:disp-aaa" in b.refs and len(b.text) <= BLOCK_CAP
    assert "tending=2" in b.log_fields()
    check = check_prime_citations("I lean [disposition:disp-aaa] and [disposition:disp-zzz]", b.refs)
    assert check.cited_valid == ("disposition:disp-aaa",) and check.rejected_unknown == ("disposition:disp-zzz",)


def test_tending_lines_are_dropped_first_when_over_the_cap():
    full = _render(_state("x" * 300))
    with_tending = _render(_state("x" * 300), [T1, T2])
    assert len(with_tending.text) <= BLOCK_CAP
    # The block's own lines are all kept; only tending lines give way.
    kept_without = [ln.split("[")[-1] for ln in full.text.splitlines()[1:]]
    kept_with = [ln.split("[")[-1] for ln in with_tending.text.splitlines()[1:] if "tending:" not in ln]
    assert kept_with == kept_without
    assert with_tending.dropped_lines - full.dropped_lines == 2 - with_tending.tending


def test_no_tending_leaves_the_block_unchanged():
    assert _render(_state()).text == _render(_state(), []).text
    assert _render(_state()).tending == 0
