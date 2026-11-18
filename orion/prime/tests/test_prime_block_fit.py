"""The Prime block's fit check (integration patch 3 follow-on). Extra lines (tending) are shown only when the block's
own lines all survive the re-render into the room they leave. Two own lines can carry the SAME ref (the body line
and "Asks: none open." both cite the tick), so a check on refs alone let the body line be dropped silently to make
room for a tending line."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.prime.block import BLOCK_CAP, render_prime_block
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from orion.prime.tending import TendingV1

NOW = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
T1 = TendingV1("disp-aaa", "people over pipeline, 11 of 14", 12.0)
T2 = TendingV1("disp-bbb", "reading over walkway, 10 of 12", 10.5)


def _state(words: str, n: int = 3) -> PrimeStateV1:
    at = NOW - timedelta(hours=1)
    return PrimeStateV1(concerns=[PrimeConcernV1(concern_id=f"c{i}", text=words, opened_at=at, last_evidence_at=at)
                                  for i in range(n)],
                        last_thought_at=NOW - timedelta(minutes=30), thoughts_total=9, last_reading="q",
                        seen_binding=[])


def _render(state, tending=()):
    return render_prime_block(state, ticked_at=NOW - timedelta(seconds=5), now=NOW, tending=tending)


def _own(text: str) -> list[str]:
    """The block's own lines, with free text removed (it may be cut shorter to make room)."""
    return [ln.split(":", 1)[0] + " " + ln.rsplit("[", 1)[-1] for ln in text.splitlines()[1:] if "tending:" not in ln]


def test_a_tending_line_never_costs_one_of_the_blocks_own_lines_even_one_sharing_a_ref():
    # Found on main 2026-11-16: two short concerns and two 90-char tending lines. The body line was dropped to make
    # room, and the refs-only check passed because "Asks: none open." still showed the same [prime_tick:...] ref.
    long_t = [TendingV1("disp-aaa", "y" * 90, 12.0), TendingV1("disp-bbb", "z" * 90, 10.5)]
    for state in (_state("xxxxx", n=2), _state("x" * 300), _state("x" * 40, n=1)):
        without = _render(state)
        assert "- Body: nothing binding." in without.text
        for tending in ([T1], [T1, T2], long_t[:1], long_t):
            with_t = _render(state, tending)
            assert len(with_t.text) <= BLOCK_CAP
            assert _own(with_t.text) == _own(without.text), with_t.text


def test_the_reproducing_case_shows_one_tending_line_and_keeps_the_body_line():
    # Two short concerns + two 90-char tending lines: on main both tending lines showed and the body line went.
    b = _render(_state("xxxxx", n=2), [TendingV1("disp-aaa", "y" * 90, 12.0), TendingV1("disp-bbb", "z" * 90, 10.5)])
    assert b.tending == 1 and b.dropped_lines == 1 and "- Body: nothing binding." in b.text


def test_tending_still_shows_when_there_is_room():
    b = _render(_state("short"), [T1])
    assert b.tending == 1 and "[disposition:disp-aaa]" in b.text and "- Body: nothing binding." in b.text
