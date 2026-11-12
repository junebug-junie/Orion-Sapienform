"""The Prime block (orion/prime/block.py, living Prime design Part 1): stale, cap, citations, absent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.prime.block import (
    BLOCK_CAP,
    PRIME_STATE_KEY,
    check_prime_citations,
    fetch_prime_block,
    parse_prime_redis,
    render_prime_block,
)
from orion.prime.state import (
    PRIME_CONCERNS_KEY,
    PrimeConcernV1,
    PrimeNominationV1,
    PrimeRegulationOpenV1,
    PrimeStateV1,
    concerns_view,
)

NOW = datetime(2026, 11, 12, 4, 15, tzinfo=timezone.utc)
ARC = "390d630d1fc992b8"


def _concern(cid: str, text: str, *, hours: float, arc: str | None = None) -> PrimeConcernV1:
    at = NOW - timedelta(hours=hours)
    return PrimeConcernV1(concern_id=cid, text=text, arc_id=arc, opened_at=at, last_evidence_at=at)


def _state(**kw) -> PrimeStateV1:
    base = dict(last_thought_at=NOW - timedelta(minutes=47), thoughts_total=235,
                last_reading="The system is quiet with no strain and zero heat.", seen_binding=[])
    base.update(kw)
    return PrimeStateV1(**base)


def _render(state, **kw):
    kw.setdefault("ticked_at", NOW - timedelta(seconds=10))
    return render_prime_block(state, now=NOW, **kw)


def test_quiet_state_renders_every_section_with_a_ref_on_every_line():
    b = _render(_state())
    assert b.status == "ok"
    lines = b.text.splitlines()
    assert lines[0].startswith("PRIME")
    for line in lines[1:]:
        assert line.rstrip().endswith("]"), line  # every line cites
    assert "Holding: no concerns now." in b.text
    assert "Last thought (47 min ago; no move): The system is quiet" in b.text
    assert "[thought:#235]" in b.text
    assert "Body: nothing binding." in b.text
    assert "thought:#235" in b.refs


def test_concerns_cite_their_id_and_arc_and_show_age():
    s = _state(concerns=[_concern("fb88ad23b93c4182", "Is the heat real?", hours=5.5, arc=ARC)])
    b = _render(s)
    line = next(ln for ln in b.text.splitlines() if "Holding" in ln)
    assert line == f"- Holding (5 h 30 min): Is the heat real? [prime_concern:fb88ad23b93c4182] [arc:{ARC}]"
    assert {"prime_concern:fb88ad23b93c4182", f"arc:{ARC}"} <= set(b.refs)


def test_last_thought_with_an_open_ask_cites_the_exact_thought_and_settle_state():
    at = NOW - timedelta(minutes=5)
    nom = PrimeNominationV1(act_id="act_x", thought_id="ab12cd34ef56ab78", focus_id="f-1", question="Why so hot?",
                            why="heat", arm="treated", opened_at=at, queued=True)
    b = _render(_state(last_thought_at=at, open_nominations=[nom]))
    assert "Last thought (5 min ago; asked curiosity a question)" in b.text
    assert "[thought:ab12cd34ef56ab78]" in b.text
    assert "- Asked curiosity, queued, waiting for a curiosity run: Why so hot? [curiosity_focus:f-1]" in b.text
    assert "curiosity_focus:f-1" in b.refs


def test_regulate_signal_shows_settle_state():
    at = NOW - timedelta(minutes=5)
    reg = PrimeRegulationOpenV1(act_id="act_r", signal_id="sig1", thought_id="t1", family="durable.curiosity",
                                direction="slow", why="hot", arm="treated", magnitude=0.5, opened_at=at,
                                expires_at=NOW + timedelta(hours=2), written=True)
    b = _render(_state(last_thought_at=at, open_regulations=[reg]))
    assert "- Regulate slow durable.curiosity, in force until 06:15 UTC [prime_regulate:sig1] [thought:t1]" in b.text
    assert "regulate slow durable.curiosity" in b.text  # the thought's move


def test_cap_holds_with_three_long_concerns_and_refs_are_never_cut():
    long = "x" * 300
    s = _state(concerns=[_concern(f"c{i}aaaaaaaaaaaaaa", long, hours=i + 1, arc=ARC) for i in range(3)],
               last_reading="y" * 600, seen_binding=["lane:chat", "heat"])
    b = _render(s)
    assert len(b.text) <= BLOCK_CAP
    for ref in b.refs:
        assert f"[{ref}]" in b.text
    assert "[prime_concern:c0aaaaaaaaaaaaaa]" in b.text
    assert "…" in b.text  # free text was cut, not the refs


def test_tiny_cap_drops_lower_lines_first_and_still_fits():
    s = _state(concerns=[_concern(f"c{i}aaaaaaaaaaaaaa", "z" * 200, hours=1, arc=ARC) for i in range(3)])
    b = _render(s, cap=260)
    assert len(b.text) <= 260
    assert b.dropped_lines > 0
    assert "[prime_concern:c0aaaaaaaaaaaaaa]" in b.text  # the first concern outranks the body line
    assert "Body:" not in b.text


def test_stale_state_renders_only_the_not_ticked_line():
    s = _state(concerns=[_concern("c1", "old worry", hours=1)])
    b = _render(s, ticked_at=NOW - timedelta(minutes=11))
    assert b.status == "stale"
    assert "Prime has not ticked since Nov 12 04:04 UTC (11 min ago)" in b.text
    assert "old worry" not in b.text and "c1" not in b.text
    assert b.refs == ()


def test_stale_threshold_is_configurable():
    s = _state()
    assert _render(s, ticked_at=NOW - timedelta(minutes=3), stale_after=timedelta(minutes=2)).status == "stale"
    assert _render(s, ticked_at=NOW - timedelta(minutes=3)).status == "ok"


def test_absent_cases_are_labelled_never_guessed():
    assert _render(None).status == "absent"
    assert "absent" in _render(None).text
    unknown_tick = render_prime_block(_state(), ticked_at=None, now=NOW)
    assert unknown_tick.status == "absent" and "tick time is unknown" in unknown_tick.text
    reason = _render(_state(), absent_reason="Redis down")
    assert reason.status == "absent" and "Redis down" in reason.text


def test_parse_prime_redis_handles_missing_and_garbage():
    assert parse_prime_redis(None, None)[2]
    assert "unreadable" in parse_prime_redis("{not json", None)[2]
    s = _state()
    view = concerns_view(s, now=NOW).model_dump_json()
    state, ticked, reason = parse_prime_redis(s.model_dump_json(), view)
    assert reason is None and state == s and ticked == NOW
    assert parse_prime_redis(s.model_dump_json(), "garbage")[1] is None


class _FakeRedis:
    def __init__(self, data=None, exc=None, delay=0.0):
        self.data, self.exc, self.delay = data or {}, exc, delay

    async def get(self, key):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.data.get(key)


def test_fetch_reads_both_keys_and_fails_open():
    s = _state(concerns=[_concern("c1aaaaaaaaaaaaaa", "watch heat", hours=1)])
    view = concerns_view(s, now=NOW - timedelta(seconds=5)).model_dump_json()
    ok = asyncio.run(fetch_prime_block(_FakeRedis({PRIME_STATE_KEY: s.model_dump_json(), PRIME_CONCERNS_KEY: view}),
                                       now=NOW))
    assert ok.status == "ok" and "prime_concern:c1aaaaaaaaaaaaaa" in ok.refs
    down = asyncio.run(fetch_prime_block(_FakeRedis(exc=ConnectionError("down")), now=NOW))
    assert down.status == "absent" and "ConnectionError" in down.text
    slow = asyncio.run(fetch_prime_block(_FakeRedis(delay=0.2), now=NOW, timeout_sec=0.01))
    assert slow.status == "absent" and "did not load" in slow.text
    empty = asyncio.run(fetch_prime_block(_FakeRedis(), now=NOW))
    assert empty.status == "absent"


def test_citation_gate_counts_shown_and_rejects_unknown_prime_refs():
    s = _state(concerns=[_concern("c1aaaaaaaaaaaaaa", "watch heat", hours=1, arc=ARC)])
    b = _render(s)
    good = check_prime_citations("I'm holding the heat question [prime_concern:c1aaaaaaaaaaaaaa].", b.refs)
    assert good.label == "verified" and good.cited_valid == ("prime_concern:c1aaaaaaaaaaaaaa",)
    bad = check_prime_citations("I worry about money [prime_concern:deadbeefdeadbeef] [thought:#9].", b.refs)
    assert bad.label == "UNVERIFIED"
    assert set(bad.rejected_unknown) == {"prime_concern:deadbeefdeadbeef", "thought:#9"}
    none = check_prime_citations("Just chatting. A thought: hello. [arc:390d630d1fc992b8]", b.refs)
    assert none.label == "no_prime_claims"  # arcs are the arc gate's; "thought: " is prose
    assert good.shown == len([r for r in b.refs if not r.startswith("arc:")])
