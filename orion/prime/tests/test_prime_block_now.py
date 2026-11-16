"""The Prime block's two lines from Orion's now (integration patch 3, Q5): what other organs lived, from the now
Prime read at its last thought. They drop first, never show the chat organ, and their refs pass the citation gate."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.prime.block import BLOCK_CAP, PRIME_STATE_KEY, check_prime_citations, fetch_prime_block, render_prime_block
from orion.prime.state import (PRIME_CONCERNS_KEY, PRIME_NOW_SEEN_KEY, PrimeConcernV1, PrimeStateV1, concerns_view,
                               now_seen)
from orion.prime.tending import TendingV1
from orion.schemas.temporal_self import TemporalNowV1

NOW = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
T1 = TendingV1("disp-aaa", "people over pipeline, 11 of 14", 12.0)


def _now_view() -> TemporalNowV1:
    def item(eid, organ, kind, label, minutes, sal):
        return {"event_id": eid, "organ": organ, "source_kind": kind, "label": label, "salience": sal,
                "occurred_at": (NOW - timedelta(minutes=minutes)).isoformat()}

    return TemporalNowV1.model_validate({
        "shown_id": "shown-1", "reader_organ": "prime", "since": (NOW - timedelta(hours=6)).isoformat(),
        "as_of": NOW.isoformat(),
        "items": [item("chat_turn:c1", "chat", "chat_turn", "chat turn by juniper", 3, 0.99),
                  item("street_event:w1", "camera", "street_event", "street event: a mail truck stopped", 40, 0.9),
                  item("prediction_surprise:7", "prediction", "prediction_surprise", "cabinet forecast missed", 90, 0.5),
                  item("visual_run:v1", "painting", "visual_run", "visual run", 100, None)]})


SEEN = now_seen(_now_view(), seen_at=NOW - timedelta(minutes=30))


def _state(words: str = "short", n: int = 3) -> PrimeStateV1:
    at = NOW - timedelta(hours=1)
    return PrimeStateV1(concerns=[PrimeConcernV1(concern_id=f"c{i}", text=words, opened_at=at, last_evidence_at=at)
                                  for i in range(n)],
                        last_thought_at=NOW - timedelta(minutes=30), thoughts_total=9, last_reading="quiet",
                        seen_binding=[])


def _render(state, tending=(), seen=SEEN):
    return render_prime_block(state, ticked_at=NOW - timedelta(seconds=5), now=NOW, tending=tending, now_seen=seen)


def test_now_seen_keeps_the_top_two_and_never_the_chat_organ():
    assert [i.event_id for i in SEEN.items] == ["street_event:w1", "prediction_surprise:7"]
    assert SEEN.shown_id == "shown-1"


def test_block_shows_two_other_organ_lines_with_experience_refs():
    b = _render(_state(n=1))
    assert "- Other organs (camera, 40 min ago): street event: a mail truck stopped [experience:street_event:w1]" in b.text
    assert "[experience:prediction_surprise:7]" in b.text and "chat turn by juniper" not in b.text
    assert b.now_lines == 2 and "experience:street_event:w1" in b.refs and len(b.text) <= BLOCK_CAP
    assert "now=2" in b.log_fields()


def test_the_citation_gate_accepts_a_shown_experience_and_rejects_an_unshown_one():
    b = _render(_state(n=1))
    check = check_prime_citations("the truck [experience:street_event:w1]; the dog [experience:vision_percept:zz]",
                                  b.refs)
    assert check.cited_valid == ("experience:street_event:w1",)
    assert check.rejected_unknown == ("experience:vision_percept:zz",)


def test_a_full_block_shows_fewer_now_lines_rather_than_drop_one_of_its_own():
    full = _render(_state(), seen=None)
    b = _render(_state())
    assert b.now_lines == 1
    assert [ln for ln in b.text.splitlines() if "Other organs" not in ln] == full.text.splitlines()


def test_now_lines_drop_before_tending_and_before_the_blocks_own_lines():
    full = _render(_state("x" * 300), seen=None)
    tight = _render(_state("x" * 300), [T1])
    assert len(tight.text) <= BLOCK_CAP
    own_without = [ln.split("[")[-1] for ln in full.text.splitlines()[1:]]
    own_with = [ln.split("[")[-1] for ln in tight.text.splitlines()[1:]
                if "tending:" not in ln and "Other organs" not in ln]
    assert own_with == own_without
    # Tending survives whenever a now line does (the now's lines go first).
    if tight.now_lines:
        assert tight.tending == 1


def test_no_now_seen_renders_exactly_the_block_as_before():
    assert _render(_state(), seen=None).text == render_prime_block(
        _state(), ticked_at=NOW - timedelta(seconds=5), now=NOW).text


def test_fetch_reads_the_now_seen_key_and_survives_a_bad_one():
    state = _state(n=1)
    kv = {PRIME_STATE_KEY: state.model_dump_json(),
          PRIME_CONCERNS_KEY: concerns_view(state, now=NOW).model_dump_json(),
          PRIME_NOW_SEEN_KEY: SEEN.model_dump_json()}

    class R:
        async def get(self, key):
            return kv.get(key)

    good = asyncio.run(fetch_prime_block(R(), now=NOW + timedelta(seconds=5)))
    assert good.status == "ok" and good.now_lines == 2
    kv[PRIME_NOW_SEEN_KEY] = "{not json"
    bad = asyncio.run(fetch_prime_block(R(), now=NOW + timedelta(seconds=5)))
    assert bad.status == "ok" and bad.now_lines == 0
