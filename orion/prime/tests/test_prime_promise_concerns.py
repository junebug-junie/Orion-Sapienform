"""Prime can keep a promise Orion made (approved by Juniper 2026-11-13).

Live 11-12: chat turn a129b72f-... promised "While you're off in Hyrule I'll take a lazy browse through the last
24 hours". The hand-back opened concern d8db9d7b1bba4cf7 with no promise sentence in it, Prime read it as "a new chat
thread opened", chose no move, and the concern faded at 24 h. Three fixes, one test group each."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import get_args

from orion.prime import handback as hb
from orion.prime.block import render_prime_block
from orion.prime.intention_sources import turn_promise
from orion.prime.state import (CONCERN_TTL, PROMISE_CONCERN_TTL, PROMISE_EVENT_PREFIX, QUIET_FLOOR, PrimeConcernV1,
                               PrimeSnapshotV1, PrimeStateV1, concern_ttl, concerns_view, observe)
from orion.prime.thought import SYSTEM_PROMPT, FrameInputs, build_frame, system_prompt
from orion.schemas.cockpit_sighting import CockpitHopV1

CORR = "a129b72f-e698-425a-a435-3388cd7cd759"
# The live harness_turn_trace row (read 2026-11-13): created_at 2026-11-12 07:08:40Z, promises_made below.
QUOTE = "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"
PROMISED = datetime(2026, 11, 12, 7, 8, 40, tzinfo=timezone.utc)
SESSION = "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"


def _hops(corr=CORR, session=SESSION, *, mode="orion", promises=({"offer": False, "quote": QUOTE},), final=None,
          at=PROMISED):
    ingress = {"session_id": session, "user_message": "ok im off to play zelda for a bit", "attachment_count": 0}
    if mode is not None:
        ingress["mode"] = mode
    fin = {"correlation_id": corr, "final_text": final or f"Have fun!\n\n{QUOTE} and see what stands out.",
           "compliance_verdict": "completed",
           "reflection": {"promises_made": [dict(p) for p in promises], "promises_kept": []}}
    return [CockpitHopV1(correlation_id=corr, seq=0, ts=at - timedelta(seconds=30), stage="ingress",
                         visor_line="ingress", status="ok", summary={}, raw=ingress,
                         producer="orion-hub").model_dump(mode="json"),
            CockpitHopV1(correlation_id=corr, seq=9, ts=at, stage="finalize", visor_line="finalize",
                         status="ok", summary={}, raw=fin, producer="orion-hub").model_dump(mode="json")]


def _fold(hops, now=PROMISED + timedelta(seconds=15)):
    signals = [s for s in (hb.signal_from_hop(h) for h in hops) if s is not None]
    return hb.fold_turn_signals(PrimeStateV1(), hb.PrimeTurnsStateV1(), signals, now)


def _ordinary(cid="0123456789abcdef", at=PROMISED):
    return PrimeConcernV1(concern_id=cid, text="watching the cabinet heat", event_ref="event:heat-1",
                          opened_at=at, last_evidence_at=at)


# --- 1. the promise sentence reaches Prime ------------------------------------------------------------------


def test_the_11_12_promise_concern_carries_the_promise_sentence_labelled_with_who_and_when():
    state, _, _ = _fold(_hops())
    (c,) = state.concerns
    assert c.concern_id == "d8db9d7b1bba4cf7"  # the live concern's id: derived from the turn
    assert c.promise_text == f'a promise I made to Juniper in chat at 2026-11-12 07:08Z: "{QUOTE}"'
    assert c.text.startswith("a promise I made in a chat with Juniper") and "thread I opened" not in c.text
    frame = build_frame(state, FrameInputs(now=PROMISED + timedelta(hours=1)))
    assert f"PROMISE: a promise I made to Juniper in chat at 2026-11-12 07:08Z: \"{QUOTE}\"" in frame


def test_the_promise_sentence_stays_out_of_the_prime_block_and_the_published_view():
    """Choice-as-EFE acceptance S-1.8: ordinary turns' motor prefix carries no promise quote."""
    state, _, _ = _fold(_hops())
    now = PROMISED + timedelta(minutes=1)
    block = render_prime_block(state, ticked_at=now, now=now)
    assert "d8db9d7b1bba4cf7" in block.text and QUOTE not in block.text and "Hyrule" not in block.text
    assert "Hyrule" not in concerns_view(state, now=now).model_dump_json()
    assert all("Hyrule" not in line for line in state.pending_changes)


def test_a_reply_that_opens_with_the_promise_still_keeps_the_quote_out_of_the_block():
    state, _, _ = _fold(_hops(final=f"{QUOTE}. Have fun!"))
    c = state.concerns[0]
    assert "Hyrule" not in c.text and c.text.endswith('Juniper: "ok im off to play zelda for a bit"')
    now = PROMISED + timedelta(minutes=1)
    assert "Hyrule" not in render_prime_block(state, ticked_at=now, now=now).text
    assert QUOTE in c.promise_text


def test_every_kind_of_turn_has_promise_words():
    assert set(hb.PROMISE_TO) == set(get_args(hb.TurnWho)) == set(hb.WHO_WORDS)


def test_a_promise_made_outside_a_chat_with_juniper_is_not_labelled_as_one_to_her():
    state, _, _ = _fold(_hops(session="orion_curiosity", mode=None))
    assert state.concerns[0].promise_text.startswith("a promise I made in a curiosity run at ")


def test_several_promises_are_all_quoted_and_no_quote_gives_no_promise_text():
    two = ({"offer": False, "quote": "I'll check the logs"}, {"offer": True, "quote": "want me to?"},
           {"offer": False, "quote": "I'll write it up"})
    state, _, _ = _fold(_hops(promises=two))
    assert state.concerns[0].promise_text.endswith('"I\'ll check the logs" / "I\'ll write it up"')
    assert "want me to?" not in state.concerns[0].promise_text
    assert hb.promise_text("chat_juniper", [], PROMISED) is None


def test_a_long_quote_is_cut_and_fits_the_field():
    long = {"offer": False, "quote": "I will " + "look again " * 80}
    state, _, _ = _fold(_hops(promises=(long,) * 3))
    assert len(state.concerns[0].promise_text) <= 600 and state.concerns[0].promise_text.endswith("...")


# --- 2. promise concerns live 72 h, every other concern 24 h -------------------------------------------------


def test_a_promise_concern_survives_48h_unkept_and_fades_after_72h():
    state, _, _ = _fold(_hops())
    opened = state.concerns[0].opened_at
    assert observe(state, PrimeSnapshotV1(), opened + timedelta(hours=48)).concerns == state.concerns
    assert observe(state, PrimeSnapshotV1(), opened + timedelta(hours=72)).concerns == state.concerns
    gone = observe(state, PrimeSnapshotV1(), opened + timedelta(hours=72, seconds=1))
    assert gone.concerns == [] and gone.concern_record[-1].end == "faded"
    assert any("not kept for 3 days" in line for line in gone.pending_changes)


def test_an_ordinary_concern_still_fades_at_24h():
    state = PrimeStateV1(concerns=[_ordinary()])
    assert observe(state, PrimeSnapshotV1(), PROMISED + timedelta(hours=24)).concerns == state.concerns
    gone = observe(state, PrimeSnapshotV1(), PROMISED + timedelta(hours=24, seconds=1))
    assert gone.concerns == [] and gone.concern_record[-1].note == "faded: not kept for a day"


def test_both_kinds_held_together_fade_on_their_own_clocks():
    promise, _, _ = _fold(_hops())
    state = promise.model_copy(update={"concerns": [*promise.concerns, _ordinary()]})
    at = PROMISED + timedelta(hours=30)
    assert [c.concern_id for c in observe(state, PrimeSnapshotV1(), at).concerns] == ["d8db9d7b1bba4cf7"]


def test_an_unkept_promise_older_than_a_day_lives_on_but_does_not_block_the_next_promise():
    state, turns, _ = _fold(_hops())
    later = PROMISED + timedelta(hours=30)
    signals = [hb.signal_from_hop(h) for h in _hops(corr="next-1", at=later)]
    state = observe(state, PrimeSnapshotV1(), later)
    assert [c.concern_id for c in state.concerns] == ["d8db9d7b1bba4cf7"]  # alive at 30 h (72 h TTL)
    state, turns, folded = hb.fold_turn_signals(state, turns, signals, later + timedelta(seconds=15))
    assert folded[0].opened_concern_id is not None and folded[0].thread_refused is None
    assert len(state.concerns) == 2
    # Within a day it still holds the one slot, as before.
    early = PROMISED + timedelta(hours=20)
    st0, t0, _ = _fold(_hops())
    _, _, f2 = hb.fold_turn_signals(st0, t0, [hb.signal_from_hop(h) for h in _hops(corr="next-2", at=early)], early)
    assert f2[0].thread_refused == "already holding 1 concern opened by a turn"


def test_the_ttls_and_the_quiet_floor():
    assert CONCERN_TTL == timedelta(hours=24) and QUIET_FLOOR == timedelta(hours=6)
    # The promise TTL is the intention ledger's promise horizon, and the kind is the adapter's prefix.
    assert PROMISE_CONCERN_TTL == timedelta(seconds=turn_promise.HORIZON_SEC)
    assert PROMISE_EVENT_PREFIX == turn_promise.PREFIX == hb.TURN_EVENT_PREFIX
    assert concern_ttl(_ordinary()) == CONCERN_TTL
    assert concern_ttl(_ordinary().model_copy(update={"event_ref": None})) == CONCERN_TTL


def test_a_concern_saved_before_this_change_loads_and_gets_the_promise_ttl_from_its_event_ref():
    old = {"concern_id": "d8db9d7b1bba4cf7", "text": "a thread I opened in a chat with Juniper (1 promise, ...)",
           "event_ref": f"turn:{CORR}", "opened_at": "2026-11-12T07:08:57Z", "last_evidence_at": "2026-11-12T07:08:57Z"}
    raw = json.dumps({"schema_version": "prime.state.v1", "concerns": [old]})
    state = PrimeStateV1.model_validate_json(raw)
    c = state.concerns[0]
    assert c.promise_text is None and concern_ttl(c) == PROMISE_CONCERN_TTL
    assert "promise_text" not in c.model_dump(mode="json")  # a concern with none saves as before
    assert "PROMISE:" not in build_frame(state, FrameInputs(now=PROMISED + timedelta(hours=1)))


# --- 3. the prompt says how a promise gets kept -------------------------------------------------------------


def test_the_prompt_says_a_curiosity_question_is_how_a_promise_to_look_gets_kept():
    line = "When a concern is a promise you made to look into something, asking that question is how the promise"
    assert line in SYSTEM_PROMPT
    assert line in system_prompt(None, events=True) and line in system_prompt(None, take_on=True)
    assert "fades after a day (a promise: after 3 days)" in SYSTEM_PROMPT
