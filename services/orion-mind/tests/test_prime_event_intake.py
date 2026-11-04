"""Attention as the want competition, patch 2: Prime's concerns come from funded event wants
(PRIME_CONCERNS_FROM_EVENT_WANTS) instead of Temporal Self arcs. Fake bus, model and market."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from orion.prime.state import PrimeStateV1
from orion.prime.thought import FrameInputs, build_frame, system_prompt
from test_prime_loop import STATE_KEY, T0, runtime

EVENTS_URL = "http://dr:8121/physiology/event-wants?hours=24&limit=8"
EVENTS = {"schema_version": "physiology.funded_event_wants.v1", "as_of": T0.isoformat(), "considered": 5, "funded_open": 3,
          "wants": [
              {"want_id": "w1", "family": "vision.ask", "event_ref": "orion_ask:k1", "salience": 0.95, "value_rank": 0.9,
               "net": 0.9, "decided_at": (T0 - timedelta(minutes=20)).isoformat(), "expires_at": None,
               "label": "an open camera question to Juniper: Whose car is this?"},
              {"want_id": "w2", "family": "hub.world_pulse_read", "event_ref": "world_pulse_read_seed:s1", "salience": 0.0,
               "value_rank": 0.4, "net": 0.4, "decided_at": (T0 - timedelta(hours=1)).isoformat(), "expires_at": None,
               "label": "a read finished: Bees learn"}]}

ADD_EVENT = {"reading": "a car keeps coming back", "concerns": {"add": [
    {"text": "whose car is it", "event_id": "orion_ask:k1"},
    {"text": "made-up worry", "event_id": "orion_ask:nope"},
    {"text": "arc worry", "arc_id": "arc1"}]}, "move": None}


def _event_runtime(answer, *, on=True, events=EVENTS):
    pages = {EVENTS_URL: events} if events is not None else {}
    rt = runtime(answer=answer, extra_pages=pages)
    rt.s.PRIME_CONCERNS_FROM_EVENT_WANTS = on
    rt.s.PRIME_EVENT_WANTS_HOURS = 24.0
    rt.s.PRIME_EVENT_WANTS_LIMIT = 8
    return rt


def test_event_mode_frame_lists_funded_events_instead_of_arcs():
    rt = _event_runtime(ADD_EVENT)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    frame, system = call["user_prompt"], call["system_prompt"]
    assert "EVENTS WORTH EFFORT" in frame and "ATTENTION (Temporal Self" not in frame
    assert "event_id orion_ask:k1: an open camera question to Juniper: Whose car is this?" in frame
    assert "(1 more funded events not shown)" in frame
    assert "event_id" in system and "arc_id must be" not in system


def test_a_new_concern_must_name_a_funded_event():
    rt = _event_runtime(ADD_EVENT)
    asyncio.run(rt.tick())
    assert [(c.text, c.event_ref, c.arc_id) for c in rt.state.concerns] == [("whose car is it", "orion_ask:k1", None)]
    saved = PrimeStateV1.model_validate_json(rt.bus.redis.kv[STATE_KEY])
    assert saved.concerns[0].event_ref == "orion_ask:k1"


def test_an_unread_menu_is_a_gap_and_adds_no_concern():
    rt = _event_runtime(ADD_EVENT, events=None)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "EVENTS WORTH EFFORT" in call["user_prompt"] and "(unavailable: HTTPError" in call["user_prompt"]
    assert rt.state.concerns == []


def test_off_is_the_arc_path_as_before():
    rt = _event_runtime(ADD_EVENT, on=False)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "ATTENTION (Temporal Self" in call["user_prompt"] and "EVENTS WORTH EFFORT" not in call["user_prompt"]
    assert call["system_prompt"] == system_prompt(None)
    # Off: the arc concern is added (event_id is ignored); nothing has an event_ref.
    assert {c.text for c in rt.state.concerns} >= {"arc worry"} and all(c.event_ref is None for c in rt.state.concerns)
    assert "event_ref" not in rt.bus.redis.kv[STATE_KEY]      # a rollback still loads the state


def test_an_empty_menu_says_so():
    frame = build_frame(PrimeStateV1(), FrameInputs(now=T0, event_mode=True, events={"wants": []}))
    assert "(none: nothing the body economy funded is open now)" in frame


def test_one_event_holds_one_concern_and_event_mode_refs_name_acts_only():
    dup = {"reading": "r", "concerns": {"add": [{"text": "first", "event_id": "orion_ask:k1"},
                                                {"text": "again", "event_id": "orion_ask:k1"}]}, "move": None}
    rt = _event_runtime(dup)
    asyncio.run(rt.tick())
    assert [c.text for c in rt.state.concerns] == ["first"]
    system = system_prompt(None, events=True)
    assert '"kind": "arc"' not in system and '"refs": [{"kind": "act", "id": "..."}]' in system


def test_a_malformed_menu_is_a_gap_not_a_frame():
    rt = _event_runtime(ADD_EVENT, events={"wants": [{"event_ref": "x"}]})
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "(unavailable: not a funded event wants payload" in call["user_prompt"] and rt.state.concerns == []
