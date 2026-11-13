"""Personality rev 4 in Prime's loop: the choice point (one personality.choice.v1 per thought that chose
among its concerns) and the tending lines in Prime's own frame. Fake bus, model and market."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from orion.prime.state import PrimeConcernV1
from orion.schemas.personality import PERSONALITY_CHOICE_CHANNEL, PERSONALITY_CHOICE_KIND, PersonalityChoiceV1
from test_prime_event_intake import EVENTS, EVENTS_URL
from test_prime_loop import T0, runtime

FRAME_URL = "http://dr:8121/personality/frame"


def _concern(cid, text, event_ref=None):
    at = T0 - timedelta(hours=1)
    return PrimeConcernV1(concern_id=cid, text=text, event_ref=event_ref, opened_at=at, last_evidence_at=at)


HELD = [_concern("c-read", "the bee paper I read", "world_pulse_read_seed:s1"),
        _concern("c-car", "whose car is it", "orion_ask:k1")]


def _rt(answer, *, held=HELD, holdback=0.0, event_mode=True, pages=None, flag=False):
    rt = runtime(answer=answer, holdback=holdback, extra_pages={EVENTS_URL: EVENTS, **(pages or {})})
    rt.s.PRIME_CONCERNS_FROM_EVENT_WANTS = event_mode
    rt.s.PRIME_EVENT_WANTS_HOURS = 24.0
    rt.s.PRIME_EVENT_WANTS_LIMIT = 8
    rt.s.PRIME_PERSONALITY_LINE_ENABLED = flag
    rt.state = rt.state.model_copy(update={"concerns": list(held), "pending_changes": ["attention moved"]})
    return rt


def _choices(rt):
    return [env for ch, env in rt.bus.published if ch == PERSONALITY_CHOICE_CHANNEL]


ASK_CAR = {"reading": "the car again", "pick": "c-car",
           "move": {"kind": "curiosity_focus", "question": "Whose car parks out front?", "why": "it keeps coming",
                    "concern_id": "c-car"}}


def test_a_pick_among_two_concerns_publishes_one_choice_with_market_ranks_and_situation():
    rt = _rt(ASK_CAR)
    asyncio.run(rt.tick())
    [env] = _choices(rt)
    assert env.kind == PERSONALITY_CHOICE_KIND
    choice = PersonalityChoiceV1.model_validate(env.payload)
    thought_id = rt.state.last_thought_id
    assert choice.choice_id == f"prime:{thought_id}" and choice.lane == "prime"
    # The market lists orion_ask:k1 (net 0.9) above the read seed (0.4): the car concern is rank 1.
    assert (choice.chosen.key, choice.chosen.market_rank, choice.chosen.text) == ("c-car", 1, "whose car is it")
    assert [(o.key, o.market_rank) for o in choice.alternatives] == [("c-read", 2)]
    assert choice.chosen_market_rank == 1 and not choice.declined_all and not choice.care_promoted
    assert choice.situation["strain"] == 0.3 and choice.situation["strain_constraint"] == "lane:agent"
    assert choice.situation["event_mode"] is True and choice.situation["body_read"] is True
    infer = [a for k, a in rt.bus.acts() if a["verb"] == "infer"][0]
    assert choice.act_id == infer["act_id"] and choice.correlation_id == f"prime:{thought_id}"


def test_no_move_with_concerns_held_is_declined_all():
    rt = _rt({"reading": "nothing to do", "pick": "none", "move": None})
    asyncio.run(rt.tick())
    [env] = _choices(rt)
    choice = PersonalityChoiceV1.model_validate(env.payload)
    assert choice.declined_all and choice.chosen is None
    assert sorted(o.key for o in choice.alternatives) == ["c-car", "c-read"]


def test_a_held_back_question_still_records_the_choice():
    """Holdback (PRIME_HOLDBACK_FRACTION) decides whether the question is SENT, after Orion chose."""
    rt = _rt(ASK_CAR, holdback=1.0)
    asyncio.run(rt.tick())
    assert rt.bus.rpcs == [] and rt.state.open_nominations[0].arm == "control"
    [env] = _choices(rt)
    assert env.payload["chosen"]["key"] == "c-car"


def test_one_concern_picked_is_no_choice_and_no_pick_key_falls_back_to_the_move():
    rt = _rt(ASK_CAR, held=HELD[1:])
    asyncio.run(rt.tick())
    assert _choices(rt) == []
    legacy = {k: v for k, v in ASK_CAR.items() if k != "pick"}
    rt = _rt(legacy)
    asyncio.run(rt.tick())
    [env] = _choices(rt)
    assert env.payload["chosen"]["key"] == "c-car"


def test_a_publish_failure_never_breaks_the_thought(caplog):
    rt = _rt(ASK_CAR)
    real = rt.bus.publish

    async def flaky(channel, envelope):
        if channel == PERSONALITY_CHOICE_CHANNEL:
            raise ConnectionError("bus blip")
        await real(channel, envelope)

    rt.bus.publish = flaky
    with caplog.at_level("INFO"):
        asyncio.run(rt.tick())
    assert "prime_choice_publish_failed" in caplog.text and "choice=publish_failed" in caplog.text
    assert rt.state.open_nominations and rt.state.thoughts_total == 1


def test_prompt_asks_for_the_pick():
    rt = _rt(ASK_CAR)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert '"pick": "<concern_id> | none"' in call["system_prompt"]


FRAME = {"schema_version": "personality.frame.v1", "generated_at": T0.isoformat(), "dispositions": [
    {"disposition_id": "disp-aaa", "label": "people over pipeline, 11 of 14", "origin": "choice", "status": "active",
     "weight": 12.5, "evidence_count": 14, "first_seen_at": T0.isoformat(), "last_reinforced_at": T0.isoformat()},
    {"disposition_id": "disp-bbb", "label": "reading over walkway, 3 of 4", "origin": "choice", "status": "active",
     "weight": 3.0, "evidence_count": 4, "first_seen_at": T0.isoformat(), "last_reinforced_at": T0.isoformat()}]}


def test_tending_lines_reach_prime_frame_when_the_flag_is_on():
    rt = _rt(ASK_CAR, flag=True, pages={FRAME_URL: FRAME})
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert "WHAT YOU TEND TO CHOOSE" in call["user_prompt"]
    assert "- tending: people over pipeline, 11 of 14 [disposition:disp-aaa]" in call["user_prompt"]
    assert "disp-bbb" not in call["user_prompt"]  # under 10 weighted choices


def test_tending_read_failure_is_a_gap_and_flag_off_shows_nothing():
    rt = _rt(ASK_CAR, flag=True)  # no frame page: 404
    asyncio.run(rt.tick())
    assert "(unavailable: the personality frame could not be read" in rt.llm.calls[0]["user_prompt"]
    assert _choices(rt)  # the thought and its choice go on
    rt = _rt(ASK_CAR, flag=False, pages={FRAME_URL: FRAME})
    asyncio.run(rt.tick())
    assert "WHAT YOU TEND TO CHOOSE" not in rt.llm.calls[0]["user_prompt"]
