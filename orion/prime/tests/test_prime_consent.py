"""Orion's consent frame, answer parse and act (orion/prime/consent.py). Pure."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.prime.consent import (
    FRAME_CAP,
    build_consent_frame,
    consent_act,
    consent_object_ref,
    parse_consent_answer,
)
from orion.schemas.claude_tools import PrimeConsentPastAnswerV1, PrimeConsentRequestV1
from orion.self_monitor.adapters import act_faculty

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)


def _req(**kw) -> PrimeConsentRequestV1:
    base = dict(call_id="call-00000001", turn_request_id="room-claude-req-1", room_id="hub-room",
                claude_session_id="sess-a", tool="body_economy", category="body",
                tool_summary="your body economy report", reason="Juniper asked whether the meter blocks consults",
                topic="@claude is the meter blocking consults?", topic_from="Juniper")
    base.update(kw)
    return PrimeConsentRequestV1(**base)


def test_the_faculty_is_act_prime_infer_consent():
    act = consent_act("call-00000001", horizon_sec=35)
    assert (act.actor, act.verb) == ("prime", "infer")
    assert act_faculty(act.actor, act.verb, act.object_ref) == "act.prime.infer.consent"
    assert act.object_ref == consent_object_ref("call-00000001")
    assert act.expect.signal_id == "artifact_produced" and act.expect.horizon_sec == 65
    # Distinct act ids per call; never the same id as a Prime thought.
    assert consent_act("call-00000002", horizon_sec=35).act_id != act.act_id


def test_frame_carries_the_ask_topic_history_and_how_to_stop_being_asked():
    past = [PrimeConsentPastAnswerV1(at=T0, tool="body_economy", answer="yes", reason="fine")]
    frame = build_consent_frame(_req(recent_answers=past, history_note="you have said yes 20 of 20 times"),
                                "PRIME (my slow loop between turns), ticked 20 s ago:\n- concern: heat")
    assert frame.index("YOUR PRESENT") < frame.index("THE ASK")
    assert "Claude wants to use `body_economy` (category `body`) because: Juniper asked" in frame
    assert "@claude is the meter blocking consults?" in frame and "from Juniper" in frame
    assert "body_economy: yes (once): fine" in frame and "20 of 20" in frame
    assert "dig claude-grants set body allow|ask|deny because" in frame


def test_frame_is_capped_even_with_hostile_lengths():
    big = "x" * 5000
    frame = build_consent_frame(_req(reason="r" * 300, topic="t" * 300,
                                     recent_answers=[PrimeConsentPastAnswerV1(at=T0, tool="t" * 60, answer="no",
                                                                              reason="n" * 200)] * 5,
                                     history_note="h" * 300), big)
    assert len(frame) <= FRAME_CAP


@pytest.mark.parametrize("raw,want", [
    ({"answer": "yes", "scope": "conversation", "reason": "fine"}, ("yes", "conversation", "fine")),
    ({"answer": "YES", "reason": "ok"}, ("yes", "once", "ok")),
    ('{"answer": "no", "scope": "once", "reason": "not today"}', ("no", "conversation", "not today")),
    ({"answer": "no", "reason": ""}, ("no", "conversation", "")),
])
def test_parse_accepts_the_one_object(raw, want):
    got, why = parse_consent_answer(raw)
    assert why is None and (got["answer"], got["scope"], got["reason"]) == want


@pytest.mark.parametrize("raw", [None, "maybe", [], {"answer": "maybe"}, {"answer": "yes", "scope": "forever"}])
def test_parse_refuses_anything_else(raw):
    got, why = parse_consent_answer(raw)
    assert got is None and why


def test_reason_is_clipped_to_200():
    got, _ = parse_consent_answer({"answer": "yes", "reason": "w " * 300})
    assert len(got["reason"]) <= 200
