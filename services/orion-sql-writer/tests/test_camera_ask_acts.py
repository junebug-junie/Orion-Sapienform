"""Camera asks as graded acts (attention patch 4a): the grade, the rule, Juniper's own cards. No database.

Row shapes are copied from live ``orion_ask`` (read-only, 2026-11-13), ids shortened.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.grammar.camera_ask_acts import (
    ACTOR,
    JUNIPER_STORY_CARD_REF,
    ORION_MADE_CAMERA_ASK_SQL,
    SIGNAL,
    ask_act,
    ask_grade,
    ask_opened_event,
    ask_rule,
    is_juniper_card,
    is_orion_camera_ask,
    rule_from_test_before,
    rule_outcome_values,
)
from orion.physiology.appetite import GRADED_SOURCES

CREATED = datetime(2026, 11, 2, 7, 0, 9, tzinfo=timezone.utc)
EXPIRES = CREATED + timedelta(days=7)

# Live shapes (2026-11-13).
VEHICLE_DONT_KNOW = {"source_kind": "vision_individual", "status": "dismissed", "answer": None,
                     "answer_detail": {"dont_know": True},
                     "evidence_refs": ["sighting:bc2c", "sighting:6a71", "sighting:be68", "sighting:5dc2", "sighting:c77d"],
                     "created_at": CREATED, "expires_at": EXPIRES, "answered_at": CREATED + timedelta(hours=11, minutes=43)}
CONFIRM_NO = {"source_kind": "vision_person_confirm", "status": "answered", "answer": "no",
              "answer_detail": {"pictures": [{"index": 0, "verdict": "different", "image_ref": "thumb:5041"}]},
              "evidence_refs": ["sighting:e617", "sighting:ecc4", "sighting:9745"],
              "created_at": CREATED, "expires_at": EXPIRES, "answered_at": CREATED + timedelta(minutes=38)}
SAME_YES = {"source_kind": "vision_same_individual", "status": "answered", "answer": "yes",
            "answer_detail": {"names": {"a4e8": "Teddy", "f524": "Teddy"}},
            "evidence_refs": ["individual:a4e8", "individual:f524", "face_cosine:0.558"],
            "created_at": CREATED, "expires_at": EXPIRES, "answered_at": CREATED + timedelta(hours=3, minutes=40)}
JUNIPER_STORY_NAME = {"source_kind": "vision_individual", "status": "answered", "answer": "Amazon delivery person",
                      "answer_detail": {"via": "motion_story", "name": "Amazon delivery person", "action": "name",
                                        "story_id": "0255398290"},
                      "evidence_refs": ["motion_story:0255398290", "individual:7780de9f"],
                      "created_at": CREATED, "expires_at": None, "answered_at": CREATED}


def test_the_grade_mapping():
    now = CREATED + timedelta(days=1)
    g = ask_grade(CONFIRM_NO, now=now)
    assert (g.value, g.source, g.observed_at) == (1.0, "resolver", CONFIRM_NO["answered_at"])   # "no" is an answer
    g = ask_grade(SAME_YES, now=now)
    assert (g.value, g.source) == (1.0, "resolver")
    g = ask_grade(VEHICLE_DONT_KNOW, now=now)
    assert (g.value, g.source, g.observed_at) == (0.0, "resolver", VEHICLE_DONT_KNOW["answered_at"])
    assert "does not know" in g.note
    plain_dismiss = {**VEHICLE_DONT_KNOW, "answer_detail": None}
    assert (ask_grade(plain_dismiss, now=now).value, ask_grade(plain_dismiss, now=now).source) == (0.0, "resolver")


def test_still_open_is_not_closed_and_silence_at_expiry_is_a_zero_grade():
    open_ask = {**SAME_YES, "status": "open", "answered_at": None, "answer": None, "answer_detail": None}
    assert ask_grade(open_ask, now=EXPIRES - timedelta(seconds=1)) is None
    g = ask_grade(open_ask, now=EXPIRES)
    assert (g.value, g.source, g.observed_at) == (0.0, "resolver", EXPIRES)
    # The writer's tick sets status 'expired' once expires_at passes: the same grade.
    g = ask_grade({**open_ask, "status": "expired"}, now=EXPIRES + timedelta(minutes=5))
    assert (g.value, g.source, g.observed_at) == (0.0, "resolver", EXPIRES)
    assert g.source in GRADED_SOURCES        # Juniper's silence IS a grade (PR #3213 question 3)


def test_withdrawn_by_orion_is_not_a_grade():
    open_ask = {**SAME_YES, "answered_at": None, "answer": None, "answer_detail": None}
    # A split undo expires the open asks about the individual it deletes, before expires_at.
    g = ask_grade({**open_ask, "status": "expired"}, now=CREATED + timedelta(days=2))
    assert (g.value, g.source) == (0.0, "expired") and g.source not in GRADED_SOURCES
    g = ask_grade({**open_ask, "status": "withdrawn"}, now=CREATED + timedelta(days=2))
    assert g.source == "expired"
    assert ask_grade(None, now=CREATED).source == "expired"       # the row is gone
    assert ask_grade({**open_ask, "status": "weird"}, now=EXPIRES + timedelta(days=1)) is None


def test_answer_detail_may_arrive_as_json_text():
    g = ask_grade({**VEHICLE_DONT_KNOW, "answer_detail": json.dumps({"dont_know": True})}, now=EXPIRES)
    assert "does not know" in g.note


def test_juniper_story_cards_are_not_orion_asks():
    assert is_juniper_card(JUNIPER_STORY_NAME["evidence_refs"], JUNIPER_STORY_NAME["answer_detail"])
    assert not is_orion_camera_ask("vision_individual", JUNIPER_STORY_NAME["evidence_refs"],
                                   JUNIPER_STORY_NAME["answer_detail"])
    # The marker alone (cards written from patch 4a on), as JSON text as asyncpg/psycopg may return it.
    marked = json.dumps(["motion_story:1", "individual:2", JUNIPER_STORY_CARD_REF])
    assert is_juniper_card(marked, None) and not is_orion_camera_ask("vision_individual", marked, None)
    for row in (VEHICLE_DONT_KNOW, CONFIRM_NO, SAME_YES):
        assert is_orion_camera_ask(row["source_kind"], row["evidence_refs"], row["answer_detail"])
    assert not is_orion_camera_ask("vision_story_person", [], None)
    assert not is_orion_camera_ask("memory_confirmation", [], None)
    # The SQL predicate says the same thing structurally, never by question text.
    assert "question" not in ORION_MADE_CAMERA_ASK_SQL
    assert JUNIPER_STORY_CARD_REF in ORION_MADE_CAMERA_ASK_SQL and "story_id" in ORION_MADE_CAMERA_ASK_SQL


@pytest.mark.parametrize("kind, refs, ikind, rule", [
    ("vision_person_confirm", ["sighting:1", "rule:camera_prior"], "person", "confirm"),
    ("vision_same_individual", ["individual:a", "individual:b", "face_cosine:0.6"], "person", "same"),
    ("vision_same_individual", ["handoff:h", "story:s", "rule:handoff"], None, "same"),
    ("vision_individual", ["rule:face", "sighting:1"], "person", "face"),
    ("vision_individual", ["rule:office_session"], "person", "office"),
    ("vision_individual", ["sighting:1"], "vehicle", "vehicle"),
    ("vision_individual", ["sighting:1"], "person", "sighting"),
    ("vision_individual", ["sighting:1"], None, "sighting"),
])
def test_rules(kind, refs, ikind, rule):
    assert ask_rule(kind, refs, ikind) == rule
    assert ask_rule(kind, json.dumps(refs), ikind) == rule


def test_laplace_per_rule_matches_live_counts():
    # Live, 30 days to 2026-11-13: vehicle 5 answered of 45 closed; confirm 141 of 145.
    closes = [("vehicle", 1.0)] * 5 + [("vehicle", 0.0)] * 40 + [("confirm", 1.0)] * 141 + [("confirm", 0.0)] * 4
    v = rule_outcome_values(closes + [("not-a-rule", 1.0)])
    assert v["vehicle"] == pytest.approx(6 / 47) and v["confirm"] == pytest.approx(142 / 147)
    assert v["face"] == 0.5                      # rest point: no closes, the prior
    assert set(v) == {"confirm", "same", "face", "office", "vehicle", "sighting"}


def test_the_act_and_its_open_event():
    act = ask_act("ask-1", rule="vehicle", created_at=CREATED, expires_at=EXPIRES)
    assert (act.actor, act.verb, act.expect.signal_id) == (ACTOR, "ask", SIGNAL)
    assert act.object_ref == "orion_ask:ask-1" and act.expect.horizon_sec == 7 * 86400
    assert rule_from_test_before(act.test_before) == "vehicle"
    assert act.act_id == ask_act("ask-1", rule="confirm", created_at=CREATED, expires_at=None).act_id  # key = ask id
    ev = ask_opened_event("ask-1", rule="vehicle", created_at=CREATED, expires_at=EXPIRES)
    assert ev.event_kind == "act_opened" and ev.correlation_id == "ask-1" and ev.observed_at == CREATED
    assert rule_from_test_before("outreach:self") is None and rule_from_test_before(None) is None
