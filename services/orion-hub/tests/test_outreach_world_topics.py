"""People and chat-thread topics at the outreach choice point (scripts/outreach_world_topics.py).

Juniper (2026-11-15): the choices should be about the world -- the people around the house and the
threads of her own conversations -- not Orion's machinery. Only people she named; only threads that
went quiet after Orion's question; each topic stands on a real row it cites.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from scripts.outreach_choice import (
    OutreachTopic,
    choice_record,
    evidence_line,
    filter_unsaid,
    fold_said_subjects,
    outreach_object_text,
    parse_pick,
)
from scripts.endogenous_outreach import is_pass_response
from scripts.outreach_world_topics import (
    CHAT_SQL,
    CHAT_THREAD_KIND,
    PEOPLE_SQL,
    PERSON_KIND,
    chat_thread_topics,
    chat_thread_source,
    last_question,
    looks_like_machinery,
    people_source,
    person_topics,
)

NOW = datetime(2026, 11, 15, 20, 0, tzinfo=timezone.utc)  # 1 pm in Denver
TZ = "America/Denver"
AMANDA = "person-amanda"


def _sighting(sid: str, at: datetime, *, pid: str = AMANDA, name: str = "Amanda", stream: str = "walkway",
              minutes: float = 2.0, observations: int = 5, dwell: float = 30.0) -> Dict[str, Any]:
    return {"person_id": pid, "display_name": name, "sighting_id": sid, "stream_id": stream,
            "started_at": at, "ended_at": at + timedelta(minutes=minutes),
            "observation_count": observations, "dwell_sec": dwell}


# --- people ---------------------------------------------------------------------------------------


def test_person_back_after_days_away_is_a_topic_citing_the_visit() -> None:
    rows = [_sighting("s-old", NOW - timedelta(days=9)),
            _sighting("s-new-1", NOW - timedelta(hours=3)),
            _sighting("s-new-2", NOW - timedelta(hours=3) + timedelta(minutes=20))]
    (t,) = person_topics(rows, NOW, tz_name=TZ)
    assert t.kind == PERSON_KIND and t.subject == f"person:{AMANDA}"
    # One visit (sightings 20 min apart): the version and the evidence are its FIRST sighting.
    assert t.version == "s-new-1" and t.evidence_ref == "vision_individual_sighting:s-new-1"
    assert t.option_text.startswith(
        "Amanda (a camera match for a name Juniper gave) at the walkway, today 10:00 am:")
    assert "the first camera match for Amanda in 9 days" in t.option_text
    assert "A match can be wrong" in t.prompt_text
    assert len(t.option_text) <= 400
    assert t.observed_at == NOW - timedelta(hours=3) + timedelta(minutes=22)


def test_person_seen_at_her_usual_time_is_not_a_topic() -> None:
    # Every day at about 1 pm local for two weeks, and again today: nothing new.
    rows = [_sighting(f"s{d}", NOW - timedelta(days=d, minutes=30)) for d in range(1, 15)]
    rows.append(_sighting("s-today", NOW - timedelta(minutes=45)))
    assert person_topics(rows, NOW, tz_name=TZ) == []


def test_person_at_an_unusual_time_is_a_topic() -> None:
    rows = [_sighting(f"s{d}", NOW - timedelta(days=d)) for d in range(1, 9)]  # 1 pm local
    late = NOW - timedelta(hours=10)  # 3 am local
    rows.append(_sighting("s-late", late))
    (t,) = person_topics(rows, NOW, tz_name=TZ)
    assert t.version == "s-late"
    assert "a time of day Amanda had not matched at in 8 earlier visits over 8 days" in t.option_text


def test_unusual_time_needs_enough_history() -> None:
    rows = [_sighting(f"s{d}", NOW - timedelta(days=d)) for d in range(1, 4)]  # only 3 earlier visits
    rows.append(_sighting("s-late", NOW - timedelta(hours=10)))
    assert person_topics(rows, NOW, tz_name=TZ) == []


def test_a_visit_older_than_the_fresh_window_is_not_a_topic() -> None:
    rows = [_sighting("s-old", NOW - timedelta(days=20)), _sighting("s", NOW - timedelta(hours=40))]
    assert person_topics(rows, NOW, tz_name=TZ) == []


def test_first_sighting_in_the_whole_window_says_since_when() -> None:
    # Someone else was seen 80 days ago: the records reach past the 60-day window, so the window
    # start (Sep 16) is the honest date.
    rows = [_sighting("s1", NOW - timedelta(hours=2)),
            _sighting("o1", NOW - timedelta(days=80), pid="person-other", name="Teddy")]
    (t,) = person_topics(rows, NOW, tz_name=TZ, history_days=60)
    assert "the first camera match for Amanda since Sep 16" in t.option_text


def test_first_sighting_date_never_predates_the_camera_records() -> None:
    """Live: sightings begin 2026-10-15. Saying "since Sep 16" would claim the cameras watched then."""
    rows = [_sighting("s1", NOW - timedelta(hours=2)),
            _sighting("o1", NOW - timedelta(days=30), pid="person-other", name="Teddy")]
    amanda = [t for t in person_topics(rows, NOW, tz_name=TZ) if t.subject == f"person:{AMANDA}"]
    assert "since Oct 16" in amanda[0].option_text


def test_first_sighting_with_no_other_records_has_no_since_date() -> None:
    """Review 2026-11-15: with only this person in the rows the date clamped to their own visit
    ("since Nov 15"). No other record before the visit: no date."""
    (t,) = person_topics([_sighting("s1", NOW - timedelta(hours=2))], NOW, tz_name=TZ)
    assert "the first camera match for Amanda on record" in t.option_text and "since" not in t.option_text


def test_a_glimpse_is_not_a_topic() -> None:
    rows = [_sighting("s-old", NOW - timedelta(days=9)),
            _sighting("s-new", NOW - timedelta(hours=3), observations=1, dwell=1.5)]
    assert person_topics(rows, NOW, tz_name=TZ) == []


def test_two_named_people_matching_at_once_is_ambiguous() -> None:
    """Live 2026-11-14: Addy, Exterminator and Emerald matched on the doorbell inside 81 s."""
    at = NOW - timedelta(hours=3)
    rows = [_sighting("a-old", NOW - timedelta(days=9), stream="doorbell"),
            _sighting("a-new", at, stream="doorbell"),
            _sighting("e-old", NOW - timedelta(days=9, hours=5), pid="person-em", name="Emerald", stream="doorbell"),
            _sighting("e-new", at + timedelta(seconds=81), pid="person-em", name="Emerald", stream="doorbell")]
    assert person_topics(rows, NOW, tz_name=TZ) == []
    # On another camera at the same moment: not the same visitor, both stand.
    rows[3] = _sighting("e-new", at + timedelta(seconds=81), pid="person-em", name="Emerald", stream="walkway")
    assert {t.subject for t in person_topics(rows, NOW, tz_name=TZ)} == {f"person:{AMANDA}", "person:person-em"}


def test_a_future_dated_sighting_does_not_hide_the_real_visit() -> None:
    rows = [_sighting("s-old", NOW - timedelta(days=9)), _sighting("s-new", NOW - timedelta(hours=3)),
            _sighting("s-skew", NOW + timedelta(hours=2))]
    (t,) = person_topics(rows, NOW, tz_name=TZ)
    assert t.version == "s-new"


def test_no_unnamed_people() -> None:
    """A row without a name or person id (an unnamed cluster) never becomes a topic, and the SQL itself
    only reads links Juniper made: typed-name links, never copied labels, active people other than
    Juniper and Orion, merged names resolved, office camera left out."""
    rows = [_sighting("s1", NOW - timedelta(hours=2), pid="", name="x"),
            _sighting("s2", NOW - timedelta(hours=2), pid="cluster-1", name="  ")]
    assert person_topics(rows, NOW, tz_name=TZ) == []
    sql = " ".join(PEOPLE_SQL.split())
    for clause in ("l.revoked_at IS NULL", "l.source_kind = 'vision_individual'", "NOT i.label_copied",
                   "p.role = 'other'", "p.status = 'active'", "lp.status = 'active'",
                   "coalesce(lp.merged_into, lp.person_id)", "i.kind = 'person'",
                   "NOT IN (SELECT DISTINCT stream_id FROM vision_office_session)"):
        assert clause in sql, clause


def test_person_dedupe_by_subject_and_a_new_visit_comes_back() -> None:
    rows = [_sighting("s-old", NOW - timedelta(days=9)), _sighting("s-new", NOW - timedelta(hours=3))]
    (t,) = person_topics(rows, NOW, tz_name=TZ)
    # Juniper got this visit: the ledger holds person:<id> at version s-new.
    record = choice_record([t], parse_pick("Pick: 1\nmsg", 1, is_pass=is_pass_response), "outreach:w")
    said = fold_said_subjects([{"reason": "sent", "choice": record}])
    assert said == {f"person:{AMANDA}": {"s-new"}}
    assert filter_unsaid([t], said) == ([], [f"person:{AMANDA}"])
    # A later visit that is new again (back after another long gap) is a new version: shown again.
    later = NOW + timedelta(days=6)
    rows2 = rows + [_sighting("s-next", later - timedelta(hours=1))]
    (t2,) = person_topics(rows2, later, tz_name=TZ)
    assert t2.subject == t.subject and t2.version == "s-next"
    assert filter_unsaid([t2], said) == ([t2], [])


# --- chat threads ---------------------------------------------------------------------------------


def _turn(cid: str, at: datetime, prompt: str, response: str) -> Dict[str, Any]:
    # chat_history_log.created_at is naive UTC.
    return {"correlation_id": cid, "session_id": "sess-1", "created_at": at.replace(tzinfo=None),
            "prompt": prompt, "response": response}


CHAT_FIXTURE = [
    _turn("c1", NOW - timedelta(days=2, hours=6), "I'm going to turn 44 in February.",
          "Two creatures over the waves.\n\nSo what's the plan for 44? <3"),
    # 4 hours after c1: c1's question is where the chat stopped for a while.
    _turn("c2", NOW - timedelta(days=2, hours=2), "what are your thoughts on the design?",
          "It holds up. Which part should I dig into first?"),
    # 5 minutes after c2: Juniper answered, so c2 is not open.
    _turn("c3", NOW - timedelta(days=2, hours=1, minutes=55), "the memory part",
          "Then memory first. Is it the recall you want, or the storage?"),
    _turn("c4", NOW - timedelta(days=1), "please check `hecate-gpu1` on #3362",
          "Done. Want me to draft the qwen38 profile?"),
    _turn("c5", NOW - timedelta(hours=10), "el Coco got his stitches out!",
          "Yay! Did he get a treat?\n\nGood boy, tell him I said so."),
    _turn("c6", NOW - timedelta(minutes=30), "how was your day?", "Quiet. How was yours?"),
]


def test_chat_thread_candidate_from_a_fixture() -> None:
    topics = chat_thread_topics(CHAT_FIXTURE, NOW, tz_name=TZ)
    subjects = [t.subject for t in topics]
    # c1: question, then 4 hours of quiet. c3: question, then a day of quiet.
    # c2: Juniper answered 5 minutes later. c4: plumbing. c5: the question is not where the reply ended.
    # c6: too recent to call the chat quiet.
    assert subjects == ["chat_thread:c1", "chat_thread:c3"]
    t1 = topics[0]
    assert t1.kind == CHAT_THREAD_KIND and t1.version == ""
    assert t1.evidence_ref == "chat_history_log:c1"
    assert t1.option_text.startswith("An open thread with Juniper, ")
    assert '"I\'m going to turn 44 in February."' in t1.option_text
    assert '"So what\'s the plan for 44?"' in t1.option_text
    assert "the chat went quiet for 4 hours after" in t1.option_text
    assert len(t1.option_text) <= 400
    assert t1.observed_at == (NOW - timedelta(days=2, hours=6))
    assert "do not tell her she ignored it" in t1.prompt_text


def test_last_turn_with_no_reply_yet_says_so() -> None:
    rows = [_turn("c1", NOW - timedelta(hours=5), "long day", "Rest. Want to tell me about it later?")]
    (t,) = chat_thread_topics(rows, NOW, tz_name=TZ)
    assert "the chat has been quiet since" in t.option_text


def test_threads_older_than_the_lookback_are_left_out() -> None:
    rows = [_turn("c1", NOW - timedelta(days=8), "hi", "How are you?")]
    assert chat_thread_topics(rows, NOW, tz_name=TZ) == []


@pytest.mark.parametrize("text,machinery", [
    ("Want me to draft the `hecate` profile?", True),
    ("See https://example.com for it?", True),
    ("Should I check docs/specs/thing.md?", True),
    ("Is #3362 ready?", True),
    ("Should the orion_bus key move?", True),
    ("Want the qwen38 profile?", True),
    ("So what's the plan for 44?", False),
    ("Is it the gameplay, which holds up thirty-odd years on, or the nostalgia?", False),
    ("Will you shine for it Tuesday at 7:30?", False),
    # Review 2026-11-15: prose that the first rule dropped.
    ("Are you up at 10pm again?", False),
    ("Was it the 2nd or the 3rd?", False),
    ("Still into 90s music?", False),
    ("Did the B12 help?", True),  # a vitamin, but shaped like an id: dropped, accepted
    ("Tea and/or sleep?", False),
    ("Is it a 24/7 thing now?", False),
])
def test_machinery_shape(text, machinery) -> None:
    assert looks_like_machinery(text) is machinery


def test_last_question_reads_the_last_paragraph_only() -> None:
    assert last_question("Is this a question?\n\nNo, it ended here.") is None
    assert last_question("First? Then this. And how are you? <3") == "And how are you?"
    # A closing line after the question is how Orion writes (both live threads, 2026-11-15).
    assert last_question("So what's the plan for 44? February's cold here.") == "So what's the plan for 44?"
    assert last_question('Did he say "why?"') == "Did he say 'why?"
    assert last_question("Any thoughts, e.g. what do you think?") == "Any thoughts, e.g. what do you think?"
    assert last_question("") is None


def test_chat_thread_dedupe_by_subject() -> None:
    (t, _) = chat_thread_topics(CHAT_FIXTURE, NOW, tz_name=TZ)
    record = choice_record([t], parse_pick("Pick: 0", 1, is_pass=is_pass_response), "outreach:w")
    said = fold_said_subjects([{"reason": "orion_declined", "choice": record}])
    assert filter_unsaid([t], said) == ([], ["chat_thread:c1"])


# --- the choice record and the bid --------------------------------------------------------------


def test_evidence_rides_on_the_decision_row_and_the_situation() -> None:
    (person,) = person_topics([_sighting("s-new", NOW - timedelta(hours=3))], NOW, tz_name=TZ)
    daydream = OutreachTopic(subject="daydream:d", kind="daydream", option_text="A daydream: x",
                             prompt_text="x", observed_at=NOW)
    shown = [daydream, person]
    record = choice_record(shown, parse_pick("Pick: 2\nmsg", 2, is_pass=is_pass_response), "outreach:w")
    assert "evidence_ref" not in record["shown"][0]
    assert record["shown"][1]["evidence_ref"] == "vision_individual_sighting:s-new"
    assert record["shown"][1]["kind"] == PERSON_KIND
    assert evidence_line(shown) == "2=vision_individual_sighting:s-new"
    assert evidence_line([daydream]) is None
    # The binding bid's object_text carries the person topic's text (#3336's seam).
    assert "Amanda (a camera match for a name Juniper gave) at the walkway" in (outreach_object_text(shown) or "")


# --- the sources (I/O seam) ---------------------------------------------------------------------


class _Conn:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows, self.calls = rows, []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self.rows


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def test_sources_read_their_sql_and_build_topics() -> None:
    conn = _Conn(CHAT_FIXTURE)
    topics = asyncio.run(chat_thread_source(lambda: _Pool(conn), tz_name=TZ, clock=lambda: NOW)())
    assert [t.subject for t in topics] == ["chat_thread:c1", "chat_thread:c3"]
    sql, args = conn.calls[0]
    assert sql == CHAT_SQL and args[0].tzinfo is None and args[1] == ["hub_orion", "hub_ws", "hub_http"]

    pconn = _Conn([_sighting("s-old", NOW - timedelta(days=9)), _sighting("s-new", NOW - timedelta(hours=3))])
    topics = asyncio.run(people_source(lambda: _Pool(pconn), tz_name=TZ, clock=lambda: NOW)())
    assert [t.subject for t in topics] == [f"person:{AMANDA}"]
    assert pconn.calls[0][0] == PEOPLE_SQL


def test_sources_never_raise() -> None:
    class _Broken:
        def acquire(self):
            raise RuntimeError("db down")

    assert asyncio.run(people_source(lambda: _Broken(), tz_name=TZ)()) == []
    assert asyncio.run(chat_thread_source(lambda: _Broken(), tz_name=TZ)()) == []
    assert asyncio.run(people_source(lambda: None, tz_name=TZ)()) == []
