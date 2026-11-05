"""Office sessions (office camera design PR #3127 step B): pure rules.

The database cycle is in test_vision_office_sessions_pg.py (real Postgres). Household names other
than Juniper never appear in tracked files: "Sibling" is a placeholder household member.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(REPO), str(SERVICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_office_sessions as vos  # noqa: E402
from app.vision_office_sessions import Record, who_of_session  # noqa: E402

T0 = datetime(2026, 11, 5, 1, 24, 7, tzinfo=timezone.utc)   # 18:24 Denver (MST, UTC-7)
TZ = ZoneInfo("America/Denver")
PRIOR = {"Juniper": 0.97, "Sibling": 0.01}


def at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def tr(to: str, sec: float, frm, n: int, stream: str = "cam0") -> dict:
    return {"transition_id": f"t{n}", "presence_id": stream, "from_state": frm, "to_state": to, "occurred_at": at(sec)}


# --------------------------------------------------------------------------- spans


def test_one_visit_present_recent_absent_ends_at_the_last_sighting():
    (s,) = vos.build_spans([tr("absent", -600, None, 0), tr("present", 0, "absent", 1), tr("recent", 441, "present", 2),
                            tr("absent", 551, "recent", 3)])
    assert s.session_id == "cam0-t1" and s.started_at == at(0) and s.ended_at == at(441) and s.end_reason == "left"
    assert s.refs == ["t1", "t2", "t3"]


def test_back_within_the_grace_period_is_the_same_visit_and_keepalives_extend_it():
    (s,) = vos.build_spans([tr("present", 0, "absent", 1), tr("present", 300, "present", 2),
                            tr("recent", 400, "present", 3), tr("present", 420, "recent", 4),
                            tr("recent", 900, "present", 5), tr("absent", 1000, "recent", 6)])
    assert s.ended_at == at(900) and s.last_seen_at == at(900)


def test_open_visit_dark_camera_and_producer_restart():
    (s,) = vos.build_spans([tr("present", 0, "absent", 1)])
    assert s.ended_at is None and s.end_reason is None
    (s,) = vos.build_spans([tr("present", 0, "absent", 1), tr("unknown", 60, "present", 2)])
    assert s.ended_at == at(60) and s.end_reason == "camera_dark"
    a, b = vos.build_spans([tr("present", 0, "absent", 1), tr("recent", 30, "present", 2), tr("present", 600, None, 3)])
    assert a.ended_at == at(30) and a.end_reason == "producer_restart"
    assert b.started_at == at(600) and b.ended_at is None


def test_a_stretch_opened_before_the_rows_given_is_not_a_visit():
    assert vos.build_spans([tr("recent", 0, "present", 1), tr("absent", 100, "recent", 2)]) == []
    # A keepalive with no open visit opens nothing (its start is unknown).
    assert vos.build_spans([tr("present", 0, "present", 1), tr("recent", 50, "present", 2)]) == []


def test_a_visit_whose_producer_died_closes_stale_at_its_last_row():
    rows = [tr("present", 0, "absent", 1), tr("present", 600, "present", 2)]
    (s,) = vos.build_spans(rows, now=at(600 + 1499))
    assert s.ended_at is None
    (s,) = vos.build_spans(rows, now=at(600 + 1501))
    assert s.ended_at == at(600) and s.end_reason == "stale"


def test_one_crowded_frame_is_not_a_second_person():
    assert vos.sustained_peak([1, 1, 2, 1]) == 1
    assert vos.sustained_peak([1, 2, 2, 1]) == 2
    assert vos.sustained_peak([3, 2, 1]) == 2
    assert vos.sustained_peak([2]) == 2 and vos.sustained_peak([]) == 0


# --------------------------------------------------------------------------- timeline


def test_caption_phrase_drops_the_opener_only():
    assert vos.caption_phrase("The person in the image is sitting at a desk, opening a box.") == \
        "sitting at a desk, opening a box"
    assert vos.caption_phrase("A person is reaching into a cabinet.") == "reaching into a cabinet"
    assert vos.caption_phrase("The man appears to be typing.") == "typing"
    assert vos.caption_phrase("Two people are talking.") == "two people are talking"
    assert vos.caption_phrase("  ") == ""


def test_timeline_uses_the_frame_time_and_folds_consecutive_repeats_only():
    def cap(eid, sec, text, frame_sec=None):
        refs = [f"frame_ts:{at(frame_sec).isoformat()}"] if frame_sec is not None else []
        return {"event_id": eid, "narrative": text, "evidence_refs": refs, "created_at": at(sec)}

    tl = vos.build_timeline([
        cap("e3", 70, "The person in the image is sitting at a desk, working on a computer.", 60),
        cap("e1", 9, "The person in the image is organizing items in a storage bin.", 0),
        cap("e2", 40, "The person in the image is sitting at a desk, working on a computer.", 30),
        cap("e4", 95, "The person in the image is opening a cabinet door."),           # no frame_ts: written time
        cap("e5", 120, "The person in the image is sitting at a desk, working on a computer.", 110),
        cap("e6", 130, "", 125),
    ])
    assert [(e["text"], e["repeats"], e["event_id"]) for e in tl] == [
        ("organizing items in a storage bin", 1, "e1"),
        ("sitting at a desk, working on a computer", 2, "e2"),
        ("opening a cabinet door", 1, "e4"),
        ("sitting at a desk, working on a computer", 1, "e5"),   # not consecutive: kept
    ]
    assert tl[0]["at"] == at(0).isoformat() and tl[2]["at"] == at(95).isoformat()


def test_activities_fit_the_label_and_spread_over_the_visit():
    tl = [{"text": f"doing thing number {i}, with some detail, and appears to be busy"} for i in range(30)]
    text = vos.activities_text(tl, 120)
    assert len(text) <= 120 and text.startswith("doing thing number 0, with some detail")
    assert "doing thing number 29, with some detail" in text and text.endswith("more)")
    assert ", and appears" not in text   # the hedge clause is cut
    assert vos.activities_text([{"text": "a"}, {"text": "A"}, {"text": "b"}], 100) == "a; b"
    assert vos.activities_text([], 100) == ""


def test_label_stays_within_the_arc_label_cap():
    tl = [{"text": f"activity {i}, " + "x" * 80} for i in range(20)]
    label = vos.session_label("Juniper (probably)", "the office", tl)
    assert label.startswith("Juniper (probably) in the office: activity 0") and len(label) <= vos.LABEL_MAX
    assert vos.session_label("someone I don't know", "the office", []) == "Someone I don't know in the office"


# --------------------------------------------------------------------------- who


def rec(iid: str, guess=None, band="unsure", basis="prior", score=0.7, label=None, ruled=()) -> Record:
    return Record(individual_id=iid, label=label, guess=guess, band=band if guess else None,
                  basis=basis if guess else None, score=score if guess else None, ruled_out=frozenset(ruled))


def test_a_split_visit_of_one_person_reads_probably_juniper():
    # 11-05 live: 10 records for one 8-minute visit, every guess the prior's, one person in view.
    w = who_of_session([rec(f"r{i}", "Juniper") for i in range(10)], peak_people=1, prior=PRIOR)
    assert w["text"] == "Juniper (probably)" and w["unknown"] == 0
    assert w["people"][0]["basis"] == "prior" and len(w["people"][0]["individual_ids"]) == 10


def test_two_people_in_view_never_read_as_two_probably_juniper():
    w = who_of_session([rec("r1", "Juniper"), rec("r2", "Juniper", score=0.5)], peak_people=2, prior=PRIOR)
    assert w["text"] == "Juniper (probably) and someone I don't know"
    assert w["unknown"] == 1 and w["unknown_reason"] == "count"


def test_a_named_record_and_a_second_person():
    w = who_of_session([rec("r1", label="Juniper"), rec("r2", "Juniper")], peak_people=2, prior=PRIOR)
    assert w["text"] == "Juniper and someone I don't know"
    assert w["people"][0]["basis"] == "named" and sorted(w["people"][0]["individual_ids"]) == ["r1", "r2"]


def test_evidence_outside_the_household_is_a_stranger_and_not_also_juniper():
    w = who_of_session([rec("r1", "Neighbor", basis="evidence")], peak_people=1, prior=PRIOR)
    assert w["text"] == "someone I don't know" and w["unknown_reason"] == "evidence"
    assert w["stranger_ids"] == ["r1"]
    # The same stranger split into a second record that got the prior's guess: still one head.
    w = who_of_session([rec("r1", "Juniper"), rec("r2", "Neighbor", basis="evidence")], peak_people=1, prior=PRIOR)
    assert w["text"] == "someone I don't know" and w["unassigned_ids"] == ["r1"]
    # An album person who is not usually here is "maybe", not "someone I don't know".
    w = who_of_session([rec("r1", "Neighbor", basis="evidence")], peak_people=1, prior=PRIOR, known_names=["Neighbor"])
    assert w["text"] == "Neighbor (maybe)" and w["unknown"] == 0


def test_a_named_person_is_not_also_probably_juniper():
    # One head: the name Juniper gave explains it; the prior adds nobody.
    for other in (rec("r2", "Juniper"), rec("r2")):
        w = who_of_session([rec("r1", label="Sibling"), other], peak_people=1, prior=PRIOR)
        assert w["text"] == "Sibling" and w["unknown"] == 0
    w = who_of_session([rec("r1", label="Sibling"), rec("r2", "Juniper")], peak_people=2, prior=PRIOR)
    assert w["text"] == "Sibling and Juniper (probably)"


def test_household_member_by_score_and_a_confident_match():
    w = who_of_session([rec("r1", "Sibling", basis="evidence")], peak_people=1, prior=PRIOR)
    assert w["text"] == "Sibling (maybe)" and w["unknown"] == 0
    w = who_of_session([rec("r1", "Sibling", band="accept", basis="evidence"), rec("r2", "Juniper")],
                       peak_people=2, prior=PRIOR)
    assert w["text"] == "Sibling (matched) and Juniper (probably)" and w["unknown"] == 0


def test_no_record_at_all_is_the_prior_person():
    # Visits before the office crop pipeline (PR #3125): presence saw someone, no record exists.
    assert who_of_session([], peak_people=1, prior=PRIOR)["text"] == "Juniper (probably)"
    assert who_of_session([], peak_people=None, prior=PRIOR)["text"] == "Juniper (probably)"
    assert who_of_session([], peak_people=2, prior=PRIOR)["text"] == "Juniper (probably) and someone I don't know"


def test_a_record_with_no_guess_yet():
    assert who_of_session([rec("r1")], peak_people=1, prior=PRIOR)["text"] == "Juniper (probably)"
    # Juniper said this record is not her: the prior cannot cover it.
    w = who_of_session([rec("r1", ruled={"juniper"})], peak_people=1, prior=PRIOR)
    assert w["text"] == "someone I don't know"
    # A camera without a prior: no guess at all is someone unknown.
    assert who_of_session([rec("r1")], peak_people=1, prior={})["text"] == "someone I don't know"
    assert who_of_session([rec("r1")], peak_people=1, prior={"Juniper": 0.4})["text"] == "someone I don't know"


def test_stationary_people_do_not_count_as_heads():
    w = who_of_session([rec("r1", "Juniper")], peak_people=2, props=1, prior=PRIOR)
    assert w["text"] == "Juniper (probably)"


def test_three_strangers_in_view():
    w = who_of_session([rec("r1", "Juniper")], peak_people=4, prior=PRIOR)
    assert w["text"] == "Juniper (probably) and 3 people I don't know" and w["unknown"] == 3


# --------------------------------------------------------------------------- the ask


def test_ask_candidates_strangers_first_then_crowded_least_like_the_guess():
    recs = [rec("a", "Juniper", score=0.7), rec("b", "Juniper", score=0.5), rec("c", "Juniper", score=0.4),
            rec("n", label="Juniper"), rec("s", "Neighbor", basis="evidence")]
    w = who_of_session(recs, peak_people=2, prior=PRIOR)
    order = vos.ask_candidates(recs, w, {"a": True, "b": True, "c": False})
    assert order == ["s", "b", "a", "c"]   # never the named record


def test_ask_question_words():
    q = vos.office_ask_question(place="the office", started_at=T0, ended_at=at(441), tz=TZ,
                                others=["Juniper (probably)"], unknown=1)
    assert q == ("I think someone I don't recognize was in the office from 18:24 to 18:31. "
                 "Also there: Juniper (probably). Do you know who this is?")
    q = vos.office_ask_question(place="the office", started_at=T0, ended_at=None, tz=TZ, others=[], unknown=2)
    assert q == "I think 2 people I don't recognize are in the office since 18:24. Do you know who this is?"


# --------------------------------------------------------------------------- contracts


def test_place_words_agree_with_the_temporal_self():
    from orion.temporal_self.sources import ROOM_PLACE_WORDS, ROOM_STREAMS

    assert vos.PLACE_WORDS == ROOM_PLACE_WORDS and set(vos.PLACE_WORDS) <= ROOM_STREAMS


def test_the_street_rule_cap_skips_office_asks():
    from app.vision_individuals import ASKED_TODAY_SQL

    assert f"NOT (evidence_refs @> '[\"{vos.OFFICE_RULE_REF}\"]'::jsonb)" in ASKED_TODAY_SQL
    assert "evidence_refs @> CAST(:office AS jsonb)" in vos.OFFICE_ASKED_TODAY_SQL


def test_settings_ship_on_and_match_env_example():
    from app.settings import Settings

    env = (SERVICE / ".env_example").read_text()
    keys = {"VISION_OFFICE_SESSIONS_ENABLED": "true", "VISION_OFFICE_SESSIONS_INTERVAL_SEC": "60",
            "VISION_OFFICE_SESSIONS_STREAMS": "cam0", "VISION_OFFICE_SESSIONS_SETTLE_SEC": "660",
            "VISION_OFFICE_SESSIONS_LOOKBACK_HOURS": "24", "VISION_OFFICE_ASK_DAILY_CAP": "4"}
    for k, v in keys.items():
        assert f"\n{k}={v}\n" in env, k
    for k in keys:
        os.environ.pop(k, None)
    s = Settings()
    assert s.vision_office_sessions_enabled is True and s.vision_office_sessions_streams == "cam0"
    assert s.vision_office_sessions_settle_sec > s.vision_person_model_interval_sec
    from app.vision_office_sessions_loop import office_sessions_config

    cfg = office_sessions_config(s)
    assert cfg.streams == ("cam0",) and cfg.ask_daily_cap == 4
