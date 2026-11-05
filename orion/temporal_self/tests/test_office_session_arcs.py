"""Office sessions label the office company arc (office camera design PR #3127 step B).

T0 is 09:00 local (America/Denver, MDT) on 2026-10-09. A visit: present at T0, recent at +8 min,
absent at +10 min; the session row is final at +19 min (VISION_OFFICE_SESSIONS_SETTLE_SEC = 660 s
after the visit ended).
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta

from orion.temporal_self import sources as S
from orion.temporal_self.arcs import advance_clock, event_available_at, fold, initial_state
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.stance_cue import project_stance_cue
from orion.temporal_self.tests.fixtures import T0, at

LABEL = ("Juniper (probably) in the office: sitting at a desk, opening a box; reaching into a cabinet; "
         "standing in a closet, reaching into a storage bin")


def presence(to: str, t: datetime, frm: str | None, n: int, *, confirmed: bool = False, label: str | None = None,
             stream: str = "cam0") -> dict:
    return {"transition_id": f"tr{n:03d}-{to}", "presence_id": stream, "from_state": frm, "to_state": to,
            "occurred_at": t, "identity_confirmed": confirmed, "label": label}


def visit(start: float = 0, *, confirmed: bool = False) -> list:
    return S.adapt("presence_transition", [
        presence("present", at(start), "absent", int(start * 10) + 1, confirmed=confirmed,
                 label="juniper" if confirmed else None),
        presence("recent", at(start + 8), "present", int(start * 10) + 2),
        presence("absent", at(start + 10), "recent", int(start * 10) + 3),
    ])


def session_row(start: float = 0, **over) -> dict:
    row = {"session_id": f"cam0-tr{int(start * 10) + 1:03d}-present", "stream_id": "cam0", "status": "closed",
           "started_at": at(start), "ended_at": at(start + 8), "closed_at": at(start + 19),
           "who_text": "Juniper (probably)", "label": LABEL, "unknown_people": 0, "ask_id": None}
    row.update(over)
    return row


def company(state):
    return sorted((a for a in state.arcs.values() if a.kind == "company"), key=lambda a: a.began_at)


# --------------------------------------------------------------------------- adapter


def test_adapter_reads_final_room_rows_only():
    e = S.office_session(session_row())
    assert e.source_kind == "office_session" and e.subject_ref == "cam0" and e.label == LABEL
    assert e.occurred_at == at(0) and e.ended_at == at(8)
    assert e.payload["who"] == "Juniper (probably)" and e.payload["place"] == "the office"
    assert e.payload["activities"].startswith("sitting at a desk, opening a box; reaching into a cabinet")
    # Available when the row became final, not when the visit ended.
    assert event_available_at(e) == at(19)
    assert S.office_session(session_row(status="open")) is None
    assert S.office_session(session_row(closed_at=None)) is None
    assert S.office_session(session_row(stream_id="walkway")) is None
    assert S.office_session(session_row(label="")) is None


def test_adapter_without_activities():
    e = S.office_session(session_row(label="Juniper (probably) in the office"))
    assert e.payload["activities"] == "" and e.label == "Juniper (probably) in the office"


# --------------------------------------------------------------------------- reducer


def test_session_labels_the_closed_company_arc_of_the_same_visit():
    s = fold(initial_state(), events=visit() + S.adapt("office_session", [session_row()]))
    (arc,) = company(s)
    assert arc.status == "closed" and arc.subject_label == LABEL
    assert "vision_office_session:cam0-tr001-present" in arc.related_refs
    # The presence rows stay the arc's evidence; the session is not one.
    assert all(r.startswith("vision_presence_transition:") for r in arc.evidence_refs)


def test_label_lands_in_a_later_fold_too_and_a_refold_is_a_no_op():
    s = fold(initial_state(), events=visit())
    (arc,) = company(s)
    assert arc.subject_label == ""
    sess = S.adapt("office_session", [session_row()])
    s2 = fold(s, events=sess)
    assert company(s2)[0].subject_label == LABEL
    s3 = fold(s2, events=sess)  # the same row read again: skipped by its order key
    assert s3.arcs == s2.arcs and s3.skipped_today == s2.skipped_today + 1


def test_two_visits_each_get_their_own_label():
    other = "Juniper (probably) and someone I don't know in the office: standing in a room, looking around"
    events = visit(0) + visit(60) + S.adapt("office_session", [
        session_row(0), session_row(60, label=other, who_text="Juniper (probably) and someone I don't know",
                                    unknown_people=1)])
    first, second = company(fold(initial_state(), events=events))
    assert first.subject_label == LABEL and second.subject_label == other


def test_no_overlapping_closed_arc_changes_nothing():
    # A session for a stretch the reducer never saw (no presence rows), and one whose arc is still open.
    s = fold(initial_state(), events=S.adapt("office_session", [session_row(200)]))
    assert company(s) == []
    rows = S.adapt("presence_transition", [presence("present", at(0), "absent", 1)])
    s = fold(initial_state(), events=rows + S.adapt("office_session", [session_row(0, closed_at=at(1))]))
    (arc,) = company(s)
    assert arc.status == "open" and arc.subject_label == ""


def test_a_confirmed_identity_keeps_its_name_and_gains_the_activities():
    s = fold(initial_state(), events=visit(confirmed=True) + S.adapt("office_session", [session_row()]))
    (arc,) = company(s)
    assert arc.subject_label.startswith("juniper in the office: sitting at a desk, opening a box")
    # With an unknown person in the visit, the session's own who wins over the confirmed name.
    who = "Juniper (probably) and someone I don't know"
    s = fold(initial_state(), events=visit(confirmed=True) + S.adapt("office_session", [
        session_row(label=f"{who} in the office", who_text=who, unknown_people=1)]))
    assert company(s)[0].subject_label == f"{who} in the office"


def test_other_cameras_arcs_are_never_labelled():
    rows = S.adapt("presence_transition", [presence("present", at(0), "absent", 1, stream="cam0"),
                                           presence("absent", at(8), "present", 2, stream="cam0")])
    sess = S.office_session(session_row())
    walk = sess.model_copy(update={"subject_ref": "cam9"})
    (arc,) = company(fold(initial_state(), events=rows + [walk]))
    assert arc.subject_label == ""


# --------------------------------------------------------------------------- readers


def test_recent_arcs_says_who_and_what_without_the_company_prefix():
    s = fold(initial_state(), events=visit() + S.adapt("office_session", [session_row()]))
    s = advance_clock(s, at(30))
    frame = build_frame(s, at(30))
    block = project_recent_arcs(frame, at(30))
    (line,) = [ln for ln in block.text.splitlines() if "office" in ln]
    assert line.startswith("- 09:00-09:08 (8 min) Juniper (probably) in the office: sitting at a desk")
    assert "company:" not in line


def test_stance_cue_open_company_clause_is_unchanged():
    # The session labels only CLOSED arcs, so "Someone has been here since" never reads a session label.
    rows = S.adapt("presence_transition", [presence("present", at(0), "absent", 1)])
    s = fold(initial_state(), events=rows)
    frame = build_frame(s, at(5))
    cue = project_stance_cue(frame, at(5))
    assert "Someone has been here since 09:00" in cue.text


def test_recent_arcs_keeps_the_old_wording_for_a_bare_confirmed_name():
    s = fold(initial_state(), events=visit(confirmed=True))
    frame = build_frame(advance_clock(s, at(30)), at(30))
    f = copy.deepcopy(frame)
    block = project_recent_arcs(f, at(30))
    assert "company: juniper" in block.text


def test_available_at_kinds_cover_the_store():
    from orion.temporal_self.arcs import AVAILABLE_AT_KINDS

    assert {"street_event", "office_session"} <= AVAILABLE_AT_KINDS
    assert timedelta(0) <= event_available_at(S.office_session(session_row())) - at(8)
