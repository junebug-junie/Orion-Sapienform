"""Self-monitor organ patch 2: faculty arcs from the self_monitor node's faculty_status_changed rows
(adapter, reducer, midnight carry, staleness, Day letter, recent arcs, citation truth).

Rows are made by the producer's own builder (``orion.self_monitor.host.event_row``) and read back the
way the chronicle reads them (``event_id``, ``occurred_at``, ``payload_json``), so a drift between the
writer and the reducer fails here. T0 is 09:00 local (America/Denver) on 2026-10-09.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.self_monitor.host import FACULTY_KEEPALIVE_SEC, FacultyChange, event_row
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import ReducerConfig, _covers, advance_clock, drain_closed_days, fold, initial_state
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.tests.fixtures import T0, at

GRADE = "act.actions.journal.make.walkway_grade"


def row(frm, to, t: datetime, *, faculty: str = GRADE, since: datetime | None = None, keepalive=False,
        dropped=False) -> dict:
    r = event_row(FacultyChange(faculty, frm, to, since or t, keepalive=keepalive, dropped=dropped),
                  day_id="2026-10-09", now=t)
    return {"event_id": r["event_id"], "occurred_at": r["occurred_at"], "payload_json": r["payload"]}


def evs(rows):
    return S.adapt("faculty_status_changed", rows)


def faculties(state):
    return sorted((a for a in state.arcs.values() if a.kind == "faculty"), key=lambda a: (a.began_at, a.arc_id))


def test_adapter_keeps_the_rows_own_event_id():
    r = row("reliable", "broken", T0)
    (e,) = evs([r])
    assert e.event_id == r["event_id"] and e.source_table == "self_monitor"
    assert e.subject_ref == f"faculty:{GRADE}" and e.verdict == "broken" and e.payload["since"] == T0.isoformat()
    assert S.faculty_status_changed({**r, "event_id": "arousal_transition:x"}) is None
    bad = row("reliable", "nonsense-status", T0)
    assert S.faculty_status_changed(bad) is None


def test_broken_opens_an_arc_that_comes_back_when_reliable():
    s = fold(initial_state(), events=evs([row("reliable", "broken", at(0)), row("broken", "reliable", at(120))]))
    (arc,) = faculties(s)
    assert arc.subject_label == f"{GRADE} broken" and arc.episode_began_at == at(0)
    assert arc.status == "closed" and arc.closed_reason == "process_ended" and arc.ended_at == at(120)


def test_a_move_between_notable_statuses_closes_one_arc_and_opens_the_next():
    s = fold(initial_state(), events=evs([row(None, "unreliable", at(0)), row("unreliable", "broken", at(60)),
                                          row("broken", "unknown", at(90))]))
    a, b = faculties(s)
    assert (a.subject_label, a.ended_at, a.closed_reason) == (f"{GRADE} unreliable", at(60), "process_ended")
    assert (b.subject_label, b.began_at, b.ended_at) == (f"{GRADE} broken", at(60), at(90))


def test_quiet_rows_and_other_faculties_open_nothing():
    s = fold(initial_state(), events=evs([row(None, "reliable", at(0)), row("reliable", "unknown", at(10)),
                                          row(None, "miscalibrated", at(20), faculty="turn.self.steps")]))
    (arc,) = faculties(s)
    assert arc.subject_ref == "faculty:turn.self.steps"


def test_keepalives_hold_and_silence_closes_at_the_last_row():
    keep = timedelta(seconds=FACULTY_KEEPALIVE_SEC)
    rows = [row(None, "broken", at(0)), row("broken", "broken", at(0) + keep, since=at(0), keepalive=True)]
    s = fold(initial_state(), events=evs(rows))
    (arc,) = faculties(s)
    assert arc.status == "open" and arc.last_seen_at == at(0) + keep and len(arc.evidence_refs) == 2
    # The organ stopped: no row for faculty_stale_sec (30 h, so the arc was carried over midnight).
    s = advance_clock(s, at(0) + keep + timedelta(seconds=ReducerConfig().faculty_stale_sec + 60))
    (arc,) = faculties(s)
    assert arc.carried_from_previous_day and arc.last_seen_at == at(0) + keep
    # Closed at its last row, which predates this day's continuation: it ends where it began.
    assert arc.closed_reason == "source_stale" and arc.ended_at == arc.began_at


def test_a_repeated_open_row_extends_instead_of_opening_twice():
    s = fold(initial_state(), events=evs([row(None, "broken", at(0)), row(None, "broken", at(30))]))
    assert len(faculties(s)) == 1


def test_faculty_arcs_bind_no_context():
    s = fold(initial_state(), events=evs([row(None, "broken", at(0)), row("broken", "reliable", at(60))]))
    (arc,) = faculties(s)
    assert not _covers(s, arc, at(30), ReducerConfig())


def test_arc_carries_over_midnight_with_its_real_start_and_replays_identically():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)      # 23:00 local 10-09
    midnight = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    rows = [row(None, "unreliable", late), row("unreliable", "reliable", midnight + timedelta(hours=3))]
    e = evs(rows)
    s = advance_clock(fold(initial_state(), events=e[:1]), midnight + timedelta(minutes=30))
    s, days = drain_closed_days(s)
    (old,) = [a for a in days[0].arcs if a.kind == "faculty"]
    assert old.closed_reason == "day_boundary" and old.ended_at == midnight
    (cont,) = faculties(s)
    assert cont.carried_from_arc_id == old.arc_id and cont.episode_began_at == late and cont.status == "open"
    s = fold(s, events=e[1:])
    (cont,) = faculties(s)
    assert cont.closed_reason == "process_ended" and cont.ended_at == midnight + timedelta(hours=3)
    block = render_closed_day_block(days[0])
    # Plain name, then the id, then the status (Juniper 2026-11-03); the stored label is unchanged.
    assert (f"Track record: walkway-grade journal runs ({GRADE}) unreliable from 23:00 still so at midnight "
            f"[arc:{old.arc_id}].") in block.text
    assert old.subject_label == f"{GRADE} unreliable" and block.labels[old.subject_label] == [old.arc_id]
    # The citation gate matches what is shown: a letter naming the faculty by its id or its plain
    # name without [arc:] reads UNVERIFIED, and with it reads verified.
    from orion.temporal_self.citations import check_arc_citations

    shown = [k[len("arc:"):] for k in block.records if k.startswith("arc:")]
    for uncited in (f"My {GRADE} was unreliable all evening.", "My walkway-grade journal runs were unreliable."):
        _, check = check_arc_citations(uncited, shown, labels=block.labels)
        assert check.label == "UNVERIFIED", uncited
    _, check = check_arc_citations(f"My {GRADE} was unreliable [arc:{old.arc_id}].", shown, labels=block.labels)
    assert check.label == "verified"
    # Folding the same rows twice changes nothing.
    end = midnight + timedelta(hours=4)
    whole = advance_clock(fold(initial_state(), events=e), end)
    again = advance_clock(fold(fold(initial_state(), events=e), events=e), end)
    assert again.arcs == whole.arcs


def test_recent_arcs_and_citation_truth_name_the_stretch():
    from orion.temporal_self.citation_truth import ArcFacts, check_decision

    rows = [row(None, "broken", at(0)), row("broken", "reliable", at(40))]
    now = at(90)
    s = advance_clock(fold(initial_state(), events=evs(rows)), now)
    recent = project_recent_arcs(build_frame(s, now), now)
    assert f"track record: walkway-grade journal runs ({GRADE}) broken" in recent.text
    (arc,) = faculties(s)
    facts = {arc.arc_id: ArcFacts.from_row(arc.model_dump(mode="json"))}
    good = f"My track record says the walkway grade was broken from 09:00 to 09:40 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": good}, arcs=facts)
    assert [c.outcome for c in checks] == ["match"]
    wrong = f"My track record says the walkway grade was broken from 07:30 to 09:40 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": wrong}, arcs=facts)
    assert [c.outcome for c in checks] == ["mismatch"]


def test_faculty_arcs_never_push_orion_s_own_arcs_out_of_the_frame():
    from orion.schemas.temporal_self import ARCS_IN_FRAME_CAP

    rows = [row(None, "broken", at(i), faculty=f"f{i}") for i in range(ARCS_IN_FRAME_CAP + 5)]
    s = fold(initial_state(), events=evs(rows))
    frame = build_frame(s, at(200))
    assert len(frame.arcs_today) == ARCS_IN_FRAME_CAP and frame.arcs_today_total == ARCS_IN_FRAME_CAP + 5
