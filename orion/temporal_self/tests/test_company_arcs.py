"""Seam S1: company arcs from vision_presence_transition rows (adapter, reducer, stance cue).

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09; the local day runs 06:00Z 10-09 to
06:00Z 10-10, so every expected clock below is hand-checkable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.temporal_self import STANCE_CUE_MAX_CHARS
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import advance_clock, drain_closed_days, fold, initial_state
from orion.temporal_self.frame import build_frame
from orion.temporal_self.stance_cue import project_stance_cue
from orion.temporal_self.tests.fixtures import T0, at

_N = [0]


def row(to: str, t: datetime, frm: str | None = "x", *, confirmed: bool = False, label: str | None = None,
        stream: str = "cam0") -> dict:
    _N[0] += 1
    return {"transition_id": f"tr{_N[0]:04d}-{to}", "presence_id": stream, "from_state": frm, "to_state": to,
            "occurred_at": t, "identity_confirmed": confirmed, "label": label}


def evs(rows):
    return S.adapt("presence_transition", rows)


def company(state):
    return sorted((a for a in state.arcs.values() if a.kind == "company"), key=lambda a: a.began_at)


# --------------------------------------------------------------------------- adapter


def test_adapter_drops_unconfirmed_labels_and_non_room_streams():
    e = S.presence_transition(row("present", T0, None, confirmed=False, label="juniper"))
    assert e.label == "" and e.payload["identity_confirmed"] is False
    assert e.subject_ref == "cam0" and e.verdict == "present" and e.payload["from_state"] is None
    e = S.presence_transition(row("present", T0, "absent", confirmed=True, label="juniper"))
    assert e.label == "juniper" and e.payload["identity_confirmed"] is True
    # Confirmed with no label is not a confirmed identity.
    assert S.presence_transition(row("present", T0, confirmed=True, label=None)).payload["identity_confirmed"] is False
    assert S.presence_transition(row("present", T0, stream="walkway")) is None
    assert S.presence_transition(row("dozing", T0)) is None


# --------------------------------------------------------------------------- reducer


def test_opens_on_present_recent_is_grace_closes_on_absent_at_last_sighting():
    rows = [
        row("absent", at(-10), None),
        row("present", at(0), "absent"),
        row("recent", at(30), "present"),
        row("present", at(31), "recent"),  # back within the grace period: same arc
        row("recent", at(90), "present"),
        row("absent", at(92), "recent"),  # grace ran out 2 min after the last sighting
    ]
    s = fold(initial_state(), events=evs(rows))
    (arc,) = company(s)
    assert arc.status == "closed" and arc.closed_reason == "process_ended"
    assert arc.began_at == at(0) and arc.ended_at == at(90)  # left at the recent row, not at absent
    assert arc.cumulative_dwell_sec == 90 * 60
    assert arc.subject_ref == "cam0" and arc.subject_label == ""
    assert len(arc.evidence_refs) == 5  # every row from present on
    assert "company" not in s.active


def test_unknown_closes_as_source_stale():
    s = fold(initial_state(), events=evs([row("present", at(0), None), row("unknown", at(40), "present")]))
    (arc,) = company(s)
    assert arc.closed_reason == "source_stale" and arc.ended_at == at(40)


def test_producer_restart_closes_the_open_arc_unobserved():
    # vision-window crashed while present (no unknown row); its first row after restart has from_state NULL.
    s = fold(initial_state(), events=evs([row("present", at(0), None), row("recent", at(10), "present"),
                                           row("present", at(60), None)]))
    first, second = company(s)
    assert first.closed_reason == "source_stale" and first.ended_at == at(10)
    assert second.status == "open" and second.began_at == at(60)
    assert s.active["company"] == second.arc_id


def test_a_lost_closing_row_never_merges_two_visits():
    # present, recent, (absent row lost), then a new visit whose from_state says the room was empty.
    s = fold(initial_state(), events=evs([row("present", at(0), None), row("recent", at(1), "present"),
                                           row("present", at(180), "absent")]))
    first, second = company(s)
    assert first.closed_reason == "process_ended" and first.ended_at == at(1) and first.cumulative_dwell_sec == 60
    assert second.status == "open" and second.began_at == at(180)
    # Same with a lost unknown row: the dark gap is never company.
    s = fold(initial_state(), events=evs([row("present", at(0), None), row("present", at(90), "unknown")]))
    first, second = company(s)
    assert first.closed_reason == "source_stale" and first.ended_at == at(0)


def test_absent_straight_from_present_closes_at_the_absent_row():
    s = fold(initial_state(), events=evs([row("present", at(0), None), row("absent", at(5), "present")]))
    (arc,) = company(s)
    assert arc.ended_at == at(5) and arc.closed_reason == "process_ended"


def test_label_only_from_a_confirmed_row():
    rows = [row("present", at(0), None, confirmed=False, label="someone-else"),  # unconfirmed: dropped
            row("present", at(1), "present", confirmed=True, label="juniper")]  # mid-stretch upgrade
    (arc,) = company(fold(initial_state(), events=evs(rows)))
    assert arc.subject_label == "juniper" and arc.status == "open"
    (arc,) = company(fold(initial_state(), events=evs([row("present", at(0), None)])))
    assert arc.subject_label == ""


def test_stray_rows_with_no_open_arc_do_nothing():
    s = fold(initial_state(), events=evs([row("recent", at(0), None), row("absent", at(2), "recent"),
                                           row("unknown", at(4), "absent")]))
    assert company(s) == []


def test_company_arcs_bind_no_context():
    from orion.temporal_self.tests.fixtures import metacog

    s = fold(initial_state(), events=evs([row("present", at(0), None), row("recent", at(30), "present")])
             + [metacog("m1", at(10))])
    (arc,) = company(s)
    assert arc.context_event_ids == []


def test_replay_identity_fold_twice_and_chunked():
    rows = [row("present", at(0), None), row("recent", at(20), "present"), row("present", at(21), "recent"),
            row("present", at(22), "present", confirmed=True, label="juniper"), row("unknown", at(50), "present"),
            row("present", at(70), "unknown"), row("absent", at(80), "present")]
    e = evs(rows)
    end = at(120)
    whole = advance_clock(fold(initial_state(), events=e), end)
    again = advance_clock(fold(fold(initial_state(), events=e), events=e), end)
    assert again.arcs == whole.arcs  # re-folding the same rows is a no-op
    s = initial_state()
    for lo in range(0, 120, 15):
        chunk = [x for x in e if at(lo) <= x.occurred_at < at(lo + 15)]
        s = advance_clock(fold(s, events=chunk), at(lo + 15))
    assert s.arcs == whole.arcs
    assert build_frame(s, end) == build_frame(whole, end)
    assert [a.closed_reason for a in company(whole)] == ["source_stale", "process_ended"]


def test_stretch_carries_over_midnight():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)  # 23:00 local 10-09
    midnight = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    s = fold(initial_state(), events=evs([row("present", late, None)]))
    s = advance_clock(s, midnight + timedelta(minutes=30))
    s, days = drain_closed_days(s)
    (old,) = [a for a in days[0].arcs if a.kind == "company"]
    assert old.closed_reason == "day_boundary" and old.ended_at == midnight and old.cumulative_dwell_sec == 3600
    (cont,) = company(s)
    assert cont.status == "open" and cont.carried_from_arc_id == old.arc_id and cont.began_at == midnight
    assert s.active["company"] == cont.arc_id
    s = fold(s, events=evs([row("absent", midnight + timedelta(minutes=40), "present")]))
    (cont,) = company(s)
    assert cont.status == "closed" and cont.ended_at == midnight + timedelta(minutes=40)


# --------------------------------------------------------------------------- stance cue


def _cue(rows, now):
    s = advance_clock(fold(initial_state(), events=evs(rows)), now)
    return project_stance_cue(build_frame(s, now), now)


def test_stance_cue_someone_when_unconfirmed():
    now = at(30)
    cue = _cue([row("present", at(0), None)], now)
    arc_id = cue.arc_ids[-1]
    assert f"Someone has been here since 09:00 [arc:{arc_id}]." in cue.text
    assert len(cue.text) <= STANCE_CUE_MAX_CHARS


def test_stance_cue_names_only_a_confirmed_person_and_never_as_a_label():
    now = at(30)
    cue = _cue([row("present", at(0), None, confirmed=True, label="juniper")], now)
    assert "Juniper has been here since 09:00 [arc:" in cue.text
    assert cue.labels == {}  # a person's name must not become a day-claim match label


def test_stance_cue_no_clause_after_close_and_before_midnight_wording():
    cue = _cue([row("present", at(0), None), row("absent", at(5), "present")], at(30))
    assert "been here" not in cue.text
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)
    now = datetime(2026, 10, 10, 6, 20, tzinfo=timezone.utc)
    cue = _cue([row("present", late, None)], now)
    assert "Someone has been here since before midnight [arc:" in cue.text


def test_company_clause_checks_as_true_against_its_own_arc():
    from orion.temporal_self.citation_truth import ArcFacts, check_decision

    now = at(30)
    s = advance_clock(fold(initial_state(), events=evs([row("present", at(0), None)])), now)
    (arc,) = company(s)
    facts = {arc.arc_id: ArcFacts.from_row(arc.model_dump(mode="json"))}
    good = f"Someone has been here since 09:00 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": good}, arcs=facts)
    assert [c.outcome for c in checks] == ["match"]
    wrong = f"Someone has been here since 07:30 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": wrong}, arcs=facts)
    assert [c.outcome for c in checks] == ["mismatch"]
