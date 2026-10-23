"""Strain loop part 1: strain arcs from the regulation step's arousal_transition rows (adapter,
reducer, stance cue, Day letter, reverie block, citation truth).

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09; the local day runs 06:00Z 10-09 to
06:00Z 10-10, so every expected clock below is hand-checkable. Rows are shaped exactly as
``services/orion-durable-runs/app/regulation_store.py`` writes them (level rows and cause rows).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.schemas.temporal_self import STANCE_CUE_MAX_CHARS
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import advance_clock, drain_closed_days, fold, initial_state
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.stance_cue import project_stance_cue
from orion.temporal_self.tests.fixtures import T0, at


def _reading(level: str, since: datetime, causes=(), observed: datetime | None = None) -> dict:
    return {"arousal_level": level, "since": since.isoformat(), "observed_at": (observed or since).isoformat(),
            "reasons": list(causes) + ["no_juniper_turn_45m"], "cabinet_temp_c": 34.6, "gpu_queue_depth": 11}


def level(frm: str | None, to: str, t: datetime, causes=()) -> dict:
    """A level row: ``transition_row`` (occurred_at = since = t)."""
    return {"event_id": f"arousal_transition:{t.isoformat()}:{to}", "occurred_at": t,
            "payload_json": json.dumps({"from": frm, "to": to, "reading": _reading(to, t, causes)})}


def cause(t: datetime, since: datetime, causes, joined=(), seed=False) -> dict:
    """A cause row: ``cause_row`` (occurred_at = observed time, same episode ``since``)."""
    return {"event_id": f"arousal_transition:{since.isoformat()}:strained:at:{t.isoformat()}", "occurred_at": t,
            "payload_json": {"from": "strained", "to": "strained", "joined": list(joined), "seed": seed,
                             "reading": _reading("strained", since, causes, observed=t)}}


def keep(since: datetime, lo: datetime, hi: datetime, causes=("gpu_queue_sustained",)) -> list[dict]:
    """The regulate node's keepalives every 30 min strictly inside (lo, hi)."""
    out, t = [], lo + timedelta(minutes=30)
    while t < hi:
        out.append(cause(t, since, causes))
        t += timedelta(minutes=30)
    return out


def evs(rows):
    return S.adapt("arousal_transition", rows)


def strains(state):
    return sorted((a for a in state.arcs.values() if a.kind == "strain"), key=lambda a: a.began_at)


GPU, HOT = "gpu_queue_sustained", "cabinet_hot"

# --------------------------------------------------------------------------- adapter


def test_adapter_keeps_the_rows_own_event_id_and_only_strain_causes():
    r = level("idle", "strained", T0, causes=[GPU])
    e = S.arousal_transition(r)
    # The chronicle stores folded events in the same table: a new id would duplicate the row forever.
    assert e.event_id == r["event_id"]
    assert e.source_kind == "arousal_transition" and e.source_table == "regulation"
    assert e.subject_ref == "orion:arousal" and e.verdict == "strained" and e.label == "strained"
    assert e.payload["causes"] == [GPU] and e.payload["since"] == T0.isoformat()
    assert S.arousal_transition(dict(r, event_id="immune_finding:x")) is None
    bad = dict(r, payload_json=json.dumps({"from": "idle", "to": "dazed", "reading": {}}))
    assert S.arousal_transition(bad) is None


# --------------------------------------------------------------------------- reducer


def test_opens_on_strained_causes_join_and_closes_on_idle():
    rows = [level("idle", "strained", at(0), [GPU]), cause(at(20), at(0), [HOT, GPU], joined=[HOT]),
            level("strained", "idle", at(45))]
    s = advance_clock(fold(initial_state(), events=evs(rows)), at(60))
    (arc,) = strains(s)
    assert arc.subject_label == f"{GPU}+{HOT}"  # in the order each cause joined
    assert arc.began_at == at(0) and arc.episode_began_at == at(0)
    assert arc.status == "closed" and arc.closed_reason == "process_ended" and arc.ended_at == at(45)
    assert len(arc.evidence_refs) == 3 and "strain" not in s.active


def test_unknown_closes_as_source_stale_and_stray_rows_do_nothing():
    s = fold(initial_state(), events=evs([level("idle", "engaged", at(-5)), level("engaged", "idle", at(-1)),
                                          level("idle", "strained", at(0), [GPU]),
                                          level("strained", "unknown", at(30))]))
    (arc,) = strains(s)
    assert arc.closed_reason == "source_stale" and arc.ended_at == at(30)


def test_a_new_episode_start_closes_the_old_arc_at_its_last_row():
    # Regulation restarted the episode (its last reading too old to carry): the gap was not seen.
    rows = [level("idle", "strained", at(0), [GPU]), cause(at(10), at(0), [GPU, HOT], joined=[HOT]),
            level("strained", "strained", at(200), [HOT])]
    s = fold(initial_state(), events=evs(rows))
    old, new = strains(s)
    assert old.closed_reason == "source_stale" and old.ended_at == at(10)
    assert new.status == "open" and new.episode_began_at == at(200) and new.subject_label == HOT
    assert s.active["strain"] == new.arc_id


def test_seed_row_opens_an_episode_whose_onset_predates_the_chronology():
    onset = at(-60 * 45)  # strained since two days earlier, before the chronicle's watermark
    s = fold(initial_state(), events=evs([cause(at(0), onset, [HOT, GPU], seed=True)]))
    (arc,) = strains(s)
    assert arc.began_at == at(0) and arc.episode_began_at == onset and arc.status == "open"
    assert arc.subject_label == f"{HOT}+{GPU}"


def test_strain_arcs_bind_no_context():
    """A strain episode is a body state, not a stretch of Orion's own process: nothing binds to it
    by time, even at an instant its recorded segment covers."""
    from orion.temporal_self.arcs import ReducerConfig, _covers

    s = fold(initial_state(), events=evs([level("idle", "strained", at(0), [GPU]),
                                          cause(at(20), at(0), [GPU, HOT], joined=[HOT])]))
    (arc,) = strains(s)
    assert arc.segments[-1].began_at <= at(10) <= arc.segments[-1].ended_at
    assert not _covers(s, arc, at(10), ReducerConfig())


def test_replay_identity_fold_twice_and_chunked():
    rows = [level("idle", "strained", at(0), [GPU]), cause(at(20), at(0), [HOT], joined=[HOT]),
            level("strained", "idle", at(50)), level("idle", "strained", at(70), [HOT]),
            level("strained", "unknown", at(100))]
    e = evs(rows)
    end = at(120)
    whole = advance_clock(fold(initial_state(), events=e), end)
    again = advance_clock(fold(fold(initial_state(), events=e), events=e), end)
    assert again.arcs == whole.arcs
    s = initial_state()
    for lo in range(0, 120, 15):
        chunk = [x for x in e if at(lo) <= x.occurred_at < at(lo + 15)]
        s = advance_clock(fold(s, events=chunk), at(lo + 15))
    assert s.arcs == whole.arcs
    assert build_frame(s, end) == build_frame(whole, end)
    assert [a.closed_reason for a in strains(whole)] == ["process_ended", "source_stale"]


def test_episode_carries_over_midnight_with_its_real_start():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)  # 23:00 local 10-09
    midnight = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    s = fold(initial_state(), events=evs([level("idle", "strained", late, [GPU]),
                                          *keep(late, late, midnight + timedelta(minutes=31))]))
    s = advance_clock(s, midnight + timedelta(minutes=30))
    s, days = drain_closed_days(s)
    (old,) = [a for a in days[0].arcs if a.kind == "strain"]
    assert old.closed_reason == "day_boundary" and old.ended_at == midnight
    (cont,) = strains(s)
    assert cont.status == "open" and cont.carried_from_arc_id == old.arc_id and cont.began_at == midnight
    assert cont.last_seen_at == midnight + timedelta(minutes=30)  # the newest keepalive: stale counts from it
    assert cont.episode_began_at == late and cont.subject_label == GPU and s.active["strain"] == cont.arc_id
    s = fold(s, events=evs([cause(midnight + timedelta(minutes=35), late, [GPU, HOT], joined=[HOT]),
                            level("strained", "idle", midnight + timedelta(minutes=40))]))
    (cont,) = strains(s)
    assert cont.status == "closed" and cont.ended_at == midnight + timedelta(minutes=40)
    assert cont.subject_label == f"{GPU}+{HOT}"


# --------------------------------------------------------------------------- readers


def _frame(rows, now):
    return build_frame(advance_clock(fold(initial_state(), events=evs(rows)), now), now)


def test_stance_cue_open_episode_names_causes_and_start():
    now = at(30)
    cue = project_stance_cue(_frame([level("idle", "strained", at(0), [GPU]),
                                     cause(at(20), at(0), [HOT, GPU], joined=[HOT])], now), now)
    arc_id = cue.arc_ids[-1]
    assert f"Strained since 09:00 (GPU jobs backed up, cabinet too hot) [arc:{arc_id}]." in cue.text
    assert len(cue.text) <= STANCE_CUE_MAX_CHARS
    assert cue.labels == {}  # the cause codes are not in the text, so they are no match label


def test_stance_cue_names_the_weekday_of_an_episode_begun_on_an_earlier_day():
    onset = datetime(2026, 10, 7, 18, 7, tzinfo=timezone.utc)  # Wed 12:07 local
    now = at(30)
    cue = project_stance_cue(_frame([cause(at(0), onset, [HOT], seed=True)], now), now)
    assert "Strained since Wed 12:07 (cabinet too hot) [arc:" in cue.text


def test_stance_cue_closed_episode_today_and_none_without_strain():
    now = at(90)
    cue = project_stance_cue(_frame([level("idle", "strained", at(0), [GPU]),
                                     level("strained", "idle", at(40))], now), now)
    assert "Was strained 09:00 to 09:40 (GPU jobs backed up) [arc:" in cue.text
    cue = project_stance_cue(_frame([level("idle", "engaged", at(0))], now), now)
    assert "trained" not in cue.text


def test_strain_clause_checks_as_true_against_its_own_arc():
    from orion.temporal_self.citation_truth import ArcFacts, check_decision

    onset = datetime(2026, 10, 7, 18, 7, tzinfo=timezone.utc)  # 12:07 local, two days earlier
    now = at(30)
    s = advance_clock(fold(initial_state(), events=evs([cause(at(0), onset, [HOT], seed=True)])), now)
    (arc,) = strains(s)
    facts = {arc.arc_id: ArcFacts.from_row(arc.model_dump(mode="json"))}
    good = f"Strained since Wed 12:07 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": good}, arcs=facts)
    assert [c.outcome for c in checks] == ["match"]
    wrong = f"Strained since Wed 07:30 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": wrong}, arcs=facts)
    assert [c.outcome for c in checks] == ["mismatch"]


def test_day_letter_and_reverie_block_name_the_episode():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)
    rows = [level("idle", "strained", at(0), [GPU]), level("strained", "idle", at(40)),
            level("idle", "strained", late, [HOT]),
            *keep(late, late, datetime(2026, 10, 10, 6, 31, tzinfo=timezone.utc), causes=(HOT,))]
    s = advance_clock(fold(initial_state(), events=evs(rows)), datetime(2026, 10, 10, 6, 30, tzinfo=timezone.utc))
    s, days = drain_closed_days(s)
    block = render_closed_day_block(days[0])
    assert "Strained: 09:00 to 09:40 (GPU jobs backed up) [arc:" in block.text
    assert "23:00 still strained at midnight (cabinet too hot) [arc:" in block.text

    now = at(90)
    recent = project_recent_arcs(_frame(rows[:2], now), now)
    assert "strained (GPU jobs backed up)" in recent.text


# --------------------------------------------------------------------------- review fixes


def test_keepalives_hold_an_episode_and_silence_closes_it_at_the_last_row():
    rows = [level("idle", "strained", at(0), [GPU]), *keep(at(0), at(0), at(100))]  # rows to 01:30
    s = advance_clock(fold(initial_state(), events=evs(rows)), at(150))
    (arc,) = strains(s)
    assert arc.status == "open" and arc.last_seen_at == at(90)
    # A lost closing row, a dead producer or the kind switched off: no row for 75 min.
    s = advance_clock(s, at(90 + 76))
    (arc,) = strains(s)
    assert arc.closed_reason == "source_stale" and arc.ended_at == at(90)
    assert "strain" not in s.active


def test_not_strained_after_an_outage_ends_the_arc_at_its_last_row():
    gap = level("strained", "idle", at(60))
    payload = json.loads(gap["payload_json"])
    payload["reading"]["reasons"] = ["prev_dropped:gap", "no_juniper_turn_45m"]
    gap["payload_json"] = json.dumps(payload)
    s = fold(initial_state(), events=evs([level("idle", "strained", at(0), [GPU]), cause(at(30), at(0), [GPU]), gap]))
    (arc,) = strains(s)
    assert arc.closed_reason == "source_stale" and arc.ended_at == at(30)


def test_citation_phrase_ignores_ordinary_strained():
    from orion.temporal_self.citation_truth import KIND_PHRASES

    strain = [p for p, k in KIND_PHRASES if k == "strain"]
    assert any(p.search("Was strained 09:00 to 09:40") for p in strain)
    assert any(p.search("Strained since Wed 12:07") for p in strain)
    assert not any(p.search("the GPU queue was strained by the batch") for p in strain)


# --------------------------------------------------------------------------- causes present now


def test_cue_names_a_cleared_cause_as_earlier():
    # The cabinet cools mid-episode: the label keeps every cause seen, the cue says what is true now.
    rows = [level("idle", "strained", at(0), [HOT, GPU]), cause(at(30), at(0), [GPU])]
    now = at(40)
    s = advance_clock(fold(initial_state(), events=evs(rows)), now)
    (arc,) = strains(s)
    assert arc.subject_label == f"{HOT}+{GPU}" and arc.causes_now == [GPU]
    cue = project_stance_cue(build_frame(s, now), now)
    assert "Strained since 09:00 (now: GPU jobs backed up; earlier: cabinet too hot) [arc:" in cue.text


def test_cue_says_settling_while_strain_only_holds():
    rows = [level("idle", "strained", at(0), [GPU]), cause(at(30), at(0), [])]
    now = at(40)
    cue = project_stance_cue(_frame(rows, now), now)
    assert "Strained since 09:00 (no cause right now, settling; earlier: GPU jobs backed up) [arc:" in cue.text


def test_cue_keeps_plain_wording_when_nothing_cleared_or_not_recorded():
    rows = [level("idle", "strained", at(0), [GPU]), cause(at(20), at(0), [GPU, HOT], joined=[HOT])]
    now = at(30)
    s = advance_clock(fold(initial_state(), events=evs(rows)), now)
    cue = project_stance_cue(build_frame(s, now), now)
    assert "Strained since 09:00 (GPU jobs backed up, cabinet too hot) [arc:" in cue.text
    # An arc folded before causes_now existed: never read as "no cause right now".
    (arc,) = strains(s)
    arc.causes_now = None
    cue = project_stance_cue(build_frame(s, now), now)
    assert "Strained since 09:00 (GPU jobs backed up, cabinet too hot) [arc:" in cue.text
    assert "settling" not in cue.text


def test_causes_now_carry_over_midnight():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)
    midnight = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    rows = [level("idle", "strained", late, [HOT, GPU]), cause(late + timedelta(minutes=30), late, [GPU]),
            *keep(late, late + timedelta(minutes=30), midnight + timedelta(minutes=31))]
    s = advance_clock(fold(initial_state(), events=evs(rows)), midnight + timedelta(minutes=31))
    (cont,) = strains(s)
    assert cont.carried_from_previous_day and cont.causes_now == [GPU]
